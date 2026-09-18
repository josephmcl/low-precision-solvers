#include "common/ozaki.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace ozaki {

/*  Compute type for the split products. LPS_OZAKI_COMPUTE selects:
      tf32 (default) -- 11-bit operands, fp32 accumulate  [shipped]
      fp32           -- 24-bit operands, fp32 accumulate  [diagnostic]
      emu            -- bf16x9 emulation, fp32 accumulate [diagnostic]
    The accumulator is fp32 in all three, so a difference between them isolates
    the operand-width ceiling from the accumulator one. */
static cublasComputeType_t ozaki_compute_type() {
    char const *v = std::getenv("LPS_OZAKI_COMPUTE");
    if (v != nullptr) {
        if (std::strcmp(v, "fp32") == 0) return CUBLAS_COMPUTE_32F;
        if (std::strcmp(v, "emu")  == 0) return CUBLAS_COMPUTE_32F_EMULATED_16BFX9;
    }
    return CUBLAS_COMPUTE_32F_FAST_TF32;
}


using harness::problem;

/*  ---- validation ------------------------------------------------------- */

bool validate(config const &cfg) {

    bool ok = true;

    if (cfg.bits > 11) {
        std::cout << "[ozaki] bits = " << cfg.bits
                  << " exceeds TF32's 11 significant bits; pieces are not"
                     " representable and accuracy degrades to plain TF32\n";
        ok = false;
    }
    if (cfg.n_groups > cfg.n_pieces) {
        std::cout << "[ozaki] n_groups = " << cfg.n_groups
                  << " exceeds n_pieces = " << cfg.n_pieces << "\n";
        ok = false;
    }

    /*  The bound is sufficient, not necessary, and whether it binds depends on
        the OPERANDS, not just the configuration. Measured at n=4096:

            operand                 bound 20   bound 28.6/29.6
            near-random A * X       exact      2.9e-06   <- binds hard
            L * U from a getrf      7.3e-16    7.3e-16   <- does not bind

        Triangular factors have a far narrower per-row dynamic range and their
        products carry mixed signs, so the running fp32 value stays well under
        2^24 even when the worst-case count says it should not. A random matrix
        does not get that.

        So this warns rather than fails: above the bound is legitimate and 2.4x
        faster for the R build, and wrong by four orders for a dense residual.
        Verify on YOUR operands — the sweep is cheap and the failure is silent
        (above the bound, adding pieces stops changing the answer, which is the
        signature to watch for). */
    double const bound = 2. * cfg.bits + std::log2(static_cast<double>(cfg.block));
    if (bound > 23.)
        std::cout << "[ozaki] note: 2*bits + log2(block) = " << bound
                  << " > 23, so fp32 accumulation is not provably exact."
                     " Whether it matters depends on the operands; verify"
                     " against a reference before relying on it.\n";

    return ok;
}

config from_environment() {

    config cfg;

    if (char const *v = std::getenv("LPS_OZ_BITS"))   cfg.bits     = std::atoi(v);
    if (char const *v = std::getenv("LPS_OZ_PIECES")) cfg.n_pieces = std::atoi(v);
    if (char const *v = std::getenv("LPS_OZ_BLOCK"))  cfg.block    = std::atoi(v);
    if (char const *v = std::getenv("LPS_OZ_MERGE"))  cfg.merge_tail = std::atoi(v);
    if (char const *v = std::getenv("LPS_OZ_TRIANGULAR"))
        cfg.triangular = std::atoi(v) != 0;
    if (char const *v = std::getenv("LPS_OZ_CONTRACTION_BOUND"))
        cfg.contraction_bound = std::atoi(v) != 0;
    cfg.n_groups = cfg.n_pieces;

    return cfg;
}

std::size_t bytes_of(format const f) {

    switch (f) {
        case format::fp32: return 4;
        case format::b24:  return 3;
        case format::bf16: return 2;
    }
    return 4;
}

/*  Read one element of a packed array as fp32. b24 keeps the top three bytes
    of the fp32 word, so unpacking is a shift; bf16 keeps the top two. */
template <format F>
__device__ __forceinline__ float read_packed(
    void const        *d_a,
    std::size_t const  idx) {

    if (F == format::fp32)
        return static_cast<float const *>(d_a)[idx];

    if (F == format::bf16)
        return __bfloat162float(
            static_cast<__nv_bfloat16 const *>(d_a)[idx]);

    unsigned char const *b =
        static_cast<unsigned char const *>(d_a) + 3 * idx;
    unsigned const bits =
        (static_cast<unsigned>(b[0]) << 8)  |
        (static_cast<unsigned>(b[1]) << 16) |
        (static_cast<unsigned>(b[2]) << 24);
    return __uint_as_float(bits);
}

/*  Round-to-nearest into the packed format. The bias added before truncation
    is what makes this rounding rather than truncation — truncating instead
    biases every element toward zero, which on a residual matrix is a
    systematic error rather than a random one. */
template <format F>
__global__ void compress_kernel(
    void              *d_out,
    float const       *d_in,
    std::size_t const  n_elements) {

    for (std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < n_elements;
         idx += static_cast<std::size_t>(blockDim.x) * gridDim.x) {

        float const v = d_in[idx];

        if (F == format::fp32) {
            static_cast<float *>(d_out)[idx] = v;
        }
        else if (F == format::bf16) {
            static_cast<__nv_bfloat16 *>(d_out)[idx] = __float2bfloat16(v);
        }
        else {
            unsigned bits = __float_as_uint(v);
            unsigned const lsb = (bits >> 8) & 1u;
            bits += 0x7Fu + lsb;                 /* round to nearest even */
            unsigned char *b =
                static_cast<unsigned char *>(d_out) + 3 * idx;
            b[0] = static_cast<unsigned char>((bits >> 8)  & 0xFFu);
            b[1] = static_cast<unsigned char>((bits >> 16) & 0xFFu);
            b[2] = static_cast<unsigned char>((bits >> 24) & 0xFFu);
        }
    }
}

template <format F>
__global__ void unpack_rows_kernel(
    float             *d_out,
    void const        *d_in,
    std::size_t const  n,
    std::size_t const  row_0,
    std::size_t const  n_rows) {

    std::size_t const total = n_rows * n;

    /*  i is the fast index of both source and destination (column-major on
        each side), so consecutive threads touch consecutive addresses. */
    for (std::size_t t = blockIdx.x * blockDim.x + threadIdx.x;
         t < total;
         t += static_cast<std::size_t>(blockDim.x) * gridDim.x) {

        std::size_t const i = t % n_rows;
        std::size_t const j = t / n_rows;
        d_out[i + j * n_rows] =
            read_packed<F>(d_in, (row_0 + i) + j * n);
    }
}

void unpack_rows(
    float             *d_out,
    void const        *d_in,
    format const       f,
    std::size_t const  n,
    std::size_t const  row_0,
    std::size_t const  n_rows,
    problem           &prob) {

    (void) prob;
    int const g = launch::grid_for(n_rows * n);

    switch (f) {
        case format::fp32:
            unpack_rows_kernel<format::fp32><<<g, launch::BLOCK_SIZE>>>(
                d_out, d_in, n, row_0, n_rows); break;
        case format::b24:
            unpack_rows_kernel<format::b24><<<g, launch::BLOCK_SIZE>>>(
                d_out, d_in, n, row_0, n_rows); break;
        case format::bf16:
            unpack_rows_kernel<format::bf16><<<g, launch::BLOCK_SIZE>>>(
                d_out, d_in, n, row_0, n_rows); break;
    }
    KERNEL_CHECK();
}

void compress(
    void              *d_out,
    float const       *d_in,
    std::size_t const  n_elements,
    format const       f,
    problem           &prob) {
    (void) prob;
    compress(d_out, d_in, n_elements, f);
}

void compress(
    void              *d_out,
    float const       *d_in,
    std::size_t const  n_elements,
    format const       f) {
    int const g = launch::grid_for(n_elements);

    switch (f) {
        case format::fp32:
            compress_kernel<format::fp32><<<g, launch::BLOCK_SIZE>>>(
                d_out, d_in, n_elements); break;
        case format::b24:
            compress_kernel<format::b24><<<g, launch::BLOCK_SIZE>>>(
                d_out, d_in, n_elements); break;
        case format::bf16:
            compress_kernel<format::bf16><<<g, launch::BLOCK_SIZE>>>(
                d_out, d_in, n_elements); break;
    }
    KERNEL_CHECK();
}

/*  ---- scale factors ---------------------------------------------------- */

/*  Largest |a| in each row of the requested triangle. One block per row: the
    block strides the row, reduces in shared memory, and writes one scale.

    Full rows, not per block, because the exponent grid must be constant along
    the summed index (see the header). */
template <format F>
__global__ void row_max_kernel(
    float             *d_mu,
    void const        *d_a,
    std::size_t const  n,
    std::size_t const  lda,
    int const          which) {

    __shared__ float s[launch::BLOCK_SIZE];

    std::size_t const i = blockIdx.x;

    float acc = 0.f;
    for (std::size_t j = threadIdx.x; j < n; j += blockDim.x) {

        /*  lower: unit-diagonal L, strictly below. upper: U, on and above. */
        if (which == 1 && j >= i) continue;
        if (which == 2 && j <  i) continue;

        acc = fmaxf(acc, fabsf(read_packed<F>(d_a, i + j * lda)));
    }

    s[threadIdx.x] = acc;
    __syncthreads();

    for (int q = launch::BLOCK_SIZE / 2; q > 0; q >>= 1) {
        if (static_cast<int>(threadIdx.x) < q)
            s[threadIdx.x] = fmaxf(s[threadIdx.x], s[threadIdx.x + q]);
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        /*  A unit diagonal contributes 1 to L's row scale even when the
            strictly-lower part is empty; without it row 0 gets scale 0 and
            every piece of it is zero. */
        float m = s[0];
        if (which == 1)
            m = fmaxf(m, 1.f);
        d_mu[i] = (m > 0.f)? m : 1.f;
    }
}

/*  Largest |x| in each column of the fp64 operand. */
__global__ void column_max_kernel(
    float             *d_nu,
    double const      *d_x,
    std::size_t const  n) {

    __shared__ double s[launch::BLOCK_SIZE];

    std::size_t const j = blockIdx.x;

    double acc = 0.;
    for (std::size_t i = threadIdx.x; i < n; i += blockDim.x)
        acc = fmax(acc, fabs(d_x[i + j * n]));

    s[threadIdx.x] = acc;
    __syncthreads();

    for (int q = launch::BLOCK_SIZE / 2; q > 0; q >>= 1) {
        if (static_cast<int>(threadIdx.x) < q)
            s[threadIdx.x] = fmax(s[threadIdx.x], s[threadIdx.x + q]);
        __syncthreads();
    }

    if (threadIdx.x == 0)
        d_nu[j] = (s[0] > 0.)? static_cast<float>(s[0]) : 1.f;
}

void row_max(
    float             *d_mu,
    void const        *d_a,
    format const       f,
    std::size_t const  n,
    std::size_t const  lda,
    shape const        which,
    problem           &prob) {
    (void) prob;
    row_max(d_mu, d_a, f, n, lda, which);
}

void row_max(
    float             *d_mu,
    void const        *d_a,
    format const       f,
    std::size_t const  n,
    std::size_t const  lda,
    shape const        which) {

    int const g = static_cast<int>(n);
    int const w = static_cast<int>(which);

    switch (f) {
        case format::fp32:
            row_max_kernel<format::fp32><<<g, launch::BLOCK_SIZE>>>(
                d_mu, d_a, n, lda, w); break;
        case format::b24:
            row_max_kernel<format::b24><<<g, launch::BLOCK_SIZE>>>(
                d_mu, d_a, n, lda, w); break;
        case format::bf16:
            row_max_kernel<format::bf16><<<g, launch::BLOCK_SIZE>>>(
                d_mu, d_a, n, lda, w); break;
    }
    KERNEL_CHECK();
}

void column_max(
    float             *d_nu,
    double const      *d_x,
    std::size_t const  n,
    std::size_t const  n_rhs,
    problem           &prob) {
    (void) prob;
    column_max(d_nu, d_x, n, n_rhs);
}

void column_max(
    float             *d_nu,
    double const      *d_x,
    std::size_t const  n,
    std::size_t const  n_rhs) {

    column_max_kernel<<<static_cast<int>(n_rhs), launch::BLOCK_SIZE>>>(
        d_nu,
        d_x,
        n);
    KERNEL_CHECK();
}

/*  ---- splitting -------------------------------------------------------- */

/*  Split one contraction block of A into n_pieces on the row's grid.

    Grid is (rows, block columns); one thread owns one element. The scale is
    per row, and threads in a block span rows, so the exp2f cannot be hoisted
    here — it can in the X split below, where the scale is per column, and
    that hoist is worth 3x. */
template <format F>
__global__ void split_a_kernel(
    float             *d_pieces,
    void const        *d_a,
    float const       *d_mu,
    std::size_t const  lda,
    std::size_t const  stride,
    std::size_t const  n_rows,
    std::size_t const  row_0,
    std::size_t const  col_0,
    std::size_t const  n_cols,
    int const          n_pieces,
    int const          bits,
    int const          which) {

    std::size_t const r = blockIdx.x * blockDim.x + threadIdx.x;
    std::size_t const c = blockIdx.y;

    if (r >= n_rows)
        return;

    /*  Slots are padded to the full block width, not truncated to n_cols, so
        every piece sits at the same stride and A_0..A_s can be handed to one
        GEMM as a single contiguous k-extent. The padding is zeroed, so it
        contributes nothing to the product. */
    if (c >= n_cols) {
        for (int p = 0; p != n_pieces; ++p)
            d_pieces[r + c * n_rows + static_cast<std::size_t>(p) * stride] = 0.f;
        return;
    }

    std::size_t const i = row_0 + r;
    std::size_t const j = col_0 + c;

    float a = 0.f;
    bool  live = true;
    if (which == 1 && j >= i) live = false;   /* strictly lower */
    if (which == 2 && j <  i) live = false;   /* on and above   */
    if (live)
        a = read_packed<F>(d_a, i + j * lda);

    int const e = ilogbf(d_mu[i]);

    for (int p = 0; p != n_pieces; ++p) {
        float const S = 1.5f * exp2f(static_cast<float>(e - bits * (p + 1) + 23));
        float const t = (a + S) - S;
        d_pieces[r + c * n_rows + static_cast<std::size_t>(p) * stride] = t;
        a -= t;
    }
}

/*  Split one contraction block of the fp64 operand into n_pieces on the
    column's grid.

    The scale depends only on the column, and blockIdx.y is the column, so the
    n_pieces exp2 calls are computed once per block into shared memory instead
    of once per element. That hoist alone was worth 3.1x when this kernel
    turned out to dominate the whole solve — it was larger than every
    tensor-core GEMM combined, which is why the profile came before the
    optimization. */
__global__ void split_x_kernel(
    float             *d_pieces,
    double const      *d_x,
    float const       *d_nu,
    std::size_t const  n,
    std::size_t const  stride,
    std::size_t const  row_0,
    std::size_t const  n_rows,
    int const          n_pieces,
    int const          bits) {

    extern __shared__ double s_scale[];

    std::size_t const c = blockIdx.y;

    if (threadIdx.x == 0) {
        int const e = ilogbf(d_nu[c]);
        for (int p = 0; p != n_pieces; ++p)
            s_scale[p] = 1.5 * exp2(
                static_cast<double>(e - bits * (p + 1) + 52));
    }
    __syncthreads();

    std::size_t const r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_rows)
        return;

    /*  The split constant carries the fp64 mantissa width, 2^(g+52). Using
        the fp32 constant 2^(g+23) here is a silent four-order accuracy loss:
        it rounds the iterate to fp32 before splitting, which reinjects 2^-24
        into a residual whose whole purpose is to be finer than that. */
    double x = d_x[(row_0 + r) + c * n];

    for (int p = 0; p != n_pieces; ++p) {
        double const S = s_scale[p];
        double const t = (x + S) - S;
        d_pieces[r + c * n_rows + static_cast<std::size_t>(p) * stride] =
            static_cast<float>(t);
        x -= t;
    }
}

/*  Add an fp32 group partial into the fp64 accumulator. */
__global__ void accumulate_kernel(
    double            *d_acc,
    float const       *d_partial,
    std::size_t const  n_acc,
    std::size_t const  row_0,
    std::size_t const  n_rows,
    std::size_t const  n_cols) {

    std::size_t const total = n_rows * n_cols;

    for (std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += static_cast<std::size_t>(blockDim.x) * gridDim.x) {

        std::size_t const r = idx % n_rows;
        std::size_t const c = idx / n_rows;
        d_acc[(row_0 + r) + c * n_acc] +=
            static_cast<double>(d_partial[r + c * n_rows]);
    }
}

/*  ---- workspace -------------------------------------------------------- */

workspace::workspace(
    std::size_t const  n,
    std::size_t const  n_rhs,
    config const      &cfg,
    problem           &prob)
    : workspace(n, n_rhs, cfg) {
    (void) prob;
}

workspace::workspace(
    std::size_t const  n,
    std::size_t const  n_rhs,
    config const      &cfg)
    : cfg(cfg) {

    std::size_t const b  = static_cast<std::size_t>(cfg.block);
    std::size_t const np = static_cast<std::size_t>(cfg.n_pieces);

    d_pieces_a = static_cast<float *>(_acquire(n * b * np * sizeof(float)));
    d_pieces_x = static_cast<float *>(_acquire(b * n_rhs * np * sizeof(float)));

    /*  Only when asked: half the bytes of the fp32 arrays, so the transient
        footprint the `block` knob bounds does not grow by enabling this. */
    if (cfg.fp16_pieces) {
        d_pieces_a16 = _acquire(n * b * np * sizeof(__half));
        d_pieces_x16 = _acquire(b * n_rhs * np * sizeof(__half));
        d_rowscale   = static_cast<double *>(_acquire(n * sizeof(double)));
        d_colscale   = static_cast<double *>(_acquire(n_rhs * sizeof(double)));
    }
    d_partial  = static_cast<float *>(_acquire(n * n_rhs * sizeof(float)));
    d_mu       = static_cast<float *>(_acquire(n * sizeof(float)));
    d_nu       = static_cast<float *>(_acquire(n_rhs * sizeof(float)));
}

workspace::~workspace() {

    for (std::size_t i = 0; i != _d_owned.size(); ++i)
        CUDA_CHECK(cudaFree(_d_owned[i]));
}

void *workspace::_acquire(std::size_t const bytes) {

    void *d_p = nullptr;
    if (!CUDA_CHECK(cudaMalloc(&d_p, bytes)))
        return nullptr;

    _d_owned.push_back(d_p);
    _bytes += bytes;
    return d_p;
}

/*  ---- the product ------------------------------------------------------ */

/*  Stop the pipeline early, for phase timing.

    0 = full, 1 = splits only, 2 = splits + GEMMs (no fp64 fold). Differencing
    the three run times gives the split / GEMM / accumulate breakdown without
    per-kernel events, which would serialize the stream and distort exactly the
    overlap being measured. Set LPS_OZ_STOP. */
static int stop_after() {

    static int const value = []{
        char const *v = std::getenv("LPS_OZ_STOP");
        return (v == nullptr)? 0 : std::atoi(v);
    }();
    return value;
}

/*  fp16 piece splitters.

    Identical arithmetic to the fp32 splitters above -- the SAME add-and-
    subtract-S extraction, the same pieces -- differing only in that each piece
    is written pre-scaled by 2^(bits*p - e) and stored as __half.

    That scaling is what makes fp16 usable at all: piece p naturally sits near
    2^(e - bits*p), which at bits=6, p=8 is 2^-48 below the row scale and far
    under fp16's 2^-14 normal floor. Pre-scaling puts every piece at O(1), so
    the exponent range stops mattering entirely.

    It is LOSSLESS. A piece is a multiple of 2^(e - bits*(p+1)) bounded by
    2^(e - bits*p), hence at most `bits` significant bits (6 or 9 here) against
    fp16's 11. Multiplying by a power of two is exact. The scale is restored
    downstream: 2^(-bits*s) by the GEMM's alpha, 2^(e_i) and 2^(f_j) by the
    fp64 fold.                                                                */
template <format F>
__global__ void split_a_kernel_fp16(
    __half            *d_pieces,
    void const        *d_a,
    float const       *d_mu,
    std::size_t const  lda,
    std::size_t const  stride,
    std::size_t const  n_rows,
    std::size_t const  row_0,
    std::size_t const  col_0,
    std::size_t const  n_cols,
    int const          n_pieces,
    int const          bits,
    int const          which) {

    std::size_t const r = blockIdx.x * blockDim.x + threadIdx.x;
    std::size_t const c = blockIdx.y;

    if (r >= n_rows)
        return;

    if (c >= n_cols) {
        for (int p = 0; p != n_pieces; ++p)
            d_pieces[r + c * n_rows + static_cast<std::size_t>(p) * stride] =
                __float2half(0.f);
        return;
    }

    std::size_t const i = row_0 + r;
    std::size_t const j = col_0 + c;

    float a = 0.f;
    bool  live = true;
    if (which == 1 && j >= i) live = false;
    if (which == 2 && j <  i) live = false;
    if (live)
        a = read_packed<F>(d_a, i + j * lda);

    int const e = ilogbf(d_mu[i]);

    for (int p = 0; p != n_pieces; ++p) {
        float const S = 1.5f * exp2f(static_cast<float>(e - bits * (p + 1) + 23));
        float const t = (a + S) - S;
        /*  exact: a power-of-two rescale of a value with <= bits mantissa */
        float const h = t * exp2f(static_cast<float>(bits * p - e));
        d_pieces[r + c * n_rows + static_cast<std::size_t>(p) * stride] =
            __float2half(h);
        a -= t;
    }
}

__global__ void split_x_kernel_fp16(
    __half            *d_pieces,
    double const      *d_x,
    float const       *d_nu,
    std::size_t const  n,
    std::size_t const  stride,
    std::size_t const  row_0,
    std::size_t const  n_rows,
    int const          n_pieces,
    int const          bits) {

    extern __shared__ double s_scale16[];

    std::size_t const c = blockIdx.y;
    int const f = ilogbf(d_nu[c]);

    if (threadIdx.x == 0)
        for (int p = 0; p != n_pieces; ++p)
            s_scale16[p] = 1.5 * exp2(
                static_cast<double>(f - bits * (p + 1) + 52));
    __syncthreads();

    std::size_t const r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n_rows)
        return;

    double x = d_x[(row_0 + r) + c * n];

    for (int p = 0; p != n_pieces; ++p) {
        double const S = s_scale16[p];
        double const t = (x + S) - S;
        double const h = t * exp2(static_cast<double>(bits * p - f));
        d_pieces[r + c * n_rows + static_cast<std::size_t>(p) * stride] =
            __float2half(static_cast<float>(h));
        x -= t;
    }
}

/*  Fill 2^ilogbf(v[i]) for a scale vector. One pass over n elements, once per
    accumulate_product call, replacing the same transcendentals recomputed per
    OUTPUT ELEMENT per group per block. */
__global__ void exponent_scale_kernel(
    double            *d_out,
    float const       *d_in,
    std::size_t const  n_elements) {

    for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n_elements;
         i += static_cast<std::size_t>(blockDim.x) * gridDim.x)
        d_out[i] = exp2(static_cast<double>(ilogbf(d_in[i])));
}

/*  Fold for the fp16 path: restores the per-row and per-column exponents the
    splitters divided out. The per-GROUP factor 2^(-bits*s) is already in the
    partial, applied as the GEMM's alpha.

    Two loads and two multiplies. The earlier version recomputed
    exp2(ilogbf(mu)) and exp2(ilogbf(nu)) HERE, per element, which tripled the
    fold and inverted the optimization on fast-tf32 parts. */
__global__ void accumulate_kernel_scaled(
    double            *d_acc,
    float const       *d_partial,
    double const      *d_rowscale,
    double const      *d_colscale,
    std::size_t const  n_acc,
    std::size_t const  row_0,
    std::size_t const  n_rows,
    std::size_t const  n_cols) {

    std::size_t const total = n_rows * n_cols;

    for (std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total;
         idx += static_cast<std::size_t>(blockDim.x) * gridDim.x) {

        std::size_t const r = idx % n_rows;
        std::size_t const c = idx / n_rows;
        d_acc[(row_0 + r) + c * n_acc] +=
            static_cast<double>(d_partial[r + c * n_rows])
            * d_rowscale[row_0 + r] * d_colscale[c];
    }
}

void accumulate_product(
    double            *d_acc,
    void const        *d_a,
    double const      *d_x,
    std::size_t const  lda,
    std::size_t const  n_rhs,
    shape const        which,
    workspace         &ws,
    problem           &prob,
    std::size_t const  contraction_limit,
    format const       a_format) {

    accumulate_product(d_acc, d_a, d_x, lda, n_rhs, which, ws,
                       prob.blas, prob.n, contraction_limit, a_format);
}

void accumulate_product(
    double            *d_acc,
    void const        *d_a,
    double const      *d_x,
    std::size_t const  lda,
    std::size_t const  n_rhs,
    shape const        which,
    workspace         &ws,
    cublasHandle_t const blas,
    std::size_t const  n_dim,
    std::size_t const  contraction_limit,
    format const       a_format) {

    std::size_t const n  = n_dim;
    config     const &cfg = ws.cfg;

    std::size_t const k_end =
        (!cfg.contraction_bound || contraction_limit == 0 ||
         contraction_limit > n)? n : contraction_limit;

    shape const eff = cfg.triangular? which : shape::full;

    std::size_t const b  = static_cast<std::size_t>(cfg.block);
    std::size_t const np = static_cast<std::size_t>(cfg.n_pieces);
    /*  Pieces for the second operand; see ozaki.h. Defaults to n_pieces. */
    int const np_x = (cfg.n_pieces_x > 0)? cfg.n_pieces_x : cfg.n_pieces;

    float const one = 1.f, zero = 0.f;

    /*  fp16 path: the row and column exponents are constant for the whole
        call, so they are computed ONCE here rather than per output element in
        every fold. That distinction is not cosmetic -- doing it in the fold
        tripled the fold cost (4.8 -> 15.0 ms measured) and was enough to make
        the entire fp16 cascade a net LOSS on parts with fast tf32. */
    if (cfg.fp16_pieces && ws.d_rowscale != nullptr) {
        exponent_scale_kernel<<<launch::grid_for(n), launch::BLOCK_SIZE>>>(
            ws.d_rowscale, ws.d_mu, n);
        exponent_scale_kernel<<<launch::grid_for(n_rhs), launch::BLOCK_SIZE>>>(
            ws.d_colscale, ws.d_nu, n_rhs);
        KERNEL_CHECK();
    }

    for (std::size_t c_0 = 0; c_0 < k_end; c_0 += b) {

        std::size_t const n_c = (k_end - c_0 < b)? k_end - c_0 : b;

        /*  Rows of the output this contraction block can touch. For L
            (strictly lower) columns c only feed rows > c, so rows below c_0
            are structurally zero; for U they feed rows < c_0 + n_c. Skipping
            them is where the triangular advantage comes from. */
        std::size_t row_0  = 0;
        std::size_t n_rows = n;
        if (eff == shape::lower) {
            row_0  = c_0;
            n_rows = n - c_0;
        }
        else if (eff == shape::upper) {
            row_0  = 0;
            n_rows = c_0 + n_c;
        }

        /*  Both operands now use full-block-width slots, so a group's pieces
            are contiguous and can be issued as one GEMM. */
        std::size_t const stride_a = n_rows * b;
        std::size_t const stride_x = b * n_rhs;

        dim3 const grid_a(
            static_cast<unsigned>((n_rows + launch::BLOCK_SIZE - 1) /
                                  launch::BLOCK_SIZE),
            static_cast<unsigned>(n_c));
        bool const f16 = cfg.fp16_pieces && ws.d_pieces_a16 != nullptr;
        /*  One-shot, to stderr: whether the fp16 path is actually live. A
            switch that silently does nothing is the failure mode this project
            has hit repeatedly with vendor emulation flags. */
        if (std::getenv("LPS_OZAKI_VERBOSE") != nullptr) {
            static int said = 0;
            if (said < 4) {
                ++said;
                std::fprintf(stderr,
                    "[ozaki] fp16_pieces=%d buf=%p -> f16 %s (bits=%d "
                    "pieces=%d block=%d)\n",
                    int(cfg.fp16_pieces), ws.d_pieces_a16,
                    f16 ? "LIVE" : "OFF", cfg.bits, cfg.n_pieces, cfg.block);
            }
        }
        __half *const pa16 = static_cast<__half *>(ws.d_pieces_a16);
        __half *const px16 = static_cast<__half *>(ws.d_pieces_x16);

        if (f16) {
            switch (a_format) {
                case format::fp32:
                    split_a_kernel_fp16<format::fp32><<<grid_a, launch::BLOCK_SIZE>>>(
                        pa16, d_a, ws.d_mu, lda, stride_a, n_rows, row_0, c_0,
                        n_c, cfg.n_pieces, cfg.bits, static_cast<int>(which)); break;
                case format::b24:
                    split_a_kernel_fp16<format::b24><<<grid_a, launch::BLOCK_SIZE>>>(
                        pa16, d_a, ws.d_mu, lda, stride_a, n_rows, row_0, c_0,
                        n_c, cfg.n_pieces, cfg.bits, static_cast<int>(which)); break;
                case format::bf16:
                    split_a_kernel_fp16<format::bf16><<<grid_a, launch::BLOCK_SIZE>>>(
                        pa16, d_a, ws.d_mu, lda, stride_a, n_rows, row_0, c_0,
                        n_c, cfg.n_pieces, cfg.bits, static_cast<int>(which)); break;
            }
            KERNEL_CHECK();
        }
        else
        switch (a_format) {
            case format::fp32:
                split_a_kernel<format::fp32><<<grid_a, launch::BLOCK_SIZE>>>(
                    ws.d_pieces_a, d_a, ws.d_mu, lda, stride_a, n_rows,
                    row_0, c_0, n_c, cfg.n_pieces, cfg.bits,
                    static_cast<int>(which)); break;
            case format::b24:
                split_a_kernel<format::b24><<<grid_a, launch::BLOCK_SIZE>>>(
                    ws.d_pieces_a, d_a, ws.d_mu, lda, stride_a, n_rows,
                    row_0, c_0, n_c, cfg.n_pieces, cfg.bits,
                    static_cast<int>(which)); break;
            case format::bf16:
                split_a_kernel<format::bf16><<<grid_a, launch::BLOCK_SIZE>>>(
                    ws.d_pieces_a, d_a, ws.d_mu, lda, stride_a, n_rows,
                    row_0, c_0, n_c, cfg.n_pieces, cfg.bits,
                    static_cast<int>(which)); break;
        }
        KERNEL_CHECK();

        dim3 const grid_x(
            static_cast<unsigned>((n_c + launch::BLOCK_SIZE - 1) /
                                  launch::BLOCK_SIZE),
            static_cast<unsigned>(n_rhs));
        if (f16) {
            split_x_kernel_fp16<<<grid_x, launch::BLOCK_SIZE,
                                  np * sizeof(double)>>>(
                px16, d_x, ws.d_nu, n, stride_x, c_0, n_c, np_x, cfg.bits);
            KERNEL_CHECK();
        }
        else
        split_x_kernel<<<grid_x, launch::BLOCK_SIZE,
                         np * sizeof(double)>>>(
            ws.d_pieces_x,
            d_x,
            ws.d_nu,
            n,
            stride_x,
            c_0,
            n_c,
            np_x,
            cfg.bits);
        KERNEL_CHECK();

        if (stop_after() == 1)
            continue;

        /*  Scale-grouped accumulation. Products with p+q = s share an
            exponent grid, so cuBLAS folds them into d_partial with beta=1
            exactly; only one fp64 pass per group is then needed instead of
            one per product. That is 21 folds down to n_groups. */
        /*  Groups below merge_tail each get their own fp64 fold; groups from
            merge_tail up accumulate into the same fp32 partial and fold once
            at the end. */
        int const tail = (cfg.merge_tail < 0)? cfg.n_groups
                       : (cfg.merge_tail > cfg.n_groups)? cfg.n_groups
                       : cfg.merge_tail;
        bool tail_open = false;

        for (int s = 0; s != cfg.n_groups; ++s) {

            bool const merging = (s >= tail);

            /*  Reset the partial only when starting a fold, i.e. at every
                group below the tail, and once at the tail's first group. */
            bool first = merging? !tail_open : true;

            for (int p = 0; p <= s; ++p) {

                int const q = s - p;
                if (p >= cfg.n_pieces || q >= np_x)
                    continue;

                float const beta = first? zero : one;

                /*  fp16 pieces were stored pre-scaled by 2^(bits*p) each, so
                    their product carries 2^(bits*s) too much. s is shared by
                    every product in the group, so one scalar alpha removes it
                    exactly -- no per-element work, and the group's beta=1
                    folding still accumulates correctly because each product
                    is scaled before it is added. */
                float const alpha = f16
                    ? exp2f(-static_cast<float>(cfg.bits * s))
                    : one;

                CUBLAS_CHECK(cublasGemmEx(
                    blas,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    static_cast<int>(n_rows),
                    static_cast<int>(n_rhs),
                    static_cast<int>(n_c),
                    &alpha,
                    f16 ? static_cast<void const *>(
                              pa16 + static_cast<std::size_t>(p) * stride_a)
                        : static_cast<void const *>(
                              ws.d_pieces_a + static_cast<std::size_t>(p) * stride_a),
                    f16 ? CUDA_R_16F : CUDA_R_32F, static_cast<int>(n_rows),
                    f16 ? static_cast<void const *>(
                              px16 + static_cast<std::size_t>(q) * stride_x)
                        : static_cast<void const *>(
                              ws.d_pieces_x + static_cast<std::size_t>(q) * stride_x),
                    f16 ? CUDA_R_16F : CUDA_R_32F, static_cast<int>(n_c),
                    &beta,
                    ws.d_partial, CUDA_R_32F, static_cast<int>(n_rows),
                    /*  WHICH CEILING BINDS THE PIECE COUNT?

                        Ozaki needs L*U to ~48-54 bits and gets 2*bits per
                        piece-pair. `bits` is capped twice: by the OPERAND
                        width (tf32 carries 11 significand bits) and by the
                        ACCUMULATOR width (2*bits + log2(K) <= 24, fp32's
                        mantissa). Which binds decides whether this device's
                        faster low-precision units can help at all -- fp16 has
                        the same 11 bits and the same fp32 accumulator, so it
                        could only make products faster, never fewer.

                        Swappable so the two can be told apart: plain 32F
                        gives 24-bit operands with the SAME accumulator. If
                        accuracy does not move, the accumulator binds and no
                        wider operand format reduces the piece count. */
                    f16 ? CUBLAS_COMPUTE_32F : ozaki_compute_type(),
                    CUBLAS_GEMM_DEFAULT));

                first = false;
            }

            if (merging) {
                tail_open = true;
                continue;   /* fold after the last group */
            }

            if (stop_after() == 2)
                continue;

            if (!first) {
                if (f16)
                    accumulate_kernel_scaled<<<launch::grid_for(n_rows * n_rhs),
                                               launch::BLOCK_SIZE>>>(
                        d_acc, ws.d_partial, ws.d_rowscale, ws.d_colscale,
                        n, row_0, n_rows, n_rhs);
                else
                    accumulate_kernel<<<launch::grid_for(n_rows * n_rhs),
                                        launch::BLOCK_SIZE>>>(
                        d_acc, ws.d_partial, n, row_0, n_rows, n_rhs);
                KERNEL_CHECK();
            }
        }

        if (tail_open && stop_after() != 2) {
            if (f16)
                accumulate_kernel_scaled<<<launch::grid_for(n_rows * n_rhs),
                                           launch::BLOCK_SIZE>>>(
                    d_acc, ws.d_partial, ws.d_rowscale, ws.d_colscale,
                    n, row_0, n_rows, n_rhs);
            else
                accumulate_kernel<<<launch::grid_for(n_rows * n_rhs),
                                    launch::BLOCK_SIZE>>>(
                    d_acc, ws.d_partial, n, row_0, n_rows, n_rhs);
            KERNEL_CHECK();
        }
    }
}

} /* namespace ozaki */
