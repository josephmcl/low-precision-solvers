#include "gpu/ozaki.h"

#include "gpu/definitions.h"
#include "gpu/error.h"
#include "gpu/problem.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <cmath>
#include <cstdlib>
#include <cstddef>
#include <iostream>
#include <vector>

namespace ozaki {

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
    cfg.n_groups = cfg.n_pieces;

    return cfg;
}

/*  ---- scale factors ---------------------------------------------------- */

/*  Largest |a| in each row of the requested triangle. One block per row: the
    block strides the row, reduces in shared memory, and writes one scale.

    Full rows, not per block, because the exponent grid must be constant along
    the summed index (see the header). */
__global__ void row_max_kernel(
    float             *d_mu,
    float const       *d_a,
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

        float const v = fabsf(d_a[i + j * lda]);
        acc = fmaxf(acc, v);
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
    float const       *d_a,
    std::size_t const  n,
    std::size_t const  lda,
    shape const        which,
    problem           &prob) {

    (void) prob;

    row_max_kernel<<<static_cast<int>(n), launch::BLOCK_SIZE>>>(
        d_mu,
        d_a,
        n,
        lda,
        static_cast<int>(which));
    KERNEL_CHECK();
}

void column_max(
    float             *d_nu,
    double const      *d_x,
    std::size_t const  n,
    std::size_t const  n_rhs,
    problem           &prob) {

    (void) prob;

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
__global__ void split_a_kernel(
    float             *d_pieces,
    float const       *d_a,
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

    if (r >= n_rows || c >= n_cols)
        return;

    std::size_t const i = row_0 + r;
    std::size_t const j = col_0 + c;

    float a = 0.f;
    bool  live = true;
    if (which == 1 && j >= i) live = false;   /* strictly lower */
    if (which == 2 && j <  i) live = false;   /* on and above   */
    if (live)
        a = d_a[i + j * lda];

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
    : cfg(cfg) {

    (void) prob;

    std::size_t const b  = static_cast<std::size_t>(cfg.block);
    std::size_t const np = static_cast<std::size_t>(cfg.n_pieces);

    d_pieces_a = static_cast<float *>(_acquire(n * b * np * sizeof(float)));
    d_pieces_x = static_cast<float *>(_acquire(b * n_rhs * np * sizeof(float)));
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

void accumulate_product(
    double            *d_acc,
    float const       *d_a,
    double const      *d_x,
    std::size_t const  lda,
    std::size_t const  n_rhs,
    shape const        which,
    workspace         &ws,
    problem           &prob,
    std::size_t const  contraction_limit) {

    std::size_t const n  = prob.n;
    config     const &cfg = ws.cfg;

    std::size_t const k_end =
        (contraction_limit == 0 || contraction_limit > n)? n : contraction_limit;

    std::size_t const b  = static_cast<std::size_t>(cfg.block);
    std::size_t const np = static_cast<std::size_t>(cfg.n_pieces);

    float const one = 1.f, zero = 0.f;

    for (std::size_t c_0 = 0; c_0 < k_end; c_0 += b) {

        std::size_t const n_c = (k_end - c_0 < b)? k_end - c_0 : b;

        /*  Rows of the output this contraction block can touch. For L
            (strictly lower) columns c only feed rows > c, so rows below c_0
            are structurally zero; for U they feed rows < c_0 + n_c. Skipping
            them is where the triangular advantage comes from. */
        std::size_t row_0  = 0;
        std::size_t n_rows = n;
        if (which == shape::lower) {
            row_0  = c_0;
            n_rows = n - c_0;
        }
        else if (which == shape::upper) {
            row_0  = 0;
            n_rows = c_0 + n_c;
        }

        std::size_t const stride_a = n_rows * b;
        std::size_t const stride_x = b * n_rhs;

        dim3 const grid_a(
            static_cast<unsigned>((n_rows + launch::BLOCK_SIZE - 1) /
                                  launch::BLOCK_SIZE),
            static_cast<unsigned>(n_c));
        split_a_kernel<<<grid_a, launch::BLOCK_SIZE>>>(
            ws.d_pieces_a,
            d_a,
            ws.d_mu,
            lda,
            stride_a,
            n_rows,
            row_0,
            c_0,
            n_c,
            cfg.n_pieces,
            cfg.bits,
            static_cast<int>(which));
        KERNEL_CHECK();

        dim3 const grid_x(
            static_cast<unsigned>((n_c + launch::BLOCK_SIZE - 1) /
                                  launch::BLOCK_SIZE),
            static_cast<unsigned>(n_rhs));
        split_x_kernel<<<grid_x, launch::BLOCK_SIZE,
                         np * sizeof(double)>>>(
            ws.d_pieces_x,
            d_x,
            ws.d_nu,
            n,
            stride_x,
            c_0,
            n_c,
            cfg.n_pieces,
            cfg.bits);
        KERNEL_CHECK();

        /*  Scale-grouped accumulation. Products with p+q = s share an
            exponent grid, so cuBLAS folds them into d_partial with beta=1
            exactly; only one fp64 pass per group is then needed instead of
            one per product. That is 21 folds down to n_groups. */
        for (int s = 0; s != cfg.n_groups; ++s) {

            bool first = true;
            for (int p = 0; p <= s; ++p) {

                int const q = s - p;
                if (p >= cfg.n_pieces || q >= cfg.n_pieces)
                    continue;

                float const beta = first? zero : one;

                CUBLAS_CHECK(cublasGemmEx(
                    prob.blas,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    static_cast<int>(n_rows),
                    static_cast<int>(n_rhs),
                    static_cast<int>(n_c),
                    &one,
                    ws.d_pieces_a + static_cast<std::size_t>(p) * stride_a,
                    CUDA_R_32F, static_cast<int>(n_rows),
                    ws.d_pieces_x + static_cast<std::size_t>(q) * stride_x,
                    CUDA_R_32F, static_cast<int>(n_c),
                    &beta,
                    ws.d_partial, CUDA_R_32F, static_cast<int>(n_rows),
                    CUBLAS_COMPUTE_32F_FAST_TF32,
                    CUBLAS_GEMM_DEFAULT));

                first = false;
            }

            if (!first) {
                accumulate_kernel<<<launch::grid_for(n_rows * n_rhs),
                                    launch::BLOCK_SIZE>>>(
                    d_acc,
                    ws.d_partial,
                    n,
                    row_0,
                    n_rows,
                    n_rhs);
                KERNEL_CHECK();
            }
        }
    }
}

} /* namespace ozaki */
