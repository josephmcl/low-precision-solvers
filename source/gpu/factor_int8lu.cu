#include "common/int8lu_arm.h"

#include "common/convert.h"
#include "common/oii_gemm.h"
#include "common/definitions.h"
#include "common/error.h"
#include "common/timing.h"

/*  int8lu.cuh calls std::sort without including <algorithm>; upstream never
    saw it because its one driver includes <algorithm> first. Vendor files are
    kept verbatim, so the include goes here. */
#include <algorithm>
#include <vector>

#include "df32_kernels.cuh"
#include "df_ext_b4.cuh"
#include "int8lu.cuh"

namespace int8lu_arm {

namespace {

/*  out = in[perm[.]], applying the row permutation to a DF32 vector. */
__global__ void gather_kernel(
    int const         n,
    int const        *perm,
    float const      *in_hi,
    float const      *in_lo,
    float            *out_hi,
    float            *out_lo) {

    int const i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    out_hi[i] = in_hi[perm[i]];
    out_lo[i] = in_lo[perm[i]];
}

/*  x += d, both DF32. The IR update, carried in the wide format so the
    correction's trailing digits survive. */
__global__ void add_correction_kernel(
    int const    n,
    float const *d_hi,
    float const *d_lo,
    float       *x_hi,
    float       *x_lo) {

    int const i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;

    df32 const r = df_add(df_make(x_hi[i], x_lo[i]), df_make(d_hi[i], d_lo[i]));
    x_hi[i] = r.hi;
    x_lo[i] = r.lo;
}

/*  max |v|, as atomicMax on the float bit pattern. Valid because the values
    compared are magnitudes: IEEE ordering is monotonic on non-negatives. */
__global__ void max_abs_kernel(
    int const    n,
    float const *v,
    float       *out) {

    __shared__ float s[launch::BLOCK_SIZE];

    int const t = threadIdx.x;
    float m = 0.f;
    for (int i = blockIdx.x * blockDim.x + t; i < n;
         i += blockDim.x * gridDim.x)
        m = fmaxf(m, fabsf(v[i]));

    s[t] = m;
    __syncthreads();

    for (int q = blockDim.x / 2; q > 0; q >>= 1) {
        if (t < q)
            s[t] = fmaxf(s[t], s[t + q]);
        __syncthreads();
    }

    if (t == 0)
        atomicMax(reinterpret_cast<int *>(out), __float_as_int(s[0]));
}

/*  Output boundary: the only place a DF32 value becomes an fp64 one. */
__global__ void combine_kernel(
    int const    n,
    float const *x_hi,
    float const *x_lo,
    double      *x) {

    int const i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        x[i] = static_cast<double>(x_hi[i]) + static_cast<double>(x_lo[i]);
}

} /* anonymous namespace */

struct state {

    std::size_t    n       = 0;
    int            b       = 0;
    int            kfac    = 0;
    kernel         which   = kernel::sliced;
    oii::state    *oii_s   = nullptr;

    /*  S-rung diagonal-block inverses, n*IB floats each, built once per
        factorization. Null when srung_ib is 0. */
    int    srung_ib = 0;
    float *d_dlh = nullptr, *d_dll = nullptr;
    float *d_duh = nullptr, *d_dul = nullptr;
    cublasHandle_t blas    = nullptr;
    Int8LUScratch  scratch = {};

    /*  Residual operator and carrier, both row-major DF32. Distinct because
        the factorization consumes the carrier. */
    float *d_ah = nullptr, *d_al = nullptr;
    float *d_hi = nullptr, *d_lo = nullptr;

    int *d_piv  = nullptr;   /* getrf-style interchanges                 */
    int *d_perm = nullptr;   /* composed permutation, host-built         */

    /*  Solve workspace. */
    std::size_t rhs_cap = 0;     /* columns the solve workspace holds */
    float *d_bh = nullptr, *d_bl = nullptr;
    float *d_xh = nullptr, *d_xl = nullptr;
    float *d_rh = nullptr, *d_rl = nullptr;
    float *d_yh = nullptr, *d_yl = nullptr;
    float *d_zh = nullptr, *d_zl = nullptr;
    float *d_th = nullptr, *d_tl = nullptr;
    float *d_nrm = nullptr;

    std::vector<void *> owned;

    /*  Phase markers. One event per boundary, synchronized once at the end,
        so the breakdown costs a record per phase and no serialization. */
    std::vector<cudaEvent_t> marks;
    std::size_t              n_marks = 0;
    phase_times              times   = {};

    /*  Phase each span ENDS in; -1 opens the sequence. Tagging rather than
        assuming a fixed stride per iteration, because the loop can exit at
        the convergence test with only part of an iteration recorded. */
    std::vector<int> tags;

    cudaEvent_t mark(int const phase) {
        if (n_marks < tags.size()) tags[n_marks] = phase;
        else                       tags.push_back(phase);
        if (n_marks == marks.size()) {
            cudaEvent_t e = nullptr;
            CUDA_CHECK(cudaEventCreate(&e));
            marks.push_back(e);
        }
        cudaEvent_t const e = marks[n_marks++];
        CUDA_CHECK(cudaEventRecord(e));
        return e;
    }

    /*  Sum the spans by tag. Call after a synchronization. */
    void fold(double *bucket[]) {
        for (std::size_t i = 1; i != n_marks; ++i)
            if (tags[i] >= 0)
                *bucket[tags[i]] += span(i - 1, i);
        n_marks = 0;
    }

    double span(std::size_t const a, std::size_t const b) const {
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, marks[a], marks[b]));
        return static_cast<double>(ms);
    }

    void *acquire(std::size_t const bytes) {
        void *p = nullptr;
        if (!CUDA_CHECK(cudaMalloc(&p, bytes)))
            return nullptr;
        owned.push_back(p);
        return p;
    }
};

state *create(
    std::size_t const n,
    int const         b,
    int const         kfac,
    kernel const      which,
    int const         srung_ib) {

    if (srung_ib != 0 && (srung_ib <= 0 || n % srung_ib != 0)) {
        std::fprintf(stderr, "[int8lu] srung_ib=%d does not divide n=%zu\n",
                     srung_ib, n);
        return nullptr;
    }

    /*  CLAMP THE PANEL WIDTH BEFORE ANYTHING ALLOCATES.

        int8lu_scratch_alloc computes `size_t M = (size_t)(n - b)` from
        two ints. With b > n that is negative, wraps to about 2^64, and
        the cudaMalloc that follows asks for a nonsense size. Upstream
        never sees it because int8lu_factor clamps `if(b<=0||b>n) b=n`
        -- but that clamp is inside the FACTOR, which runs long after
        the scratch is allocated, so the wrapper has to do it here.

        Found by a boundary test in lps-consistency, which triggered it
        rather than reporting it: the machine became unreachable during
        the run. The clamp matches the vendor's own rule so the
        factorization behaves identically; what changes is that the
        allocation is sane. */
    int b_use = b;
    if (b_use <= 0 || static_cast<std::size_t>(b_use) > n)
        b_use = static_cast<int>(n);

    state *s = new state;
    s->n        = n;
    s->b        = b_use;
    s->kfac     = kfac;
    s->which    = which;
    s->srung_ib = srung_ib;

    if (!CUBLAS_CHECK(cublasCreate(&s->blas))) {
        delete s;
        return nullptr;
    }

    std::size_t const nn = n * n;
    s->d_ah = static_cast<float *>(s->acquire(nn * sizeof(float)));
    s->d_al = static_cast<float *>(s->acquire(nn * sizeof(float)));
    s->d_hi = static_cast<float *>(s->acquire(nn * sizeof(float)));
    s->d_lo = static_cast<float *>(s->acquire(nn * sizeof(float)));

    s->d_piv  = static_cast<int *>(s->acquire(n * sizeof(int)));
    s->d_perm = static_cast<int *>(s->acquire(n * sizeof(int)));

    float **const work[] = {&s->d_bh, &s->d_bl, &s->d_xh, &s->d_xl,
                            &s->d_rh, &s->d_rl, &s->d_yh, &s->d_yl,
                            &s->d_zh, &s->d_zl, &s->d_th, &s->d_tl};
    for (std::size_t i = 0; i != sizeof work / sizeof *work; ++i)
        *work[i] = static_cast<float *>(s->acquire(n * sizeof(float)));

    s->d_nrm = static_cast<float *>(s->acquire(sizeof(float)));

    if (srung_ib != 0) {
        std::size_t const nb = n * static_cast<std::size_t>(srung_ib)
                             * sizeof(float);
        s->d_dlh = static_cast<float *>(s->acquire(nb));
        s->d_dll = static_cast<float *>(s->acquire(nb));
        s->d_duh = static_cast<float *>(s->acquire(nb));
        s->d_dul = static_cast<float *>(s->acquire(nb));
        if (s->d_dul == nullptr) {
            destroy(s);
            return nullptr;
        }
    }

    int8lu_scratch_alloc(s->scratch, static_cast<int>(n), b_use, kfac);

    if (which == kernel::oii) {
        /*  Sized for the first trailing update, which is the largest;
            every later one is smaller and passes its own m and n.

            n <= b means the whole matrix is one panel and there is no
            trailing update at all. Guarded because `n - b` is computed
            in size_t and would wrap to an enormous allocation rather
            than fail. */
        if (n <= static_cast<std::size_t>(b_use)) {
            std::fprintf(stderr, "[int8lu] oii arm needs n > b "
                                 "(n=%zu b=%d): no trailing update\n",
                         n, b_use);
            destroy(s);
            return nullptr;
        }
        s->oii_s = oii::create(n - static_cast<std::size_t>(b_use), b_use,
                               n - static_cast<std::size_t>(b_use), kfac);
        if (s->oii_s == nullptr) {
            destroy(s);
            return nullptr;
        }
    }
    return s;
}

void destroy(state *s) {

    if (s == nullptr)
        return;

    oii::destroy(s->oii_s);

    for (std::size_t i = 0; i != s->marks.size(); ++i)
        CUDA_CHECK(cudaEventDestroy(s->marks[i]));

    int8lu_scratch_free(s->scratch);
    for (std::size_t i = 0; i != s->owned.size(); ++i)
        CUDA_CHECK(cudaFree(s->owned[i]));
    if (s->blas != nullptr)
        CUBLAS_CHECK(cublasDestroy(s->blas));

    delete s;
}

void prepare(
    state        *s,
    double const *d_a) {

    if (s == nullptr)
        return;

    std::size_t const nn = s->n * s->n;

    /*  One transpose+split into the residual operator, then a straight copy
        into the carrier: the two hold the same matrix and only the carrier
        is destroyed. */
    s->times = phase_times();
    s->n_marks = 0;
    s->mark(-1);

    convert::transpose_split_df32(s->d_ah, s->d_al, d_a, s->n);
    CUDA_CHECK(cudaMemcpy(s->d_hi, s->d_ah, nn * sizeof(float),
                          cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(s->d_lo, s->d_al, nn * sizeof(float),
                          cudaMemcpyDeviceToDevice));

    s->mark(0);
    CUDA_CHECK(cudaDeviceSynchronize());
    s->times.prepare += s->span(0, 1);
    s->n_marks = 0;
}

/*  ---- slice-cascade saturation on the production carrier -------------

    i_sat is the level at which the DF32 accumulator stops changing BITS.
    Not a chi criterion, not a residual threshold: the thing itself.

    This replicates k_slice_rows exactly -- per-row max over |hi|, scale
    max/127, rintf, clamp to +-127, exact two_prod, and the residual
    updated with df_sub_acc -- and adds a change detector on the
    accumulator. Replicating rather than calling it is deliberate: the
    vendored kernel writes slices and scales and keeps no accumulator,
    and the numerics here are line-for-line the same.

    The carrier has to govern the WHOLE cascade, residual included. The
    simulation measured what happens otherwise: with the slicer left in
    fp64 and only the accumulator widened, i_sat ran past 24 levels
    against a predicted 17, because the contributions floored at
    u * rowmax instead of continuing to shrink geometrically. Here the
    residual is DF32 because the arm's residual is DF32.

    The segment width is the panel's 256-column k-extent, because that is
    the extent the trailing update actually slices over. Saturation
    depends on the dynamic range the scale has to cover, so measuring it
    over a different extent would measure a different quantity. */
__global__ void k_saturation(
    int const    n,
    int const    c0,
    int const    width,
    int const    max_depth,
    float const *ah,
    float const *al,
    int         *last_change,
    int         *uf_level,
    float       *g_ratio,
    float       *scale_out) {   /* optional, depth x n */

    /*  Sized for the only launch configuration this kernel has, 256
        threads; the reductions below read blockDim.x so they stay
        correct if that changes, but these arrays would not. */
    __shared__ float rh[256], rl[256], acch[256], accl[256], red[256];
    __shared__ float gmx[256];
    __shared__ int   any;

    int const i = blockIdx.x;
    int const t = threadIdx.x;

    float const v_h = (t < width)? ah[static_cast<std::size_t>(i) * n + c0 + t] : 0.f;
    float const v_l = (t < width)? al[static_cast<std::size_t>(i) * n + c0 + t] : 0.f;
    rh[t] = v_h;
    rl[t] = v_l;
    acch[t] = 0.f;
    accl[t] = 0.f;
    gmx[t]  = -1e30f;
    int last = 0;
    int uf   = 0;
    __syncthreads();

    for (int s = 1; s <= max_depth; ++s) {

        float m = (t < width)? fabsf(rh[t]) : 0.f;
        red[t] = m;
        __syncthreads();
        for (int o = blockDim.x >> 1; o > 0; o >>= 1) {
            if (t < o)
                red[t] = fmaxf(red[t], red[t + o]);
            __syncthreads();
        }

        float sc = (red[0] > 0.f)? red[0] / 127.f : 1.f;
        /*  The residual reaches the denormal floor and the scale flushes
            to zero long before a fixed depth cap is hit; the division
            would then poison the whole cascade with inf. Guarded, and
            the fact that it happened is REPORTED rather than swallowed,
            because a run that underflowed is measuring the exponent
            range and not the saturation depth. */
        /*  The residual has reached the denormal floor: there is nothing
            left to slice, and the division would poison the cascade with
            inf. Record WHEN it happened per row rather than raising one
            global flag -- a row whose residual vanished has saturated,
            and only underflow BEFORE the last change invalidates the
            measurement. The first version flagged the whole grid on any
            row underflowing and reported nothing usable. */
        if (sc == 0.f) {
            sc = 1.f;
            if (uf == 0)
                uf = s;
        }

        if (t == 0) {
            any = 0;
            if (scale_out != nullptr)
                scale_out[static_cast<std::size_t>(s - 1) * gridDim.x + i]
                    = sc;
        }
        __syncthreads();

        if (t < width) {
            float q = rintf(rh[t] / sc);
            q = fminf(fmaxf(q, -127.f), 127.f);
            df32 const x    = two_prod(sc, q);
            df32 const prev = df_make(acch[t], accl[t]);
            df32 const a    = df_add_acc(prev, x);
            if (a.hi != prev.hi || a.lo != prev.lo)
                atomicOr(&any, 1);

            /*  The simulation's Gamma reading, kept in LOG space:
                log_beta(|x^(i)| / |a^(i-1)|) + i, whose maximum over
                entries is g_max directly. Forming the ratio times
                beta^i and taking the log afterwards overflows fp32 at
                i = 16 -- beta^16 is 1e38 -- and reports inf for every
                cell, which is what the first version did. */
            float const pv = fabsf(prev.hi + prev.lo);
            float const xv = fabsf(x.hi + x.lo);
            if (pv > 0.f && xv > 0.f)
                gmx[t] = fmaxf(gmx[t],
                               __logf(xv / pv) / 5.53733f + (float)s);

            acch[t] = a.hi;
            accl[t] = a.lo;
            df32 const r = df_sub_acc(df_make(rh[t], rl[t]), x);
            rh[t] = r.hi;
            rl[t] = r.lo;
        }
        __syncthreads();

        if (any != 0)
            last = s;
    }

    /*  Row maxima of the Gamma reading. */
    __syncthreads();
    for (int o = blockDim.x >> 1; o > 0; o >>= 1) {
        if (t < o)
            gmx[t] = fmaxf(gmx[t], gmx[t + o]);
        __syncthreads();
    }

    if (t == 0) {
        last_change[i] = last;
        uf_level[i]    = uf;
        g_ratio[i]     = gmx[0];
    }
}

/*  ---- multi-RHS triangular solve -------------------------------------

    The solve was a loop over columns: k right-hand sides cost k
    independent chains and the measured per-RHS time was flat at 19.35 ms
    from k=1 to k=16, i.e. no amortization at all. The phase split at k=16
    says where it all is -- TRSV 97.5 percent, residual 2.0, gather 0.2,
    update 0.3 -- so these two kernel pairs are the whole job and the
    residual stays per column.

    Each column's arithmetic is UNCHANGED: same loop bounds, same warp
    reduction order, same sub-block sequence. The batched path is
    therefore bit-identical to the column loop, which is what the gate
    checks. The win is not arithmetic, it is that one pass over L\U now
    serves k columns and that the diagonal apply, which ran in a SINGLE
    block, now runs in k.

    Vectors are n x nrhs, column major with stride ldv, so a single-vector
    kernel still works on any one column by pointer offset. */

/*  Off-diagonal update, one warp per (row, rhs). Consecutive warps take
    the same row across right-hand sides, so the row of L\U is fetched
    once and serves all of them. */
__global__ void k_trsv_off_L_mrhs(
    int const    n,
    int const    i0,
    int const    ni,
    int const    nrhs,
    int const    ldv,
    float const *luh,
    float const *lul,
    float const *yh,
    float const *yl,
    float const *bh,
    float const *bl,
    float       *rh,
    float       *rl) {

    int const gw   = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int const lane = threadIdx.x & 31;
    if (gw >= ni * nrhs)
        return;
    int const r = i0 + gw / nrhs;
    std::size_t const off = static_cast<std::size_t>(gw % nrhs) * ldv;

    df32 acc = df_make(0.f, 0.f);
    for (int j = lane; j < i0; j += 32)
        acc = df_add_acc(acc, df_mul(
            df_make(luh[static_cast<std::size_t>(r) * n + j],
                    lul[static_cast<std::size_t>(r) * n + j]),
            df_make(yh[off + j], yl[off + j])));
    for (int o = 16; o > 0; o >>= 1) {
        float const oh = __shfl_down_sync(0xffffffffu, acc.hi, o);
        float const ol = __shfl_down_sync(0xffffffffu, acc.lo, o);
        acc = df_add_acc(acc, df_make(oh, ol));
    }
    if (lane == 0) {
        df32 const v = df_sub_acc(df_make(bh[off + r], bl[off + r]), acc);
        rh[off + r] = v.hi;
        rl[off + r] = v.lo;
    }
}

__global__ void k_trsv_off_U_mrhs(
    int const    n,
    int const    i0,
    int const    ni,
    int const    nrhs,
    int const    ldv,
    float const *luh,
    float const *lul,
    float const *yh,
    float const *yl,
    float const *bh,
    float const *bl,
    float       *rh,
    float       *rl) {

    int const gw   = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    int const lane = threadIdx.x & 31;
    if (gw >= ni * nrhs)
        return;
    int const r = i0 + gw / nrhs;
    std::size_t const off = static_cast<std::size_t>(gw % nrhs) * ldv;

    df32 acc = df_make(0.f, 0.f);
    for (int j = i0 + ni + lane; j < n; j += 32)
        acc = df_add_acc(acc, df_mul(
            df_make(luh[static_cast<std::size_t>(r) * n + j],
                    lul[static_cast<std::size_t>(r) * n + j]),
            df_make(yh[off + j], yl[off + j])));
    for (int o = 16; o > 0; o >>= 1) {
        float const oh = __shfl_down_sync(0xffffffffu, acc.hi, o);
        float const ol = __shfl_down_sync(0xffffffffu, acc.lo, o);
        acc = df_add_acc(acc, df_make(oh, ol));
    }
    if (lane == 0) {
        df32 const v = df_sub_acc(df_make(bh[off + r], bl[off + r]), acc);
        rh[off + r] = v.hi;
        rl[off + r] = v.lo;
    }
}

/*  Diagonal apply, one BLOCK per right-hand side. The single-RHS kernel
    launches one block on a 132-SM device; this is the same body with the
    vectors offset, so k right-hand sides fill k blocks. */
__global__ void k_diag_apply_L_mrhs(
    int const    n,
    int const    i0,
    int const    ni,
    int const    ib,
    int const    ldv,
    float const *luh,
    float const *lul,
    float const *dh,
    float const *dl,
    float const *rh,
    float const *rl,
    float       *yh,
    float       *yl) {

    __shared__ float ys[256], yls[256], rrh[256], rrl[256];
    int const t = threadIdx.x;
    std::size_t const off = static_cast<std::size_t>(blockIdx.x) * ldv;

    for (int sb = 0; sb < ni; sb += ib) {
        int const sw = (ni - sb < ib)? (ni - sb) : ib;
        if (t < sw) {
            df32 rr = df_make(rh[off + i0 + sb + t], rl[off + i0 + sb + t]);
            for (int j = 0; j < sb; ++j)
                rr = df_sub_acc(rr, df_mul(
                    df_make(luh[static_cast<std::size_t>(i0 + sb + t) * n + (i0 + j)],
                            lul[static_cast<std::size_t>(i0 + sb + t) * n + (i0 + j)]),
                    df_make(ys[j], yls[j])));
            rrh[t] = rr.hi;
            rrl[t] = rr.lo;
        }
        __syncthreads();
        if (t < sw) {
            float const *mh = dh + static_cast<std::size_t>((i0 + sb) / ib) * ib * ib;
            float const *ml = dl + static_cast<std::size_t>((i0 + sb) / ib) * ib * ib;
            df32 acc = df_make(0.f, 0.f);
            for (int q = 0; q < sw; ++q)
                acc = df_add_acc(acc, df_mul(df_make(mh[t * ib + q], ml[t * ib + q]),
                                             df_make(rrh[q], rrl[q])));
            ys[sb + t]  = acc.hi;
            yls[sb + t] = acc.lo;
        }
        __syncthreads();
    }
    for (int m = t; m < ni; m += blockDim.x) {
        yh[off + i0 + m] = ys[m];
        yl[off + i0 + m] = yls[m];
    }
}

__global__ void k_diag_apply_U_mrhs(
    int const    n,
    int const    i0,
    int const    ni,
    int const    ib,
    int const    ldv,
    float const *luh,
    float const *lul,
    float const *dh,
    float const *dl,
    float const *rh,
    float const *rl,
    float       *yh,
    float       *yl) {

    __shared__ float ys[256], yls[256], rrh[256], rrl[256];
    int const t = threadIdx.x;
    std::size_t const off = static_cast<std::size_t>(blockIdx.x) * ldv;

    for (int sb = ((ni - 1) / ib) * ib; sb >= 0; sb -= ib) {
        int const sw = (sb + ib <= ni)? ib : (ni - sb);
        if (t < sw) {
            df32 rr = df_make(rh[off + i0 + sb + t], rl[off + i0 + sb + t]);
            for (int j = sb + sw; j < ni; ++j)
                rr = df_sub_acc(rr, df_mul(
                    df_make(luh[static_cast<std::size_t>(i0 + sb + t) * n + (i0 + j)],
                            lul[static_cast<std::size_t>(i0 + sb + t) * n + (i0 + j)]),
                    df_make(ys[j], yls[j])));
            rrh[t] = rr.hi;
            rrl[t] = rr.lo;
        }
        __syncthreads();
        if (t < sw) {
            float const *mh = dh + static_cast<std::size_t>((i0 + sb) / ib) * ib * ib;
            float const *ml = dl + static_cast<std::size_t>((i0 + sb) / ib) * ib * ib;
            df32 acc = df_make(0.f, 0.f);
            for (int q = 0; q < sw; ++q)
                acc = df_add_acc(acc, df_mul(df_make(mh[t * ib + q], ml[t * ib + q]),
                                             df_make(rrh[q], rrl[q])));
            ys[sb + t]  = acc.hi;
            yls[sb + t] = acc.lo;
        }
        __syncthreads();
    }
    for (int m = t; m < ni; m += blockDim.x) {
        yh[off + i0 + m] = ys[m];
        yl[off + i0 + m] = yls[m];
    }
}

/*  The blocked drivers, batched. Same block sequence as the vendored
    single-vector ones, so each column sees the identical order. */
void trsv_L_mrhs(
    int const n, int const blk, int const ib, int const nrhs, int const ldv,
    float const *luh, float const *lul, float const *dlh, float const *dll,
    float const *bh, float const *bl, float *yh, float *yl,
    float *rh, float *rl) {

    for (int i0 = 0; i0 < n; i0 += blk) {
        int const ni = (i0 + blk <= n)? blk : (n - i0);
        int const w  = ni * nrhs * 32;
        k_trsv_off_L_mrhs<<<(w + 255) / 256, 256>>>(
            n, i0, ni, nrhs, ldv, luh, lul, yh, yl, bh, bl, rh, rl);
        KERNEL_CHECK();
        k_diag_apply_L_mrhs<<<nrhs, 256>>>(
            n, i0, ni, ib, ldv, luh, lul, dlh, dll, rh, rl, yh, yl);
        KERNEL_CHECK();
    }
}

void trsv_U_mrhs(
    int const n, int const blk, int const ib, int const nrhs, int const ldv,
    float const *luh, float const *lul, float const *duh, float const *dul,
    float const *bh, float const *bl, float *yh, float *yl,
    float *rh, float *rl) {

    for (int i0 = ((n - 1) / blk) * blk; i0 >= 0; i0 -= blk) {
        int const ni = (i0 + blk <= n)? blk : (n - i0);
        int const w  = ni * nrhs * 32;
        k_trsv_off_U_mrhs<<<(w + 255) / 256, 256>>>(
            n, i0, ni, nrhs, ldv, luh, lul, yh, yl, bh, bl, rh, rl);
        KERNEL_CHECK();
        k_diag_apply_U_mrhs<<<nrhs, 256>>>(
            n, i0, ni, ib, ldv, luh, lul, duh, dul, rh, rl, yh, yl);
        KERNEL_CHECK();
    }
}

/*  Blocked right-looking LU with the Ozaki-II trailing update.

    The panel, the bulk row interchange and the U12 block solve are the
    vendored ones, called in the vendored order -- this is deliberately a
    copy of int8lu_factor's per-panel loop with one block replaced, so the
    two arms differ in the trailing update and in nothing else. Anything
    the sliced arm gets from its panel, this gets identically.

    What is NOT reproduced: the whole-factor CUDA graph, the async and
    persistent panels, the look-ahead and the FUSE backends. Those are
    built around the sliced update's schedule. The comparison this
    supports is against the per-panel sliced path.

    Returns false if a trailing update was rejected -- Ozaki-II has no
    meaning on a zero row of L21 or a zero column of U12, and a silent
    wrong answer there would be worse than a failed factorization. */
bool oii_factor(
    int const       n,
    int const       b,
    float          *d_ah,
    float          *d_al,
    int            *d_piv,
    Int8LUScratch  &sc,
    oii::state     *os) {

    for (int p = 0; p < n; p += b) {

        int const bb = (p + b <= n)? b : (n - p);

        launch_panel_coop(n, p, bb, d_ah, d_al, d_piv + p, sc.gv, sc.gi);
        if (n - bb > 0)
            LAUNCH((k_laswp_bulk<<<(n - bb + 255) / 256, 256>>>(
                        n, p, bb, d_piv + p, d_ah, d_al)));

        if (p + bb >= n)
            continue;

        u12_blocked(n, p, bb, d_ah, d_al);

        int const mp = n - p - bb;
        int const np = mp;

        /*  L21 = carrier[p+bb : n, p : p+bb], U12 = carrier[p : p+bb,
            p+bb : n], both read in place at leading dimension n, and the
            product subtracted into the trailing block. No gather: that is
            what the strides are for. */
        std::size_t const l21 = static_cast<std::size_t>(p + bb) * n + p;
        std::size_t const u12 = static_cast<std::size_t>(p) * n + (p + bb);
        std::size_t const s22 = static_cast<std::size_t>(p + bb) * n
                              + (p + bb);

        if (!oii::gemm_df32(os, mp, np,
                            d_ah + l21, d_al + l21, n,
                            d_ah + u12, d_al + u12, n,
                            d_ah + s22, d_al + s22, n,
                            true)) {
            std::fprintf(stderr,
                         "[oii-arm] trailing update rejected at panel "
                         "p=%d (mp=%d)\n", p, mp);
            return false;
        }
    }
    return true;
}

double factor(state *s) {

    if (s == nullptr)
        return 0.;

    timing::stopwatch watch;
    watch.start();

    if (s->which == kernel::oii) {
        if (!oii_factor(static_cast<int>(s->n), s->b, s->d_hi, s->d_lo,
                        s->d_piv, s->scratch, s->oii_s))
            return -1.;
    } else {
        int8lu_factor(s->blas, static_cast<int>(s->n), s->b, s->kfac,
                      UPD_INT8, s->d_hi, s->d_lo, s->d_piv, s->scratch);
    }

    /*  The inverses depend only on the factor, so they are built here and
        reused by every refinement step of every right-hand side. Timed
        with the factorization because that is where the work is: charging
        them to the solve would flatter it. */
    if (s->srung_ib != 0)
        srung_invert(static_cast<int>(s->n), s->srung_ib,
                     s->d_hi, s->d_lo,
                     s->d_dlh, s->d_dll, s->d_duh, s->d_dul);

    double const ms = watch.stop();
    s->times.factor += ms;
    double const t_perm0 = ms;

    /*  Compose the sequential interchanges into a permutation. Serial by
        nature and n ints against an O(n^3) factorization, so the host does
        it; a device scan would buy nothing. Outside the timed region only
        because getrf's own pivot list is what it consumes. */
    std::size_t const n = s->n;
    std::vector<int> piv(n), perm(n);
    CUDA_CHECK(cudaMemcpy(piv.data(), s->d_piv, n * sizeof(int),
                          cudaMemcpyDeviceToHost));
    for (std::size_t i = 0; i != n; ++i)
        perm[i] = static_cast<int>(i);
    for (std::size_t k = 0; k != n; ++k) {
        int const j = piv[k];
        if (j >= 0 && static_cast<std::size_t>(j) < n)
            std::swap(perm[k], perm[static_cast<std::size_t>(j)]);
    }
    CUDA_CHECK(cudaMemcpy(s->d_perm, perm.data(), n * sizeof(int),
                          cudaMemcpyHostToDevice));

    /*  The compose is host-serial between two blocking copies, so the
        stopwatch's device timeline measures it fairly. */
    s->times.perm += watch.stop() - t_perm0;

    return ms;
}

/*  Grow the solve workspace to hold `k` columns. The triangular solve's
    scratch (d_th/d_tl) stays one vector wide only in the single-column
    path; the batched kernels write per column, so it grows too. */
bool ensure_rhs(state *s, std::size_t const k) {

    if (k <= s->rhs_cap)
        return true;

    /*  Growing keeps the previous buffers in `owned` until destroy
        rather than freeing them. Bounded by the number of DISTINCT k a
        state is asked for, which is one in every current caller; it is
        retention, not a leak. */

    float **const work[] = {&s->d_bh, &s->d_bl, &s->d_xh, &s->d_xl,
                            &s->d_rh, &s->d_rl, &s->d_yh, &s->d_yl,
                            &s->d_zh, &s->d_zl, &s->d_th, &s->d_tl};
    std::size_t const bytes = s->n * k * sizeof(float);
    for (std::size_t i = 0; i != sizeof work / sizeof *work; ++i) {
        void *p = s->acquire(bytes);
        if (p == nullptr)
            return false;
        *work[i] = static_cast<float *>(p);
    }
    s->rhs_cap = k;
    return true;
}

double solve(
    state        *s,
    double       *d_x,
    double const *d_b,
    std::size_t const k,
    std::size_t  *n_iterations) {

    if (s == nullptr)
        return 0.;

    int const   n = static_cast<int>(s->n);
    int const   T = launch::BLOCK_SIZE;
    int const   g = (n + T - 1) / T;
    std::size_t used = 0;

    CUDA_CHECK(cudaDeviceSynchronize());
    s->n_marks = 0;
    s->mark(-1);

    timing::stopwatch outer;
    outer.start();

    /*  Grow the solve workspace to k columns, n x k column major so a
        single-vector kernel still works on one column by pointer offset.
        Allocated here and not in create() because create() does not know
        k; it happens once per distinct k, not per solve. */
    if (!ensure_rhs(s, k))
        return 0.;

    std::size_t const nk = s->n * k;
    std::size_t const ld = s->n;

    LAUNCH((k_split_f64<<<(int)((nk + T - 1) / T), T>>>(
        (int)nk, d_b, s->d_bh, s->d_bl)));
    CUDA_CHECK(cudaMemset(s->d_xh, 0, nk * sizeof(float)));
    CUDA_CHECK(cudaMemset(s->d_xl, 0, nk * sizeof(float)));

    /*  The batched pass runs every column, converged ones included, so
        it reads the correction vector of columns that never had a
        gather -- a column that meets its test on the first residual is
        exactly that case, and its slot would otherwise hold whatever
        cudaMalloc returned. The values are discarded either way, but a
        NaN read is not free and reading uninitialized memory is not
        something to leave in on the grounds that the answer survives. */
    CUDA_CHECK(cudaMemset(s->d_yh, 0, nk * sizeof(float)));
    CUDA_CHECK(cudaMemset(s->d_yl, 0, nk * sizeof(float)));
    CUDA_CHECK(cudaMemset(s->d_zh, 0, nk * sizeof(float)));
    CUDA_CHECK(cudaMemset(s->d_zl, 0, nk * sizeof(float)));

    /*  ||b||_inf per column, for each column's relative stopping test. */
    std::vector<float> b_norm(k, 1.f);
    for (std::size_t c = 0; c != k; ++c) {
        CUDA_CHECK(cudaMemset(s->d_nrm, 0, sizeof(float)));
        LAUNCH((max_abs_kernel<<<g, T>>>(n, s->d_bh + c * ld, s->d_nrm)));
        float v = 0.f;
        CUDA_CHECK(cudaMemcpy(&v, s->d_nrm, sizeof(float),
                              cudaMemcpyDeviceToHost));
        b_norm[c] = (v > 0.f)? v : 1.f;
    }
    s->mark(3);

    /*  Per-column refinement state. Every column is carried through every
        pass of the batched solve, but a column that has met its own test
        stops taking the correction -- so its answer is what the column
        loop would have produced, bit for bit, and the batching costs it
        only wasted work. */
    std::size_t const cap = 60;
    std::vector<double>      best(k, 1e30);
    std::vector<int>         stalled(k, 0);
    std::vector<char>        done(k, 0);
    std::vector<std::size_t> it_col(k, 0);

    for (std::size_t it = 0; it != cap; ++it) {

        for (std::size_t c = 0; c != k; ++c) {
            if (done[c])
                continue;
            LAUNCH((k_df_residual_warp<<<(n * 32 + T - 1) / T, T>>>(
                n, s->d_ah, s->d_al, s->d_xh + c * ld, s->d_xl + c * ld,
                s->d_bh + c * ld, s->d_bl + c * ld,
                s->d_rh + c * ld, s->d_rl + c * ld)));
        }

        std::size_t n_active = 0;
        for (std::size_t c = 0; c != k; ++c) {
            if (done[c])
                continue;
            CUDA_CHECK(cudaMemset(s->d_nrm, 0, sizeof(float)));
            LAUNCH((max_abs_kernel<<<g, T>>>(n, s->d_rh + c * ld,
                                             s->d_nrm)));
            float r_norm = 0.f;
            CUDA_CHECK(cudaMemcpy(&r_norm, s->d_nrm, sizeof(float),
                                  cudaMemcpyDeviceToHost));
            double const rel = static_cast<double>(r_norm) /
                               static_cast<double>(b_norm[c]);
            if (rel < 1e-14) {
                done[c] = 1;
                continue;
            }
            if (rel < 0.7 * best[c]) {
                best[c] = rel;
                stalled[c] = 0;
            } else if (++stalled[c] >= 2 && it > 3) {
                done[c] = 1;
                continue;
            }
            ++n_active;
        }
        s->mark(0);

        if (n_active == 0)
            break;

        for (std::size_t c = 0; c != k; ++c) {
            if (done[c])
                continue;
            LAUNCH((gather_kernel<<<g, T>>>(n, s->d_perm,
                                            s->d_rh + c * ld,
                                            s->d_rl + c * ld,
                                            s->d_yh + c * ld,
                                            s->d_yl + c * ld)));
        }
        s->mark(1);

        /*  One batched pass over every column, converged ones included:
            compacting the active set would move columns in memory and
            buy nothing, since the pass is dominated by the shared walk
            over L\U rather than by the per-column arithmetic. */
        if (s->srung_ib != 0) {
            trsv_L_mrhs(n, 256, s->srung_ib, (int)k, (int)ld,
                        s->d_hi, s->d_lo, s->d_dlh, s->d_dll,
                        s->d_yh, s->d_yl, s->d_zh, s->d_zl,
                        s->d_th, s->d_tl);
            trsv_U_mrhs(n, 256, s->srung_ib, (int)k, (int)ld,
                        s->d_hi, s->d_lo, s->d_duh, s->d_dul,
                        s->d_zh, s->d_zl, s->d_yh, s->d_yl,
                        s->d_th, s->d_tl);
        } else {
            for (std::size_t c = 0; c != k; ++c) {
                if (done[c])
                    continue;
                trsv_L_blocked(n, 256, s->d_hi, s->d_lo,
                               s->d_yh + c * ld, s->d_yl + c * ld,
                               s->d_zh + c * ld, s->d_zl + c * ld,
                               s->d_th, s->d_tl);
                trsv_U_blocked(n, 256, s->d_hi, s->d_lo,
                               s->d_zh + c * ld, s->d_zl + c * ld,
                               s->d_yh + c * ld, s->d_yl + c * ld,
                               s->d_th, s->d_tl);
            }
        }
        s->mark(2);

        for (std::size_t c = 0; c != k; ++c) {
            if (done[c])
                continue;
            LAUNCH((add_correction_kernel<<<g, T>>>(
                n, s->d_yh + c * ld, s->d_yl + c * ld,
                s->d_xh + c * ld, s->d_xl + c * ld)));
            ++it_col[c];
        }
        s->mark(3);
    }

    for (std::size_t c = 0; c != k; ++c) {
        LAUNCH((combine_kernel<<<g, T>>>(n, s->d_xh + c * ld,
                                         s->d_xl + c * ld,
                                         d_x + c * s->n)));
        if (it_col[c] > used)
            used = it_col[c];
    }
    s->mark(3);

    double const ms = outer.stop();

    double *bucket[4] = {&s->times.residual, &s->times.gather,
                         &s->times.trsv,     &s->times.update};
    s->fold(bucket);

    if (n_iterations != nullptr)
        *n_iterations = used;

    return ms;
}

phase_times const &profile(state const *s) {

    static phase_times const empty = {};
    return (s != nullptr)? s->times : empty;
}

/*  Does the cascade in k_saturation actually reproduce k_slice_rows?

    Everything item 6 reports rests on that replication, and a
    replication is worth exactly as much as the check that it matches.
    The per-level SCALE is the signature: it is max|hi| / 127 over the
    row, and it is determined entirely by the residual cascade, so if
    every scale agrees bit for bit at every level then the residuals
    agree at every level and so does everything downstream.

    Compares against the vendored kernel on the real L21 block. Returns
    the number of differing scales. */
std::size_t saturation_verify(state *s, int const depth) {

    if (s == nullptr)
        return 1;

    int const n   = static_cast<int>(s->n);
    int const b   = (n < 256)? n : 256;
    int const r0  = b;
    int const mp  = n - r0;
    if (mp <= 0)
        return 1;

    std::int8_t *lq = nullptr;
    float       *ls = nullptr, *sc = nullptr;
    int         *d_last = nullptr, *d_uf = nullptr;
    float       *d_g = nullptr;
    CUDA_CHECK(cudaMalloc(&lq, static_cast<std::size_t>(depth) * mp * b));
    CUDA_CHECK(cudaMalloc(&ls, static_cast<std::size_t>(depth) * mp * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&sc, static_cast<std::size_t>(depth) * n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_last, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_uf, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_g, n * sizeof(float)));

    /*  The vendored slicer, on exactly the block the trailing update
        slices: rows [b, n) over columns [0, b). */
    LAUNCH((k_slice_rows<<<mp, 256>>>(n, r0, 0, mp, b, depth,
                                      s->d_hi, s->d_lo, lq, ls)));

    k_saturation<<<n, 256>>>(n, 0, b, depth, s->d_hi, s->d_lo,
                             d_last, d_uf, d_g, sc);
    KERNEL_CHECK();

    std::vector<float> h_ls(static_cast<std::size_t>(depth) * mp);
    std::vector<float> h_sc(static_cast<std::size_t>(depth) * n);
    CUDA_CHECK(cudaMemcpy(h_ls.data(), ls, h_ls.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_sc.data(), sc, h_sc.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));

    std::size_t bad = 0;
    for (int lev = 0; lev != depth; ++lev)
        for (int i = 0; i != mp; ++i) {
            float const v = h_ls[static_cast<std::size_t>(lev) * mp + i];
            float const w = h_sc[static_cast<std::size_t>(lev) * n + r0 + i];
            if (v != w)
                ++bad;
        }

    cudaFree(lq); cudaFree(ls); cudaFree(sc);
    cudaFree(d_last); cudaFree(d_uf); cudaFree(d_g);
    return bad;
}

bool saturation(
    state      *s,
    int const   c0,
    int const   max_depth,
    int        *out,
    float      *g_max) {

    if (s == nullptr || out == nullptr)
        return false;

    int const n = static_cast<int>(s->n);
    int const width = (c0 + 256 <= n)? 256 : (n - c0);
    if (width <= 0)
        return false;

    int   *d_last = nullptr, *d_uf = nullptr;
    float *d_g = nullptr;
    if (!CUDA_CHECK(cudaMalloc(&d_last, n * sizeof(int))) ||
        !CUDA_CHECK(cudaMalloc(&d_uf, n * sizeof(int))) ||
        !CUDA_CHECK(cudaMalloc(&d_g, n * sizeof(float))))
        return false;

    k_saturation<<<n, 256>>>(n, c0, width, max_depth,
                             s->d_hi, s->d_lo, d_last, d_uf, d_g, nullptr);
    KERNEL_CHECK();

    std::vector<int> uf(n);
    CUDA_CHECK(cudaMemcpy(out, d_last, n * sizeof(int),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(uf.data(), d_uf, n * sizeof(int),
                          cudaMemcpyDeviceToHost));
    if (g_max != nullptr)
        CUDA_CHECK(cudaMemcpy(g_max, d_g, n * sizeof(float),
                              cudaMemcpyDeviceToHost));
    cudaFree(d_last);
    cudaFree(d_uf);
    cudaFree(d_g);

    /*  i_sat is one past the last level that changed anything. A row is
        valid unless its residual underflowed at or before that level --
        underflow afterwards just means an exhausted residual, which IS
        saturation. */
    bool clean = true;
    for (int i = 0; i != n; ++i) {
        if (uf[i] != 0 && uf[i] <= out[i])
            clean = false;
        out[i] += 1;
    }
    return clean;
}

void copy_factor(
    state *s,
    float *hi,
    float *lo,
    int   *perm) {

    if (s == nullptr)
        return;

    std::size_t const nn = s->n * s->n;
    CUDA_CHECK(cudaMemcpy(hi, s->d_hi, nn * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(lo, s->d_lo, nn * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(perm, s->d_perm, s->n * sizeof(int),
                          cudaMemcpyDeviceToHost));
}

} /* namespace int8lu_arm */
