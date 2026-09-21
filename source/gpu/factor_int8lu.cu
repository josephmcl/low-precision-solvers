#include "common/int8lu_arm.h"

#include "common/convert.h"
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
    cublasHandle_t blas    = nullptr;
    Int8LUScratch  scratch = {};

    /*  Residual operator and carrier, both row-major DF32. Distinct because
        the factorization consumes the carrier. */
    float *d_ah = nullptr, *d_al = nullptr;
    float *d_hi = nullptr, *d_lo = nullptr;

    int *d_piv  = nullptr;   /* getrf-style interchanges                 */
    int *d_perm = nullptr;   /* composed permutation, host-built         */

    /*  Solve workspace. */
    float *d_bh = nullptr, *d_bl = nullptr;
    float *d_xh = nullptr, *d_xl = nullptr;
    float *d_rh = nullptr, *d_rl = nullptr;
    float *d_yh = nullptr, *d_yl = nullptr;
    float *d_zh = nullptr, *d_zl = nullptr;
    float *d_th = nullptr, *d_tl = nullptr;
    float *d_nrm = nullptr;

    std::vector<void *> owned;

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
    int const         kfac) {

    state *s = new state;
    s->n    = n;
    s->b    = b;
    s->kfac = kfac;

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

    int8lu_scratch_alloc(s->scratch, static_cast<int>(n), b, kfac);
    return s;
}

void destroy(state *s) {

    if (s == nullptr)
        return;

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
    convert::transpose_split_df32(s->d_ah, s->d_al, d_a, s->n);
    CUDA_CHECK(cudaMemcpy(s->d_hi, s->d_ah, nn * sizeof(float),
                          cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(s->d_lo, s->d_al, nn * sizeof(float),
                          cudaMemcpyDeviceToDevice));
}

double factor(state *s) {

    if (s == nullptr)
        return 0.;

    timing::stopwatch watch;
    watch.start();

    int8lu_factor(s->blas, static_cast<int>(s->n), s->b, s->kfac,
                  UPD_INT8, s->d_hi, s->d_lo, s->d_piv, s->scratch);

    double const ms = watch.stop();

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

    return ms;
}

double solve(
    state        *s,
    double       *d_x,
    double const *d_b,
    std::size_t  *n_iterations) {

    if (s == nullptr)
        return 0.;

    int const   n = static_cast<int>(s->n);
    int const   T = launch::BLOCK_SIZE;
    int const   g = (n + T - 1) / T;
    std::size_t used = 0;

    LAUNCH((k_split_f64<<<g, T>>>(n, d_b, s->d_bh, s->d_bl)));
    CUDA_CHECK(cudaMemset(s->d_xh, 0, n * sizeof(float)));
    CUDA_CHECK(cudaMemset(s->d_xl, 0, n * sizeof(float)));

    /*  ||b||_inf, for the relative stopping test. */
    CUDA_CHECK(cudaMemset(s->d_nrm, 0, sizeof(float)));
    LAUNCH((max_abs_kernel<<<g, T>>>(n, s->d_bh, s->d_nrm)));
    float b_norm = 0.f;
    CUDA_CHECK(cudaMemcpy(&b_norm, s->d_nrm, sizeof(float),
                          cudaMemcpyDeviceToHost));
    if (b_norm <= 0.f)
        b_norm = 1.f;

    CUDA_CHECK(cudaDeviceSynchronize());

    timing::stopwatch watch;
    watch.start();

    /*  A cap, not a schedule: the loop stops on its own test and reports the
        count it used. */
    std::size_t const cap = 60;
    double best = 1e30;
    int    stalled = 0;

    for (std::size_t it = 0; it != cap; ++it) {

        LAUNCH((k_df_residual_warp<<<(n * 32 + T - 1) / T, T>>>(
            n, s->d_ah, s->d_al, s->d_xh, s->d_xl, s->d_bh, s->d_bl,
            s->d_rh, s->d_rl)));

        CUDA_CHECK(cudaMemset(s->d_nrm, 0, sizeof(float)));
        LAUNCH((max_abs_kernel<<<g, T>>>(n, s->d_rh, s->d_nrm)));
        float r_norm = 0.f;
        CUDA_CHECK(cudaMemcpy(&r_norm, s->d_nrm, sizeof(float),
                              cudaMemcpyDeviceToHost));

        double const rel = static_cast<double>(r_norm) /
                           static_cast<double>(b_norm);
        if (rel < 1e-14)
            break;

        /*  Two tests, as the other iterative methods here use: stalling
            alone never fires once the correction reaches zero, convergence
            alone never fires on a problem that plateaus. */
        if (rel < 0.7 * best) {
            best = rel;
            stalled = 0;
        }
        else if (++stalled >= 2 && it > 3)
            break;

        LAUNCH((gather_kernel<<<g, T>>>(n, s->d_perm, s->d_rh, s->d_rl,
                                        s->d_yh, s->d_yl)));
        trsv_L_blocked(n, 256, s->d_hi, s->d_lo, s->d_yh, s->d_yl,
                       s->d_zh, s->d_zl, s->d_th, s->d_tl);
        trsv_U_blocked(n, 256, s->d_hi, s->d_lo, s->d_zh, s->d_zl,
                       s->d_yh, s->d_yl, s->d_th, s->d_tl);
        LAUNCH((add_correction_kernel<<<g, T>>>(n, s->d_yh, s->d_yl,
                                                s->d_xh, s->d_xl)));
        ++used;
    }

    LAUNCH((combine_kernel<<<g, T>>>(n, s->d_xh, s->d_xl, d_x)));

    double const ms = watch.stop();

    if (n_iterations != nullptr)
        *n_iterations = used;

    return ms;
}

} /* namespace int8lu_arm */
