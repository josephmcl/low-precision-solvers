#include "common/oii_gemm.h"

#include "common/error.h"

#include <cublas_v2.h>

#include <cstdio>

namespace oii {

namespace {

int const TPB = 256;

/*  cuBLAS int8 GEMM requires the contraction dimension -- which is the
    leading dimension of both operands in this layout -- to be a multiple
    of 4, and returns CUBLAS_STATUS_NOT_SUPPORTED otherwise. Every k-strided
    buffer is therefore allocated at the padded stride and zero filled once;
    the kernels only ever write h < k, so the pad stays zero and contributes
    nothing to any product. */
inline std::size_t pad4(std::size_t const k) {
    return (k + 3) & ~static_cast<std::size_t>(3);
}

/*  Algorithm 2 line 3: row maxima of |A|, A row major m x k. One block per
    row; the reduction is over k, which is the panel width in the arm and
    so never large enough to want a two-stage reduction. */
__global__ void k_absmax_rows(
    int const     m,
    int const     k,
    double const *a,
    double       *out) {

    __shared__ double red[TPB];
    int const i = blockIdx.x;
    if (i >= m)
        return;

    double v = 0.;
    for (int h = threadIdx.x; h < k; h += blockDim.x)
        v = fmax(v, fabs(a[static_cast<std::size_t>(i) * k + h]));
    red[threadIdx.x] = v;
    __syncthreads();
    for (int o = blockDim.x >> 1; o > 0; o >>= 1) {
        if (threadIdx.x < o)
            red[threadIdx.x] = fmax(red[threadIdx.x], red[threadIdx.x + o]);
        __syncthreads();
    }
    if (threadIdx.x == 0)
        out[i] = red[0];
}

/*  Column maxima of |B|, B row major k x n. */
__global__ void k_absmax_cols(
    int const     k,
    int const     n,
    double const *b,
    double       *out) {

    __shared__ double red[TPB];
    int const j = blockIdx.x;
    if (j >= n)
        return;

    double v = 0.;
    for (int h = threadIdx.x; h < k; h += blockDim.x)
        v = fmax(v, fabs(b[static_cast<std::size_t>(h) * n + j]));
    red[threadIdx.x] = v;
    __syncthreads();
    for (int o = blockDim.x >> 1; o > 0; o >>= 1) {
        if (threadIdx.x < o)
            red[threadIdx.x] = fmax(red[threadIdx.x], red[threadIdx.x + o]);
        __syncthreads();
    }
    if (threadIdx.x == 0)
        out[j] = red[0];
}

/*  Lines 5 and 6. mup = 5 - floor(log2(max)), Abar = ceil(|A| 2^mup), which
    the source bounds by 2^6 so it is an int8. Abar stays ROW major. */
__global__ void k_prescale_a(
    int const     m,
    int const     k,
    int const     kp,
    double const *a,
    double const *amax,
    int          *mup,
    signed char  *abar) {

    int const i = blockIdx.x;
    if (i >= m)
        return;

    int const s = 5 - static_cast<int>(floor(log2(amax[i])));
    if (threadIdx.x == 0)
        mup[i] = s;
    double const f = exp2(static_cast<double>(s));
    for (int h = threadIdx.x; h < k; h += blockDim.x)
        abar[static_cast<std::size_t>(i) * kp + h] =
            static_cast<signed char>(
                ceil(fabs(a[static_cast<std::size_t>(i) * k + h]) * f));
}

/*  Same for B, but Bbar comes out COLUMN major so cuBLAS can take it OP_N
    with ld = k against the OP_T A side. */
__global__ void k_prescale_b(
    int const     k,
    int const     kp,
    int const     n,
    double const *b,
    double const *bmax,
    int          *nup,
    signed char  *bbar) {

    int const j = blockIdx.x;
    if (j >= n)
        return;

    int const s = 5 - static_cast<int>(floor(log2(bmax[j])));
    if (threadIdx.x == 0)
        nup[j] = s;
    double const f = exp2(static_cast<double>(s));
    for (int h = threadIdx.x; h < k; h += blockDim.x)
        bbar[static_cast<std::size_t>(j) * kp + h] =
            static_cast<signed char>(
                ceil(fabs(b[static_cast<std::size_t>(h) * n + j]) * f));
}

/*  Lines 8 to 12. Dbar = single_triangle_up(Cbar) -- the directed rounding
    is the point, an int-to-float conversion that rounded to nearest could
    understate the magnitude and put condition (5) at risk. Cbar is column
    major m x n.

    `rows` selects which margin is reduced: the row maxima give e and hence
    mu, the column maxima give f and hence nu. */
__global__ void k_shifts(
    int const     m,
    int const     n,
    int const    *cbar,
    int const    *pre,
    double const  pprime,
    double const  c_step,
    bool const    rows,
    int          *shift) {

    __shared__ float red[TPB];
    int const idx = blockIdx.x;
    int const len = rows? n : m;
    if (idx >= (rows? m : n))
        return;

    float v = 0.f;
    for (int t = threadIdx.x; t < len; t += blockDim.x) {
        int const c = rows? cbar[static_cast<std::size_t>(t) * m + idx]
                          : cbar[static_cast<std::size_t>(idx) * m + t];
        v = fmaxf(v, __int2float_ru(c));
    }
    red[threadIdx.x] = v;
    __syncthreads();
    for (int o = blockDim.x >> 1; o > 0; o >>= 1) {
        if (threadIdx.x < o)
            red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + o]);
        __syncthreads();
    }
    if (threadIdx.x != 0)
        return;

    float const e = log2f(red[0]);
    /*  The fma is fp64 on fp32-valued arguments, as the source specifies;
        rounding it in fp32 would move the floor by one on ties. */
    double const step = fma(c_step, static_cast<double>(e), pprime);
    shift[idx] = pre[idx] + static_cast<int>(floor(step));
}

/*  Lines 13 and 14: A' = trunc(2^mu A) and B' = trunc(B 2^nu), as int32.
    31 bits at N = 8, so this is the narrowest container that holds them. */
__global__ void k_scale_trunc_a(
    int const     m,
    int const     k,
    int const     kp,
    double const *a,
    int const    *mu,
    int          *ap) {

    int const i = blockIdx.x;
    if (i >= m)
        return;
    double const f = exp2(static_cast<double>(mu[i]));
    for (int h = threadIdx.x; h < k; h += blockDim.x)
        ap[static_cast<std::size_t>(i) * kp + h] = static_cast<int>(
            trunc(a[static_cast<std::size_t>(i) * k + h] * f));
}

__global__ void k_scale_trunc_b(
    int const     k,
    int const     kp,
    int const     n,
    double const *b,
    int const    *nu,
    int          *bp) {

    int const j = blockIdx.x;
    if (j >= n)
        return;
    double const f = exp2(static_cast<double>(nu[j]));
    for (int h = threadIdx.x; h < k; h += blockDim.x)
        bp[static_cast<std::size_t>(j) * kp + h] =
            static_cast<int>(trunc(b[static_cast<std::size_t>(h) * n + j] * f));
}

/*  Residues for every modulus at once. int8 with the natural wrap: the
    symmetric range is [-floor(p/2), floor(p/2)], which is inside int8 for
    every modulus but 256, where the single value 128 wraps to -128. That
    is harmless and the source says so -- -128 == 128 (mod 256) -- and the
    int32 accumulation that follows is reduced mod p again anyway. */
__global__ void k_residues(
    int const        len,
    int const        n_moduli,
    int const       *p,
    int const       *src,
    signed char     *dst) {

    int const t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= len)
        return;
    int const v = src[t];
    for (int l = 0; l != n_moduli; ++l) {
        int r = v % p[l];
        if (r < 0)
            r += p[l];
        if (r > p[l] / 2)
            r -= p[l];
        dst[static_cast<std::size_t>(l) * len + t] =
            static_cast<signed char>(r);
    }
}

/*  Algorithm 3 plus line 15 of Algorithm 1: reduce each modulus' int32
    product into the symmetric range, reconstruct, and undo the shifts.
    Products are column major m x n; C is written ROW major. */
__global__ void k_crt_unscale(
    int const                   m,
    int const                   n,
    int const                   n_moduli,
    int const                  *prod,
    ozaki2::crt_constants const c,
    int const                  *mu,
    int const                  *nu,
    bool const                  free_cfg,
    double                     *out) {

    std::size_t const g = static_cast<std::size_t>(blockIdx.x) * blockDim.x
                        + threadIdx.x;
    if (g >= static_cast<std::size_t>(m) * n)
        return;
    int const i = static_cast<int>(g / n), j = static_cast<int>(g % n);

    double w[ozaki2::MAX_MODULI];
    for (int l = 0; l != n_moduli; ++l) {
        int const v = prod[static_cast<std::size_t>(l) * m * n
                           + static_cast<std::size_t>(j) * m + i];
        w[l] = static_cast<double>(ozaki2::sym_mod(v, c.p[l]));
    }

    double cpp;
    if (free_cfg) {
        float hi, lo, lost;
        ozaki2::crt_fp32free(w, c, ozaki2::MAX_LIMB, hi, lo, lost);
        cpp = static_cast<double>(hi) + static_cast<double>(lo);
    } else {
        cpp = ozaki2::crt_fp64(w, c);
    }

    out[g] = cpp * exp2(-static_cast<double>(mu[i]))
                 * exp2(-static_cast<double>(nu[j]));
}

} /* anonymous namespace */

struct state {
    std::size_t m = 0, k = 0, n = 0, kp = 0;
    int n_moduli = 0;
    ozaki2::crt_constants c;
    cublasHandle_t blas = nullptr;

    double *amax = nullptr, *bmax = nullptr;
    int    *mup = nullptr, *nup = nullptr, *mu = nullptr, *nu = nullptr;
    int    *dp = nullptr;                 /* the moduli, on the device */
    signed char *abar = nullptr, *bbar = nullptr;
    int    *cbar = nullptr;
    int    *ap = nullptr, *bp = nullptr;
    signed char *ares = nullptr, *bres = nullptr;
    int    *prod = nullptr;
};

namespace {

template <typename T>
bool grab(T *&p, std::size_t const count) {
    return CUDA_CHECK(cudaMalloc(&p, count * sizeof(T)));
}

} /* anonymous namespace */

state *create(
    std::size_t const m,
    std::size_t const k,
    std::size_t const n,
    int const         n_moduli) {

    state *s = new state;
    s->m = m; s->k = k; s->n = n; s->kp = pad4(k);
    s->n_moduli = (n_moduli < ozaki2::MAX_MODULI)? n_moduli
                                                 : ozaki2::MAX_MODULI;
    s->c = ozaki2::make_crt_constants(s->n_moduli);

    if (!CUBLAS_CHECK(cublasCreate(&s->blas))) {
        destroy(s);
        return nullptr;
    }

    std::size_t const nm = static_cast<std::size_t>(s->n_moduli);
    bool ok = grab(s->amax, m) && grab(s->bmax, n)
           && grab(s->mup, m)  && grab(s->nup, n)
           && grab(s->mu, m)   && grab(s->nu, n)
           && grab(s->dp, ozaki2::MAX_MODULI)
           && grab(s->abar, m * s->kp) && grab(s->bbar, s->kp * n)
           && grab(s->cbar, m * n)
           && grab(s->ap, m * s->kp)   && grab(s->bp, s->kp * n)
           && grab(s->ares, nm * m * s->kp)
           && grab(s->bres, nm * s->kp * n)
           && grab(s->prod, nm * m * n);
    if (!ok) {
        destroy(s);
        return nullptr;
    }

    /*  Zero once: the pad columns are never written again. */
    CUDA_CHECK(cudaMemset(s->abar, 0, m * s->kp));
    CUDA_CHECK(cudaMemset(s->bbar, 0, s->kp * n));
    CUDA_CHECK(cudaMemset(s->ap, 0, m * s->kp * sizeof(int)));
    CUDA_CHECK(cudaMemset(s->bp, 0, s->kp * n * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(s->dp, s->c.p,
                          ozaki2::MAX_MODULI * sizeof(int),
                          cudaMemcpyHostToDevice));
    return s;
}

void destroy(state *s) {

    if (s == nullptr)
        return;
    cudaFree(s->amax); cudaFree(s->bmax);
    cudaFree(s->mup);  cudaFree(s->nup);
    cudaFree(s->mu);   cudaFree(s->nu);   cudaFree(s->dp);
    cudaFree(s->abar); cudaFree(s->bbar); cudaFree(s->cbar);
    cudaFree(s->ap);   cudaFree(s->bp);
    cudaFree(s->ares); cudaFree(s->bres); cudaFree(s->prod);
    if (s->blas != nullptr)
        cublasDestroy(s->blas);
    delete s;
}

bool gemm(
    state        *s,
    double const *d_a,
    double const *d_b,
    double       *d_c,
    config const  cfg) {

    if (s == nullptr)
        return false;

    int const m = static_cast<int>(s->m);
    int const k = static_cast<int>(s->k);
    int const n = static_cast<int>(s->n);
    int const kp = static_cast<int>(s->kp);

    k_absmax_rows<<<m, TPB>>>(m, k, d_a, s->amax);
    KERNEL_CHECK();
    k_absmax_cols<<<n, TPB>>>(k, n, d_b, s->bmax);
    KERNEL_CHECK();
    k_prescale_a<<<m, TPB>>>(m, k, kp, d_a, s->amax, s->mup, s->abar);
    KERNEL_CHECK();
    k_prescale_b<<<n, TPB>>>(k, kp, n, d_b, s->bmax, s->nup, s->bbar);
    KERNEL_CHECK();

    int const one = 1, zero = 0;
    if (!CUBLAS_CHECK(cublasGemmEx(
            s->blas, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &one,
            s->abar, CUDA_R_8I, kp, s->bbar, CUDA_R_8I, kp, &zero,
            s->cbar, CUDA_R_32I, m, CUBLAS_COMPUTE_32I,
            CUBLAS_GEMM_DEFAULT)))
        return false;

    k_shifts<<<m, TPB>>>(m, n, s->cbar, s->mup,
                                 s->c.Pprime, s->c.c_step, true, s->mu);
    KERNEL_CHECK();
    k_shifts<<<n, TPB>>>(m, n, s->cbar, s->nup,
                                 s->c.Pprime, s->c.c_step, false, s->nu);
    KERNEL_CHECK();

    k_scale_trunc_a<<<m, TPB>>>(m, k, kp, d_a, s->mu, s->ap);
    KERNEL_CHECK();
    k_scale_trunc_b<<<n, TPB>>>(k, kp, n, d_b, s->nu, s->bp);
    KERNEL_CHECK();

    int const la = m * kp, lb = kp * n;
    k_residues<<<(la + TPB - 1) / TPB, TPB>>>(
                la, s->n_moduli, s->dp, s->ap, s->ares);
    KERNEL_CHECK();
    k_residues<<<(lb + TPB - 1) / TPB, TPB>>>(
                lb, s->n_moduli, s->dp, s->bp, s->bres);
    KERNEL_CHECK();

    for (int l = 0; l != s->n_moduli; ++l)
        if (!CUBLAS_CHECK(cublasGemmEx(
                s->blas, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &one,
                s->ares + static_cast<std::size_t>(l) * la, CUDA_R_8I, kp,
                s->bres + static_cast<std::size_t>(l) * lb, CUDA_R_8I, kp,
                &zero,
                s->prod + static_cast<std::size_t>(l) * m * n,
                CUDA_R_32I, m, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT)))
            return false;

    std::size_t const out = static_cast<std::size_t>(m) * n;
    k_crt_unscale<<<(out + TPB - 1) / TPB, TPB>>>(
                m, n, s->n_moduli, s->prod, s->c, s->mu, s->nu,
                cfg == config::ff, d_c);
    KERNEL_CHECK();

    return CUDA_CHECK(cudaGetLastError());
}

void copy_scaling(state const *s, scaling_view &out) {

    if (s == nullptr)
        return;
    out.mu.resize(s->m);
    out.nu.resize(s->n);
    out.ap.resize(s->m * s->k);
    out.bp.resize(s->k * s->n);
    std::vector<int> pa(s->m * s->kp), pb(s->kp * s->n);
    CUDA_CHECK(cudaMemcpy(out.mu.data(), s->mu, s->m * sizeof(int),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(out.nu.data(), s->nu, s->n * sizeof(int),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(pa.data(), s->ap, pa.size() * sizeof(int),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(pb.data(), s->bp, pb.size() * sizeof(int),
                          cudaMemcpyDeviceToHost));
    for (std::size_t i = 0; i != s->m; ++i)
        for (std::size_t h = 0; h != s->k; ++h)
            out.ap[i * s->k + h] = pa[i * s->kp + h];
    for (std::size_t j = 0; j != s->n; ++j)
        for (std::size_t h = 0; h != s->k; ++h)
            out.bp[j * s->k + h] = pb[j * s->kp + h];
}

} /* namespace oii */
