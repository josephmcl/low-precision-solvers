#include "common/rhs_reference.h"
#include "common/error.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <vector>

namespace harness {

namespace {

/*  Double-double on the device: the same error-free transformations as
    reference.h (TwoSum, FastTwoSum, TwoProd by FMA, AccurateDWPlusDW). The
    build refuses fast-math, so these are exact as written. */
struct ddv { double hi, lo; };

__device__ __forceinline__ ddv d_two_sum(double a, double b) {
    double const s = a + b, bb = s - a; return {s, (a - (s - bb)) + (b - bb)};
}
__device__ __forceinline__ ddv d_fast_two_sum(double a, double b) {
    double const s = a + b; return {s, b - (s - a)};
}
__device__ __forceinline__ ddv d_add(ddv x, ddv y) {
    ddv const s = d_two_sum(x.hi, y.hi), t = d_two_sum(x.lo, y.lo);
    ddv const v = d_fast_two_sum(s.hi, s.lo + t.hi);
    return d_fast_two_sum(v.hi, t.lo + v.lo);
}
/*  a (double) times (xh + xl), as a double-double */
__device__ __forceinline__ ddv d_mul(double a, double xh, double xl) {
    double const p = a * xh, e = __fma_rn(a, xh, -p);
    return d_fast_two_sum(p, __fma_rn(a, xl, e));
}

/*  out = c - A x, A column major n x n, x = xh + xl, c = ch + cl; one thread a
    row, so a warp reads 32 consecutive entries of a column. */
__global__ void k_dd_residual(std::size_t n, double const *A, double const *xh, double const *xl,
                              double const *ch, double const *cl, double *oh, double *ol) {
    std::size_t const i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (i >= n) return;
    ddv acc = {ch ? ch[i] : 0., cl ? cl[i] : 0.};
    for (std::size_t j = 0; j != n; ++j) {
        ddv p = d_mul(A[i + j * n], xh[j], xl ? xl[j] : 0.);
        acc = d_add(acc, {-p.hi, -p.lo});
    }
    oh[i] = acc.hi; ol[i] = acc.lo;
}

__global__ void k_uniform(std::size_t m, double *x, unsigned seed) {
    std::size_t const i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (i >= m) return;
    unsigned t = static_cast<unsigned>(i) * 0x9e3779b9u ^ (seed * 0x85ebca6bu + 0x27d4eb2fu);
    t ^= t >> 16; t *= 0x7feb352du; t ^= t >> 15; t *= 0x846ca68bu; t ^= t >> 16;
    x[i] = (static_cast<double>(t >> 8) / 16777216. - 0.5) * 2.;
}

__global__ void k_round(std::size_t m, double const *h, double const *l, double *out) {
    std::size_t const i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (i < m) out[i] = h[i] + l[i];
}

__global__ void k_neg_round(std::size_t m, double const *h, double const *l, double *out) {
    std::size_t const i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (i < m) out[i] = -(h[i] + l[i]);
}

/*  x (dd) += d */
__global__ void k_dd_accumulate(std::size_t m, double *xh, double *xl, double const *d) {
    std::size_t const i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (i >= m) return;
    ddv const r = d_add({xh[i], xl[i]}, {d[i], 0.});
    xh[i] = r.hi; xl[i] = r.lo;
}

/*  y = |M| |v| for a column-major n x n M */
__global__ void k_abs_matvec(std::size_t n, double const *M, double const *v, double *y) {
    std::size_t const i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (i >= n) return;
    double s = 0.;
    for (std::size_t j = 0; j != n; ++j) s += std::fabs(M[i + j * n]) * std::fabs(v[j]);
    y[i] = s;
}

/*  row sums of |M| */
__global__ void k_abs_rowsum(std::size_t n, double const *M, double *y) {
    std::size_t const i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (i >= n) return;
    double s = 0.;
    for (std::size_t j = 0; j != n; ++j) s += std::fabs(M[i + j * n]);
    y[i] = s;
}

__global__ void k_unit_diagonal(std::size_t n, double *M) {
    std::size_t const i = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
    if (i < n) M[i + i * n] = 1.;
}

unsigned blocks(std::size_t m) { return static_cast<unsigned>((m + 255) / 256); }

double inf_norm(std::size_t m, double const *d) {
    std::vector<double> h(m);
    CUDA_CHECK(cudaMemcpy(h.data(), d, m * sizeof(double), cudaMemcpyDeviceToHost));
    double r = 0.;
    for (double v : h) r = std::max(r, std::fabs(v));
    return r;
}

/*  FP64 LU of a copy of A, kept for the session of one problem */
struct lu_t {
    double *a = nullptr; int *ipiv = nullptr; int *info = nullptr; double *work = nullptr;
    void factor(problem &prob) {
        std::size_t const n = prob.n; int const ni = static_cast<int>(n);
        CUDA_CHECK(cudaMalloc(&a, n * n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&ipiv, n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&info, sizeof(int)));
        CUDA_CHECK(cudaMemcpy(a, prob.d_a, n * n * sizeof(double), cudaMemcpyDeviceToDevice));
        int lw = 0;
        CUSOLVER_CHECK(cusolverDnDgetrf_bufferSize(prob.solver, ni, ni, a, ni, &lw));
        CUDA_CHECK(cudaMalloc(&work, static_cast<std::size_t>(lw) * sizeof(double)));
        CUSOLVER_CHECK(cusolverDnDgetrf(prob.solver, ni, ni, a, ni, work, ipiv, info));
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    void solve(problem &prob, double *b, int nrhs, bool trans = false) {
        int const ni = static_cast<int>(prob.n);
        CUSOLVER_CHECK(cusolverDnDgetrs(prob.solver, trans ? CUBLAS_OP_T : CUBLAS_OP_N, ni, nrhs, a, ni, ipiv, b, ni, info));
    }
    ~lu_t() { cudaFree(a); cudaFree(ipiv); cudaFree(info); cudaFree(work); }
};

} /* namespace */

void make_xtrue_rhs(problem &prob, double *d_xtrue, unsigned seed) {
    std::size_t const n = prob.n, k = prob.k;
    k_uniform<<<blocks(n * k), 256>>>(n * k, d_xtrue, seed + 1234567u);
    KERNEL_CHECK();
    double *h = nullptr, *l = nullptr;
    CUDA_CHECK(cudaMalloc(&h, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&l, n * sizeof(double)));
    for (std::size_t c = 0; c != k; ++c) {
        /*  -(0 - A x) = A x, accumulated in double-double, then rounded */
        k_dd_residual<<<blocks(n), 256>>>(n, prob.d_a, d_xtrue + c * n, nullptr, nullptr, nullptr, h, l);
        KERNEL_CHECK();
        k_neg_round<<<blocks(n), 256>>>(n, h, l, prob.d_b + c * n);
        KERNEL_CHECK();
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaFree(h); cudaFree(l);
}

reference_info reference_solution(problem &prob, double const *d_xtrue, double *d_xref) {
    std::size_t const n = prob.n, k = prob.k;
    reference_info info;
    lu_t lu; lu.factor(prob);
    double *xh, *xl, *rh, *rl, *d;
    CUDA_CHECK(cudaMalloc(&xh, n * sizeof(double))); CUDA_CHECK(cudaMalloc(&xl, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&rh, n * sizeof(double))); CUDA_CHECK(cudaMalloc(&rl, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d, n * sizeof(double)));
    double worst_err = 0.;
    for (std::size_t c = 0; c != k; ++c) {
        double const *b = prob.d_b + c * n;
        CUDA_CHECK(cudaMemcpy(xh, b, n * sizeof(double), cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemset(xl, 0, n * sizeof(double)));
        lu.solve(prob, xh, 1);
        int it = 0; double last = 1., prev = 1e300;
        for (; it != 12; ++it) {
            k_dd_residual<<<blocks(n), 256>>>(n, prob.d_a, xh, xl, b, nullptr, rh, rl);
            KERNEL_CHECK();
            k_round<<<blocks(n), 256>>>(n, rh, rl, d);
            KERNEL_CHECK();
            lu.solve(prob, d, 1);
            k_dd_accumulate<<<blocks(n), 256>>>(n, xh, xl, d);
            KERNEL_CHECK();
            double const dn = inf_norm(n, d), xn = inf_norm(n, xh);
            last = (xn > 0.) ? dn / xn : 0.;
            if (last < 1e-30 || last >= prev) { ++it; break; }
            prev = last;
        }
        k_round<<<blocks(n), 256>>>(n, xh, xl, d_xref + c * n);
        KERNEL_CHECK();
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<double> xr(n), xt(n);
        CUDA_CHECK(cudaMemcpy(xr.data(), d_xref + c * n, n * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(xt.data(), d_xtrue + c * n, n * sizeof(double), cudaMemcpyDeviceToHost));
        double num = 0., den = 0.;
        for (std::size_t i = 0; i != n; ++i) { num = std::max(num, std::fabs(xr[i] - xt[i])); den = std::max(den, std::fabs(xt[i])); }
        worst_err = std::max(worst_err, den > 0. ? num / den : 0.);
        info.iterations = std::max(info.iterations, it);
        info.last_correction = std::max(info.last_correction, last);
    }
    info.err_vs_xtrue = worst_err;
    info.converged = info.last_correction < 1e-24;
    cudaFree(xh); cudaFree(xl); cudaFree(rh); cudaFree(rl); cudaFree(d);
    return info;
}

residual_norms dd_residual(problem &prob, double const *d_xh, double const *d_xl, std::size_t const column) {
    std::size_t const n = prob.n;
    double *rh = nullptr, *rl = nullptr;
    CUDA_CHECK(cudaMalloc(&rh, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&rl, n * sizeof(double)));
    k_dd_residual<<<blocks(n), 256>>>(n, prob.d_a, d_xh, d_xl, prob.d_b + column * n, nullptr, rh, rl);
    KERNEL_CHECK();
    std::vector<double> h(n), l(n), xh(n), xl(n, 0.);
    CUDA_CHECK(cudaMemcpy(h.data(), rh, n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(l.data(), rl, n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(xh.data(), d_xh, n * sizeof(double), cudaMemcpyDeviceToHost));
    if (d_xl != nullptr)
        CUDA_CHECK(cudaMemcpy(xl.data(), d_xl, n * sizeof(double), cudaMemcpyDeviceToHost));
    cudaFree(rh); cudaFree(rl);
    residual_norms out;
    for (std::size_t i = 0; i != n; ++i) {
        double const r = h[i] + l[i], x = xh[i] + xl[i];
        out.r_inf = std::max(out.r_inf, std::fabs(r)); out.r_2 += r * r;
        out.x_inf = std::max(out.x_inf, std::fabs(x)); out.x_2 += x * x;
    }
    out.r_2 = std::sqrt(out.r_2); out.x_2 = std::sqrt(out.x_2);
    return out;
}

conditioning measure_conditioning(problem &prob, double const *d_x) {
    std::size_t const n = prob.n; int const ni = static_cast<int>(n);
    conditioning out;
    lu_t lu; lu.factor(prob);
    double *v, *w;
    CUDA_CHECK(cudaMalloc(&v, n * sizeof(double))); CUDA_CHECK(cudaMalloc(&w, n * sizeof(double)));
    double const one = 1., zero = 0.;
    int const iters = 60;
    /*  sigma_max^2: power iteration on A^T A */
    k_uniform<<<blocks(n), 256>>>(n, v, 99u); KERNEL_CHECK();
    double lam = 0.;
    for (int i = 0; i != iters; ++i) {
        double nv = 0.;
        CUBLAS_CHECK(cublasDnrm2(prob.blas, ni, v, 1, &nv));
        double const s = 1. / nv;
        CUBLAS_CHECK(cublasDscal(prob.blas, ni, &s, v, 1));
        CUBLAS_CHECK(cublasDgemv(prob.blas, CUBLAS_OP_N, ni, ni, &one, prob.d_a, ni, v, 1, &zero, w, 1));
        CUBLAS_CHECK(cublasDgemv(prob.blas, CUBLAS_OP_T, ni, ni, &one, prob.d_a, ni, w, 1, &zero, v, 1));
        CUBLAS_CHECK(cublasDnrm2(prob.blas, ni, v, 1, &lam));
    }
    out.sigma_max = std::sqrt(lam);
    /*  1 / sigma_min^2: power iteration on (A^T A)^-1 = A^-1 A^-T through the LU */
    k_uniform<<<blocks(n), 256>>>(n, v, 77u); KERNEL_CHECK();
    double mu = 0.;
    for (int i = 0; i != iters; ++i) {
        double nv = 0.;
        CUBLAS_CHECK(cublasDnrm2(prob.blas, ni, v, 1, &nv));
        double const s = 1. / nv;
        CUBLAS_CHECK(cublasDscal(prob.blas, ni, &s, v, 1));
        lu.solve(prob, v, 1, true);     /* A^-T v */
        lu.solve(prob, v, 1, false);    /* A^-1 A^-T v */
        CUBLAS_CHECK(cublasDnrm2(prob.blas, ni, v, 1, &mu));
    }
    out.sigma_min = 1. / std::sqrt(mu);
    out.kappa_2 = out.sigma_max / out.sigma_min;
    out.iterations = iters;
    /*  explicit inverse: getrs with the identity */
    double *inv = nullptr;
    CUDA_CHECK(cudaMalloc(&inv, n * n * sizeof(double)));
    CUDA_CHECK(cudaMemset(inv, 0, n * n * sizeof(double)));
    k_unit_diagonal<<<blocks(n), 256>>>(n, inv); KERNEL_CHECK();
    lu.solve(prob, inv, ni);
    k_abs_rowsum<<<blocks(n), 256>>>(n, prob.d_a, w); KERNEL_CHECK();
    double const a_inf = inf_norm(n, w);
    k_abs_rowsum<<<blocks(n), 256>>>(n, inv, w); KERNEL_CHECK();
    out.kappa_inf = a_inf * inf_norm(n, w);
    k_abs_matvec<<<blocks(n), 256>>>(n, prob.d_a, d_x, w); KERNEL_CHECK();      /* |A||x| */
    k_abs_matvec<<<blocks(n), 256>>>(n, inv, w, v); KERNEL_CHECK();            /* |A^-1| |A| |x| */
    double const xn = inf_norm(n, d_x);
    out.cond_ax = (xn > 0.) ? inf_norm(n, v) / xn : 0.;
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaFree(inv); cudaFree(v); cudaFree(w);
    return out;
}

} /* namespace harness */
