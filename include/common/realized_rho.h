#pragma once
/*  The realized power iteration on the stationary operator as solved:
    v <- (LU)^-1 A v - v, normalised each step; the last norm is the
    estimate of rho(M). Shared by the campaign's verdict and the
    closed-loop driver's probe so both read one number.

    A v runs on the device. It used to run on the host, as a triple loop
    over the fp64 reference with the inner index striding by n, which made
    the probe the most expensive thing in the solver by three orders of
    magnitude: 10.4 s per 60-step probe at n = 8192 against 6.5 ms for the
    whole factor and solve, scaling as n^3.14 rather than n^2 because the
    access pattern misses cache on every read. The reference operator is
    already resident and column major, which is what dgemv wants, so the
    product is one cuBLAS call and the vector work stays on the device too:
    one scalar comes back per step instead of two vectors.

    This is not bit-identical to the host loop -- dgemv sums in a different
    order -- so it is a verdict-identical change, and the value agrees to
    the last few digits. */
#include "common/qlu.h"
#include "common/error.h"
#include "common/problem.h"

#include <cmath>
#include <vector>

namespace harness {

inline double realized_rho(qlu::state *st, harness::problem &prob,
                           std::vector<double> const &A, std::size_t const n,
                           int const steps) {
    if (steps <= 0)
        return 0.;
    double *d_v  = static_cast<double *>(prob.acquire(n * sizeof(double)));
    double *d_w  = static_cast<double *>(prob.acquire(n * sizeof(double)));
    double *d_av = static_cast<double *>(prob.acquire(n * sizeof(double)));

    /*  One handle for the process. The probe is called once per rung by the
        closed loop and once per pair by the campaign, and cublasCreate is
        not cheap enough to pay per call. */
    static cublasHandle_t bl = nullptr;
    if (bl == nullptr)
        CUBLAS_CHECK(cublasCreate(&bl));

    /*  The same starting vector as before, so a rho from this build is
        comparable with one from the host version. */
    std::vector<double> v(n);
    for (std::size_t i = 0; i != n; ++i)
        v[i] = std::sin(0.3 * static_cast<double>(i) + 1.);
    double d0 = 0.;
    for (double const q : v) d0 += q * q;
    d0 = std::sqrt(d0);
    for (double &q : v) q /= d0;
    CUDA_CHECK(cudaMemcpy(d_v, v.data(), n * sizeof(double),
                          cudaMemcpyHostToDevice));

    double const one = 1., zero = 0., minus_one = -1.;
    int const ni = static_cast<int>(n);
    double lam = 0.;
    for (int t = 0; t != steps; ++t) {
        CUBLAS_CHECK(cublasDgemv(bl, CUBLAS_OP_N, ni, ni, &one,
                                 prob.d_a, ni, d_v, 1, &zero, d_av, 1));
        qlu::apply_inverse(st, d_w, d_av);
        CUBLAS_CHECK(cublasDaxpy(bl, ni, &minus_one, d_v, 1, d_w, 1));
        double nv = 0.;
        CUBLAS_CHECK(cublasDnrm2(bl, ni, d_w, 1, &nv));
        if (nv == 0.)
            return 0.;
        lam = nv;
        double const inv = 1. / nv;
        CUBLAS_CHECK(cublasDcopy(bl, ni, d_w, 1, d_v, 1));
        CUBLAS_CHECK(cublasDscal(bl, ni, &inv, d_v, 1));
    }
    (void)A;   /*  kept in the signature: the callers hold the host copy and
                   the DF32 operator the solver applies is not this one */
    return lam;
}

} // namespace harness
