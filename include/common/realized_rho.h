#pragma once
/*  The realized power iteration on the stationary operator as solved:
    v <- (LU)^-1 A v - v, normalised each step; the last norm is the
    estimate of rho(M). Host GEMV, device apply_inverse. Shared by the
    campaign's verdict and the closed-loop driver's probe so both read
    one number. */
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
    double *d_v = static_cast<double *>(prob.acquire(n * sizeof(double)));
    double *d_w = static_cast<double *>(prob.acquire(n * sizeof(double)));
    std::vector<double> v(n), Av(n), Mv(n);
    for (std::size_t i = 0; i != n; ++i)
        v[i] = std::sin(0.3 * static_cast<double>(i) + 1.);
    double d0 = 0.;
    for (double const q : v) d0 += q * q;
    d0 = std::sqrt(d0);
    for (double &q : v) q /= d0;

    double lam = 0.;
    for (int t = 0; t != steps; ++t) {
        for (std::size_t i = 0; i != n; ++i) {
            double acc = 0.;
            for (std::size_t j = 0; j != n; ++j)
                acc += A[j * n + i] * v[j];
            Av[i] = acc;
        }
        CUDA_CHECK(cudaMemcpy(d_v, Av.data(), n * sizeof(double),
                              cudaMemcpyHostToDevice));
        qlu::apply_inverse(st, d_w, d_v);
        CUDA_CHECK(cudaMemcpy(Mv.data(), d_w, n * sizeof(double),
                              cudaMemcpyDeviceToHost));
        double nv = 0.;
        for (std::size_t i = 0; i != n; ++i) {
            Mv[i] -= v[i];
            nv += Mv[i] * Mv[i];
        }
        nv = std::sqrt(nv);
        if (nv == 0.)
            return 0.;
        lam = nv;
        for (std::size_t i = 0; i != n; ++i)
            v[i] = Mv[i] / nv;
    }
    return lam;
}

} // namespace harness
