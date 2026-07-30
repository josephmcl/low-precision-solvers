#include "common/solver.h"
#include "common/metrics.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <vector>

/*  Where do the fp64-free schemes break down as a function of conditioning?
    Sweep the diagonal shift, estimate kappa, and score every method at each
    point.

    This is the axis the selection rule is missing. The rule so far reads "these
    schemes pay when fp32:fp64 is large", which predicts speed and says nothing
    about when the answer stops being usable — and the answer does stop: on a
    near-random matrix R-IR floors near 1e-10 while split-MPIR, configured
    correctly, reaches 2.5e-17. Four named matrix classes cannot locate that
    transition; a continuous sweep can.

    A = rand(-1,1) + shift*I. Large shift is diagonally dominant; shift -> 0
    approaches singular. */

namespace {

using harness::problem;

/*  kappa_2 by power iteration, using an fp64 factorization of A.

    sigma_max: power iteration on A^T A.
    sigma_min: the same on (A^T A)^-1, applied through the LU rather than an
    explicit inverse — two triangular solves per step.

    Power iteration converges linearly and is not accurate to many digits, but
    the quantity of interest here spans ten orders and the answer only needs to
    be right to a factor of about two. Reported as an order of magnitude for
    that reason. */
double estimate_kappa(problem &prob) {

    std::size_t const n = prob.n;
    int const ni = static_cast<int>(n);

    double *d_a = static_cast<double *>(prob.acquire(n * n * sizeof(double)));
    double *d_v = static_cast<double *>(prob.acquire(n * sizeof(double)));
    double *d_w = static_cast<double *>(prob.acquire(n * sizeof(double)));
    int    *d_ipiv = static_cast<int *>(prob.acquire(n * sizeof(int)));
    int    *d_info = static_cast<int *>(prob.acquire(sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_a, prob.d_a, n * n * sizeof(double),
                          cudaMemcpyDeviceToDevice));

    std::vector<double> host(n, 1.);
    for (std::size_t i = 0; i != n; ++i)
        host[i] = 1. + 0.001 * static_cast<double>(i % 97);
    CUDA_CHECK(cudaMemcpy(d_v, host.data(), n * sizeof(double),
                          cudaMemcpyHostToDevice));

    double const one = 1., zero = 0.;
    auto normalize = [&](double *d_x) {
        double nrm = 0.;
        CUBLAS_CHECK(cublasDnrm2(prob.blas, ni, d_x, 1, &nrm));
        if (nrm > 0.) {
            double const inv = 1. / nrm;
            CUBLAS_CHECK(cublasDscal(prob.blas, ni, &inv, d_x, 1));
        }
        return nrm;
    };

    /*  sigma_max: v <- A^T (A v), normalized. */
    normalize(d_v);
    double sigma_max = 0.;
    for (int it = 0; it != 40; ++it) {
        CUBLAS_CHECK(cublasDgemv(prob.blas, CUBLAS_OP_N, ni, ni,
                                 &one, prob.d_a, ni, d_v, 1, &zero, d_w, 1));
        CUBLAS_CHECK(cublasDgemv(prob.blas, CUBLAS_OP_T, ni, ni,
                                 &one, prob.d_a, ni, d_w, 1, &zero, d_v, 1));
        sigma_max = std::sqrt(normalize(d_v));
    }

    /*  Factor once, then inverse-iterate: v <- (A^T A)^-1 v via A^-1 A^-T. */
    int lwork = 0;
    CUSOLVER_CHECK(cusolverDnDgetrf_bufferSize(
        prob.solver, ni, ni, d_a, ni, &lwork));
    double *d_work = static_cast<double *>(
        prob.acquire(static_cast<std::size_t>(lwork) * sizeof(double)));
    CUSOLVER_CHECK(cusolverDnDgetrf(
        prob.solver, ni, ni, d_a, ni, d_work, d_ipiv, d_info));

    CUDA_CHECK(cudaMemcpy(d_v, host.data(), n * sizeof(double),
                          cudaMemcpyHostToDevice));
    normalize(d_v);

    double sigma_min = 0.;
    for (int it = 0; it != 40; ++it) {
        CUSOLVER_CHECK(cusolverDnDgetrs(prob.solver, CUBLAS_OP_T,
                                        ni, 1, d_a, ni, d_ipiv, d_v, ni,
                                        d_info));
        CUSOLVER_CHECK(cusolverDnDgetrs(prob.solver, CUBLAS_OP_N,
                                        ni, 1, d_a, ni, d_ipiv, d_v, ni,
                                        d_info));
        double const growth = normalize(d_v);
        sigma_min = 1. / std::sqrt(growth);
    }

    return (sigma_min > 0.)? sigma_max / sigma_min : 0.;
}

} /* namespace */

int main(int argc, char **argv) {

    std::cout.setf(std::ios::unitbuf);

    std::size_t const n = (argc > 1)?
        static_cast<std::size_t>(std::atoll(argv[1])) : 4096;
    std::size_t const k = (argc > 2)?
        static_cast<std::size_t>(std::atoll(argv[2])) : 256;

    /*  Shifts spanning diagonally dominant (shift = n) down toward singular.
        Logarithmic, because kappa moves by orders across this range. */
    std::vector<double> shifts;
    for (double s = static_cast<double>(n); s >= 0.5; s /= 4.)
        shifts.push_back(s);
    shifts.push_back(0.25);
    shifts.push_back(0.125);

    std::cout << "conditioning sweep   n=" << n << "  k=" << k << "\n"
              << "A = rand(-1,1) + shift*I; kappa_2 by power iteration\n\n";

    std::cout << "  " << std::right
              << std::setw(9)  << "shift"
              << std::setw(11) << "kappa_2"
              << std::setw(14) << "direct fp64"
              << std::setw(14) << "split-MPIR"
              << std::setw(14) << "R-IR"
              << std::setw(14) << "vendor IRS" << "\n";

    std::vector<solver::method> const &methods = solver::registry();

    for (std::size_t si = 0; si != shifts.size(); ++si) {

        problem prob(n, k, harness::matrix_kind::near_random, 7u, shifts[si]);

        double const kappa = estimate_kappa(prob);

        double *d_x = static_cast<double *>(
            prob.acquire(n * k * sizeof(double)));

        std::cout << "  " << std::right << std::scientific
                  << std::setprecision(1) << std::setw(9) << shifts[si]
                  << std::setw(11) << kappa;

        for (std::size_t mi = 0; mi != methods.size(); ++mi) {
            solver::state st;
            solver::run(d_x, prob.d_b, methods[mi], st, prob);
            metrics::report const err = metrics::evaluate(d_x, nullptr, prob);
            std::cout << std::setw(14) << std::setprecision(2) << err.backward;
        }
        std::cout << "\n";
    }

    std::cout << "\n  Backward error against the untouched fp64 A. Direct fp64"
                 " is the control:\n  where IT degrades, the problem is hard"
                 " rather than the method being bad.\n";

    return 0;
}
