#include "common/problem.h"
#include "common/error.h"

#include <cublas_v2.h>
#include <cusolverDn.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

/*  Does the spectral generator actually produce the kappa it was asked for?

    WHY THIS EXISTS. `results/rtx5090/spectral.csv` sweeps five families over
    log10(kappa) = 1..12, and its x-axis is the REQUESTED value. Nothing had
    ever checked it against the realised one. The families are built by applying
    Householder reflectors to a prescribed singular-value spectrum, so kappa is
    correct by construction *if* the reflectors are orthogonal to working
    precision and the spectrum is laid out as intended — two assumptions, both
    plausible, neither measured. A generator that quietly compressed the
    spectrum would leave every conclusion in that file drawn against a wrong
    axis, and would look like nothing at all.

    WHY IT DOES NOT RE-RUN THE SOLVERS. Measuring kappa needs only the matrix,
    not a solve. This tool constructs each (family, param, n) exactly as
    `lps-profile` does, measures kappa, and emits a small table that is JOINED
    onto the existing spectral.csv by `profile/inject_kappa.py`. Re-running the
    240-row sweep to add one column would cost hours and would also re-roll
    every timing measurement in it for no reason.

    CONVERGENCE IS REPORTED, NOT ASSUMED. Inverse iteration converges at a rate
    set by sigma_{n-1}/sigma_n and an unconverged estimate is systematically
    LOW, never high — it would understate kappa and so make the generator look
    better than it is. The `converged` column exists so that cannot pass
    silently; treat a false there as "no measurement", not as a small one.

    Build:  make bin/lps-kappacheck
    Run:    bin/lps-kappacheck --families randsvd,clust_pos --sizes 4096 > k.csv
    Join:   python3 profile/inject_kappa.py results/rtx5090/spectral.csv k.csv  */

namespace {

using harness::problem;

/*  kappa_2 by power iteration on A^T A, and inverse iteration through an LU.
    Lifted from test/main_kappa.cpp; kept here rather than shared because that
    file also owns a Skeel estimate and a whole solver sweep this tool does not
    want to link against. */
double estimate_kappa(problem &prob, bool &converged) {

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

    normalize(d_v);
    double sigma_max = 0., previous = 0.;
    converged = false;
    for (int it = 0; it != 2000; ++it) {
        CUBLAS_CHECK(cublasDgemv(prob.blas, CUBLAS_OP_N, ni, ni,
                                 &one, prob.d_a, ni, d_v, 1, &zero, d_w, 1));
        CUBLAS_CHECK(cublasDgemv(prob.blas, CUBLAS_OP_T, ni, ni,
                                 &one, prob.d_a, ni, d_w, 1, &zero, d_v, 1));
        sigma_max = std::sqrt(normalize(d_v));
        if (it > 4 && std::fabs(sigma_max - previous) <= 1e-8 * sigma_max)
            break;
        previous = sigma_max;
    }

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
    previous = 0.;
    for (int it = 0; it != 2000; ++it) {
        CUSOLVER_CHECK(cusolverDnDgetrs(prob.solver, CUBLAS_OP_T,
                                        ni, 1, d_a, ni, d_ipiv, d_v, ni,
                                        d_info));
        CUSOLVER_CHECK(cusolverDnDgetrs(prob.solver, CUBLAS_OP_N,
                                        ni, 1, d_a, ni, d_ipiv, d_v, ni,
                                        d_info));
        double const growth = normalize(d_v);
        sigma_min = 1. / std::sqrt(growth);
        if (it > 4 && std::fabs(sigma_min - previous) <= 1e-8 * sigma_min) {
            converged = true;
            break;
        }
        previous = sigma_min;
    }

    return (sigma_min > 0.)? sigma_max / sigma_min : 0.;
}

/*  These two mirror main_profile.cpp. They are duplicated rather than shared
    because sharing them would mean exporting the profile driver's internals;
    if a family is added there and not here, this tool reports `near_random`
    for it and the ratio column will be visibly wrong rather than silently
    plausible. */
harness::matrix_kind kind_of(std::string const &name) {
    if (name == "graded")    return harness::matrix_kind::graded_diagonal;
    if (name == "spd")       return harness::matrix_kind::spd_graded;
    if (name == "wilkinson") return harness::matrix_kind::wilkinson;
    if (name == "diag")      return harness::matrix_kind::diag_dominant;
    if (name == "randsvd")   return harness::matrix_kind::randsvd_log;
    if (name == "clust_pos") return harness::matrix_kind::eig_clustered_pos;
    if (name == "clust_sgn") return harness::matrix_kind::eig_clustered_signed;
    if (name == "arith_pos") return harness::matrix_kind::eig_arith_pos;
    if (name == "arith_sgn") return harness::matrix_kind::eig_arith_signed;
    return harness::matrix_kind::near_random;
}

bool is_spectral(std::string const &f) {
    return f == "randsvd" || f == "clust_pos" || f == "clust_sgn" ||
           f == "arith_pos" || f == "arith_sgn";
}

std::vector<double> params_for(std::string const &fam) {
    std::vector<double> p;
    if (fam == "graded" || fam == "spd")
        for (double d = 0.; d <= 12.; d += 2.) p.push_back(d);
    else if (fam == "wilkinson")
        for (double m = 4.; m <= 40.; m += 8.) p.push_back(m);
    else if (is_spectral(fam))
        for (double d = 1.; d <= 12.; d += 1.) p.push_back(d);
    else
        p.push_back(-1.);
    return p;
}

std::vector<std::string> split_list(std::string const &s) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) if (!item.empty()) out.push_back(item);
    return out;
}

} /* namespace */

int main(int argc, char **argv) {

    std::cout.setf(std::ios::unitbuf);

    std::vector<std::string> families =
        {"randsvd", "clust_pos", "clust_sgn", "arith_pos", "arith_sgn"};
    std::vector<std::string> sizes = {"4096"};

    for (int i = 1; i < argc; ++i) {
        std::string const a = argv[i];
        auto next = [&]() { return (i + 1 < argc)? argv[++i] : ""; };
        if      (a == "--families") families = split_list(next());
        else if (a == "--sizes")    sizes    = split_list(next());
        else if (a == "--help") {
            std::cerr <<
              "usage: lps-kappacheck [--families f,...] [--sizes n,...]\n"
              "  measures realised kappa_2 for each (family, param, n)\n";
            return 0;
        }
    }

    std::cout << "matrix,param,n,kappa_requested,kappa_measured,ratio,"
                 "converged\n";

    for (std::string const &fam : families) {
        for (double const param : params_for(fam)) {
            for (std::string const &sz : sizes) {

                std::size_t const n = std::atoll(sz.c_str());
                if (n == 0) continue;   /* an n=0 problem drives cuBLAS into
                                           "illegal value" and emits a row of
                                           zeros that parses as data */

                /*  k=1: this tool never solves, and a full right-hand side
                    block would be pure allocation. */
                problem prob(n, 1, kind_of(fam), 7u, param);

                bool converged = false;
                double const measured = estimate_kappa(prob, converged);

                /*  For the spectral families `param` IS log10(kappa). For the
                    others there is no requested value to compare against, so
                    the ratio is meaningless and is emitted as -1 rather than
                    as a number that invites comparison. */
                double const requested =
                    is_spectral(fam)? std::pow(10., param) : -1.;
                double const ratio =
                    (requested > 0. && measured > 0.)? measured / requested
                                                     : -1.;

                std::cout << fam << ',' << param << ',' << n << ','
                          << requested << ',' << measured << ',' << ratio
                          << ',' << (converged? 1 : 0) << '\n';
            }
        }
    }

    return 0;
}
