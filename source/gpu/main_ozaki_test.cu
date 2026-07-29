#include "gpu/definitions.h"
#include "gpu/error.h"
#include "gpu/metrics.h"
#include "gpu/ozaki.h"
#include "gpu/problem.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <vector>

/*  Validate the Ozaki product against an fp64 reference, before either method
    depends on it.

    The reference is Dgemm on the SAME operands, promoted: fl64(fl32(A)) * X.
    Comparing against fl64(A) * X instead would fold the demotion of A into the
    error and report a floor of 2^-24 no matter how good the split is — which
    would look like the split not working. */

namespace {

__global__ void demote_kernel(
    float             *d_af,
    double const      *d_a,
    std::size_t const  n_total) {

    for (std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < n_total;
         idx += static_cast<std::size_t>(blockDim.x) * gridDim.x)
        d_af[idx] = static_cast<float>(d_a[idx]);
}

/*  Promote, applying the same triangular mask the Ozaki path reads through, so
    the reference and the test see identical operands. */
__global__ void promote_kernel(
    double            *d_a,
    float const       *d_af,
    std::size_t const  n,
    int const          which) {

    std::size_t const n_total = n * n;

    for (std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < n_total;
         idx += static_cast<std::size_t>(blockDim.x) * gridDim.x) {

        std::size_t const i = idx % n;
        std::size_t const j = idx / n;

        double v = static_cast<double>(d_af[idx]);
        if (which == 1 && j >= i) v = 0.;
        if (which == 2 && j <  i) v = 0.;
        d_a[idx] = v;
    }
}

char const *name_of(ozaki::shape const s) {

    switch (s) {
        case ozaki::shape::full:  return "full";
        case ozaki::shape::lower: return "lower (strict)";
        case ozaki::shape::upper: return "upper";
    }
    return "?";
}

} /* namespace */

int main(int argc, char **argv) {

    std::size_t const n = (argc > 1)?
        static_cast<std::size_t>(std::atoll(argv[1])) : 4096;
    std::size_t const k = (argc > 2)?
        static_cast<std::size_t>(std::atoll(argv[2])) : 256;

    harness::problem prob(n, k, harness::matrix_kind::near_random);

    float  *d_af  = static_cast<float *>(
        prob.acquire(n * n * sizeof(float)));
    double *d_am  = static_cast<double *>(
        prob.acquire(n * n * sizeof(double)));
    double *d_ref = static_cast<double *>(
        prob.acquire(n * k * sizeof(double)));
    double *d_acc = static_cast<double *>(
        prob.acquire(n * k * sizeof(double)));

    demote_kernel<<<launch::grid_for(n * n), launch::BLOCK_SIZE>>>(
        d_af, prob.d_a, n * n);
    KERNEL_CHECK();

    std::cout << "ozaki product vs fp64 reference   n = " << n
              << "   k = " << k << "\n"
              << "reference is Dgemm on fl64(fl32(A)) * X, the same operands\n\n";

    ozaki::shape const shapes[] = {
        ozaki::shape::full, ozaki::shape::lower, ozaki::shape::upper};

    for (int si = 0; si != 1; ++si) {

        ozaki::shape const which = shapes[si];

        promote_kernel<<<launch::grid_for(n * n), launch::BLOCK_SIZE>>>(
            d_am, d_af, n, static_cast<int>(which));
        KERNEL_CHECK();

        double const one = 1., zero = 0.;
        CUBLAS_CHECK(cublasDgemm(
            prob.blas,
            CUBLAS_OP_N, CUBLAS_OP_N,
            static_cast<int>(n), static_cast<int>(k), static_cast<int>(n),
            &one,
            d_am,     static_cast<int>(n),
            prob.d_b, static_cast<int>(n),
            &zero,
            d_ref,    static_cast<int>(n)));

        double const norm_ref = metrics::norm(d_ref, n * k, prob);

        std::cout << "  " << std::left << std::setw(16) << name_of(which)
                  << std::right
                  << std::setw(7)  << "pieces"
                  << std::setw(7)  << "block"
                  << std::setw(9)  << "prods"
                  << std::setw(13) << "rel err" << "\n";

        /*  Sweeping the piece count is the real check. Accuracy must improve
            monotonically until it hits the fp64 reference's own floor; a flat
            or non-monotonic curve means the pieces are not landing on a common
            grid, which is the failure mode that makes a naive split look like
            it works while delivering plain TF32. */
        int const blocks[] = {64, 256, 1024, 3072};
        for (int bi = 0; bi != 4; ++bi)
        for (int n_pieces = 3; n_pieces <= 6; n_pieces += 3) {

            ozaki::config cfg;
            cfg.n_pieces = n_pieces;
            cfg.bits     = 9;
            cfg.block    = blocks[bi];
            cfg.n_groups = n_pieces;

            ozaki::workspace ws(n, k, cfg, prob);

            CUDA_CHECK(cudaMemset(d_acc, 0, n * k * sizeof(double)));

            ozaki::row_max(ws.d_mu, d_af, n, n, which, prob);
            ozaki::column_max(ws.d_nu, prob.d_b, n, k, prob);
            CUDA_CHECK(cudaDeviceSynchronize());

            ozaki::accumulate_product(
                d_acc, d_af, prob.d_b, n, k, which, ws, prob);
            CUDA_CHECK(cudaDeviceSynchronize());

            double const err = metrics::norm_difference(
                d_acc, d_ref, n * k, prob);

            std::cout << "  " << std::left << std::setw(16) << ""
                      << std::right
                      << std::setw(7) << n_pieces
                      << std::setw(7) << cfg.block
                      << std::setw(9) << cfg.n_products()
                      << std::setw(13) << std::scientific << std::setprecision(2)
                      << ((norm_ref > 0.)? err / norm_ref : 0.)
                      << ((2*cfg.bits+static_cast<int>(std::log2((double)cfg.block)) <= 24)? "   bound ok" : "")
                      << "\n";
        }
        std::cout << "\n";
    }

    return 0;
}
