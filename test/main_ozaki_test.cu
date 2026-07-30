#include "common/definitions.h"
#include "common/error.h"
#include "common/metrics.h"
#include "common/ozaki.h"
#include "common/problem.h"
#include "common/timing.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#include <cmath>
#include <string>
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

/*  Promote the strict lower / upper triangle of a packed LU factor into fp64,
    so the reference sees the same operand the Ozaki path reads through. */
__global__ void promote_triangle_kernel(
    double            *d_out,
    float const       *d_lu,
    std::size_t const  n,
    int const          which) {

    std::size_t const n_total = n * n;

    for (std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < n_total;
         idx += static_cast<std::size_t>(blockDim.x) * gridDim.x) {

        std::size_t const i = idx % n;
        std::size_t const j = idx / n;

        double v = static_cast<double>(d_lu[idx]);
        if (which == 1 && j >= i) v = 0.;
        if (which == 2 && j <  i) v = 0.;
        d_out[idx] = v;
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
                  << std::setw(7)  << "bits"
                  << std::setw(4)  << "np"
                  << std::setw(7)  << "block"
                  << std::setw(9)  << "prods"
                  << std::setw(10) << "ms"
                  << std::setw(13) << "rel err" << "\n";

        /*  Sweeping the piece count is the real check. Accuracy must improve
            monotonically until it hits the fp64 reference's own floor; a flat
            or non-monotonic curve means the pieces are not landing on a common
            grid, which is the failure mode that makes a naive split look like
            it works while delivering plain TF32. */
        /*  (bits, pieces, block). Holding total bits ~54 while moving the
            bound away from 24 separates "the bound is still marginal" from
            "a second limit sits below it". */
        struct trial {int bits; int pieces; int block;};
        trial const trials[] = {
            { 9,  6,   64},   /* bound 24.0 -- exactly at the limit */
            { 9,  6,   32},   /* bound 23.0                          */
            { 9,  6,   16},   /* bound 22.0                          */
            { 7,  8,   64},   /* bound 20.0, 56 bits                 */
            { 6,  9,   64},   /* bound 18.0, 54 bits                 */
            { 6,  9,  256},   /* bound 20.0, larger block            */
            { 5, 11,  256}    /* bound 18.0, 55 bits                 */
        };
        for (int ti = 0; ti != 7; ++ti) {

            ozaki::config cfg;
            cfg.n_pieces = trials[ti].pieces;
            cfg.bits     = trials[ti].bits;
            cfg.block    = trials[ti].block;
            cfg.n_groups = trials[ti].pieces;

            ozaki::workspace ws(n, k, cfg, prob);

            CUDA_CHECK(cudaMemset(d_acc, 0, n * k * sizeof(double)));

            ozaki::row_max(ws.d_mu, d_af, ozaki::format::fp32, n, n, which, prob);
            ozaki::column_max(ws.d_nu, prob.d_b, n, k, prob);
            CUDA_CHECK(cudaDeviceSynchronize());

            /*  One untimed pass first: cuBLAS selects kernels per shape, and
                the block size changes the shape on every trial. */
            ozaki::accumulate_product(
                d_acc, d_af, prob.d_b, n, k, which, ws, prob);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemset(d_acc, 0, n * k * sizeof(double)));

            timing::stopwatch watch;
            watch.start();
            ozaki::accumulate_product(
                d_acc, d_af, prob.d_b, n, k, which, ws, prob);
            double const ms = watch.stop();
            CUDA_CHECK(cudaDeviceSynchronize());

            double const err = metrics::norm_difference(
                d_acc, d_ref, n * k, prob);

            std::cout << "  " << std::left << std::setw(16) << ""
                      << std::right
                      << std::setw(7) << cfg.bits
                      << std::setw(4) << cfg.n_pieces
                      << std::setw(7) << cfg.block
                      << std::setw(9) << cfg.n_products()
                      << std::setw(10) << std::fixed << std::setprecision(1)
                      << ms
                      << std::setw(13) << std::scientific << std::setprecision(2)
                      << ((norm_ref > 0.)? err / norm_ref : 0.)
                      << ("   bound " + std::to_string(2*cfg.bits+static_cast<int>(std::log2((double)cfg.block))) + ", " + std::to_string(cfg.bits_resolved()) + " bits").c_str()
                      << "\n";
        }
        std::cout << "\n";
    }

    /*  ---- the R-build case: L * U from a real factorization ----------------

        The sweep above uses a near-random A against a random X. The R build is
        L * U, whose rows have a completely different dynamic range, and the
        exactness bound's effect may not carry across. This is the operand
        structure that actually matters, because R = PA - LU is the one product
        whose error is stored rather than corrected. */
    {
        std::cout << "L * U from a real factorization (the R-build operand)\n";

        float *d_lu = static_cast<float *>(prob.acquire(n * n * sizeof(float)));
        int *d_ipiv = static_cast<int *>(prob.acquire(n * sizeof(int)));
        int *d_info = static_cast<int *>(prob.acquire(sizeof(int)));

        CUDA_CHECK(cudaMemcpy(d_lu, d_af, n * n * sizeof(float),
                              cudaMemcpyDeviceToDevice));

        int lwork = 0;
        CUSOLVER_CHECK(cusolverDnSgetrf_bufferSize(
            prob.solver, static_cast<int>(n), static_cast<int>(n),
            d_lu, static_cast<int>(n), &lwork));
        float *d_work = static_cast<float *>(
            prob.acquire(static_cast<std::size_t>(lwork) * sizeof(float)));
        CUSOLVER_CHECK(cusolverDnSgetrf(
            prob.solver, static_cast<int>(n), static_cast<int>(n),
            d_lu, static_cast<int>(n), d_work, d_ipiv, d_info));

        /*  U promoted to fp64 is the right operand. L is read out of the packed
            factor through shape::lower, which is the STRICT triangle, so the
            unit diagonal contributes U itself: L*U = strict_L*U + U. */
        double *d_u = static_cast<double *>(prob.acquire(n * n * sizeof(double)));
        promote_triangle_kernel<<<launch::grid_for(n * n), launch::BLOCK_SIZE>>>(
            d_u, d_lu, n, 2);
        KERNEL_CHECK();

        double *d_l = static_cast<double *>(prob.acquire(n * n * sizeof(double)));
        promote_triangle_kernel<<<launch::grid_for(n * n), launch::BLOCK_SIZE>>>(
            d_l, d_lu, n, 1);
        KERNEL_CHECK();

        double *d_lu_ref = static_cast<double *>(
            prob.acquire(n * n * sizeof(double)));
        double const one = 1., zero = 0.;
        CUBLAS_CHECK(cublasDgemm(
            prob.blas, CUBLAS_OP_N, CUBLAS_OP_N,
            static_cast<int>(n), static_cast<int>(n), static_cast<int>(n),
            &one, d_l, static_cast<int>(n), d_u, static_cast<int>(n),
            &zero, d_lu_ref, static_cast<int>(n)));

        double const norm_lu = metrics::norm(d_lu_ref, n * n, prob);

        double *d_out = static_cast<double *>(
            prob.acquire(n * n * sizeof(double)));

        std::cout << "  " << std::right << std::setw(7) << "bits"
                  << std::setw(4) << "np" << std::setw(7) << "block"
                  << std::setw(8) << "bound" << std::setw(10) << "ms"
                  << std::setw(13) << "rel err" << "\n";

        struct t2 {int bits; int pieces; int block;};
        t2 const cases[] = {{6, 9, 256}, {9, 6, 1536}, {9, 6, 3072}};

        for (int ci = 0; ci != 3; ++ci) {

            ozaki::config cfg;
            cfg.bits = cases[ci].bits; cfg.n_pieces = cases[ci].pieces;
            cfg.n_groups = cases[ci].pieces; cfg.block = cases[ci].block;

            ozaki::workspace ws2(n, n, cfg, prob);
            ozaki::row_max(ws2.d_mu, d_lu, ozaki::format::fp32, n, n, ozaki::shape::lower, prob);
            ozaki::column_max(ws2.d_nu, d_u, n, n, prob);
            CUDA_CHECK(cudaDeviceSynchronize());

            CUDA_CHECK(cudaMemset(d_out, 0, n * n * sizeof(double)));
            ozaki::accumulate_product(
                d_out, d_lu, d_u, n, n, ozaki::shape::lower, ws2, prob);
            CUDA_CHECK(cudaDeviceSynchronize());

            timing::stopwatch w2;
            CUDA_CHECK(cudaMemset(d_out, 0, n * n * sizeof(double)));
            w2.start();
            ozaki::accumulate_product(
                d_out, d_lu, d_u, n, n, ozaki::shape::lower, ws2, prob);
            double const ms2 = w2.stop();

            double const err2 = metrics::norm_difference(
                d_out, d_lu_ref, n * n, prob);

            std::cout << "  " << std::right << std::setw(7) << cfg.bits
                      << std::setw(4) << cfg.n_pieces
                      << std::setw(7) << cfg.block
                      << std::setw(8) << std::fixed << std::setprecision(1)
                      << 2. * cfg.bits + std::log2((double) cfg.block)
                      << std::setw(10) << ms2
                      << std::setw(13) << std::scientific << std::setprecision(2)
                      << ((norm_lu > 0.)? err2 / norm_lu : 0.) << "\n";
        }
    }

    return 0;
}
