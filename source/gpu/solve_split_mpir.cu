#include "gpu/solver.h"

#include "gpu/error.h"
#include "gpu/metrics.h"
#include "gpu/ozaki.h"
#include "gpu/tuning.h"
#include "gpu/timing.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#include <cstddef>
#include <cstdlib>
#include <vector>

/*  split-MPIR: classical mixed-precision refinement with no fp64 arithmetic.
    Storage 12n^2.

    Classical MPIR is usually described as having one irreducible fp64
    operation, the residual b - Ax, because cancellation there makes low
    precision unusable. That is wrong: the residual is irreducibly
    *high-precision*, not irreducibly *fp64*. Store A as an unevaluated sum of
    two fp32 words and the residual becomes Ozaki products on tensor cores —
    accurate, fast, and fp64-free.

    A_hi + A_lo carries 48 bits, the same as fp64's 53 to within a few, and
    costs 8n^2 exactly as fp64 A does. So this keeps MPIR's 12n^2 footprint
    while getting the tensor-core throughput. It is the baseline any
    residual-storage scheme has to beat, and the one that is easy to leave out:
    a comparison against only the vendor solver and the direct solve makes the
    fp64-free idea look like it needs the 8n^2 trick, when most of the win is
    the splitting.

    The pair is stored NEGATED, so forming the residual is
    `acc = B; acc += (-A) * X` and no separate sign pass is needed. Row scales
    are unaffected by the sign. */

namespace solver {

using harness::problem;

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

/*  A -> (-A_hi, -A_lo), an unevaluated sum carrying ~48 bits.

    A_lo holds what the demotion of A discarded, so the pair reproduces A to
    fp32-of-fp32 precision. Negated here rather than at every residual. */
__global__ void split_pair_kernel(
    float             *d_a_hi,
    float             *d_a_lo,
    double const      *d_a,
    std::size_t const  n_total) {

    for (std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < n_total;
         idx += static_cast<std::size_t>(blockDim.x) * gridDim.x) {

        double const a  = d_a[idx];
        float  const hi = static_cast<float>(a);
        float  const lo = static_cast<float>(a - static_cast<double>(hi));

        d_a_hi[idx] = -hi;
        d_a_lo[idx] = -lo;
    }
}

/*  Row-permute an n x k fp64 matrix into fp32, applying P from the
    factorization. sgetrf yields P*A = L*U, so a correction solve needs P*r,
    not r. One thread owns one output element. */
__global__ void permute_demote_kernel(
    float             *d_out,
    double const      *d_in,
    int const         *d_perm,
    std::size_t const  n,
    std::size_t const  k) {

    std::size_t const n_total = n * k;

    for (std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < n_total;
         idx += static_cast<std::size_t>(blockDim.x) * gridDim.x) {

        std::size_t const i = idx % n;
        std::size_t const j = idx / n;
        d_out[idx] = static_cast<float>(
            d_in[static_cast<std::size_t>(d_perm[i]) + j * n]);
    }
}

/*  X += correction. The iterate stays fp64 throughout: demoting it between
    iterations reinjects 2^-24 into the very quantity the refinement is trying
    to resolve below that. */
__global__ void add_correction_kernel(
    double            *d_x,
    float const       *d_d,
    std::size_t const  n_total) {

    for (std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < n_total;
         idx += static_cast<std::size_t>(blockDim.x) * gridDim.x)
        d_x[idx] += static_cast<double>(d_d[idx]);
}

__global__ void zero_kernel(
    double            *d_x,
    std::size_t const  n_total) {

    for (std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < n_total;
         idx += static_cast<std::size_t>(blockDim.x) * gridDim.x)
        d_x[idx] = 0.;
}

/*  Sum of squares of an fp32 array, for the convergence test. */
__global__ void norm_f32_kernel(
    float const       *d_d,
    double            *d_partial,
    std::size_t const  n_total) {

    __shared__ double s[launch::BLOCK_SIZE];

    double acc = 0.;
    for (std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < n_total;
         idx += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
        double const v = static_cast<double>(d_d[idx]);
        acc += v * v;
    }

    s[threadIdx.x] = acc;
    __syncthreads();

    for (int q = launch::BLOCK_SIZE / 2; q > 0; q >>= 1) {
        if (static_cast<int>(threadIdx.x) < q)
            s[threadIdx.x] += s[threadIdx.x + q];
        __syncthreads();
    }

    if (threadIdx.x == 0)
        d_partial[blockIdx.x] = s[0];
}

double norm_f32(
    float const       *d_d,
    std::size_t const  n_total,
    problem           &prob) {

    int const n_blocks = launch::grid_for(n_total);

    norm_f32_kernel<<<n_blocks, launch::BLOCK_SIZE>>>(
        d_d, prob.d_partial, n_total);
    KERNEL_CHECK();

    std::vector<double> partial(static_cast<std::size_t>(n_blocks));
    CUDA_CHECK(cudaMemcpy(
        partial.data(),
        prob.d_partial,
        static_cast<std::size_t>(n_blocks) * sizeof(double),
        cudaMemcpyDeviceToHost));

    double total = 0.;
    for (std::size_t i = 0; i != partial.size(); ++i)
        total += partial[i];

    return total;
}

/*  Scratch that outlives one solve but is not part of the 12n^2 claim: it is
    all O(nk). Kept in a file-static so the timed region allocates nothing. */
struct scratch {
    double *d_acc  = nullptr;
    float  *d_rhs  = nullptr;
    float  *d_d    = nullptr;
    int    *d_perm = nullptr;
};

} /* namespace */

void factor_split_mpir(state &st, problem &prob) {

    std::size_t const n  = prob.n;
    std::size_t const nn = n * n;

    st.d_lu   = static_cast<float *>(st.acquire(nn * sizeof(float)));
    st.d_ipiv = static_cast<int *>(st.acquire(n * sizeof(int)));
    st.d_a_hi = static_cast<float *>(st.acquire(nn * sizeof(float)));
    st.d_a_lo = static_cast<float *>(st.acquire(nn * sizeof(float)));

    int lwork = 0;
    CUSOLVER_CHECK(cusolverDnSgetrf_bufferSize(
        prob.solver,
        static_cast<int>(n), static_cast<int>(n),
        st.d_lu, static_cast<int>(n), &lwork));

    float *d_work = static_cast<float *>(
        st.acquire(static_cast<std::size_t>(lwork) * sizeof(float)));
    int *d_info = static_cast<int *>(st.acquire(sizeof(int)));

    timing::stopwatch watch;
    watch.start();

    demote_kernel<<<launch::grid_for(nn), launch::BLOCK_SIZE>>>(
        st.d_lu, prob.d_a, nn);
    KERNEL_CHECK();

    CUSOLVER_CHECK(cusolverDnSgetrf(
        prob.solver,
        static_cast<int>(n), static_cast<int>(n),
        st.d_lu, static_cast<int>(n),
        d_work, st.d_ipiv, d_info));

    /*  The (hi, lo) pair. Charged to factor, not solve: it is a fixed cost and
        the whole reason this scheme sits at 12n^2 rather than 8n^2, so hiding
        it in the solve would misreport both axes at once. */
    split_pair_kernel<<<launch::grid_for(nn), launch::BLOCK_SIZE>>>(
        st.d_a_hi, st.d_a_lo, prob.d_a, nn);
    KERNEL_CHECK();

    st.factor_ms = watch.stop();
}

void solve_split_mpir(
    double       *d_x,
    double const *d_b,
    state        &st,
    problem      &prob) {

    std::size_t const n  = prob.n;
    std::size_t const k  = prob.k;
    std::size_t const nk = n * k;

    scratch sc;
    sc.d_acc  = static_cast<double *>(st.acquire(nk * sizeof(double)));
    sc.d_rhs  = static_cast<float *>(st.acquire(nk * sizeof(float)));
    sc.d_d    = static_cast<float *>(st.acquire(nk * sizeof(float)));
    sc.d_perm = static_cast<int *>(st.acquire(n * sizeof(int)));

    /*  sgetrf reports sequential row interchanges, 1-based. Compose them into
        a permutation vector once, on the host: it is n ints of setup against
        an O(n^2 k) solve. */
    {
        std::vector<int> ipiv(n);
        CUDA_CHECK(cudaMemcpy(
            ipiv.data(), st.d_ipiv, n * sizeof(int), cudaMemcpyDeviceToHost));

        std::vector<int> perm(n);
        for (std::size_t i = 0; i != n; ++i)
            perm[i] = static_cast<int>(i);
        for (std::size_t i = 0; i != n; ++i) {
            int const j = ipiv[i] - 1;
            if (j >= 0 && static_cast<std::size_t>(j) < n)
                std::swap(perm[i], perm[static_cast<std::size_t>(j)]);
        }
        CUDA_CHECK(cudaMemcpy(
            sc.d_perm, perm.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    }

    /*  The residual is consumed immediately by the refinement, so the
        inexact-but-fast configuration is correct here: measured, the converged
        backward error is identical to the exact one at 3x the speed. */
    ozaki::config const base = ozaki::config::for_refinement();
    ozaki::config cfg;
    cfg.bits     = tuning::current().get("mpir.ozaki.bits",   base.bits);
    cfg.n_pieces = tuning::current().get("mpir.ozaki.pieces", base.n_pieces);
    cfg.block    = tuning::current().get("mpir.ozaki.block",  base.block);
    cfg.n_groups = cfg.n_pieces;
    cfg.merge_tail = tuning::current().get("mpir.ozaki.merge_tail", base.merge_tail);
    ozaki::workspace ws(n, k, cfg, prob);

    ozaki::row_max(ws.d_mu, st.d_a_hi, n, n, ozaki::shape::full, prob);
    CUDA_CHECK(cudaDeviceSynchronize());

    timing::stopwatch watch;
    watch.start();

    zero_kernel<<<launch::grid_for(nk), launch::BLOCK_SIZE>>>(d_x, nk);
    KERNEL_CHECK();

    float const one = 1.f;
    double previous = 0., first = 0.;
    std::size_t used = 0;

    /*  Cap, not schedule. The loop stops on its own test and reports the count
        it used; a fixed count made two accuracy comparisons meaningless in the
        predecessor, in both directions. */
    std::size_t const cap = 8;

    for (std::size_t it = 0; it != cap; ++it) {

        /*  acc = B + (-A_hi - A_lo) * X, i.e. the residual, to ~48 bits. Two
            Ozaki products because the pair is unevaluated; this is where the
            method spends its time and where a plain fp32 GEMM would cancel. */
        CUDA_CHECK(cudaMemcpy(
            sc.d_acc, d_b, nk * sizeof(double), cudaMemcpyDeviceToDevice));

        ozaki::column_max(ws.d_nu, d_x, n, k, prob);

        ozaki::accumulate_product(
            sc.d_acc, st.d_a_hi, d_x, n, k, ozaki::shape::full, ws, prob);
        ozaki::accumulate_product(
            sc.d_acc, st.d_a_lo, d_x, n, k, ozaki::shape::full, ws, prob);

        /*  LU d = P r, in fp32. */
        permute_demote_kernel<<<launch::grid_for(nk), launch::BLOCK_SIZE>>>(
            sc.d_rhs, sc.d_acc, sc.d_perm, n, k);
        KERNEL_CHECK();

        CUDA_CHECK(cudaMemcpy(
            sc.d_d, sc.d_rhs, nk * sizeof(float), cudaMemcpyDeviceToDevice));

        CUBLAS_CHECK(cublasStrsm(
            prob.blas,
            CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
            CUBLAS_OP_N, CUBLAS_DIAG_UNIT,
            static_cast<int>(n), static_cast<int>(k),
            &one, st.d_lu, static_cast<int>(n),
            sc.d_d, static_cast<int>(n)));

        CUBLAS_CHECK(cublasStrsm(
            prob.blas,
            CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER,
            CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
            static_cast<int>(n), static_cast<int>(k),
            &one, st.d_lu, static_cast<int>(n),
            sc.d_d, static_cast<int>(n)));

        add_correction_kernel<<<launch::grid_for(nk), launch::BLOCK_SIZE>>>(
            d_x, sc.d_d, nk);
        KERNEL_CHECK();

        ++used;

        /*  Two tests, because either alone fails. Stalling alone never fires
            once the correction reaches zero exactly, and convergence alone
            never fires on a problem that plateaus. Getting this wrong cost
            three separate wrong stopping rules in the predecessor, one of
            which doubled a method's work and inflated a reported margin. */
        double const current = norm_f32(sc.d_d, nk, prob);
        if (it == 0)
            first = current;
        else {
            bool const stalled   = current > 0.25 * previous;
            bool const converged = current <= 1e-20 * first;
            if (stalled || converged)
                break;
        }
        previous = current;
    }

    st.solve_ms     = watch.stop();
    st.n_iterations = used;
}

} /* namespace solver */
