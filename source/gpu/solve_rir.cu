#include "gpu/solver.h"

#include "gpu/error.h"
#include "gpu/ozaki.h"
#include "gpu/tuning.h"
#include "gpu/timing.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <vector>

/*  R-IR: store R = PA - LU instead of A. Storage 8n^2.
    (fp32 LU + fp32 R; 6n^2 if R is kept in bf16.)

    Every other scheme here keeps A in some form and needs 12n^2. This one
    keeps only the factorization and its own error, which is the same
    information in fewer bits: the storage identity
    bits(eps) = bits(u_f) + bits(u_R) says a scheme's accuracy comes from the
    total bits it holds, and MPIR pays that total *plus* a redundant copy of
    what the factorization already encodes.

    Solved in fixed-point form, LU X_{m+1} = PB - R X_m, which converges at
    rho = ||(LU)^-1 R|| ~ 1e-7. Two things follow. The residual R*X never
    cancels — R is already the small quantity — so it needs no fp64 anywhere.
    But the LU solve must then be accurate, because in this form it produces
    the iterate directly rather than a correction, which is why the final pass
    carries a refinement (see below). The correction form was tried in earlier
    work and is worse: its residual PB - LU X - R X cancels, reintroducing
    exactly the problem R was meant to remove.

    WHAT R-IR BUYS AND WHAT IT COSTS. A is needed once, to form R, and can then
    be discarded permanently — so its dependence on A is transient where every
    A-based method's is permanent. That is the durable claim. The cost is a
    second Theta(n^3) pass: the R build is work of the same order as the
    factorization, so at few right-hand sides this scheme cannot win on time
    and is not meant to. */

namespace solver {

using harness::problem;

namespace {

__global__ void rir_demote_kernel(
    float             *d_af,
    double const      *d_a,
    std::size_t const  n_total) {

    for (std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < n_total;
         idx += static_cast<std::size_t>(blockDim.x) * gridDim.x)
        d_af[idx] = static_cast<float>(d_a[idx]);
}

/*  Promote a column block of U out of the packed factor into fp64, so the
    Ozaki product can take it as its right operand. U is the upper triangle
    including the diagonal. */
__global__ void rir_promote_u_block_kernel(
    double            *d_u,
    float const       *d_lu,
    std::size_t const  n,
    std::size_t const  col_0,
    std::size_t const  n_cols) {

    std::size_t const n_total = n * n_cols;

    for (std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < n_total;
         idx += static_cast<std::size_t>(blockDim.x) * gridDim.x) {

        std::size_t const i = idx % n;
        std::size_t const c = idx / n;
        std::size_t const j = col_0 + c;

        d_u[idx] = (i <= j)?
            static_cast<double>(d_lu[i + j * n]) : 0.;
    }
}

/*  R = PA - LU for one column block, written in fp32.

    d_acc holds strict_L * U for these columns; subtracting U as well supplies
    L's unit diagonal, since L*U = strict_L*U + U. PA is read straight out of
    the reference through the permutation, so no permuted copy of A is ever
    materialized — which matters, because a permuted fp64 copy would cost 8n^2
    and double the peak footprint this scheme exists to keep small. */
__global__ void rir_form_r_block_kernel(
    float             *d_r,
    double const      *d_acc,
    float const       *d_lu,
    double const      *d_a,
    int const         *d_perm,
    std::size_t const  n,
    std::size_t const  col_0,
    std::size_t const  n_cols) {

    std::size_t const n_total = n * n_cols;

    for (std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < n_total;
         idx += static_cast<std::size_t>(blockDim.x) * gridDim.x) {

        std::size_t const i = idx % n;
        std::size_t const c = idx / n;
        std::size_t const j = col_0 + c;

        double const u  = (i <= j)?
            static_cast<double>(d_lu[i + j * n]) : 0.;
        double const pa = d_a[static_cast<std::size_t>(d_perm[i]) + j * n];

        /*  d_acc = strict_L * U, so pa - acc - u = PA - L*U. */
        d_r[i + j * n] = static_cast<float>(pa - d_acc[idx] - u);
    }
}

__global__ void rir_negate_kernel(
    float             *d_m,
    std::size_t const  n_total) {

    for (std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < n_total;
         idx += static_cast<std::size_t>(blockDim.x) * gridDim.x)
        d_m[idx] = -d_m[idx];
}

__global__ void rir_permute_kernel(
    double            *d_out,
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
        d_out[idx] = d_in[static_cast<std::size_t>(d_perm[i]) + j * n];
    }
}

__global__ void rir_demote_f_kernel(
    float             *d_out,
    double const      *d_in,
    std::size_t const  n_total) {

    for (std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < n_total;
         idx += static_cast<std::size_t>(blockDim.x) * gridDim.x)
        d_out[idx] = static_cast<float>(d_in[idx]);
}

__global__ void rir_promote_kernel(
    double            *d_out,
    float const       *d_in,
    std::size_t const  n_total) {

    for (std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < n_total;
         idx += static_cast<std::size_t>(blockDim.x) * gridDim.x)
        d_out[idx] = static_cast<double>(d_in[idx]);
}

/*  Sum of squares of the difference of two fp64 arrays: the convergence
    measure for a fixed-point iteration, which yields the ITERATE and not a
    correction. Testing ||dRHS|| here instead — the natural measure for a
    correction form — barely moves between passes and fires immediately. */
__global__ void rir_diff_sq_kernel(
    double const      *d_x,
    double const      *d_y,
    double            *d_partial,
    std::size_t const  n_total) {

    __shared__ double s[launch::BLOCK_SIZE];

    double acc = 0.;
    for (std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < n_total;
         idx += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
        double const v = d_x[idx] - d_y[idx];
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

__global__ void rir_subtract_kernel(
    double            *d_x,
    double const      *d_y,
    std::size_t const  n_total) {

    for (std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < n_total;
         idx += static_cast<std::size_t>(blockDim.x) * gridDim.x)
        d_x[idx] -= d_y[idx];
}

__global__ void rir_add_f32_kernel(
    double            *d_x,
    float const       *d_d,
    std::size_t const  n_total) {

    for (std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < n_total;
         idx += static_cast<std::size_t>(blockDim.x) * gridDim.x)
        d_x[idx] += static_cast<double>(d_d[idx]);
}

double rir_diff_norm(
    double const      *d_x,
    double const      *d_y,
    std::size_t const  n_total,
    problem           &prob) {

    int const n_blocks = launch::grid_for(n_total);

    rir_diff_sq_kernel<<<n_blocks, launch::BLOCK_SIZE>>>(
        d_x, d_y, prob.d_partial, n_total);
    KERNEL_CHECK();

    std::vector<double> partial(static_cast<std::size_t>(n_blocks));
    CUDA_CHECK(cudaMemcpy(
        partial.data(), prob.d_partial,
        static_cast<std::size_t>(n_blocks) * sizeof(double),
        cudaMemcpyDeviceToHost));

    double total = 0.;
    for (std::size_t i = 0; i != partial.size(); ++i)
        total += partial[i];

    return total;
}

/*  Both triangular solves of the packed factor, in fp32, in place. */
void rir_lu_solve_f32(
    float             *d_y,
    float const       *d_lu,
    std::size_t const  n,
    std::size_t const  k,
    problem           &prob) {

    float const one = 1.f;

    CUBLAS_CHECK(cublasStrsm(
        prob.blas,
        CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
        CUBLAS_OP_N, CUBLAS_DIAG_UNIT,
        static_cast<int>(n), static_cast<int>(k),
        &one, d_lu, static_cast<int>(n),
        d_y, static_cast<int>(n)));

    CUBLAS_CHECK(cublasStrsm(
        prob.blas,
        CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER,
        CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
        static_cast<int>(n), static_cast<int>(k),
        &one, d_lu, static_cast<int>(n),
        d_y, static_cast<int>(n)));
}

} /* namespace */

void factor_rir(state &st, problem &prob) {

    std::size_t const n  = prob.n;
    std::size_t const nn = n * n;

    st.d_lu = static_cast<float *>(st.acquire(nn * sizeof(float)));
    st.d_r  = static_cast<float *>(st.acquire(nn * sizeof(float)));
    st.d_ipiv = static_cast<int *>(st.acquire(n * sizeof(int)));

    int lwork = 0;
    CUSOLVER_CHECK(cusolverDnSgetrf_bufferSize(
        prob.solver,
        static_cast<int>(n), static_cast<int>(n),
        st.d_lu, static_cast<int>(n), &lwork));

    float *d_work = static_cast<float *>(
        st.acquire(static_cast<std::size_t>(lwork) * sizeof(float)));
    int *d_info = static_cast<int *>(st.acquire(sizeof(int)));
    int *d_perm = static_cast<int *>(st.acquire(n * sizeof(int)));

    /*  R is built one column block at a time. A full n x n fp64 accumulator
        would cost 8n^2 on top of the 8n^2 this scheme keeps, doubling the peak
        footprint and destroying the only claim it has. The transient here is
        n * COLUMN_BLOCK * 8 bytes — 0.5n^2 at n=8192. */
    std::size_t const COLUMN_BLOCK = static_cast<std::size_t>(
        tuning::current().get("rir.build.column_block", 512));
    std::size_t const nb = std::min(COLUMN_BLOCK, n);

    double *d_acc = static_cast<double *>(
        st.acquire(n * nb * sizeof(double)));
    double *d_u = static_cast<double *>(
        st.acquire(n * nb * sizeof(double)));

    /*  The R build's result is STORED and reused by every solve, so its error
        is undamped — unlike a refinement residual, which the next iteration
        corrects. Measured on this operand (L*U from a real factorization),
        blocking above the exactness bound costs nothing and runs 2.4x faster,
        because triangular factors have a narrow per-row dynamic range. That is
        operand-specific: the same configuration is wrong by four orders on a
        dense random product. */
    ozaki::config const base = ozaki::config::for_refinement();
    ozaki::config cfg;
    cfg.bits     = tuning::current().get("rir.build.ozaki.bits",   base.bits);
    cfg.n_pieces = tuning::current().get("rir.build.ozaki.pieces", base.n_pieces);
    cfg.block    = tuning::current().get("rir.build.ozaki.block",  base.block);
    cfg.n_groups = cfg.n_pieces;
    ozaki::workspace ws(n, nb, cfg, prob);

    timing::stopwatch watch;
    watch.start();

    rir_demote_kernel<<<launch::grid_for(nn), launch::BLOCK_SIZE>>>(
        st.d_lu, prob.d_a, nn);
    KERNEL_CHECK();

    CUSOLVER_CHECK(cusolverDnSgetrf(
        prob.solver,
        static_cast<int>(n), static_cast<int>(n),
        st.d_lu, static_cast<int>(n),
        d_work, st.d_ipiv, d_info));

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
            d_perm, perm.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    }

    ozaki::row_max(ws.d_mu, st.d_lu, n, n, ozaki::shape::lower, prob);

    for (std::size_t j = 0; j < n; j += nb) {

        std::size_t const n_c = std::min(nb, n - j);

        rir_promote_u_block_kernel<<<launch::grid_for(n * n_c),
                                 launch::BLOCK_SIZE>>>(
            d_u, st.d_lu, n, j, n_c);
        KERNEL_CHECK();

        ozaki::column_max(ws.d_nu, d_u, n, n_c, prob);

        CUDA_CHECK(cudaMemset(d_acc, 0, n * n_c * sizeof(double)));
        /*  Only contraction indices below j + n_c reach these columns: U is
            upper triangular, so its rows at or beyond that are zero here.
            Bounding the loop halves the build. */
        ozaki::accumulate_product(
            d_acc, st.d_lu, d_u, n, n_c, ozaki::shape::lower, ws, prob,
            j + n_c);

        rir_form_r_block_kernel<<<launch::grid_for(n * n_c),
                              launch::BLOCK_SIZE>>>(
            st.d_r, d_acc, st.d_lu, prob.d_a, d_perm, n, j, n_c);
        KERNEL_CHECK();
    }

    st.factor_ms = watch.stop();

    st.d_perm = d_perm;
}

void solve_rir(
    double       *d_x,
    double const *d_b,
    state        &st,
    problem      &prob) {

    std::size_t const n  = prob.n;
    std::size_t const k  = prob.k;
    std::size_t const nk = n * k;

    int *d_perm = st.d_perm;

    double *d_pb   = static_cast<double *>(st.acquire(nk * sizeof(double)));
    double *d_prev = static_cast<double *>(st.acquire(nk * sizeof(double)));
    double *d_rhs = static_cast<double *>(st.acquire(nk * sizeof(double)));
    float  *d_y   = static_cast<float *>(st.acquire(nk * sizeof(float)));

    ozaki::config const base = ozaki::config::for_refinement();
    ozaki::config cfg;
    cfg.bits     = tuning::current().get("rir.solve.ozaki.bits",   base.bits);
    cfg.n_pieces = tuning::current().get("rir.solve.ozaki.pieces", base.n_pieces);
    cfg.block    = tuning::current().get("rir.solve.ozaki.block",  base.block);
    cfg.n_groups = cfg.n_pieces;
    ozaki::workspace ws(n, k, cfg, prob);

    /*  R*X never cancels — R is already ~2^-24 of A — which is the structural
        reason this form needs no fp64. Negated once so the residual is an
        accumulate rather than a subtract. */
    rir_negate_kernel<<<launch::grid_for(n * n), launch::BLOCK_SIZE>>>(
        st.d_r, n * n);
    KERNEL_CHECK();

    ozaki::row_max(ws.d_mu, st.d_r, n, n, ozaki::shape::full, prob);

    rir_permute_kernel<<<launch::grid_for(nk), launch::BLOCK_SIZE>>>(
        d_pb, d_b, d_perm, n, k);
    KERNEL_CHECK();

    CUDA_CHECK(cudaMemset(d_x, 0, nk * sizeof(double)));
    CUDA_CHECK(cudaDeviceSynchronize());

    timing::stopwatch watch;
    watch.start();

    std::size_t used = 0;
    double previous = 0., first = 0.;

    /*  A cap, not a schedule: the loop stops on the test below and reports
        what it used. */
    std::size_t const cap = 8;

    for (std::size_t it = 0; it != cap; ++it) {

        /*  rhs = PB - R*X. */
        CUDA_CHECK(cudaMemcpy(
            d_rhs, d_pb, nk * sizeof(double), cudaMemcpyDeviceToDevice));

        if (it != 0) {
            ozaki::column_max(ws.d_nu, d_x, n, k, prob);
            ozaki::accumulate_product(
                d_rhs, st.d_r, d_x, n, k, ozaki::shape::full, ws, prob);
        }

        rir_demote_f_kernel<<<launch::grid_for(nk), launch::BLOCK_SIZE>>>(
            d_y, d_rhs, nk);
        KERNEL_CHECK();

        rir_lu_solve_f32(d_y, st.d_lu, n, k, prob);

        CUDA_CHECK(cudaMemcpy(
            d_prev, d_x, nk * sizeof(double), cudaMemcpyDeviceToDevice));
        rir_promote_kernel<<<launch::grid_for(nk), launch::BLOCK_SIZE>>>(
            d_x, d_y, nk);
        KERNEL_CHECK();

        ++used;

        /*  Two tests, because either alone fails. The fixed point converges at
            rho ~ 1e-7 and can reach dX == 0 exactly, after which a
            stall-ratio test never fires again; and a pure tolerance test never
            fires on a problem that plateaus above it. Three wrong stopping
            rules in earlier work came from having only one of the two. */
        double const current = rir_diff_norm(d_x, d_prev, nk, prob);
        if (it == 0)
            first = current;
        else {
            bool const stalled   = current > 0.25 * previous;
            bool const converged = current <= 1e-24 * first;
            if (stalled || converged)
                break;
        }
        previous = current;
    }

    /*  The fixed-point form hands the LU solve straight to the iterate, so the
        answer inherits that solve's fp32 error and floors at ~1e-7 no matter
        how many outer passes run. One refinement of the final triangular solve
        fixes it: form the residual of LU X = rhs accurately, solve for a
        correction in fp32, add it back.

        Both products are Ozaki and both are triangular, which is where this
        scheme's one arithmetic advantage lives — the dense baselines have no
        triangle to exploit. */
    {
        double *d_t = static_cast<double *>(st.acquire(nk * sizeof(double)));

        /*  t = U * X. */
        CUDA_CHECK(cudaMemset(d_t, 0, nk * sizeof(double)));
        ozaki::row_max(ws.d_mu, st.d_lu, n, n, ozaki::shape::upper, prob);
        ozaki::column_max(ws.d_nu, d_x, n, k, prob);
        ozaki::accumulate_product(
            d_t, st.d_lu, d_x, n, k, ozaki::shape::upper, ws, prob);

        /*  d = rhs - L*t, with L's unit diagonal supplying -t. */
        rir_negate_kernel<<<launch::grid_for(n * n), launch::BLOCK_SIZE>>>(
            st.d_lu, n * n);
        KERNEL_CHECK();
        ozaki::row_max(ws.d_mu, st.d_lu, n, n, ozaki::shape::lower, prob);
        ozaki::column_max(ws.d_nu, d_t, n, k, prob);
        ozaki::accumulate_product(
            d_rhs, st.d_lu, d_t, n, k, ozaki::shape::lower, ws, prob);
        rir_negate_kernel<<<launch::grid_for(n * n), launch::BLOCK_SIZE>>>(
            st.d_lu, n * n);
        KERNEL_CHECK();

        /*  rhs currently holds (PB - R*X) - strict_L*t; subtract t for the
            unit diagonal, then solve and correct. */
        /*  A kernel rather than Daxpy: n*k exceeds INT_MAX before the
            problem exceeds device memory, and the cuBLAS signature takes int. */
        rir_subtract_kernel<<<launch::grid_for(nk), launch::BLOCK_SIZE>>>(
            d_rhs, d_t, nk);
        KERNEL_CHECK();

        rir_demote_f_kernel<<<launch::grid_for(nk), launch::BLOCK_SIZE>>>(
            d_y, d_rhs, nk);
        KERNEL_CHECK();

        rir_lu_solve_f32(d_y, st.d_lu, n, k, prob);

        rir_add_f32_kernel<<<launch::grid_for(nk), launch::BLOCK_SIZE>>>(
            d_x, d_y, nk);
        KERNEL_CHECK();
    }

    st.solve_ms     = watch.stop();
    st.n_iterations = used;

    rir_negate_kernel<<<launch::grid_for(n * n), launch::BLOCK_SIZE>>>(
        st.d_r, n * n);
    KERNEL_CHECK();
}

} /* namespace solver */
