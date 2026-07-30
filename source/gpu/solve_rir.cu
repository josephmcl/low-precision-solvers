#include "common/solver.h"

/*  R-IR: store R = PA - LU instead of A. Storage 8n^2.
    (fp32 LU + fp32 R; 6n^2 if R is kept in bf16.)

    Every other scheme here keeps A in some form and needs 12n^2. This one
    keeps only the factorization and its own error, which is the same
    information in fewer bits: bits(eps) = bits(u_f) + bits(u_R) says accuracy
    comes from the total bits held, and MPIR pays that total *plus* a redundant
    copy of what the factorization already encodes.

    Solved in fixed-point form, LU X_{m+1} = PB - R X_m, converging at
    rho = ||(LU)^-1 R|| ~ 1e-7. Two things follow. R*X never cancels — R is
    already the small quantity — so no fp64 is needed anywhere. But the LU
    solve must then be accurate, because in this form it produces the iterate
    directly rather than a correction, which is why the final pass carries a
    refinement. The correction form was tried in earlier work and is worse: its
    residual PB - LU X - R X cancels, reintroducing what R was meant to remove.

    WHAT IT BUYS AND WHAT IT COSTS. A is needed once, to form R, and can then be
    discarded permanently — its dependence on A is transient where every
    A-based method's is permanent. That is the durable claim. The cost is a
    second Theta(n^3) pass, so at few right-hand sides this cannot win on time
    and is not meant to.

    LIMIT. On ill-conditioned input this method floors near 1e-10 and no amount
    of Ozaki exactness recovers it — the limit is R itself. A poor fp32
    factorization makes ||R|| grow, and R's 24 stored bits then cover
    proportionally less of A. Unlike split-MPIR's residual, which fails on such
    input only when misconfigured, this is structural. */

namespace solver {

using harness::problem;

namespace {

/*  Promote a column block of U out of the packed factor into fp64, for the
    Ozaki product's right operand. U is the upper triangle with the diagonal. */
__global__ void promote_u_block_kernel(
    double            *d_u,
    float const       *d_lu,
    std::size_t const  n,
    std::size_t const  col_0,
    std::size_t const  n_cols) {

    std::size_t const n_elements = n * n_cols;

    for (std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < n_elements;
         idx += static_cast<std::size_t>(blockDim.x) * gridDim.x) {

        std::size_t const i = idx % n;
        std::size_t const j = col_0 + idx / n;

        d_u[idx] = (i <= j)? static_cast<double>(d_lu[i + j * n]) : 0.;
    }
}

/*  R = PA - LU for one column block, written in fp32.

    d_acc holds strict_L * U for these columns; subtracting U as well supplies
    L's unit diagonal, since L*U = strict_L*U + U. PA is read straight out of
    the reference through the permutation, so no permuted copy of A is ever
    materialized — which matters, because a permuted fp64 copy would cost 8n^2
    and double the peak footprint this scheme exists to keep small. */
__global__ void form_r_block_kernel(
    float             *d_r,
    double const      *d_acc,
    float const       *d_lu,
    double const      *d_a,
    int const         *d_perm,
    std::size_t const  n,
    std::size_t const  col_0,
    std::size_t const  n_cols) {

    std::size_t const n_elements = n * n_cols;

    for (std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < n_elements;
         idx += static_cast<std::size_t>(blockDim.x) * gridDim.x) {

        std::size_t const i = idx % n;
        std::size_t const j = col_0 + idx / n;

        double const u  = (i <= j)? static_cast<double>(d_lu[i + j * n]) : 0.;
        double const pa = d_a[static_cast<std::size_t>(d_perm[i]) + j * n];

        d_r[i + j * n] = static_cast<float>(pa - d_acc[idx] - u);
    }
}

/*  The two Ozaki sites want different settings — the build's operand is
    triangular and the solve's is dense — so they are keyed separately rather
    than sharing one config. A single setting is necessarily wrong at one of
    them. */
ozaki::config config_for(std::string const &prefix) {

    ozaki::config const base = ozaki::config::for_refinement();
    ozaki::config cfg;

    cfg.bits       = tuning::current().get(prefix + ".ozaki.bits",
                                           base.bits);
    cfg.n_pieces   = tuning::current().get(prefix + ".ozaki.pieces",
                                           base.n_pieces);
    cfg.block      = tuning::current().get(prefix + ".ozaki.block",
                                           base.block);
    cfg.n_groups   = cfg.n_pieces;
    cfg.merge_tail = tuning::current().get(prefix + ".ozaki.merge_tail",
                                           base.merge_tail);
    return cfg;
}

} /* namespace */

void factor_rir(state &st, problem &prob) {

    std::size_t const n  = prob.n;
    std::size_t const nn = n * n;

    st.d_lu   = static_cast<float *>(st.acquire(nn * sizeof(float)));
    st.d_r    = static_cast<float *>(st.acquire(nn * sizeof(float)));
    st.d_ipiv = static_cast<int *>(st.acquire(n * sizeof(int)));
    st.d_perm = static_cast<int *>(st.acquire(n * sizeof(int)));

    int const lwork = factorize::buffer_size(prob);
    float *d_work = static_cast<float *>(
        st.acquire(static_cast<std::size_t>(lwork) * sizeof(float)));
    int *d_info = static_cast<int *>(st.acquire(sizeof(int)));

    /*  R is built one column block at a time. A full n x n fp64 accumulator
        would cost 8n^2 on top of the 8n^2 this scheme keeps, doubling the peak
        footprint and destroying its only claim. The transient is
        n * column_block * 8 bytes. */
    std::size_t const nb = std::min(
        static_cast<std::size_t>(
            tuning::current().get("rir.build.column_block", 1024)),
        n);

    double *d_acc = static_cast<double *>(st.acquire(n * nb * sizeof(double)));
    double *d_u   = static_cast<double *>(st.acquire(n * nb * sizeof(double)));

    /*  The build's result is STORED and reused by every solve, so its error is
        undamped — unlike a refinement residual, which the next iteration
        corrects. On this operand (L*U from a real factorization) blocking above
        the exactness bound costs nothing and runs faster, because triangular
        factors have a narrow per-row dynamic range. Operand-specific: the same
        configuration is wrong by four orders on a dense product. */
    ozaki::config const cfg = config_for("rir.build");
    ozaki::workspace ws(n, nb, cfg, prob);

    timing::stopwatch watch;
    watch.start();

    factorize::lu_fp32(
        st.d_lu, st.d_ipiv, st.d_perm, d_work, d_info, prob.d_a, prob);

    ozaki::row_max(ws.d_mu, st.d_lu, n, n, ozaki::shape::lower, prob);

    for (std::size_t j = 0; j < n; j += nb) {

        std::size_t const n_c = std::min(nb, n - j);

        promote_u_block_kernel<<<launch::grid_for(n * n_c),
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

        form_r_block_kernel<<<launch::grid_for(n * n_c),
                              launch::BLOCK_SIZE>>>(
            st.d_r, d_acc, st.d_lu, prob.d_a, st.d_perm, n, j, n_c);
        KERNEL_CHECK();
    }

    st.factor_ms = watch.stop();
}

void solve_rir(
    double       *d_x,
    double const *d_b,
    state        &st,
    problem      &prob) {

    std::size_t const n  = prob.n;
    std::size_t const k  = prob.k;
    std::size_t const nk = n * k;

    double *d_pb   = static_cast<double *>(st.acquire(nk * sizeof(double)));
    double *d_rhs  = static_cast<double *>(st.acquire(nk * sizeof(double)));
    double *d_prev = static_cast<double *>(st.acquire(nk * sizeof(double)));
    double *d_t    = static_cast<double *>(st.acquire(nk * sizeof(double)));
    float  *d_y    = static_cast<float *>(st.acquire(nk * sizeof(float)));

    ozaki::config const cfg = config_for("rir.solve");
    ozaki::workspace ws(n, k, cfg, prob);

    bool const tf32_outer =
        tuning::current().get("rir.solve.tf32_outer", 0) != 0;

    /*  R*X never cancels — R is already ~2^-24 of A — which is the structural
        reason this form needs no fp64. Negated once so the residual is an
        accumulate rather than a subtract; restored before returning. */
    convert::negate(st.d_r, n * n, prob);
    ozaki::row_max(ws.d_mu, st.d_r, n, n, ozaki::shape::full, prob);

    convert::permute_rows(d_pb, d_b, st.d_perm, n, k, prob);
    convert::zero(d_x, nk, prob);
    CUDA_CHECK(cudaDeviceSynchronize());

    timing::stopwatch watch;
    watch.start();

    double previous = 0., first = 0.;
    std::size_t used = 0;

    /*  A cap, not a schedule. Exposed only so the stopping rule can be checked
        against a forced count: if capping below the converged count does not
        change the answer, the extra passes were doing nothing. */
    std::size_t const cap = static_cast<std::size_t>(
        tuning::current().get("rir.solve.max_outer", 8));

    for (std::size_t it = 0; it != cap; ++it) {

        CUDA_CHECK(cudaMemcpy(
            d_rhs, d_pb, nk * sizeof(double), cudaMemcpyDeviceToDevice));

        if (it != 0) {
            ozaki::column_max(ws.d_nu, d_x, n, k, prob);
            ozaki::accumulate_product(
                d_rhs, st.d_r, d_x, n, k, ozaki::shape::full, ws, prob);
        }

        convert::demote(d_y, d_rhs, nk, prob);
        factorize::lu_solve(d_y, st.d_lu, k, prob, tf32_outer);

        CUDA_CHECK(cudaMemcpy(
            d_prev, d_x, nk * sizeof(double), cudaMemcpyDeviceToDevice));
        convert::promote(d_x, d_y, nk, prob);

        ++used;

        /*  The fixed-point form yields the ITERATE, not a correction, so the
            measure is ||X_new - X_old||. Testing the residual change instead
            barely moves between passes and fires immediately.

            Sums of squares, so the tolerance is squared: 1e-20 means ||dX||
            has fallen to 1e-10 of its first step. Measured, that stops one
            pass earlier than a tighter value with the same answer. */
        double const current =
            convert::sum_squares_difference(d_x, d_prev, nk, prob);
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

    /*  The fixed-point form hands the LU solve straight to the iterate, so the
        answer inherits that solve's fp32 error and floors near 1e-7 no matter
        how many outer passes run. One refinement of the final triangular solve
        fixes it: form the residual of LU X = rhs accurately, solve for a
        correction in fp32, add it back.

        Both products are Ozaki and both are triangular, which is where this
        scheme's one arithmetic advantage lives — the dense baselines have no
        triangle to exploit. */
    {
        /*  t = U * X. */
        convert::zero(d_t, nk, prob);
        ozaki::row_max(ws.d_mu, st.d_lu, n, n, ozaki::shape::upper, prob);
        ozaki::column_max(ws.d_nu, d_x, n, k, prob);
        ozaki::accumulate_product(
            d_t, st.d_lu, d_x, n, k, ozaki::shape::upper, ws, prob);

        /*  rhs -= L * t, with L's unit diagonal supplying the -t term. */
        convert::negate(st.d_lu, n * n, prob);
        ozaki::row_max(ws.d_mu, st.d_lu, n, n, ozaki::shape::lower, prob);
        ozaki::column_max(ws.d_nu, d_t, n, k, prob);
        ozaki::accumulate_product(
            d_rhs, st.d_lu, d_t, n, k, ozaki::shape::lower, ws, prob);
        convert::negate(st.d_lu, n * n, prob);

        convert::subtract(d_rhs, d_t, nk, prob);

        /*  fp32, never tf32: this solve produces the correction that sets the
            delivered accuracy. */
        convert::demote(d_y, d_rhs, nk, prob);
        factorize::lu_solve(d_y, st.d_lu, k, prob, false);
        convert::add_correction(d_x, d_y, nk, prob);
    }

    st.solve_ms     = watch.stop();
    st.n_iterations = used;

    convert::negate(st.d_r, n * n, prob);
}

} /* namespace solver */
