#include "common/solver.h"

/*  split-MPIR: classical mixed-precision refinement with no fp64 arithmetic.
    Storage 12n^2.

    Classical MPIR is usually described as having one irreducible fp64
    operation, the residual b - Ax, because cancellation there makes low
    precision unusable. That is wrong: the residual is irreducibly
    *high-precision*, not irreducibly *fp64*. Store A as an unevaluated sum of
    two fp32 words and the residual becomes Ozaki products on tensor cores.

    A_hi + A_lo carries ~48 bits and costs 8n^2, exactly as fp64 A does, so
    this keeps MPIR's 12n^2 footprint while getting tensor-core throughput. It
    is the baseline any residual-storage scheme must beat, and the one easiest
    to leave out: comparing only against the vendor solver and a direct solve
    makes the fp64-free idea look like it needs the 8n^2 trick, when most of
    the win is the splitting.

    ACCURACY HAZARD. The residual is a DENSE product, so the Ozaki exactness
    bound binds here — unlike the triangular R build, where it does not. Above
    the bound this method still reaches fp64-class accuracy on well-conditioned
    input, because refinement corrects the inexact residual on the next pass.
    When conditioning prevents convergence that correction never arrives: on a
    near-random matrix an above-bound residual gives 1.23e-09 where a
    within-bound one gives 2.54e-17. Whatever the tuning file supplies is valid
    only for the conditioning it was measured on. */

namespace solver {

using harness::problem;

namespace {

/*  A -> (-A_hi, -A_lo), an unevaluated sum carrying ~48 bits.

    A_lo holds what demoting A discarded, so the pair reproduces A to
    fp32-of-fp32 precision. Stored NEGATED so forming the residual is
    `acc = B; acc += (-A) * X` with no separate sign pass; row scales are
    unaffected by the sign. */
__global__ void split_pair_kernel(
    float             *d_a_hi,
    float             *d_a_lo,
    double const      *d_a,
    std::size_t const  n_elements) {

    for (std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < n_elements;
         idx += static_cast<std::size_t>(blockDim.x) * gridDim.x) {

        double const a  = d_a[idx];
        float  const hi = static_cast<float>(a);
        float  const lo = static_cast<float>(a - static_cast<double>(hi));

        d_a_hi[idx] = -hi;
        d_a_lo[idx] = -lo;
    }
}

ozaki::config residual_config() {

    ozaki::config const base = ozaki::config::for_refinement();
    ozaki::config cfg;

    cfg.bits       = tuning::current().get("mpir.ozaki.bits",   base.bits);
    cfg.n_pieces   = tuning::current().get("mpir.ozaki.pieces", base.n_pieces);
    cfg.block      = tuning::current().get("mpir.ozaki.block",  base.block);
    cfg.n_groups   = cfg.n_pieces;
    /*  Read for symmetry with the ablation harness, but inert here: this
        method's operand is A itself, which has no triangle, so every call
        below passes shape::full and the switch changes nothing. The triangular
        path is R-IR's advantage and cannot be taken by an A-based scheme. */
    cfg.triangular =
        tuning::current().get("ozaki.triangular", 1) != 0;
    cfg.contraction_bound =
        tuning::current().get("ozaki.contraction_bound", 1) != 0;
    cfg.merge_tail = tuning::current().get("mpir.ozaki.merge_tail",
                                           base.merge_tail);
    return cfg;
}

} /* namespace */

void factor_split_mpir(state &st, problem &prob) {

    std::size_t const n  = prob.n;
    std::size_t const nn = n * n;

    st.d_lu   = static_cast<float *>(st.acquire(nn * sizeof(float)));
    st.d_ipiv = static_cast<int *>(st.acquire(n * sizeof(int)));
    st.d_perm = static_cast<int *>(st.acquire(n * sizeof(int)));
    st.d_a_hi = static_cast<float *>(st.acquire(nn * sizeof(float)));
    st.d_a_lo = static_cast<float *>(st.acquire(nn * sizeof(float)));

    int const lwork = factorize::buffer_size(prob);
    float *d_work = static_cast<float *>(
        st.acquire(static_cast<std::size_t>(lwork) * sizeof(float)));
    int *d_info = static_cast<int *>(st.acquire(sizeof(int)));

    timing::stopwatch watch;
    watch.start();

    factorize::lu_fp32(
        st.d_lu, st.d_ipiv, st.d_perm, d_work, d_info, prob.d_a, prob);

    /*  The (hi, lo) pair. Charged to factor, not solve: it is a fixed cost and
        the reason this scheme sits at 12n^2 rather than 8n^2, so hiding it in
        the solve would misreport both axes at once. */
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

    double *d_acc = static_cast<double *>(st.acquire(nk * sizeof(double)));
    float  *d_d   = static_cast<float *>(st.acquire(nk * sizeof(float)));

    ozaki::config const cfg = residual_config();
    ozaki::workspace ws(n, k, cfg, prob);

    ozaki::row_max(ws.d_mu, st.d_a_hi, ozaki::format::fp32, n, n, ozaki::shape::full, prob);
    CUDA_CHECK(cudaDeviceSynchronize());

    timing::stopwatch watch;
    watch.start();

    convert::zero(d_x, nk, prob);

    double previous = 0., first = 0.;
    std::size_t used = 0;

    /*  A cap, not a schedule: the loop stops on its own test below and reports
        the count it used. */
    /*  Exposed for the same reason R-IR's is: so the stopping rule can be
        checked against a forced count. Without it this method could not be
        asked whether its 3 passes are necessary, while R-IR could — an
        asymmetry that makes any pass-structure comparison unfair. */
    std::size_t const cap = static_cast<std::size_t>(
        tuning::current().get("mpir.max_outer", 8));

    for (std::size_t it = 0; it != cap; ++it) {

        /*  acc = B + (-A_hi - A_lo) * X, the residual to ~48 bits. Two Ozaki
            products because the pair is unevaluated; this is where the method
            spends its time and where a plain fp32 GEMM would cancel. */
        CUDA_CHECK(cudaMemcpy(
            d_acc, d_b, nk * sizeof(double), cudaMemcpyDeviceToDevice));

        ozaki::column_max(ws.d_nu, d_x, n, k, prob);
        ozaki::accumulate_product(
            d_acc, st.d_a_hi, d_x, n, k, ozaki::shape::full, ws, prob);
        ozaki::accumulate_product(
            d_acc, st.d_a_lo, d_x, n, k, ozaki::shape::full, ws, prob);

        /*  LU d = P r, in fp32, then X += d. */
        convert::permute_rows_demote(d_d, d_acc, st.d_perm, n, k, prob);
        factorize::lu_solve(d_d, st.d_lu, k, prob);
        convert::add_correction(d_x, d_d, nk, prob);

        ++used;

        /*  Two tests, because either alone fails. Stalling alone never fires
            once the correction reaches zero exactly; convergence alone never
            fires on a problem that plateaus. Three wrong stopping rules in
            earlier work came from having only one of the two.

            Both norms are sums of squares, so the tolerance is the square of
            the one in norm terms. */
        double const current = convert::sum_squares_f32(d_d, nk, prob);
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
