#include "common/solver.h"
#include "common/trsm.h"

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

    LIMIT — RETRACTED. This comment previously read that the method floors near
    1e-10 on ill-conditioned input, structurally, because R's 24 stored bits
    cover proportionally less of A as ||R|| grows. That was measured with an
    fp32 triangular solve, which was itself the limiter by a factor of 8e6.
    With an accurate solve there is no conditioning limit: across a shift sweep
    spanning kappa 1 to 5.6e4, R-IR tracks a direct fp64 solve to within
    2.4x-2.9x, and the measured contraction ratio rho never exceeds 4e-3
    against a limit of 1.

    The real convergence criterion is rho = ||(LU)^-1 R|| < 1, measured for
    free as ||dX_m||/||dX_{m-1}||. Neither kappa nor pivot growth appears in
    it; both were proposed here on the strength of a real correlation and both
    were artifacts of the fp32 solve. rho also sets the pass count: below 1e-3
    three polish passes suffice, near 3e-3 it takes five.

    What DOES limit the method is its representation. R-IR never holds A, so
    its best attainable answer is the exact solution of (LU + R)x = Pb. That is
    the 8n^2-against-12n^2 trade appearing as accuracy rather than as memory,
    and it is why vendor IRS — which keeps A in fp64 — reaches 2.85e-17 where
    this reaches 5.27e-15. */

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

        d_r[idx] = static_cast<float>(pa - d_acc[idx] - u);
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
    /*  For the R build the second operand is U, promoted out of the packed
        fp32 factor — 24 bits, not 53 — so it needs far fewer pieces than the
        first. -1 leaves it equal to n_pieces, which is right for the solve
        sites where the second operand is a genuine fp64 iterate. */
    cfg.n_pieces_x = tuning::current().get(prefix + ".ozaki.pieces_x", -1);
    cfg.triangular =
        tuning::current().get("ozaki.triangular", 1) != 0;
    cfg.contraction_bound =
        tuning::current().get("ozaki.contraction_bound", 1) != 0;
    cfg.merge_tail = tuning::current().get(prefix + ".ozaki.merge_tail",
                                           base.merge_tail);
    return cfg;
}

/*  Multiply R elementwise by (1 + eps*u), u uniform in [-1,1].

    A CONTROLLED perturbation, to answer directly what an inaccurate R costs.
    Every attempt so far to infer that from uncontrolled variation — comparing
    R against a reference, or against another configuration — has been
    ambiguous, because it could not say whether the difference measured was in
    R or in the thing R was compared to. Injecting a known error removes the
    reference entirely: if the delivered backward error responds as
    eps * ||R||/||A||, then R's accuracy sets the floor at that level. If it
    does not respond, R is not the limiter and no amount of build precision
    matters. */
__global__ void rir_perturb_r_kernel(
    float             *d_r,
    std::size_t const  n_total,
    float const        eps,
    unsigned const     seed) {

    for (std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < n_total;
         idx += static_cast<std::size_t>(blockDim.x) * gridDim.x) {

        unsigned t = seed + static_cast<unsigned>(idx) * 2654435761u;
        t ^= t >> 16; t *= 0x7feb352du;
        t ^= t >> 15; t *= 0x846ca68bu;
        t ^= t >> 16;

        float const u = (static_cast<float>(t >> 8) / 16777216.f - 0.5f) * 2.f;
        d_r[idx] *= (1.f + eps * u);
    }
}

} /* namespace */

void factor_rir(state &st, problem &prob) {

    std::size_t const n  = prob.n;
    std::size_t const nn = n * n;

    /*  R's width is the storage dial. fp32 -> 8n^2, b24 -> 7n^2, bf16 ->
        6n^2, against a fixed 4n^2 for the fp32 factorization. */
    {
        std::string const want =
            (tuning::current().get("rir.r_format_bf16", 0) != 0)? "bf16"
          : (tuning::current().get("rir.r_format_b24",  0) != 0)? "b24"
          : "fp32";
        st.r_format = (want == "bf16")? ozaki::format::bf16
                    : (want == "b24") ? ozaki::format::b24
                                      : ozaki::format::fp32;
    }
    std::size_t const r_bytes = ozaki::bytes_of(st.r_format);
    st.storage_n2 = 4. + static_cast<double>(r_bytes);

    st.d_lu   = static_cast<float *>(st.acquire(nn * sizeof(float)));
    st.d_r    = st.acquire(nn * r_bytes);
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

    /*  R is formed in fp32 a block at a time and compressed on the way out, so
        the full-width array never exists. Staging one column block costs
        n * nb * 4 bytes; materializing all of R in fp32 first would cost 4n^2
        and defeat the point of compressing it. */
    float *d_r_block = static_cast<float *>(
        st.acquire(n * nb * sizeof(float)));

    /*  The build's result is STORED and reused by every solve, so its error is
        undamped — unlike a refinement residual, which the next iteration
        corrects. On this operand (L*U from a real factorization) blocking above
        the exactness bound costs nothing and runs faster, because triangular
        factors have a narrow per-row dynamic range. Operand-specific: the same
        configuration is wrong by four orders on a dense product. */
    ozaki::config const cfg = config_for("rir.build");
    ozaki::workspace ws(n, nb, cfg, prob);

    /*  Hoisted out of the timed region. A cudaMalloc of n^2 floats is 256 MB
        at n=8192 and would be charged to the arithmetic — the defect the
        caller-owned-buffer rule exists to prevent, and it was sitting inert
        behind a default-off switch waiting for someone to turn it on. */
    float *d_m_inverse = (tuning::current().get("rir.solve.m_form", 0) != 0)
        ? static_cast<float *>(st.acquire(nn * sizeof(float)))
        : nullptr;

    timing::stopwatch watch;
    watch.start();

    factorize::lu_fp32(
        st.d_lu, st.d_ipiv, st.d_perm, d_work, d_info, prob.d_a, prob);

    ozaki::row_max(ws.d_mu, st.d_lu, ozaki::format::fp32, n, n, ozaki::shape::lower, prob);

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
            d_r_block, d_acc, st.d_lu, prob.d_a, st.d_perm, n, j, n_c);
        KERNEL_CHECK();

        ozaki::compress(
            static_cast<unsigned char *>(st.d_r) + j * n * r_bytes,
            d_r_block, n * n_c, st.r_format, prob);
    }

    /*  M = (LU)^-1, replacing the packed factor.

        R is built above from L and U; after that the factor is needed only to
        APPLY (LU)^-1, and applying it as one dense product is 1.94x faster
        than the blocked triangular solve because a solve serialises its
        diagonal blocks. Storage is unchanged — M is n^2 fp32 exactly as the
        packed factor was, and d_lu is dead from here on. */
    /*  Off by default: an fp32 M puts the fixed point at (M^-1 + R) X = PB
        with M^-1 about kappa*u_32 from LU, so the answer floors near 1e-9 and
        degrades with conditioning (sec.87). The form is fast and correct only
        with an fp64 inverse, which is 12n^2. */
    if (d_m_inverse != nullptr) {
        float *d_m = d_m_inverse;
        trsm::form_inverse(d_m, st.d_lu, prob);
        st.d_lu = d_m;
        st.m_form = true;
    }

    st.factor_ms = watch.stop();

    /*  Controlled perturbation of R, off unless asked. rir.perturb_r_exp is a
        negative power of ten: -5 multiplies every entry of R by 1 +- 1e-5. */
    int const pexp = tuning::current().get("rir.perturb_r_exp", 0);
    if (pexp != 0) {
        float const eps = static_cast<float>(std::pow(10., pexp));
        rir_perturb_r_kernel<<<launch::grid_for(nn), launch::BLOCK_SIZE>>>(
            static_cast<float *>(st.d_r), nn, eps, 12345u);
        KERNEL_CHECK();
    }
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
    double *d_neg  = static_cast<double *>(st.acquire(nk * sizeof(double)));
    float  *d_xf   = static_cast<float *>(st.acquire(nk * sizeof(float)));
    float  *d_rx   = static_cast<float *>(st.acquire(nk * sizeof(float)));
    float  *d_y    = static_cast<float *>(st.acquire(nk * sizeof(float)));

    ozaki::config const cfg = config_for("rir.solve");
    ozaki::workspace ws(n, k, cfg, prob);

    /*  The accurate solve and residual. Both are used only on the POLISH
        passes: the early passes have to converge the fixed point, not deliver
        accuracy, and an accurate residual feeding an fp32 triangular solve is
        precision thrown away. Measured, the last two to three passes are the
        only ones that need to be accurate. */
    trsm::config const tcfg = trsm::config::from_tuning();
    trsm::workspace tws;
    tws.acquire(prob, tcfg, n, k);
    int const pieces_r = tuning::current().get("rir.solve.rx.pieces", 4);
    /*  Separable from the accurate SOLVE so each can be measured alone. */
    bool const rx_compensated =
        tuning::current().get("rir.solve.rx.compensated", 1) != 0;

    bool const tf32_outer =
        tuning::current().get("rir.solve.tf32_outer", 0) != 0;

    /*  R*X as a single fp32 GEMM instead of an Ozaki cascade.

        The precision map for this scheme measured R*X as needing fp32 and only
        fp32 — TF32 is 176x too coarse, but plain fp32 is sufficient — and a
        cublasSgemm is exactly fp32. The Ozaki path here was inherited from
        sharing code with the mp-TRSM, which genuinely does need it; R*X does
        not. One SGEMM replaces 21 TF32 products.

        Why the error is tolerable: R is already small, ||R||/||A|| ~ 6e-8, so
        an fp32 GEMM's ~2^-24*sqrt(n) relative error on R*X lands near 3e-13
        relative to ||A||*||X||, which is what the backward error normalizes by.

        Only for an fp32-stored R: a packed R would have to be unpacked to feed
        cuBLAS, and a full-width fp32 copy costs 4n^2, which is exactly the
        storage the packing was buying back. A blocked unpack would work and is
        not implemented. */
    bool const plain_rx =
        st.r_format == ozaki::format::fp32 &&
        tuning::current().get("rir.solve.plain_rx", 1) != 0;

    /*  R*X never cancels — R is already ~2^-24 of A — which is the structural
        reason this form needs no fp64. Negated once so the residual is an
        accumulate rather than a subtract; restored before returning. */
    
    ozaki::row_max(ws.d_mu, st.d_r, st.r_format, n, n, ozaki::shape::full, prob);

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

    /*  Convergence tolerance, as a NEGATIVE power of ten on the sum of
        squares — so -20 means ||dX|| has fallen to 1e-10 of its first step.
        Tunable because the right value moved when R*X became a plain SGEMM:
        the iterate now converges to a slightly coarser floor, and a threshold
        set against the old floor spends a full pass confirming it. */
    double const tol = std::pow(
        10., static_cast<double>(
            tuning::current().get("rir.solve.tol_exp", -12)));

    /*  Cheap passes before the polish phase begins. One is enough on every
        matrix measured; it exists as a knob because the count is a property of
        the problem, not of the machine. */
    /*  POLISH IS OPT-IN, and off by default.

        The original design — cheap fp32 passes, then ONE mp-TRSM refinement at
        the end — measures 257.9 ms / 8.96e-16 at n=8192 k=2048, against
        split-MPIR's 437.0 and direct fp64's 441.8 at 8n^2. Replacing that
        single refinement with an accurate solve on every polish pass costs
        2.3x the time for 2.2x the accuracy (851.5 ms / 4.12e-16), and the M
        and G forms were then built to claw back a cost that need not have been
        paid.

        The structure was already right: iterate cheaply, correct once,
        accurately, at the end. That is the same conclusion sec.49 reached
        independently as "cheap-then-polish" and sec.60 confirmed structurally —
        it was in this file from the start, in the n_refine block below. */
    bool const use_polish =
        tuning::current().get("rir.solve.polish", 0) != 0;

    std::size_t const n_cheap = use_polish
        ? static_cast<std::size_t>(
              tuning::current().get("rir.solve.cheap_passes", 1))
        : cap;

    /*  rho = ||dX_m|| / ||dX_{m-1}||, the fixed point's measured contraction
        ratio. THE convergence criterion for this method: it converges iff
        rho < 1, and rho is available for free from two iterates the solver
        already holds — no condition estimator, no norm of an inverse.

        This replaces every kappa- and growth-based rule tried earlier. Both
        were factors in the bound rho <= kappa * ||R||/||A||, which measured
        ~4000x loose, and each looked predictive until swept properly. rho is
        also what says how many polish passes are needed: below 1e-3 three
        suffice, near 3e-3 it takes five. */
    double rho = 0.;

    for (std::size_t it = 0; it != cap; ++it) {

        bool const polish = it >= n_cheap;

        CUDA_CHECK(cudaMemcpy(
            d_rhs, d_pb, nk * sizeof(double), cudaMemcpyDeviceToDevice));

        if (it != 0) {
            if (polish && rx_compensated) {
                /*  Compensated residual: exponent-aligned splits, TF32 tensor
                    products, fp64 folds. Costs up to 34x accuracy to skip. */
                trsm::residual(d_rhs, static_cast<float const *>(st.d_r),
                               d_x, k, pieces_r, -1., tws, tcfg, prob);
            }
            else if (!plain_rx) {
                ozaki::column_max(ws.d_nu, d_x, n, k, prob);
                convert::zero(d_neg, nk, prob);
            }
            /*  `else if`, not `if`: this is a SECOND chain, and leaving it
                ungated made the polish passes subtract R*X twice — once
                compensated and once through the SGEMM below. The signature was
                a wrong answer completely insensitive to every precision knob,
                which is what a miscounted term looks like and what a precision
                bug never does. */
            else if (plain_rx) {
                float const minus_one = -1.f, zero = 0.f;
                convert::demote(d_xf, d_x, nk, prob);
                CUBLAS_CHECK(cublasSgemm(
                    prob.blas,
                    CUBLAS_OP_N, CUBLAS_OP_N,
                    static_cast<int>(n), static_cast<int>(k),
                    static_cast<int>(n),
                    &minus_one,
                    static_cast<float const *>(st.d_r), static_cast<int>(n),
                    d_xf, static_cast<int>(n),
                    &zero,
                    d_rx, static_cast<int>(n)));
                convert::add_correction(d_rhs, d_rx, nk, prob);
            }
            else {
                /*  Subtracted rather than accumulated: R is packed, so
                    negating it in place is no longer a single kernel. */
                ozaki::accumulate_product(
                    d_neg, st.d_r, d_x, n, k, ozaki::shape::full, ws, prob,
                    0, st.r_format);
                convert::subtract(d_rhs, d_neg, nk, prob);
            }
        }

        CUDA_CHECK(cudaMemcpy(
            d_prev, d_x, nk * sizeof(double), cudaMemcpyDeviceToDevice));

        if (polish) {
            if (st.m_form) {
                /*  X = M rhs, one dense compensated product. */
                convert::zero(d_x, nk, prob);
                trsm::residual(d_x, st.d_lu, d_rhs, k, tcfg.pieces_tri, 1.,
                               tws, tcfg, prob);
            }
            else {
                CUDA_CHECK(cudaMemcpy(
                    d_x, d_rhs, nk * sizeof(double), cudaMemcpyDeviceToDevice));
                trsm::solve(d_x, st.d_lu, k, tws, tcfg, prob);
            }
        }
        else if (st.m_form) {
            /*  Cheap pass: the same product in plain fp32. The early passes
                only walk the fixed point in, so precision spent here is
                destroyed by the next accurate pass anyway. */
            float const one = 1.f, zero = 0.f;
            convert::demote(d_y, d_rhs, nk, prob);
            CUBLAS_CHECK(cublasSgemm(
                prob.blas, CUBLAS_OP_N, CUBLAS_OP_N,
                static_cast<int>(n), static_cast<int>(k), static_cast<int>(n),
                &one, st.d_lu, static_cast<int>(n),
                d_y, static_cast<int>(n), &zero, d_xf, static_cast<int>(n)));
            convert::promote(d_x, d_xf, nk, prob);
        }
        else {
            convert::demote(d_y, d_rhs, nk, prob);
            factorize::lu_solve(d_y, st.d_lu, k, prob, tf32_outer);
            convert::promote(d_x, d_y, nk, prob);
        }

        ++used;

        /*  The fixed-point form yields the ITERATE, not a correction, so the
            measure is ||X_new - X_old||. Testing the residual change instead
            barely moves between passes and fires immediately.

            Sums of squares, so the tolerance is squared: 1e-20 means ||dX||
            has fallen to 1e-10 of its first step. Measured, that stops one
            pass earlier than a tighter value with the same answer. */
        double const current =
            convert::sum_squares_difference(d_x, d_prev, nk, prob);

        /*  The stopping rule must not straddle the cheap/polish switch. When
            the solve's precision changes, ||dX|| jumps because the ANSWER
            moved to a better one, not because the iteration diverged — and the
            stall test reads that jump as divergence and quits, leaving the
            polish phase one pass long. Measured: it stopped at 2 passes with
            3.28e-09 where 4 passes reach 5e-15.

            So the phase restarts the comparison, and no break is allowed until
            two polish passes have run and can be compared against each other. */
        bool const just_switched = (it == n_cheap);
        std::size_t const polish_done = polish? (it - n_cheap + 1) : 0;

        if (it == 0 || just_switched) {
            first = current;
            previous = current;
            continue;
        }

        {
            if (previous > 0. && current > 0.)
                rho = std::sqrt(current / previous);
            if (polish && polish_done < 2) {
                previous = current;
                continue;
            }
            /*  In the POLISH phase the test must be relative to the previous
                step, not to `first`.

                `first` is reset at the cheap->polish switch, and that reset
                records the jump caused by changing the solve's PRECISION — a
                number reflecting how far the fp32 answer sat from the accurate
                one, not the scale of the iteration. Testing
                `current <= tol * first` against it fires on the very next pass
                whatever the matrix, pinning the polish phase at exactly two
                passes and making `max_outer` inert. Benign problems converge
                in two and looked correct; ill-conditioned ones need more and
                silently did not get them — 1.78e-11 where four passes reach
                1e-14, with the pass-count knobs showing no effect at all,
                which is the signature of a parameter that is not connected
                rather than one that does not matter.

                "Still improving" is the honest criterion: keep going while the
                step is at least halving, stop when it is not. That adapts to
                rho without having to name it — slow convergence simply runs
                longer, and the cap remains the only hard limit. */
            if (polish) {
                /*  Stop when the ANSWER stops moving, not when the STEP stops
                    halving. Those differ: the step can keep contracting long
                    after the iterate has reached the accuracy its
                    representation allows, and "still halving" then runs three
                    extra passes for nothing — measured, 6 passes and 3 give
                    4.12e-16 alike, at 1266 ms against 510.

                    The scale-free test is ||dX|| against ||X||: once the step
                    is at fp64 rounding relative to the solution, further
                    passes cannot change it. `first` is deliberately not used —
                    it records the cheap->polish precision jump, not the scale
                    of the iteration. */
                double x_norm = 0.;
                CUBLAS_CHECK(cublasDnrm2(
                    prob.blas, static_cast<int>(nk), d_x, 1, &x_norm));
                bool const at_solution_scale =
                    current <= 1e-32 * x_norm * x_norm;
                bool const not_improving = current >= previous;
                if (at_solution_scale || not_improving)
                    break;
            }
            else {
                /*  With a polish phase to reach, the cheap phase must run its
                    budget: it converges to the fp32 solve's own floor (~1e-7)
                    long before the answer is acceptable, and exiting there
                    skips the polish entirely — measured once as 2 passes and
                    9.32e-09.

                    WITHOUT a polish phase the opposite holds. The refinement
                    below supplies the accuracy, so the cheap loop should stop
                    as soon as the iterate settles; forcing it to the cap
                    inflated the original configuration from 257.9 ms to
                    368.7 ms for nothing. */
                if (use_polish) {
                    if (current > previous)
                        break;
                }
                else {
                    bool const stalled   = current > 0.25 * previous;
                    bool const converged = current <= tol * first;
                    if (stalled || converged)
                        break;
                }
            }
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
        triangle to exploit.

        ONE step is enough only while kappa(LU) is small. Each step contracts
        by kappa*u_32, so at kappa = 5.6e4 a single step buys 3.4e-3 and the
        answer floors near 2e-11 — which is exactly where the conditioning
        sweep found R-IR sitting, insensitive to every other knob including a
        1e-4 perturbation of R. This is the ill-conditioned limiter. */
    /*  Skipped entirely when polish passes ran: those already solved the
        triangular system to fp64 accuracy, and this block's correction is
        computed with the fp32 `lu_solve` it was written to repair. Running it
        afterwards can only add error. It remains for the cheap-only path. */
    std::size_t const n_refine = (use_polish || st.m_form)? 0u : static_cast<std::size_t>(
        tuning::current().get("rir.solve.n_refine", 1));

    /*  The refinement consumes d_rhs — accumulate_product adds into it and the
        unit-diagonal trick subtracts from it — so a second pass would build on
        a destroyed residual. Save the outer loop's rhs and restore it each
        pass. d_prev is dead once the outer loop exits, so this costs nothing.

        Note the rhs held fixed here is PB - R*X_last, from the LAST outer
        pass. That is deliberate: these passes refine the TRIANGULAR solve
        LU X = rhs against a fixed right-hand side, which is a different loop
        from the outer fixed point that updates rhs as X moves. */
    CUDA_CHECK(cudaMemcpy(
        d_prev, d_rhs, nk * sizeof(double), cudaMemcpyDeviceToDevice));

    for (std::size_t rf = 0; rf != n_refine; ++rf) {

        if (rf != 0)
            CUDA_CHECK(cudaMemcpy(
                d_rhs, d_prev, nk * sizeof(double), cudaMemcpyDeviceToDevice));

        /*  t = U * X. */
        convert::zero(d_t, nk, prob);
        ozaki::row_max(ws.d_mu, st.d_lu, ozaki::format::fp32, n, n, ozaki::shape::upper, prob);
        ozaki::column_max(ws.d_nu, d_x, n, k, prob);
        ozaki::accumulate_product(
            d_t, st.d_lu, d_x, n, k, ozaki::shape::upper, ws, prob);

        /*  rhs -= L * t, with L's unit diagonal supplying the -t term. */
        convert::negate(st.d_lu, n * n, prob);
        ozaki::row_max(ws.d_mu, st.d_lu, ozaki::format::fp32, n, n, ozaki::shape::lower, prob);
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
    st.rho = rho;

    
}

} /* namespace solver */
