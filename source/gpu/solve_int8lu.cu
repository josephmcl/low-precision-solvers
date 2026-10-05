#include "common/solver.h"

#include "common/qlu.h"
#include "common/tuning.h"

/*  Registry bridge for the INT-sliced arm. The arm itself lives behind
    qlu.h because exactly one translation unit may include the vendor
    header; this file only needs the pure interface, so it compiles like any
    other method.

    Storage is 16n^2 and reported as such: the captured residual operator plus
    the carrier the factorization consumes. That is worse than every other
    method here except vendor IRS, and reducing it is the point of the R-IR
    merge, not something to round off in the table. */

namespace solver {

using harness::problem;

void factor_int8lu(state &st, problem &prob) {

    int const b    = tuning::current().get("int8lu.block", 256);
    int const kfac = tuning::current().get("int8lu.kfac", 4);

    /*  Allocation before the stopwatch: an allocator call inside a timed
        region is charged to the arithmetic, and this one is ~16n^2. */
    /*  The S-rung the campaign and the closed loop use: the largest of
        128/64/32/16 that divides n (128 is 2.5 ms a solve faster than 64 at
        n = 16384; 256 gains nothing more). This arm used to be created without it, so
        the registry timed the unblocked solve -- 61 ms a refinement step at
        n = 16384 against 23 ms with it -- and the baseline table charged the
        method for a solve nothing else runs. */
    int srung = 0;
    for (int ib : {128, 64, 32, 16})
        if (prob.n % static_cast<std::size_t>(ib) == 0) { srung = ib; break; }
    st.arm        = qlu::create(prob.n, b, kfac, qlu::kernel::sliced, srung);
    st.storage_n2 = qlu::STORAGE_N2;

    if (st.arm == nullptr) {
        std::cout << "[int8lu] could not allocate for n = " << prob.n << "\n";
        return;
    }

    timing::stopwatch watch;
    watch.start();

    /*  prepare() is timed: it transposes A out of the harness's column-major
        layout and splits it to the DF32 carrier, which is work this method
        must do to produce X — the same reason factorize::lu_fp32 times its
        demote for the other refinement schemes. It is also an artefact of two
        codebases disagreeing about layout rather than of the algorithm, so it
        is worth watching separately if it ever matters. */
    qlu::prepare(st.arm, prob.d_a);
    qlu::factor(st.arm);

    st.factor_ms = watch.stop();
}

void solve_int8lu(
    double       *d_x,
    double const *d_b,
    state        &st,
    problem      &prob) {

    if (st.arm == nullptr)
        return;

    /*  GMRES degree of the refinement: 0 is stationary refinement (the
        default), m in 2..8 is GMRES(m) (tuning key int8lu.gmres, or
        LPS_INT8LU_GMRES). */
    int const inner_j = tuning::current().get("int8lu.gmres", 0);
    std::size_t iterations = 0;
    st.solve_ms     = qlu::solve(st.arm, d_x, d_b, prob.k, &iterations, inner_j);
    st.n_iterations = iterations;
    st.status = qlu::last_converged(st.arm)? 1 : 2;
    st.raw_iterations = static_cast<int>(iterations);
}

} /* namespace solver */
