#include "common/solver.h"

#include "common/int8lu_arm.h"

/*  Registry bridge for the INT-sliced arm. The arm itself lives behind
    int8lu_arm.h because exactly one translation unit may include the vendor
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
    st.arm        = int8lu_arm::create(prob.n, b, kfac);
    st.storage_n2 = int8lu_arm::STORAGE_N2;

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
    int8lu_arm::prepare(st.arm, prob.d_a);
    int8lu_arm::factor(st.arm);

    st.factor_ms = watch.stop();
}

void solve_int8lu(
    double       *d_x,
    double const *d_b,
    state        &st,
    problem      &prob) {

    if (st.arm == nullptr)
        return;

    std::size_t iterations = 0;
    st.solve_ms     = int8lu_arm::solve(st.arm, d_x, d_b, prob.k, &iterations);
    st.n_iterations = iterations;
}

} /* namespace solver */
