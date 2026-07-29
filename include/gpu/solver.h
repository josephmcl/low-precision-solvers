#pragma once

#include "gpu/definitions.h"
#include "gpu/problem.h"

#include <cstddef>
#include <vector>

namespace solver {

using harness::problem;

/*  Method-private factored state.

    Every method's fixed cost lands here, and every device allocation it
    makes is tracked in _d_owned and released by the destructor — the same
    manual-tracking pattern the rest of the codebase uses for vendor
    buffers. A method that needs storage the members below do not cover adds
    a named member; the set is deliberately small and visible rather than
    hidden behind a void *.

    n_iterations is reported, not configured. Fixed iteration counts made two
    separate accuracy comparisons meaningless in the work this harness
    replaces: a method capped below what it needed looked inaccurate, and one
    running past convergence looked slow. Methods must stop on a convergence
    test and write the count they used here. */
struct state {

    ~state();

    state()                          = default;
    state(state const &)             = delete;
    state &operator=(state const &)  = delete;

    /*  Written by the method, read by the reporter. Event-timed inside the
        method, because a wall-clock delta around the call also charges
        whatever the driver did on either side of it.

        A split method fills factor_ms and solve_ms. A monolithic one fills
        total_ms only and leaves the other two at zero; see `method`. */
    double      factor_ms    = 0.;
    double      solve_ms     = 0.;
    double      total_ms     = 0.;
    std::size_t n_iterations = 0;

    /*  True when the method reported a factor/solve breakdown. The reporter
        prints "--" in those columns otherwise rather than a zero that would
        read as "free". */
    bool split_reported = false;

    /*  fp32 factorization, shared by every refinement scheme. */
    float *d_lu   = nullptr;
    int   *d_ipiv = nullptr;

    /*  R = PA - LU, the residual-storage scheme's matrix. */
    float *d_r = nullptr;

    /*  A as an unevaluated sum of two fp32 words, for the split residual. */
    float *d_a_hi = nullptr;
    float *d_a_lo = nullptr;

    /*  A in fp64, for the methods that keep it resident. */
    double *d_a = nullptr;

    void *acquire(std::size_t const bytes);

private:

    std::vector<void *> _d_owned;
};

/*  A method is either a factor/solve pair or a single factor_solve, plus the
    storage it holds resident. Exactly one of the two forms is provided:

      - split:      factor != nullptr && solve != nullptr, factor_solve null
      - monolithic: factor_solve != nullptr, the other two null

    Splitting is preferred and is what the fp64-free schemes do. Comparing
    one method's solve against another's factor-plus-solve overstated a
    scheme by roughly 3x before it was caught, and that mistake is invisible
    in any interface where a single call does both — so the split exists at
    the type level, not as a convention.

    factor_solve is for methods that genuinely cannot be split.
    cusolverDnIRSXgesv is the case in hand: cuSOLVER exposes no boundary
    between its factorization and its refinement, and an earlier version of
    this harness estimated one from a k=1 probe. That estimate was reported
    with a caveat nobody would carry downstream. Declaring the method
    monolithic is the honest alternative — the number it produces is a total,
    is labelled a total, and is compared only against other totals.

    Which is why the reporter ranks on total_ms and treats the breakdown as
    detail: totals are always comparable across both forms, breakdowns are
    not. Reading a monolithic total against a split method's solve column is
    the one comparison this design still permits by hand, and the "--" in
    those columns is there to make it look wrong. */
struct method {
    char const *name;

    /*  Everything independent of the right-hand side: demote, factor, build
        whatever the scheme keeps. Writes st.factor_ms. */
    void (*factor)(state &st, problem &prob);

    /*  Solve for prob.k right-hand sides from the factored state. Writes
        st.solve_ms and st.n_iterations. d_x is n x k and caller-owned. */
    void (*solve)(
        double       *d_x,
        double const *d_b,
        state        &st,
        problem      &prob);

    /*  Factor and solve in one call, for methods with no exposed boundary.
        Writes st.total_ms and st.n_iterations, and leaves split_reported
        false. */
    void (*factor_solve)(
        double       *d_x,
        double const *d_b,
        state        &st,
        problem      &prob);

    /*  Matrix-resident bytes per n^2 (see namespace storage). */
    double storage_n2;

    bool is_split() const {
        return factor != nullptr && solve != nullptr;
    }
};

/*  Run one method end to end and fill st's timing fields, dispatching on
    which form the method provides. Drivers call this rather than the
    function pointers, so total_ms is populated the same way for both forms
    and no call site has to remember the dispatch rule. */
void run(
    double       *d_x,
    double const *d_b,
    method const &m,
    state        &st,
    problem      &prob);

/*  The methods the harness scores. Adding a method means adding one entry
    here and one file pair; nothing else in the harness changes. */
std::vector<method> const &registry();

/*  Reference: cusolverDnDgetrf + cusolverDnDgetrs, fp64 throughout. Slowest
    on fp64-deprecated hardware and the accuracy reference everywhere. */
void factor_direct(state &st, problem &prob);
void solve_direct(
    double       *d_x,
    double const *d_b,
    state        &st,
    problem      &prob);

/*  Vendor baseline: cusolverDnIRSXgesv, classical mixed-precision
    refinement — fp32 factorization, fp64 residual. Monolithic; see the note
    on `method`. */
void factor_solve_vendor_irs(
    double       *d_x,
    double const *d_b,
    state        &st,
    problem      &prob);

} /* namespace solver */
