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
    buffers. A method that needs storage the four below do not cover adds a
    named member; the set is deliberately small and visible rather than
    hidden behind a void *.

    n_iterations is reported, not configured. Fixed iteration counts made
    two separate accuracy comparisons meaningless in the work this harness
    replaces: a method capped below what it needed looked inaccurate, and
    one running past convergence looked slow. Methods must stop on a
    convergence test and write the count they used here. */
struct state {

    ~state();

    state()                          = default;
    state(state const &)             = delete;
    state &operator=(state const &)  = delete;

    /*  Filled by factor(), read by the reporter. Event-timed inside the
        method, because a wall-clock delta around the call also charges
        whatever the driver did on either side of it. */
    double      factor_ms    = 0.;
    std::size_t n_iterations = 0;

    /*  fp32 factorization, shared by every refinement scheme. */
    float *d_lu   = nullptr;
    int   *d_ipiv = nullptr;

    /*  R = PA - LU, the residual-storage scheme's matrix. */
    float *d_r = nullptr;

    /*  A as an unevaluated double-fp32 sum, for the split residual. */
    float *d_a_hi = nullptr;
    float *d_a_lo = nullptr;

    /*  fp64 A, for the methods that keep it resident. */
    double *d_a = nullptr;

    void *acquire(std::size_t const bytes);

private:

    std::vector<void *> _d_owned;
};

/*  A method is two free functions and the storage it holds resident.

    factor and solve are separate entry points *by construction*, so that a
    driver cannot fuse them. Reporting a method's solve time against
    another's factor-plus-solve is the single mistake this harness exists to
    prevent — it flattered one scheme by roughly 3x before it was caught, and
    it is invisible in any interface where one call does both. Keeping them
    apart makes the fair comparison the only one that is easy to write.

    storage_n2 is carried in the same record as the timing so a capacity
    claim and a speed claim about one method cannot drift apart. */
struct method {
    char const *name;

    /*  Everything independent of the right-hand side: demote, factor,
        build whatever the scheme keeps. Writes st.factor_ms. */
    void (*factor)(state &st, problem &prob);

    /*  Solve for prob.k right-hand sides from the factored state. Writes
        st.n_iterations. d_x is n x k and caller-owned. */
    void (*solve)(
        double       *d_x,
        double const *d_b,
        state        &st,
        problem      &prob);

    /*  Matrix-resident bytes per n^2 (see namespace storage). */
    double storage_n2;
};

/*  The methods the harness scores. Adding a method means adding one entry
    here and one file pair; nothing else in the harness changes. */
std::vector<method> const &registry();

/*  Reference: cusolverDnDgetrf + cusolverDnDgetrs in fp64 throughout.
    Slowest on fp64-deprecated hardware and the accuracy reference
    everywhere. */
void factor_direct(state &st, problem &prob);
void solve_direct(
    double       *d_x,
    double const *d_b,
    state        &st,
    problem      &prob);

/*  Vendor baseline: cusolverDnIRSXgesv, classical mixed-precision
    refinement with an fp32 factorization and an fp64 residual.

    Note for anyone reading its timings: IRSXgesv is monolithic. cuSOLVER
    exposes no factor/solve boundary, so factor_vendor_irs does the warmup
    and parameter setup only, and the whole cost lands in solve. Its column
    is therefore *not* comparable to the other methods' factor/solve split,
    and the reporter labels it so. */
void factor_vendor_irs(state &st, problem &prob);
void solve_vendor_irs(
    double       *d_x,
    double const *d_b,
    state        &st,
    problem      &prob);

} /* namespace solver */
