#pragma once

#include "common/convert.h"
#include "common/factorize.h"
#include "common/ozaki.h"
#include "common/timing.h"
#include "common/tuning.h"

#include "common/error.h"

#include <cuda_runtime.h>
#include <iostream>

#include "common/definitions.h"
#include "common/problem.h"

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
    /*  Matrix-resident bytes per n^2, when the method's storage depends on a
        runtime choice rather than the fixed value in its registry entry. Zero
        means "use the registry value". Reported rather than assumed, so a
        capacity claim always matches the configuration that produced the
        timings beside it. */
    double      storage_n2   = 0.;

    /*  Measured contraction ratio of the fixed point, ||dX_m||/||dX_{m-1}||.

        THE convergence criterion for a fixed-point scheme: it converges iff
        rho < 1. Free to compute from two iterates the solver already holds, so
        a caller can check convergence at runtime and fall back — which no
        kappa-based rule allows, since kappa needs an estimator. It also
        predicts the pass count. Zero when the method does not iterate. */
    double      rho          = 0.;

    /*  d_lu holds M = (LU)^-1 rather than the packed factor. */
    bool        m_form       = false;

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

    /*  R = PA - LU, the residual-storage scheme's matrix. Void because its
        element width is a tunable: fp32, 24-bit or bf16, giving 8n^2, 7n^2 or
        6n^2 total. See ozaki::format. */
    void         *d_r = nullptr;
    ozaki::format r_format = ozaki::format::fp32;

    /*  A as an unevaluated sum of two fp32 words, for the split residual. */
    float *d_a_hi = nullptr;
    float *d_a_lo = nullptr;

    /*  A in fp64, for the methods that keep it resident. */
    double *d_a = nullptr;

    /*  Row permutation from the factorization, composed from getrf's
        sequential interchanges. Its own member rather than borrowed space in
        another pointer: the two have different types and different lifetimes,
        and reusing one for the other is how a harness starts lying about what
        it holds. */
    int *d_perm = nullptr;

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

/*  Is the vendor's emulated fp64 math mode available in THIS toolkit?

    CUSOLVER_FP64_EMULATED_FIXEDPOINT_MATH is an enumerator, not a macro, so it
    cannot be probed with #ifdef -- the version gate is the only way to ask.
    Measured: absent in CUDA 12.4, present in 13.0 and 13.2. A CUDA 12.4 H100
    failed to BUILD once this method was added, which is the wrong failure: an
    old toolkit should lose the method, not the harness.

    THE THREE FEATURES SHIPPED SEPARATELY, and the version that matters is the
    LAST one. Measured directly from the headers:

        toolkit  cuBLAS 32F_EMU  cuBLAS 64F_EMU  cuSOLVER fp32  cuSOLVER fp64
        12.4     absent          absent          absent         absent
        13.0     present         present         present        ABSENT
        13.2     present         present         present         present

    So an emulated fp64 GEMM has been reachable since 13.0, but driving Dgetrf
    through it needs 13.2. That is why the B300 investigation closed this
    exposure correctly and is now reopened: SetEmulationStrategy and Xgetrf
    were genuinely the only candidates in that toolkit, and both are no-ops.
    The API that works did not exist yet. The archived numbers were not unfair
    when measured; the software moved under them. */
#if defined(CUDART_VERSION) && CUDART_VERSION >= 13020
#define LPS_HAVE_FP64_EMULATION 1
#else
#define LPS_HAVE_FP64_EMULATION 0
#endif

#if LPS_HAVE_FP64_EMULATION
/*  The same routines with cusolverDnSetMathMode(FP64_EMULATED_FIXEDPOINT).
    Reported ALONGSIDE the default-mode reference, not instead of it: the
    default is what a naive user gets, this is what the vendor makes available,
    and on fp64-deprecated parts they differ by 3.18x at 1.3% backward error.
    See the rationale block in solve_direct.cu. */
void factor_direct_emulated(state &st, problem &prob);
void solve_direct_emulated(
    double       *d_x,
    double const *d_b,
    state        &st,
    problem      &prob);
#endif

/*  split-MPIR: A as an unevaluated fp32 pair, Ozaki residual, no fp64
    arithmetic anywhere. 12n^2, same footprint as classical MPIR. */
void factor_split_mpir(state &st, problem &prob);
void solve_split_mpir(
    double       *d_x,
    double const *d_b,
    state        &st,
    problem      &prob);

/*  R-IR: store R = PA - LU instead of A. 8n^2, the only scheme here that does
    not keep A in any form. */
void factor_rir(state &st, problem &prob);
void solve_rir(
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
