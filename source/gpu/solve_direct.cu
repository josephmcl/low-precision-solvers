#include "gpu/solver.h"

#include "gpu/error.h"
#include "gpu/timing.h"

#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <cstddef>

/*  Reference method: fp64 throughout, cusolverDnDgetrf then Dgetrs.

    This file is the template a new method copies. Two things it does that
    every method must do:

      - the factorization happens in factor() and nothing else does, so
        st.factor_ms is the method's whole fixed cost;
      - the timed region contains no allocation. Buffers come from
        st.acquire() before the stopwatch starts, because an allocator call
        inside a timed region is charged to the arithmetic. */

namespace solver {

using harness::problem;

void factor_direct(state &st, problem &prob) {

    std::size_t const n = prob.n;

    /*  Dgetrf overwrites its input, so the reference A is copied. This copy
        is part of the method's honest cost: keeping A resident in fp64 is
        what this method's 8n^2 buys. */
    st.d_a    = static_cast<double *>(st.acquire(n * n * sizeof(double)));
    st.d_ipiv = static_cast<int *>(st.acquire(n * sizeof(int)));

    int lwork = 0;
    CUSOLVER_CHECK(cusolverDnDgetrf_bufferSize(
        prob.solver,
        static_cast<int>(n),
        static_cast<int>(n),
        st.d_a,
        static_cast<int>(n),
        &lwork));

    double *d_work = static_cast<double *>(
        st.acquire(static_cast<std::size_t>(lwork) * sizeof(double)));
    int *d_info = static_cast<int *>(st.acquire(sizeof(int)));

    CUDA_CHECK(cudaMemcpy(
        st.d_a,
        prob.d_a,
        n * n * sizeof(double),
        cudaMemcpyDeviceToDevice));

    timing::stopwatch watch;
    watch.start();

    CUSOLVER_CHECK(cusolverDnDgetrf(
        prob.solver,
        static_cast<int>(n),
        static_cast<int>(n),
        st.d_a,
        static_cast<int>(n),
        d_work,
        st.d_ipiv,
        d_info));

    st.factor_ms = watch.stop();
}

void solve_direct(
    double       *d_x,
    double const *d_b,
    state        &st,
    problem      &prob) {

    std::size_t const n = prob.n;
    std::size_t const k = prob.k;

    int *d_info = static_cast<int *>(st.acquire(sizeof(int)));

    /*  Dgetrs solves in place, so the right-hand side is copied into the
        caller's output before the solve. The copy is inside the timed region:
        it is work this method must do to produce X, and hoisting it would
        credit the method with a solve it did not perform. The rule is that
        everything needed to produce X is timed — drawing the line anywhere
        else invites the same judgment call on the demote, the permutation and
        the (hi,lo) split, which cost 30 ms rather than 0.15.

        Note the asymmetry, so it is not later read as bias against the
        reference: this method is the only one that pays a B -> X copy, and it
        pays it because Dgetrs is in-place, not for any algorithmic reason.
        IRSXgesv takes separate B and X; the iterative schemes write X
        directly. Measured, the copy is 0.03-0.09% of this method's solve
        (2 * n * k * 8 bytes against ~1.8 TB/s), which is ~100x under the
        run-to-run spread, so it is charged rather than engineered around. */
    timing::stopwatch watch;
    watch.start();

    CUDA_CHECK(cudaMemcpy(
        d_x,
        d_b,
        n * k * sizeof(double),
        cudaMemcpyDeviceToDevice));

    CUSOLVER_CHECK(cusolverDnDgetrs(
        prob.solver,
        CUBLAS_OP_N,
        static_cast<int>(n),
        static_cast<int>(k),
        st.d_a,
        static_cast<int>(n),
        st.d_ipiv,
        d_x,
        static_cast<int>(n),
        d_info));

    st.solve_ms = watch.stop();

    /*  A direct solve does not iterate. Reported as zero rather than left
        unset, so the column means "did not refine" and not "forgot to
        record". */
    st.n_iterations = 0;
}

} /* namespace solver */
