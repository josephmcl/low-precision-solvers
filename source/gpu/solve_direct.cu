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
        caller's output before the solve. */
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

    /*  A direct solve does not iterate. Reported as zero rather than left
        unset, so the column means "did not refine" and not "forgot to
        record". */
    st.n_iterations = 0;
}

} /* namespace solver */
