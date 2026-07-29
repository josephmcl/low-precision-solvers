#include "gpu/solver.h"

#include "gpu/error.h"
#include "gpu/timing.h"

#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <cstddef>

/*  Vendor baseline: cusolverDnIRSXgesv.

    Classical mixed-precision refinement — fp32 factorization, fp64 residual —
    in NVIDIA's own implementation. This is the baseline that matters: a
    hand-rolled MPIR was 1.33x slower than this, and margins measured against
    the hand-rolled version were inflated by that factor.

    Monolithic by necessity. cuSOLVER exposes no boundary between the
    factorization and the refinement, so this is a factor_solve method and its
    number is a total. An earlier harness estimated the split from a k=1 probe
    and labelled it an estimate; declaring the method monolithic is the honest
    alternative. */

namespace solver {

using harness::problem;

void factor_solve_vendor_irs(
    double       *d_x,
    double const *d_b,
    state        &st,
    problem      &prob) {

    std::size_t const n = prob.n;
    std::size_t const k = prob.k;

    /*  IRSXgesv overwrites A, so it gets its own copy. Keeping A resident in
        fp64 alongside the fp32 factorization is what this method's 12n^2
        buys, and it is the footprint the capacity claim is made against. */
    st.d_a = static_cast<double *>(st.acquire(n * n * sizeof(double)));

    cusolverDnIRSParams_t params;
    cusolverDnIRSInfos_t  infos;
    CUSOLVER_CHECK(cusolverDnIRSParamsCreate(&params));
    CUSOLVER_CHECK(cusolverDnIRSInfosCreate(&infos));

    CUSOLVER_CHECK(cusolverDnIRSParamsSetSolverMainPrecision(
        params, CUSOLVER_R_64F));
    CUSOLVER_CHECK(cusolverDnIRSParamsSetSolverLowestPrecision(
        params, CUSOLVER_R_32F));
    CUSOLVER_CHECK(cusolverDnIRSParamsSetRefinementSolver(
        params, CUSOLVER_IRS_REFINE_CLASSICAL));

    /*  A cap, not a schedule. IRS stops on its own convergence test and
        reports the count it used, which is the behaviour every method here is
        held to; the cap only bounds a pathological case. */
    CUSOLVER_CHECK(cusolverDnIRSParamsSetMaxIters(params, 50));

    size_t lwork = 0;
    CUSOLVER_CHECK(cusolverDnIRSXgesv_bufferSize(
        prob.solver,
        params,
        static_cast<int>(n),
        static_cast<int>(k),
        &lwork));

    void *d_work = st.acquire((lwork > 0)? lwork : 1);
    int  *d_info = static_cast<int *>(st.acquire(sizeof(int)));

    CUDA_CHECK(cudaMemcpy(
        st.d_a,
        prob.d_a,
        n * n * sizeof(double),
        cudaMemcpyDeviceToDevice));

    int n_iterations = 0;

    timing::stopwatch watch;
    watch.start();

    CUSOLVER_CHECK(cusolverDnIRSXgesv(
        prob.solver,
        params,
        infos,
        static_cast<int>(n),
        static_cast<int>(k),
        st.d_a,
        static_cast<int>(n),
        const_cast<double *>(d_b),
        static_cast<int>(n),
        d_x,
        static_cast<int>(n),
        d_work,
        lwork,
        &n_iterations,
        d_info));

    st.total_ms = watch.stop();

    /*  Negative means it did not converge within the cap; reported as zero so
        a non-converged run cannot be read as a fast one. */
    st.n_iterations = (n_iterations > 0)?
        static_cast<std::size_t>(n_iterations) : 0;

    CUSOLVER_CHECK(cusolverDnIRSParamsDestroy(params));
    CUSOLVER_CHECK(cusolverDnIRSInfosDestroy(infos));
}

} /* namespace solver */
