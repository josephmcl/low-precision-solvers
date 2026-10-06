#include "common/solver.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

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
    /*  The inner precision: FP32 unless LPS_IRS_PRECISION names another
        (16F, 16BF, TF32, 32F), for the conditioning sweep against MPIR. */
    cusolverPrecType_t lowest = CUSOLVER_R_32F;
    if (char const *e = std::getenv("LPS_IRS_PRECISION")) {
        std::string const p(e);
        if (p == "16F") lowest = CUSOLVER_R_16F;
        else if (p == "16BF") lowest = CUSOLVER_R_16BF;
        else if (p == "TF32") lowest = CUSOLVER_R_TF32;
    }
    CUSOLVER_CHECK(cusolverDnIRSParamsSetSolverLowestPrecision(
        params, lowest));
    /*  The refinement solver: classical unless LPS_IRS_REFINE names one of
        the GMRES ones. gmres is cuSOLVER's GMRES-based refinement;
        classical_gmres is classical refinement whose correction equation is
        solved by an inner GMRES. Tolerances stay at cuSOLVER's defaults in
        every mode. */
    cusolverIRSRefinement_t refine = CUSOLVER_IRS_REFINE_CLASSICAL;
    st.irs_refine = 0;
    if (char const *e = std::getenv("LPS_IRS_REFINE")) {
        std::string const r(e);
        if (r == "gmres") { refine = CUSOLVER_IRS_REFINE_GMRES; st.irs_refine = 1; }
        else if (r == "classical_gmres") { refine = CUSOLVER_IRS_REFINE_CLASSICAL_GMRES; st.irs_refine = 2; }
    }
    CUSOLVER_CHECK(cusolverDnIRSParamsSetRefinementSolver(params, refine));

    /*  A cap, not a schedule. IRS stops on its own convergence test and
        reports the count it used, which is the behaviour every method here is
        held to; the cap only bounds a pathological case. */
    st.irs_max_iters = 50;
    if (char const *e = std::getenv("LPS_IRS_MAXITERS"))        /* a larger cap, for the sensitivity runs */
        st.irs_max_iters = std::atoi(e);
    CUSOLVER_CHECK(cusolverDnIRSParamsSetMaxIters(params, st.irs_max_iters));
    if (st.irs_refine != 0) {
        /*  The inner GMRES cap, set to cuSOLVER's documented default so that
            the row can state it. LPS_IRS_INNER overrides. */
        st.irs_max_inner = 50;
        if (char const *e = std::getenv("LPS_IRS_INNER"))
            st.irs_max_inner = std::atoi(e);
        CUSOLVER_CHECK(cusolverDnIRSParamsSetMaxItersInner(params, st.irs_max_inner));
    }

    /*  LPS_IRS_TOL replaces cuSOLVER's default stopping tolerance (the
        default is RNRM < sqrt(n) XNRM ANRM EPS in infinity norms), for runs
        that hold the vendor solver to the same backward-error level as the
        other methods. */
    if (char const *e = std::getenv("LPS_IRS_TOL")) {
        st.irs_tol = std::atof(e);
        if (st.irs_tol > 0.)
            CUSOLVER_CHECK(cusolverDnIRSParamsSetTol(params, st.irs_tol));
    }
    /*  LPS_IRS_NOFALLBACK=1: no FP64 factorization when the refinement does
        not converge, so the time is that of the failed attempt alone and x
        is whatever the refinement left. */
    if (std::getenv("LPS_IRS_NOFALLBACK") != nullptr) {
        st.irs_fallback_on = 0;
        CUSOLVER_CHECK(cusolverDnIRSParamsDisableFallback(params));
    }
    bool const want_history = (std::getenv("LPS_IRS_HISTORY") != nullptr);
    if (want_history)
        CUSOLVER_CHECK(cusolverDnIRSInfosRequestResidual(infos));

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

    cusolverStatus_t const irs_rc = cusolverDnIRSXgesv(
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
        d_info);
    /*  With the fallback disabled a run that does not converge may come back
        with a non-success code; that is the outcome being measured, not an
        error of the harness. */
    if (st.irs_fallback_on != 0 || irs_rc == CUSOLVER_STATUS_SUCCESS)
        CUSOLVER_CHECK(irs_rc);
    else
        std::printf("IRSSTATUS,fallback_disabled,return_code,%d,returned_iterations,%d\n", static_cast<int>(irs_rc), n_iterations);

    st.total_ms = watch.stop();

    /*  Negative means it did not converge within the cap; reported as zero so
        a non-converged run cannot be read as a fast one. */
    st.n_iterations = (n_iterations > 0)?
        static_cast<std::size_t>(n_iterations) : 0;
    st.raw_iterations = n_iterations;
    st.status = (n_iterations > 0)? 1 : (st.irs_fallback_on != 0)? 3 : 2;
    {
        cusolver_int_t total = 0, outer = 0;
        CUSOLVER_CHECK(cusolverDnIRSInfosGetNiters(infos, &total));
        CUSOLVER_CHECK(cusolverDnIRSInfosGetOuterNiters(infos, &outer));
        st.irs_niters = static_cast<int>(total);
        st.irs_outer_niters = static_cast<int>(outer);

        /*  The residual history cuSOLVER keeps when asked: (maxiters + 1) rows
            of (iterations so far, residual norm), column major, of which
            the first outer + 1 rows are filled. Both columns are printed as
            stored, so a different layout would show. */
        if (want_history) {
            void *hist = nullptr;
            cusolver_int_t cap = 0;
            CUSOLVER_CHECK(cusolverDnIRSInfosGetResidualHistory(infos, &hist));
            CUSOLVER_CHECK(cusolverDnIRSInfosGetMaxIters(infos, &cap));
            double const *h = static_cast<double const *>(hist);
            int const rows = static_cast<int>(std::min<cusolver_int_t>(std::abs(outer), cap)) + 1;
            char const *prec = std::getenv("LPS_IRS_PRECISION");
            std::printf("IRSHIST,refine,%d,precision,%s,max_iters,%d,max_inner,%d,tol,%g,returned,%d,outer,%d,rows,%d,infos_max_iters,%d\n",
                        st.irs_refine, (prec != nullptr)? prec : "32F", st.irs_max_iters, st.irs_max_inner, st.irs_tol,
                        n_iterations, static_cast<int>(outer), rows, static_cast<int>(cap));
            if (h != nullptr)
                for (int i = 0; i < rows; ++i)
                    std::printf("IRSHISTROW,%d,%.6e,%.6e\n", i, h[i], h[i + static_cast<std::size_t>(cap) + 1]);
            std::fflush(stdout);
        }
    }

    CUSOLVER_CHECK(cusolverDnIRSParamsDestroy(params));
    CUSOLVER_CHECK(cusolverDnIRSInfosDestroy(infos));
}

} /* namespace solver */
