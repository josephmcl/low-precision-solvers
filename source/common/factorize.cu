#include "common/factorize.h"
#include "common/tuning.h"

namespace factorize {

using harness::problem;

int buffer_size(problem &prob) {

    int lwork = 0;
    int const n = static_cast<int>(prob.n);

    /*  getrf's workspace query does not read the matrix, so a null pointer
        would do; passing a real one avoids relying on that. */
    float *d_probe = nullptr;
    CUDA_CHECK(cudaMalloc(&d_probe, sizeof(float)));
    CUSOLVER_CHECK(cusolverDnSgetrf_bufferSize(
        prob.solver, n, n, d_probe, n, &lwork));
    CUDA_CHECK(cudaFree(d_probe));

    return lwork;
}

void lu_fp32(
    float             *d_lu,
    int               *d_ipiv,
    int               *d_perm,
    float             *d_work,
    int               *d_info,
    double const      *d_a,
    problem           &prob) {

    /*  EMULATED fp32 FACTORIZATION (Blackwell).

        sgetrf is the largest single component of both R-IR's and split-MPIR's
        factor, and cuSOLVER exposes bf16x9 emulation for it via SetMathMode --
        3 bf16 pieces, 24 significand bits, so fp32-equivalent. Measured
        standalone at n=8192: 38.08 -> 34.89 ms (1.09x) at bit-identical
        ||PA - LU||.

        NOTE the asymmetry with cuBLAS: SetEmulationStrategy is accepted and
        SILENTLY IGNORED for both Strsm and getrf (9.17 vs 9.16 ms, status 0);
        only the MATH MODE does anything. Two near-identical-looking APIs, one
        of which is a no-op here.

        Shared by every method that factors in fp32, so this improves the
        baseline too and is not a differentiator. Off by default: emulation is
        not supported on every part. */
    bool const emulate =
        tuning::current().get("factorize.sgetrf_emulated", 0) != 0;
    cusolverMathMode_t previous_math = CUSOLVER_DEFAULT_MATH;
    if (emulate) {
        cusolverDnGetMathMode(prob.solver, &previous_math);
        cusolverDnSetMathMode(prob.solver,
                              CUSOLVER_FP32_EMULATED_BF16X9_MATH);
    }


    std::size_t const n = prob.n;

    convert::demote(d_lu, d_a, n * n, prob);

    CUSOLVER_CHECK(cusolverDnSgetrf(
        prob.solver,
        static_cast<int>(n), static_cast<int>(n),
        d_lu, static_cast<int>(n),
        d_work, d_ipiv, d_info));

    /*  getrf reports SEQUENTIAL row interchanges, 1-based: row i was swapped
        with row ipiv[i]. Composing them by applying the swaps in order to an
        identity gives the permutation a solve needs. Done on the host because
        it is inherently serial and n ints against an O(n^2 k) solve is
        nothing; doing it on the device would need a scan for no gain. */
    std::vector<int> ipiv(n);
    CUDA_CHECK(cudaMemcpy(
        ipiv.data(), d_ipiv, n * sizeof(int), cudaMemcpyDeviceToHost));

    std::vector<int> perm(n);
    for (std::size_t i = 0; i != n; ++i)
        perm[i] = static_cast<int>(i);

    for (std::size_t i = 0; i != n; ++i) {
        int const j = ipiv[i] - 1;
        if (j >= 0 && static_cast<std::size_t>(j) < n)
            std::swap(perm[i], perm[static_cast<std::size_t>(j)]);
    }

    CUDA_CHECK(cudaMemcpy(
        d_perm, perm.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    if (emulate)
        cusolverDnSetMathMode(prob.solver, previous_math);

}

void lu_solve(
    float             *d_y,
    float const       *d_lu,
    std::size_t const  k,
    problem           &prob,
    bool const         tf32) {

    int const n = static_cast<int>(prob.n);
    float const one = 1.f;

    cublasMath_t previous_mode = CUBLAS_DEFAULT_MATH;
    if (tf32) {
        CUBLAS_CHECK(cublasGetMathMode(prob.blas, &previous_mode));
        CUBLAS_CHECK(cublasSetMathMode(prob.blas, CUBLAS_TF32_TENSOR_OP_MATH));
    }

    /*  L is unit-diagonal; U is not. Both come out of the one packed factor. */
    CUBLAS_CHECK(cublasStrsm(
        prob.blas,
        CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
        CUBLAS_OP_N, CUBLAS_DIAG_UNIT,
        n, static_cast<int>(k),
        &one, d_lu, n,
        d_y, n));

    CUBLAS_CHECK(cublasStrsm(
        prob.blas,
        CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER,
        CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
        n, static_cast<int>(k),
        &one, d_lu, n,
        d_y, n));

    if (tf32)
        CUBLAS_CHECK(cublasSetMathMode(prob.blas, previous_mode));
}

void lu_solve_blocked(
    float             *d_y,
    float const       *d_lu,
    std::size_t const  k,
    problem           &prob,
    int const          nb,
    bool const         emulated) {

    int const n  = static_cast<int>(prob.n);
    int const ki = static_cast<int>(k);
    float const one = 1.f, minus_one = -1.f;

    cublasComputeType_t const ct = emulated
        ? CUBLAS_COMPUTE_32F_EMULATED_16BFX9
        : CUBLAS_COMPUTE_32F;

    /*  L Y = Y, forward. L is unit-diagonal and strictly lower, packed in the
        same array as U. Each step solves one diagonal block, then removes that
        block's contribution from every row below it. */
    for (int j0 = 0; j0 < n; j0 += nb) {

        int const cb = (nb < n - j0)? nb : n - j0;

        CUBLAS_CHECK(cublasStrsm(
            prob.blas, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER,
            CUBLAS_OP_N, CUBLAS_DIAG_UNIT,
            cb, ki, &one,
            d_lu + j0 + static_cast<std::size_t>(j0) * n, n,
            d_y  + j0, n));

        int const rows = n - (j0 + cb);
        if (rows <= 0) continue;

        CUBLAS_CHECK(cublasGemmEx(
            prob.blas, CUBLAS_OP_N, CUBLAS_OP_N,
            rows, ki, cb,
            &minus_one,
            d_lu + (j0 + cb) + static_cast<std::size_t>(j0) * n,
            CUDA_R_32F, n,
            d_y + j0, CUDA_R_32F, n,
            &one,
            d_y + (j0 + cb), CUDA_R_32F, n,
            ct, CUBLAS_GEMM_DEFAULT));
    }

    /*  U Y = Y, backward. U is non-unit and on-or-above the diagonal, so the
        sweep runs from the last block up and each step removes the solved
        block's contribution from every row ABOVE it. */
    for (int j1 = n; j1 > 0; j1 -= nb) {

        int const cb = (nb < j1)? nb : j1;
        int const s  = j1 - cb;

        CUBLAS_CHECK(cublasStrsm(
            prob.blas, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER,
            CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
            cb, ki, &one,
            d_lu + s + static_cast<std::size_t>(s) * n, n,
            d_y  + s, n));

        if (s <= 0) continue;

        CUBLAS_CHECK(cublasGemmEx(
            prob.blas, CUBLAS_OP_N, CUBLAS_OP_N,
            s, ki, cb,
            &minus_one,
            d_lu + 0 + static_cast<std::size_t>(s) * n, CUDA_R_32F, n,
            d_y + s, CUDA_R_32F, n,
            &one,
            d_y, CUDA_R_32F, n,
            ct, CUBLAS_GEMM_DEFAULT));
    }
}

} /* namespace factorize */
