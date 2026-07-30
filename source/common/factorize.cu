#include "common/factorize.h"

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

} /* namespace factorize */
