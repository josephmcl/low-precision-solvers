#include "common/metrics.h"

namespace metrics {

using harness::problem;

/*  sum of squares over a flat array, one partial per block.

    One thread accumulates a grid-stride slice in a register, then the block
    reduces BLOCK_SIZE partials in shared memory. The host sums the block
    partials, so the result does not depend on grid size — a reduction that
    changes its answer with launch geometry cannot be used to compare two
    methods. */
__global__ void sum_squares_kernel(
    double const      *d_m,
    double            *d_partial,
    std::size_t const  n_elements) {

    __shared__ double s[launch::BLOCK_SIZE];

    double acc = 0.;
    for (std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < n_elements;
         idx += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
        double const v = d_m[idx];
        acc += v * v;
    }

    s[threadIdx.x] = acc;
    __syncthreads();

    for (int q = launch::BLOCK_SIZE / 2; q > 0; q >>= 1) {
        if (static_cast<int>(threadIdx.x) < q)
            s[threadIdx.x] += s[threadIdx.x + q];
        __syncthreads();
    }

    if (threadIdx.x == 0)
        d_partial[blockIdx.x] = s[0];
}

/*  sum of (m - n)^2, the same reduction over a difference formed on the fly
    so no n*k temporary is needed. */
__global__ void sum_squares_difference_kernel(
    double const      *d_m,
    double const      *d_n,
    double            *d_partial,
    std::size_t const  n_elements) {

    __shared__ double s[launch::BLOCK_SIZE];

    double acc = 0.;
    for (std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < n_elements;
         idx += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
        double const v = d_m[idx] - d_n[idx];
        acc += v * v;
    }

    s[threadIdx.x] = acc;
    __syncthreads();

    for (int q = launch::BLOCK_SIZE / 2; q > 0; q >>= 1) {
        if (static_cast<int>(threadIdx.x) < q)
            s[threadIdx.x] += s[threadIdx.x + q];
        __syncthreads();
    }

    if (threadIdx.x == 0)
        d_partial[blockIdx.x] = s[0];
}

/*  Sum the block partials on the host and take the root. */
static double finish(
    int const  n_blocks,
    problem   &prob) {

    std::vector<double> partial(static_cast<std::size_t>(n_blocks));
    CUDA_CHECK(cudaMemcpy(
        partial.data(),
        prob.d_partial,
        static_cast<std::size_t>(n_blocks) * sizeof(double),
        cudaMemcpyDeviceToHost));

    double total = 0.;
    for (std::size_t i = 0; i != partial.size(); ++i)
        total += partial[i];

    return std::sqrt(total);
}

double norm(
    double const      *d_m,
    std::size_t const  n_elements,
    problem           &prob) {

    int const n_blocks = launch::grid_for(n_elements);

    sum_squares_kernel<<<n_blocks, launch::BLOCK_SIZE>>>(
        d_m,
        prob.d_partial,
        n_elements);
    KERNEL_CHECK();

    return finish(n_blocks, prob);
}

double norm_difference(
    double const      *d_m,
    double const      *d_n,
    std::size_t const  n_elements,
    problem           &prob) {

    int const n_blocks = launch::grid_for(n_elements);

    sum_squares_difference_kernel<<<n_blocks, launch::BLOCK_SIZE>>>(
        d_m,
        d_n,
        prob.d_partial,
        n_elements);
    KERNEL_CHECK();

    return finish(n_blocks, prob);
}

report evaluate(
    double const *d_x,
    double const *d_x_ref,
    problem      &prob) {

    report out;

    std::size_t const n  = prob.n;
    std::size_t const k  = prob.k;
    std::size_t const nk = n * k;

    /*  residual = B - A*X, formed in fp64 against the untouched reference A.
        Every method is scored with this same call, so no method is measured
        against its own copy of the matrix. */
    CUDA_CHECK(cudaMemcpy(
        prob.d_residual,
        prob.d_b,
        nk * sizeof(double),
        cudaMemcpyDeviceToDevice));

    double const minus_one = -1., one = 1.;
    CUBLAS_CHECK(cublasDgemm(
        prob.blas,
        CUBLAS_OP_N, CUBLAS_OP_N,
        static_cast<int>(n), static_cast<int>(k), static_cast<int>(n),
        &minus_one,
        prob.d_a, static_cast<int>(n),
        d_x,      static_cast<int>(n),
        &one,
        prob.d_residual, static_cast<int>(n)));

    double const norm_r = norm(prob.d_residual, nk, prob);
    double const norm_b = norm(prob.d_b, nk, prob);

    out.norm_a = norm(prob.d_a, n * n, prob);
    out.norm_x = norm(d_x, nk, prob);

    /*  Both normalizations from one residual, so the pair can never describe
        different solutions. They differ by ||A|| ||X|| / ||B||, which is
        ~2.3e4 on a diagonally dominant matrix at n=8192 — large enough that
        an unlabelled number is unusable. */
    out.backward = (out.norm_a * out.norm_x > 0.)?
        norm_r / (out.norm_a * out.norm_x) : 0.;
    out.relative = (norm_b > 0.)? norm_r / norm_b : 0.;

    if (d_x_ref != nullptr) {
        double const norm_ref = norm(d_x_ref, nk, prob);
        out.forward = (norm_ref > 0.)?
            norm_difference(d_x, d_x_ref, nk, prob) / norm_ref : 0.;
    }

    return out;
}

} /* namespace metrics */
