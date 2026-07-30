#include "common/convert.h"

namespace convert {

using harness::problem;

namespace {

/*  One thread owns one element of a grid-stride loop throughout this file. */

__global__ void demote_kernel(
    float             *d_out,
    double const      *d_in,
    std::size_t const  n_elements) {

    for (std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < n_elements;
         idx += static_cast<std::size_t>(blockDim.x) * gridDim.x)
        d_out[idx] = static_cast<float>(d_in[idx]);
}

__global__ void promote_kernel(
    double            *d_out,
    float const       *d_in,
    std::size_t const  n_elements) {

    for (std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < n_elements;
         idx += static_cast<std::size_t>(blockDim.x) * gridDim.x)
        d_out[idx] = static_cast<double>(d_in[idx]);
}

__global__ void negate_kernel(
    float             *d_m,
    std::size_t const  n_elements) {

    for (std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < n_elements;
         idx += static_cast<std::size_t>(blockDim.x) * gridDim.x)
        d_m[idx] = -d_m[idx];
}

__global__ void zero_kernel(
    double            *d_x,
    std::size_t const  n_elements) {

    for (std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < n_elements;
         idx += static_cast<std::size_t>(blockDim.x) * gridDim.x)
        d_x[idx] = 0.;
}

__global__ void add_correction_kernel(
    double            *d_x,
    float const       *d_d,
    std::size_t const  n_elements) {

    for (std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < n_elements;
         idx += static_cast<std::size_t>(blockDim.x) * gridDim.x)
        d_x[idx] += static_cast<double>(d_d[idx]);
}

__global__ void subtract_kernel(
    double            *d_x,
    double const      *d_y,
    std::size_t const  n_elements) {

    for (std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < n_elements;
         idx += static_cast<std::size_t>(blockDim.x) * gridDim.x)
        d_x[idx] -= d_y[idx];
}

/*  out(i, j) = in(perm[i], j), column major. DEMOTE selects whether the
    result lands in fp32; both forms are otherwise identical, so they share
    one body rather than two that can drift. */
template <bool DEMOTE, typename OUT>
__global__ void permute_rows_kernel(
    OUT               *d_out,
    double const      *d_in,
    int const         *d_perm,
    std::size_t const  n,
    std::size_t const  k) {

    std::size_t const n_elements = n * k;

    for (std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < n_elements;
         idx += static_cast<std::size_t>(blockDim.x) * gridDim.x) {

        std::size_t const i = idx % n;
        std::size_t const j = idx / n;
        double const v = d_in[static_cast<std::size_t>(d_perm[i]) + j * n];

        d_out[idx] = DEMOTE? static_cast<OUT>(static_cast<float>(v))
                           : static_cast<OUT>(v);
    }
}

/*  Block-reduced sum of squares. The host sums the block partials, so the
    result does not depend on grid size — a reduction whose answer moves with
    launch geometry cannot be used to compare two runs. */
__global__ void sum_squares_f32_kernel(
    float const       *d_m,
    double            *d_partial,
    std::size_t const  n_elements) {

    __shared__ double s[launch::BLOCK_SIZE];

    double acc = 0.;
    for (std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < n_elements;
         idx += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
        double const v = static_cast<double>(d_m[idx]);
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

__global__ void sum_squares_difference_kernel(
    double const      *d_x,
    double const      *d_y,
    double            *d_partial,
    std::size_t const  n_elements) {

    __shared__ double s[launch::BLOCK_SIZE];

    double acc = 0.;
    for (std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < n_elements;
         idx += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
        double const v = d_x[idx] - d_y[idx];
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

double sum_partials(
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

    return total;
}

} /* namespace */

void demote(
    float             *d_out,
    double const      *d_in,
    std::size_t const  n_elements,
    problem           &prob) {

    (void) prob;
    demote_kernel<<<launch::grid_for(n_elements), launch::BLOCK_SIZE>>>(
        d_out, d_in, n_elements);
    KERNEL_CHECK();
}

void promote(
    double            *d_out,
    float const       *d_in,
    std::size_t const  n_elements,
    problem           &prob) {

    (void) prob;
    promote_kernel<<<launch::grid_for(n_elements), launch::BLOCK_SIZE>>>(
        d_out, d_in, n_elements);
    KERNEL_CHECK();
}

void negate(
    float             *d_m,
    std::size_t const  n_elements,
    problem           &prob) {

    (void) prob;
    negate_kernel<<<launch::grid_for(n_elements), launch::BLOCK_SIZE>>>(
        d_m, n_elements);
    KERNEL_CHECK();
}

void zero(
    double            *d_x,
    std::size_t const  n_elements,
    problem           &prob) {

    (void) prob;
    zero_kernel<<<launch::grid_for(n_elements), launch::BLOCK_SIZE>>>(
        d_x, n_elements);
    KERNEL_CHECK();
}

void add_correction(
    double            *d_x,
    float const       *d_d,
    std::size_t const  n_elements,
    problem           &prob) {

    (void) prob;
    add_correction_kernel<<<launch::grid_for(n_elements), launch::BLOCK_SIZE>>>(
        d_x, d_d, n_elements);
    KERNEL_CHECK();
}

void subtract(
    double            *d_x,
    double const      *d_y,
    std::size_t const  n_elements,
    problem           &prob) {

    (void) prob;
    subtract_kernel<<<launch::grid_for(n_elements), launch::BLOCK_SIZE>>>(
        d_x, d_y, n_elements);
    KERNEL_CHECK();
}

void permute_rows(
    double            *d_out,
    double const      *d_in,
    int const         *d_perm,
    std::size_t const  n,
    std::size_t const  k,
    problem           &prob) {

    (void) prob;
    permute_rows_kernel<false, double>
        <<<launch::grid_for(n * k), launch::BLOCK_SIZE>>>(
            d_out, d_in, d_perm, n, k);
    KERNEL_CHECK();
}

void permute_rows_demote(
    float             *d_out,
    double const      *d_in,
    int const         *d_perm,
    std::size_t const  n,
    std::size_t const  k,
    problem           &prob) {

    (void) prob;
    permute_rows_kernel<true, float>
        <<<launch::grid_for(n * k), launch::BLOCK_SIZE>>>(
            d_out, d_in, d_perm, n, k);
    KERNEL_CHECK();
}

double sum_squares_f32(
    float const       *d_m,
    std::size_t const  n_elements,
    problem           &prob) {

    int const n_blocks = launch::grid_for(n_elements);

    sum_squares_f32_kernel<<<n_blocks, launch::BLOCK_SIZE>>>(
        d_m, prob.d_partial, n_elements);
    KERNEL_CHECK();

    return sum_partials(n_blocks, prob);
}

double sum_squares_difference(
    double const      *d_x,
    double const      *d_y,
    std::size_t const  n_elements,
    problem           &prob) {

    int const n_blocks = launch::grid_for(n_elements);

    sum_squares_difference_kernel<<<n_blocks, launch::BLOCK_SIZE>>>(
        d_x, d_y, prob.d_partial, n_elements);
    KERNEL_CHECK();

    return sum_partials(n_blocks, prob);
}

} /* namespace convert */
