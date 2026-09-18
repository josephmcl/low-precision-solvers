#include "common/metrics.h"
#include <utility>
#include <vector>
#include <cmath>
#include <algorithm>

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


/*  Per-column sum of squares of an n x k column-major fp64 matrix.

    One block per column, block-strided load, shared-memory reduction. k is
    the right-hand-side count (up to 4096 here), so one block per column
    saturates the machine without a second pass, and n is large enough that
    the strided load is coalesced.

    Written rather than looping cublasDnrm2 over columns: that is k kernel
    launches (2048 at the reference shape), and the launch overhead alone
    exceeded the residual GEMM it is measuring. */
__global__ void column_sumsq_kernel(
    double const      *d_m,
    double            *d_out,
    std::size_t const  n) {

    extern __shared__ double s_red[];

    std::size_t const col = blockIdx.x;
    double acc = 0.;
    for (std::size_t i = threadIdx.x; i < n; i += blockDim.x) {
        double const v = d_m[i + col * n];
        acc += v * v;
    }
    s_red[threadIdx.x] = acc;
    __syncthreads();

    for (unsigned s = blockDim.x / 2; s != 0; s >>= 1) {
        if (threadIdx.x < s) s_red[threadIdx.x] += s_red[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) d_out[col] = s_red[0];
}

/*  eta_j = ||r_j|| / (||A||_F ||x_j|| + ||b_j||), returned as (max, median).

    The median is taken on the host over k values -- k <= 4096, so the sort is
    free against the GEMM that produced the residual. */
static std::pair<double,double> rigal_gaches(
    double const *d_r,
    double const *d_x,
    double const *d_b,
    double const  norm_a_f,
    problem      &prob) {

    std::size_t const n = prob.n, k = prob.k;
    unsigned const threads = 256;
    std::size_t const shmem = threads * sizeof(double);

    double *d_sq = static_cast<double *>(prob.acquire(3 * k * sizeof(double)));
    double *d_sr = d_sq, *d_sx = d_sq + k, *d_sb = d_sq + 2 * k;

    column_sumsq_kernel<<<static_cast<unsigned>(k), threads, shmem>>>(d_r, d_sr, n);
    column_sumsq_kernel<<<static_cast<unsigned>(k), threads, shmem>>>(d_x, d_sx, n);
    column_sumsq_kernel<<<static_cast<unsigned>(k), threads, shmem>>>(d_b, d_sb, n);
    KERNEL_CHECK();

    std::vector<double> h(3 * k);
    CUDA_CHECK(cudaMemcpy(h.data(), d_sq, 3 * k * sizeof(double),
                          cudaMemcpyDeviceToHost));

    std::vector<double> eta;
    eta.reserve(k);
    for (std::size_t j = 0; j != k; ++j) {
        double const rj = std::sqrt(h[j]);
        double const xj = std::sqrt(h[k + j]);
        double const bj = std::sqrt(h[2 * k + j]);
        double const den = norm_a_f * xj + bj;
        eta.push_back(den > 0. ? rj / den : 0.);
    }

    double const mx = *std::max_element(eta.begin(), eta.end());
    std::sort(eta.begin(), eta.end());
    double const med = (k % 2) ? eta[k / 2]
                               : 0.5 * (eta[k / 2 - 1] + eta[k / 2]);
    return {mx, med};
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

    /*  Standard per-RHS metric alongside the aggregate. Same residual, so
        the two can never describe different solutions. */
    {
        auto const rg = rigal_gaches(prob.d_residual, d_x, prob.d_b,
                                     out.norm_a, prob);
        out.rg_max = rg.first;
        out.rg_median = rg.second;
    }

    if (d_x_ref != nullptr) {
        double const norm_ref = norm(d_x_ref, nk, prob);
        out.forward = (norm_ref > 0.)?
            norm_difference(d_x, d_x_ref, nk, prob) / norm_ref : 0.;
    }

    return out;
}

} /* namespace metrics */
