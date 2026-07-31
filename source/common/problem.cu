#include "common/problem.h"

namespace harness {

/*  cudaDeviceProp.name, for the tuning lookup. Read here rather than passed
    in so no caller has to remember to. */
static std::string _device_name() {

    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess)
        return "unknown";
    return std::string(prop.name);
}

char const *name_of(matrix_kind const kind) {

    switch (kind) {
        case matrix_kind::diag_dominant: return "diag-dominant (+n)";
        case matrix_kind::moderate:      return "moderate (+sqrt n)";
        case matrix_kind::near_random:   return "near-random (+1)";
        case matrix_kind::graded_rows:   return "graded rows (6 dec)";
        case matrix_kind::graded_diagonal: return "graded diagonal";
        case matrix_kind::spd_graded:    return "SPD log-spectrum";
        case matrix_kind::wilkinson:     return "Wilkinson growth";
    }
    return "unknown";
}

/*  A bit-mixing hash, not an LCG.

    One thread owns one element and derives its value from (index, seed)
    alone, so the matrix is reproducible regardless of launch geometry. The
    mixer matters: a single-LCG stream laid the entries on a lattice and
    produced a "random" control matrix with condition number 1.3e20, which
    read as an accuracy failure in the methods rather than in the generator. */
__device__ __forceinline__
double uniform_pm1(unsigned const index, unsigned const seed) {

    unsigned t = seed + index * 2654435761u;
    t ^= t >> 16; t *= 0x7feb352du;
    t ^= t >> 15; t *= 0x846ca68bu;
    t ^= t >> 16;

    /*  24 bits into [-1, 1); the low bits of a mixer are the weakest, so
        the top of the word is used. */
    return (static_cast<double>(t >> 8) / 16777216. - 0.5) * 2.;
}

/*  A(i,j) = uniform(-1,1), then the diagonal shift that sets the class.
    Column major, so element index = i + j*n and one thread owns one index. */
__global__ void generate_matrix_kernel(
    double            *d_a,
    std::size_t const  n,
    int const          kind,
    double const       shift,
    unsigned const     seed) {

    std::size_t const n_total = n * n;

    for (std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < n_total;
         idx += static_cast<std::size_t>(blockDim.x) * gridDim.x) {

        std::size_t const i = idx % n;
        std::size_t const j = idx / n;

        double v = uniform_pm1(static_cast<unsigned>(idx), seed);

        /*  Graded rows: row i is scaled by 10^(6 i/n - 3), spreading the
            entries over six decades. Counter-intuitively this is the *easy*
            case for aligned splitting, because grading makes each row's
            entries share a scale. */
        if (kind == static_cast<int>(matrix_kind::graded_rows)) {
            double const e = 6. * static_cast<double>(i) /
                             static_cast<double>(n) - 3.;
            v *= pow(10., e);
        }

        /*  Graded diagonal: the off-diagonal is deliberately tiny so the
            matrix stays diagonally dominant at every scale, keeping growth
            near 1 while the diagonal spread sets kappa. */
        if (kind == static_cast<int>(matrix_kind::spd_graded)) {

            /*  H L H with w = ones/sqrt(n) expands to
                  A_ij = l_i d_ij - 2 l_j/n - 2 l_i/n + 4c/n,   c = mean(l).
                c is the geometric-series mean, in closed form so no reduction
                is needed. */
            double const dec = shift;
            double const li = pow(10., -dec * static_cast<double>(i) /
                                        static_cast<double>(n));
            double const lj = pow(10., -dec * static_cast<double>(j) /
                                        static_cast<double>(n));
            double const r  = pow(10., -dec / static_cast<double>(n));
            double const c  = (fabs(1. - r) > 1e-300)?
                (1. - pow(10., -dec)) / (1. - r) / static_cast<double>(n) : 1.;

            double a = -2. * lj / static_cast<double>(n)
                       -2. * li / static_cast<double>(n)
                       + 4. * c / static_cast<double>(n);
            if (i == j) a += li;
            d_a[idx] = a;
        }
        else if (kind == static_cast<int>(matrix_kind::wilkinson)) {

            std::size_t const m = static_cast<std::size_t>(shift);
            double a;
            if (i < m && j < m) {
                if (i == j)            a =  1.;
                else if (j == m - 1)   a =  1.;
                else if (j < i)        a = -1.;
                else                   a =  0.;
            }
            else {
                a = (i == j)? 1. : v * 1.e-8;
            }
            d_a[idx] = a;
        }
        else if (kind == static_cast<int>(matrix_kind::graded_diagonal)) {
            v *= 1.e-6;
            if (i == j)
                v += pow(10., shift * static_cast<double>(i) /
                              static_cast<double>(n));
            d_a[idx] = v;

        }
        else {
            if (i == j)
                v += shift;
            d_a[idx] = v;
        }
    }
}

__global__ void generate_rhs_kernel(
    double            *d_b,
    std::size_t const  n_total,
    unsigned const     seed) {

    for (std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < n_total;
         idx += static_cast<std::size_t>(blockDim.x) * gridDim.x)
        d_b[idx] = uniform_pm1(static_cast<unsigned>(idx), seed);
}

problem::problem(
    std::size_t const  n,
    std::size_t const  k,
    matrix_kind const  kind,
    unsigned const     seed,
    double const       shift_override)
    : n(n), k(k), kind(kind) {

    CUBLAS_CHECK(cublasCreate(&blas));
    CUSOLVER_CHECK(cusolverDnCreate(&solver));

    tuning::current().load(_device_name());

    d_a = static_cast<double *>(acquire(n * n * sizeof(double)));
    d_b = static_cast<double *>(acquire(n * k * sizeof(double)));

    d_partial = static_cast<double *>(
        acquire(static_cast<std::size_t>(launch::MAX_BLOCKS) * sizeof(double)));
    d_residual = static_cast<double *>(acquire(n * k * sizeof(double)));

    _generate(seed, shift_override);
    _warm_libraries();
}

problem::~problem() {

    for (std::size_t i = 0; i != _d_owned.size(); ++i)
        CUDA_CHECK(cudaFree(_d_owned[i]));

    if (blas != nullptr)
        CUBLAS_CHECK(cublasDestroy(blas));
    if (solver != nullptr)
        CUSOLVER_CHECK(cusolverDnDestroy(solver));
}

void *problem::acquire(std::size_t const bytes) {

    void *d_p = nullptr;
    if (!CUDA_CHECK(cudaMalloc(&d_p, bytes)))
        return nullptr;

    _d_owned.push_back(d_p);
    return d_p;
}

void problem::_generate(unsigned const seed, double const shift_override) {

    /*  The diagonal shift is what sets the conditioning, and with it whether
        the fp32 factorization leaves a small R. */
    double shift = 0.;
    switch (kind) {
        case matrix_kind::diag_dominant: shift = static_cast<double>(n); break;
        case matrix_kind::moderate:
            shift = std::sqrt(static_cast<double>(n));
            break;
        case matrix_kind::near_random:   shift = 1.; break;
        case matrix_kind::graded_rows:   shift = static_cast<double>(n); break;
        case matrix_kind::graded_diagonal: shift = 6.; break;
        case matrix_kind::spd_graded:    shift = 8.; break;
        case matrix_kind::wilkinson:     shift = 24.; break;
    }

    if (shift_override >= 0.)
        shift = shift_override;

    generate_matrix_kernel<<<launch::grid_for(n * n), launch::BLOCK_SIZE>>>(
        d_a,
        n,
        static_cast<int>(kind),
        shift,
        seed);
    KERNEL_CHECK();

    generate_rhs_kernel<<<launch::grid_for(n * k), launch::BLOCK_SIZE>>>(
        d_b,
        n * k,
        seed + 99u);
    KERNEL_CHECK();

    CUDA_CHECK(cudaDeviceSynchronize());
}

void problem::_warm_libraries() {

    warm_libraries(*this);
}

void warm_libraries(problem &prob) {

    /*  A cold cuSOLVER call costs on the order of 100 ms of lazy
        initialization. Charging that to whichever phase happens to run first
        produced three separate false findings in the predecessor of this
        code, one of which survived five rounds of "optimization" aimed at an
        artifact. So the libraries are exercised once, on a small problem,
        before any timed region exists.

        Small and separate on purpose: a warmup on the real problem would
        also populate caches, which would flatter the first method measured. */
    std::size_t const m = 64;

    double *d_small = static_cast<double *>(prob.acquire(m * m * sizeof(double)));
    int    *d_ipiv  = static_cast<int *>(prob.acquire(m * sizeof(int)));
    int    *d_info  = static_cast<int *>(prob.acquire(sizeof(int)));

    generate_matrix_kernel<<<launch::grid_for(m * m), launch::BLOCK_SIZE>>>(
        d_small,
        m,
        static_cast<int>(matrix_kind::diag_dominant),
        static_cast<double>(m),
        1u);
    KERNEL_CHECK();

    int lwork = 0;
    CUSOLVER_CHECK(cusolverDnDgetrf_bufferSize(
        prob.solver,
        static_cast<int>(m),
        static_cast<int>(m),
        d_small,
        static_cast<int>(m),
        &lwork));

    double *d_work = static_cast<double *>(
        prob.acquire(static_cast<std::size_t>(lwork) * sizeof(double)));

    CUSOLVER_CHECK(cusolverDnDgetrf(
        prob.solver,
        static_cast<int>(m),
        static_cast<int>(m),
        d_small,
        static_cast<int>(m),
        d_work,
        d_ipiv,
        d_info));

    /*  cuBLAS initializes on its own first call, so it gets one too. */
    double const one = 1., zero = 0.;
    CUBLAS_CHECK(cublasDgemm(
        prob.blas,
        CUBLAS_OP_N, CUBLAS_OP_N,
        static_cast<int>(m), static_cast<int>(m), static_cast<int>(m),
        &one,
        d_small, static_cast<int>(m),
        d_small, static_cast<int>(m),
        &zero,
        d_small, static_cast<int>(m)));

    /*  IRSXgesv initializes separately from the rest of cuSOLVER, and warming
        only Dgetrf left it cold. The first run of this harness measured a
        66% spread on IRS at k=64 against 0.4% on the direct solve, with the
        median still landing where an independent harness had it — the
        signature of one slow first call rather than genuine variance. Every
        vendor entry point a method uses needs its own warmup; a family
        warmup is not enough. */
    {
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
        CUSOLVER_CHECK(cusolverDnIRSParamsSetMaxIters(params, 2));

        size_t lwork = 0;
        CUSOLVER_CHECK(cusolverDnIRSXgesv_bufferSize(
            prob.solver, params, static_cast<int>(m), 1, &lwork));

        double *d_rhs = static_cast<double *>(prob.acquire(m * sizeof(double)));
        double *d_out = static_cast<double *>(prob.acquire(m * sizeof(double)));
        void   *d_ws  = prob.acquire((lwork > 0)? lwork : 1);

        CUDA_CHECK(cudaMemset(d_rhs, 0, m * sizeof(double)));

        int n_iterations = 0;
        cusolverDnIRSXgesv(
            prob.solver, params, infos,
            static_cast<int>(m), 1,
            d_small, static_cast<int>(m),
            d_rhs,   static_cast<int>(m),
            d_out,   static_cast<int>(m),
            d_ws, lwork, &n_iterations, d_info);

        /*  Status deliberately unchecked: a zero right-hand side may report
            non-convergence, which is meaningless here. The call is made for
            its initialization side effect, not its answer. */

        CUSOLVER_CHECK(cusolverDnIRSParamsDestroy(params));
        CUSOLVER_CHECK(cusolverDnIRSInfosDestroy(infos));
    }

    CUDA_CHECK(cudaDeviceSynchronize());
}

} /* namespace harness */
