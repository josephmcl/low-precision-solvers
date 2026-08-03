/*  Out-of-memory experiment: solving where an A-resident method cannot
    allocate at all.

    THE CLAIM. R-IR needs (LU, R) and never needs A after the build, so its
    matrix-resident footprint is 8n^2 — 4n^2 of fp32 factor plus 4n^2 of fp32
    R. Every method that keeps A in any form needs more: split-MPIR and vendor
    IRS need 12n^2, and direct fp64 needs 8n^2 of A itself, which it must hold
    resident. So there is a band of n where R-IR runs and they run out of
    memory.

    WHY THIS NEEDS ITS OWN PROGRAM. The harness's `problem` owns a resident
    fp64 A by construction — 8n^2 before any method allocates anything — so a
    method inside it can never demonstrate a footprint below 16n^2 no matter
    what it does internally. Showing the claim requires A to not exist as an
    array at all, which is a different problem source, not a different solver.

    WHAT IS DEMONSTRATED. A is defined by a closed form in `a_entry` and is
    materialised only one column block at a time, three times:

      1. to build the fp32 factor,
      2. to form R = PA - LU block by block,
      3. to evaluate the residual b - Ax at the end.

    The peak allocation is reported alongside the backward error, so the claim
    is measured rather than asserted. `--resident` runs the same solve with A
    held in fp64 for comparison; above the exclusion threshold that mode is
    expected to fail its allocation, which IS the result.

    HONEST SCOPE. This demonstrates admissibility, not speed: regenerating A
    costs roughly an order over reading it, and the R build here uses an fp64
    dgemm on the regenerated blocks rather than the Ozaki path, because the
    point being made is about memory. A production version would use the Ozaki
    build against the same streamed blocks.

    Build:  make bin/lps-oom
    Run:    bin/lps-oom 50000 64             # streamed
            bin/lps-oom 50000 64 --resident  # A held in fp64
            bin/lps-oom --sweep 20000,40000,50000,56000 --csv > oom.csv

    An allocation failure is a RESULT, not an error: it is emitted as a CSV row
    with status=oom so the out-of-memory boundary appears in the data rather
    than as a gap in it. The sweep therefore continues past a failure. */

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#define CHECK(x) do { auto e_ = (x); if (e_ != cudaSuccess) { \
    std::printf("[cuda] %s at line %d\n", cudaGetErrorString(e_), __LINE__); \
    std::exit(1); } } while (0)

namespace {

/*  Peak device allocation, tracked by hand.

    cudaMemGetInfo reports what the driver has reserved, which includes the
    context and any caching allocator, so it overstates what the method needs
    and would make the claim look better than it is. Counting the bytes this
    program asks for is the number that belongs next to the storage figure. */
std::size_t g_bytes = 0, g_peak = 0;
bool g_csv = false;
bool g_oom = false;          /*  set instead of exiting, so a sweep continues */
std::vector<void *> g_owned; /*  freed between sweep points                   */

void *dmalloc(std::size_t bytes) {
    void *p = nullptr;
    cudaError_t const e = cudaMalloc(&p, bytes);
    if (e != cudaSuccess) {
        if (!g_csv) {
            std::printf("  ALLOCATION FAILED at %.2f GB (needed %.2f GB more)\n",
                        static_cast<double>(g_bytes) / 1e9,
                        static_cast<double>(bytes) / 1e9);
            std::printf("  -> this size is out of memory for this method\n");
        }
        g_oom = true;
        return nullptr;
    }
    g_bytes += bytes;
    if (g_bytes > g_peak) g_peak = g_bytes;
    g_owned.push_back(p);
    return p;
}

/*  A(i,j), in closed form.

    Any deterministic function of (i,j) works; this one is diagonally dominant
    so the fp32 factorization is well behaved and the demonstration is about
    memory rather than conditioning. It must be cheap — it is evaluated three
    times over the whole matrix. */
__device__ __forceinline__ double a_entry(std::size_t i, std::size_t j,
                                          std::size_t n) {
    unsigned t = static_cast<unsigned>(i * 73856093u ^ j * 19349663u);
    t ^= t >> 16; t *= 0x7feb352du; t ^= t >> 15;
    double const r = (static_cast<double>(t >> 8) / 16777216.) * 2. - 1.;
    return r + ((i == j)? static_cast<double>(n) : 0.);
}

/*  Materialise one column block of A, in fp64 or fp32. */
__global__ void gen_block_f64(double *d_out, std::size_t n, std::size_t j0,
                              std::size_t n_c) {
    std::size_t const total = n * n_c;
    for (std::size_t t = blockIdx.x * blockDim.x + threadIdx.x;
         t < total; t += static_cast<std::size_t>(blockDim.x) * gridDim.x)
        d_out[t] = a_entry(t % n, j0 + t / n, n);
}

__global__ void gen_block_f32(float *d_out, std::size_t n, std::size_t j0,
                              std::size_t n_c) {
    std::size_t const total = n * n_c;
    for (std::size_t t = blockIdx.x * blockDim.x + threadIdx.x;
         t < total; t += static_cast<std::size_t>(blockDim.x) * gridDim.x)
        d_out[t] = static_cast<float>(a_entry(t % n, j0 + t / n, n));
}

/*  R = PA - LU for one column block, written fp32.

    d_acc holds strict_L * U for these columns; subtracting U supplies L's unit
    diagonal. PA is read from the REGENERATED block through the permutation,
    which is the whole point — no permuted copy of A is ever materialised. */
__global__ void form_r_block(float *d_r, double const *d_acc,
                             float const *d_lu, double const *d_a_block,
                             int const *d_perm, std::size_t n,
                             std::size_t j0, std::size_t n_c) {
    std::size_t const total = n * n_c;
    for (std::size_t t = blockIdx.x * blockDim.x + threadIdx.x;
         t < total; t += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
        std::size_t const i = t % n, j = j0 + t / n;
        double const u  = (i <= j)? static_cast<double>(d_lu[i + j * n]) : 0.;
        double const pa = d_a_block[static_cast<std::size_t>(d_perm[i]) +
                                    (t / n) * n];
        d_r[t] = static_cast<float>(pa - d_acc[t] - u);
    }
}

/*  Strict lower L, one block column, promoted to fp64. Unit diagonal is
    implicit: the diagonal entries are written as zero here and U supplies the
    +U term in form_r_block. */
__global__ void promote_l_block(double *d_l, float const *d_lu, std::size_t n,
                                std::size_t c0, std::size_t cb) {
    std::size_t const total = n * cb;
    for (std::size_t t = blockIdx.x * blockDim.x + threadIdx.x;
         t < total; t += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
        std::size_t const i = t % n, c = c0 + t / n;
        d_l[t] = (i > c)? static_cast<double>(d_lu[i + c * n]) : 0.;
    }
}

__global__ void promote_u_block(double *d_u, float const *d_lu, std::size_t n,
                                std::size_t j0, std::size_t n_c) {
    std::size_t const total = n * n_c;
    for (std::size_t t = blockIdx.x * blockDim.x + threadIdx.x;
         t < total; t += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
        std::size_t const i = t % n, j = j0 + t / n;
        d_u[t] = (i <= j)? static_cast<double>(d_lu[i + j * n]) : 0.;
    }
}

__global__ void demote(float *d_o, double const *d_i, std::size_t n_total) {
    for (std::size_t t = blockIdx.x * blockDim.x + threadIdx.x;
         t < n_total; t += static_cast<std::size_t>(blockDim.x) * gridDim.x)
        d_o[t] = static_cast<float>(d_i[t]);
}

__global__ void promote(double *d_o, float const *d_i, std::size_t n_total) {
    for (std::size_t t = blockIdx.x * blockDim.x + threadIdx.x;
         t < n_total; t += static_cast<std::size_t>(blockDim.x) * gridDim.x)
        d_o[t] = static_cast<double>(d_i[t]);
}

__global__ void add_promoted(double *d_acc, float const *d_add,
                             std::size_t n_total) {
    for (std::size_t t = blockIdx.x * blockDim.x + threadIdx.x;
         t < n_total; t += static_cast<std::size_t>(blockDim.x) * gridDim.x)
        d_acc[t] += static_cast<double>(d_add[t]);
}

__global__ void gen_rhs(double *d_b, std::size_t n_total, unsigned seed) {
    for (std::size_t t = blockIdx.x * blockDim.x + threadIdx.x;
         t < n_total; t += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
        unsigned x = seed + static_cast<unsigned>(t) * 2654435761u;
        x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15;
        d_b[t] = (static_cast<double>(x >> 8) / 16777216.) * 2. - 1.;
    }
}

} /* namespace */

struct result {
    std::size_t n = 0, k = 0;
    bool   resident = false;
    bool   oom = false;
    double peak_gb = 0., peak_n2 = 0.;
    double backward = 0., r_norm = 0.;
};

/*  One measurement. Allocation failure returns with oom set rather than
    exiting, so a sweep can record the boundary and carry on past it. */
result run_one(std::size_t const n, std::size_t const k, bool const resident) {

    std::size_t const nb = 1024;                  /* column block            */
    result res; res.n = n; res.k = k; res.resident = resident;
    g_bytes = 0; g_peak = 0; g_oom = false; g_owned.clear();

    std::size_t free_b = 0, total_b = 0;
    CHECK(cudaMemGetInfo(&free_b, &total_b));

    if (!g_csv) {
    std::printf("oom experiment   n=%zu  k=%zu  mode=%s\n", n, k,
                resident? "A RESIDENT (fp64)" : "A STREAMED");
    std::printf("device memory    %.2f GB total, %.2f GB free\n",
                total_b / 1e9, free_b / 1e9);
    std::printf("predicted        R-IR 8n^2 = %.2f GB | 12n^2 = %.2f GB"
                " | +A fp64 = %.2f GB\n",
                8. * n * n / 1e9, 12. * n * n / 1e9, 16. * n * n / 1e9);
    }

    cublasHandle_t blas; cublasCreate(&blas);
    cusolverDnHandle_t sol; cusolverDnCreate(&sol);

    int const ni = static_cast<int>(n), ki = static_cast<int>(k);
    std::size_t const nn = n * n, nk = n * k;

    /*  The 8n^2: the fp32 factor, and R. Nothing else scales with n^2. */
    float *d_lu = static_cast<float *>(dmalloc(nn * sizeof(float)));
    float *d_r  = static_cast<float *>(dmalloc(nn * sizeof(float)));

    /*  A held resident, for the comparison mode. This is the allocation that
        is expected to fail above the threshold. */
    double *d_a_full = resident
        ? static_cast<double *>(dmalloc(nn * sizeof(double))) : nullptr;

    int *d_ipiv = static_cast<int *>(dmalloc(n * sizeof(int)));
    int *d_perm = static_cast<int *>(dmalloc(n * sizeof(int)));
    int *d_info = static_cast<int *>(dmalloc(sizeof(int)));

    double *d_b   = static_cast<double *>(dmalloc(nk * sizeof(double)));
    double *d_x   = static_cast<double *>(dmalloc(nk * sizeof(double)));
    double *d_rhs = static_cast<double *>(dmalloc(nk * sizeof(double)));
    float  *d_y   = static_cast<float *>(dmalloc(nk * sizeof(float)));
    float  *d_yr  = static_cast<float *>(dmalloc(nk * sizeof(float)));

    /*  Block workspace: O(n * nb), not O(n^2). */
    double *d_ablk = static_cast<double *>(dmalloc(n * nb * sizeof(double)));
    double *d_u    = static_cast<double *>(dmalloc(n * nb * sizeof(double)));
    double *d_acc  = static_cast<double *>(dmalloc(n * nb * sizeof(double)));
    double *d_lblk = static_cast<double *>(dmalloc(n * nb * sizeof(double)));

    int lwork = 0;
    cusolverDnSgetrf_bufferSize(sol, ni, ni, d_lu, ni, &lwork);
    float *d_work = static_cast<float *>(
        dmalloc(static_cast<std::size_t>(lwork > 0? lwork : 1) * sizeof(float)));

    if (g_oom) {
        res.oom = true;
        for (void *q : g_owned) cudaFree(q);
        cublasDestroy(blas); cusolverDnDestroy(sol);
        return res;
    }
    if (!g_csv)
        std::printf("allocated        %.2f GB peak (%.1f n^2 bytes)\n\n",
                    g_peak / 1e9, static_cast<double>(g_peak) / (n * n));

    /*  1. Factor. A is generated straight into the fp32 array it overwrites,
        so no fp64 copy of A ever exists in streamed mode. */
    if (resident) {
        for (std::size_t j = 0; j < n; j += nb) {
            std::size_t const n_c = (j + nb <= n)? nb : n - j;
            gen_block_f64<<<256, 256>>>(d_a_full + j * n, n, j, n_c);
        }
        demote<<<256, 256>>>(d_lu, d_a_full, nn);
    }
    else {
        for (std::size_t j = 0; j < n; j += nb) {
            std::size_t const n_c = (j + nb <= n)? nb : n - j;
            gen_block_f32<<<256, 256>>>(d_lu + j * n, n, j, n_c);
        }
    }
    CHECK(cudaDeviceSynchronize());

    cusolverDnSgetrf(sol, ni, ni, d_lu, ni, d_work, d_ipiv, d_info);
    CHECK(cudaDeviceSynchronize());

    /*  Compose getrf's interchanges into a permutation. */
    {
        std::vector<int> ipiv(n), perm(n);
        CHECK(cudaMemcpy(ipiv.data(), d_ipiv, n * sizeof(int),
                         cudaMemcpyDeviceToHost));
        for (std::size_t i = 0; i != n; ++i) perm[i] = static_cast<int>(i);
        for (std::size_t i = 0; i != n; ++i) {
            int const p = ipiv[i] - 1;
            if (p >= 0 && static_cast<std::size_t>(p) != i)
                std::swap(perm[i], perm[p]);
        }
        CHECK(cudaMemcpy(d_perm, perm.data(), n * sizeof(int),
                         cudaMemcpyHostToDevice));
    }

    /*  2. R = PA - LU, one column block at a time, A regenerated per block. */
    double const one = 1., zero = 0., minus_one = -1.;
    for (std::size_t j = 0; j < n; j += nb) {
        std::size_t const n_c = (j + nb <= n)? nb : n - j;

        if (resident)
            CHECK(cudaMemcpy(d_ablk, d_a_full + j * n,
                             n * n_c * sizeof(double),
                             cudaMemcpyDeviceToDevice));
        else
            gen_block_f64<<<256, 256>>>(d_ablk, n, j, n_c);

        promote_u_block<<<256, 256>>>(d_u, d_lu, n, j, n_c);
        CHECK(cudaMemset(d_acc, 0, n * n_c * sizeof(double)));

        /*  strict_L * U for these columns, accumulated over L's block
            columns so neither operand is ever promoted whole. Promoting all of
            L would be O(n^2) of fp64 and would reintroduce exactly the
            footprint this program exists to avoid. */
        {
            std::size_t const c_lim = (j + n_c < n)? j + n_c : n;
            for (std::size_t c0 = 0; c0 < c_lim; c0 += nb) {
                std::size_t const cb = (c0 + nb <= c_lim)? nb : c_lim - c0;
                promote_l_block<<<256, 256>>>(d_lblk, d_lu, n, c0, cb);
                cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N,
                            ni, static_cast<int>(n_c), static_cast<int>(cb),
                            &one, d_lblk, ni,
                            d_u + c0, ni, &one, d_acc, ni);
            }
        }

        form_r_block<<<256, 256>>>(d_r + j * n, d_acc, d_lu, d_ablk,
                                   d_perm, n, j, n_c);
    }
    CHECK(cudaDeviceSynchronize());

    /*  Diagnostic: ||R||/||A|| should land near u_32 ~ 6e-8 for a
        well-conditioned A. If it is far larger the build is wrong; if it is
        near zero R is not being written at all. */
    {
        /*  Blocked: n*n exceeds INT_MAX past n ~ 46341, and cublasSnrm2
            takes an int. Unblocked this returned 0 at exactly the sizes this
            program exists to measure — silently, since the overflow is in the
            argument rather than the result. */
        double rr = 0.;
        for (std::size_t j = 0; j < n; j += nb) {
            std::size_t const n_c = (j + nb <= n)? nb : n - j;
            float bn = 0.f;
            cublasSnrm2(blas, static_cast<int>(n * n_c), d_r + j * n, 1, &bn);
            rr += static_cast<double>(bn) * static_cast<double>(bn);
        }
        float const rn32 = static_cast<float>(std::sqrt(rr));
        double an = 0.;
        for (std::size_t j = 0; j < n; j += nb) {
            std::size_t const n_c = (j + nb <= n)? nb : n - j;
            gen_block_f64<<<256, 256>>>(d_ablk, n, j, n_c);
            double bn = 0.;
            cublasDnrm2(blas, static_cast<int>(n * n_c), d_ablk, 1, &bn);
            an += bn * bn;
        }
        res.r_norm = static_cast<double>(rn32) / std::sqrt(an);
        if (!g_csv)
            std::printf("  ||R||/||A||       %.3e   (u_32 = 5.96e-08)\n",
                        res.r_norm);
    }

    /*  3. Solve, and evaluate the residual against a regenerated A. */
    gen_rhs<<<256, 256>>>(d_b, nk, 12345u);
    CHECK(cudaMemcpy(d_rhs, d_b, nk * sizeof(double), cudaMemcpyDeviceToDevice));

    /*  The R-IR fixed point: LU X_{m+1} = PB - R X_m.

        Without this the program builds R and never uses it, and the answer is
        just the fp32 triangular solve — which is what the first run reported
        (3.1e-09). R is the whole point: it is what lets a solve that never
        holds A still reach fp64-quality accuracy. */
    float const f_one = 1.f, f_zero = 0.f, f_minus = -1.f;
    CHECK(cudaMemset(d_x, 0, nk * sizeof(double)));

    int const n_iter = (std::getenv("ITERS") != nullptr)?
        std::atoi(std::getenv("ITERS")) : 3;
    for (int it = 0; it != n_iter; ++it) {

        CHECK(cudaMemcpy(d_rhs, d_b, nk * sizeof(double),
                         cudaMemcpyDeviceToDevice));
        if (it != 0) {
            /*  rhs <- PB - R X, in fp32: R is small, so this product never
                cancels and needs no compensation. */
            demote<<<256, 256>>>(d_y, d_x, nk);
            cublasSgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N, ni, ki, ni,
                        &f_minus, d_r, ni, d_y, ni, &f_zero, d_yr, ni);
            add_promoted<<<256, 256>>>(d_rhs, d_yr, nk);
        }

        demote<<<256, 256>>>(d_y, d_rhs, nk);
        cublasStrsm(blas, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                    CUBLAS_DIAG_UNIT, ni, ki, &f_one, d_lu, ni, d_y, ni);
        cublasStrsm(blas, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
                    CUBLAS_DIAG_NON_UNIT, ni, ki, &f_one, d_lu, ni, d_y, ni);
        promote<<<256, 256>>>(d_x, d_y, nk);
    }
    /*  Classical refinement on the STREAMED residual.

        The fixed point above floors at the fp32 triangular solve's accuracy,
        because it yields the iterate rather than a correction — production
        R-IR fixes that with an mp-TRSM refinement against L and U. A streamed
        method has a better option: A is regenerable, so b - Ax can be formed
        EXACTLY in fp64 one block at a time, and the classical correction form
        applies with no cancellation problem. That is a capability the resident
        method does not have and does not need, and it costs no extra n^2
        storage — only the O(n*nb) block buffer already allocated. */
    for (int ref = 0; ref != 2; ++ref) {

        CHECK(cudaMemcpy(d_rhs, d_b, nk * sizeof(double),
                         cudaMemcpyDeviceToDevice));
        for (std::size_t j = 0; j < n; j += nb) {
            std::size_t const n_c = (j + nb <= n)? nb : n - j;
            gen_block_f64<<<256, 256>>>(d_ablk, n, j, n_c);
            cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N, ni, ki,
                        static_cast<int>(n_c), &minus_one,
                        d_ablk, ni, d_x + j, ni, &one, d_rhs, ni);
        }

        demote<<<256, 256>>>(d_y, d_rhs, nk);
        cublasStrsm(blas, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N,
                    CUBLAS_DIAG_UNIT, ni, ki, &f_one, d_lu, ni, d_y, ni);
        cublasStrsm(blas, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
                    CUBLAS_DIAG_NON_UNIT, ni, ki, &f_one, d_lu, ni, d_y, ni);
        add_promoted<<<256, 256>>>(d_x, d_y, nk);
    }
    CHECK(cudaDeviceSynchronize());

    /*  Residual b - Ax, with A regenerated block by block — never resident. */
    CHECK(cudaMemcpy(d_rhs, d_b, nk * sizeof(double), cudaMemcpyDeviceToDevice));
    double a_norm_sq = 0.;
    for (std::size_t j = 0; j < n; j += nb) {
        std::size_t const n_c = (j + nb <= n)? nb : n - j;
        gen_block_f64<<<256, 256>>>(d_ablk, n, j, n_c);
        cublasDgemm(blas, CUBLAS_OP_N, CUBLAS_OP_N, ni, ki,
                    static_cast<int>(n_c), &minus_one,
                    d_ablk, ni, d_x + j, ni, &one, d_rhs, ni);
        double bn = 0.;
        cublasDnrm2(blas, static_cast<int>(n * n_c), d_ablk, 1, &bn);
        a_norm_sq += bn * bn;
    }
    double r_nrm = 0., x_nrm = 0.;
    cublasDnrm2(blas, ni * ki, d_rhs, 1, &r_nrm);
    cublasDnrm2(blas, ni * ki, d_x, 1, &x_nrm);

    res.peak_gb  = g_peak / 1e9;
    res.peak_n2  = static_cast<double>(g_peak) / (n * n);
    res.backward = r_nrm / (std::sqrt(a_norm_sq) * x_nrm);

    if (!g_csv) {
        std::printf("RESULT\n");
        std::printf("  peak allocation   %.2f GB   (%.2f n^2 bytes)\n",
                    res.peak_gb, res.peak_n2);
        std::printf("  A resident        %s\n",
                    resident? "YES (8n^2 of A)" : "NO");
        std::printf("  backward error    %.3e\n", res.backward);
    }

    for (void *q : g_owned) cudaFree(q);
    cublasDestroy(blas); cusolverDnDestroy(sol);
    return res;
}

int main(int argc, char **argv) {

    std::cout.setf(std::ios::unitbuf);

    std::vector<std::size_t> ns;
    std::size_t k = 64;
    bool resident_only = false, both = false;

    for (int i = 1; i < argc; ++i) {
        std::string const a = argv[i];
        if (a == "--csv")           g_csv = true;
        else if (a == "--resident") resident_only = true;
        else if (a == "--sweep") {
            both = true;
            std::stringstream ss(argv[++i]); std::string t;
            while (std::getline(ss, t, ',')) ns.push_back(std::atoll(t.c_str()));
        }
        else if (a[0] != '-') {
            if (ns.empty()) ns.push_back(std::atoll(a.c_str()));
            else            k = std::atoll(a.c_str());
        }
    }
    if (ns.empty()) ns.push_back(40000);

    if (g_csv)
        std::cout << "n,k,mode,status,peak_gb,peak_n2,backward,r_norm\n";

    for (std::size_t const n : ns) {
        std::vector<bool> modes;
        if (both)               modes = {false, true};
        else if (resident_only) modes = {true};
        else                    modes = {false};

        for (bool const rz : modes) {
            result const r = run_one(n, k, rz);
            if (g_csv)
                std::cout << r.n << ',' << r.k << ','
                          << (r.resident? "resident" : "streamed") << ','
                          << (r.oom? "oom" : "ok") << ','
                          << r.peak_gb << ',' << r.peak_n2 << ','
                          << r.backward << ',' << r.r_norm << '\n';
        }
    }
    return 0;
}
