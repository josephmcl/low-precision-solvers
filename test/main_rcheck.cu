#include "common/solver.h"
#include "common/metrics.h"

#include <cstdlib>
#include <string>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

/*  Validate R itself, independently of any solve.

    R = PA - LU is the one quantity R-IR stores and reuses, and its error is
    undamped — nothing downstream corrects it. If R-IR's accuracy floor on
    ill-conditioned input is a property of the method, then R must be *correct*
    and merely large. If instead the build has a defect, R will be wrong.

    Two numbers separate those cases:

      ||PA - LU - R|| / ||R||   how accurately R was formed. Should sit at
                                R's fp32 storage limit, ~6e-8, regardless of
                                conditioning. Anything much larger is a build
                                defect.

      ||R|| / ||A||             how big R is. The claimed mechanism is that
                                poor conditioning inflates this through pivot
                                growth, so R's fixed 24 bits cover
                                proportionally less of A. If this does not
                                grow as the shift falls, the mechanism is
                                wrong and something else explains the floor.

    The reference is formed in fp64 from the same packed factor R was built
    from, so this compares R against the product it is supposed to be — not
    against a differently-rounded A. */

namespace {

using harness::problem;

/*  Promote a triangle of the packed factor into fp64. which: 1 = strictly
    lower, 2 = upper including the diagonal. */
__global__ void promote_triangle_kernel(
    double            *d_out,
    float const       *d_lu,
    std::size_t const  n,
    int const          which) {

    std::size_t const n_total = n * n;

    for (std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < n_total;
         idx += static_cast<std::size_t>(blockDim.x) * gridDim.x) {

        std::size_t const i = idx % n;
        std::size_t const j = idx / n;

        double v = static_cast<double>(d_lu[idx]);
        if (which == 1 && j >= i) v = 0.;
        if (which == 2 && j <  i) v = 0.;
        d_out[idx] = v;
    }
}

/*  ref = PA - (strict_L * U + U), i.e. PA - LU with L's unit diagonal. */
__global__ void form_reference_kernel(
    double            *d_ref,
    double const      *d_slu,
    double const      *d_u,
    double const      *d_a,
    int const         *d_perm,
    std::size_t const  n) {

    std::size_t const n_total = n * n;

    for (std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < n_total;
         idx += static_cast<std::size_t>(blockDim.x) * gridDim.x) {

        std::size_t const i = idx % n;
        std::size_t const j = idx / n;
        double const pa = d_a[static_cast<std::size_t>(d_perm[i]) + j * n];

        d_ref[idx] = pa - d_slu[idx] - d_u[idx];
    }
}

__global__ void promote_r_kernel(
    double            *d_out,
    float const       *d_r,
    std::size_t const  n_total) {

    for (std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < n_total;
         idx += static_cast<std::size_t>(blockDim.x) * gridDim.x)
        d_out[idx] = static_cast<double>(d_r[idx]);
}

/*  Per-row dynamic range of a triangle, and how much of it the split can
    actually reach.

    The Ozaki split resolves bits below each row's MAXIMUM, so an entry smaller
    than rowmax * 2^-(pieces*bits) contributes nothing to any piece — it is not
    approximated, it is dropped. If pivot growth widens U's rows beyond that
    window, adding pieces cannot recover the lost entries unless the extra
    pieces extend the window downward, which they do; what more pieces cannot
    fix is a row whose range exceeds what the FORMAT can represent below its
    max. This measures which regime we are in.

    One block per row. Reports, per row: log2(max/min) over nonzero entries,
    and the count of nonzero entries below the reachable floor. */
__global__ void row_range_kernel(
    double            *d_range,
    double            *d_lost,
    float const       *d_lu,
    std::size_t const  n,
    int const          which,
    int const          covered_bits) {

    __shared__ float s_max[launch::BLOCK_SIZE];
    __shared__ float s_min[launch::BLOCK_SIZE];
    __shared__ double s_cnt[launch::BLOCK_SIZE];

    std::size_t const i = blockIdx.x;

    float hi = 0.f, lo = 3.4e38f;
    for (std::size_t j = threadIdx.x; j < n; j += blockDim.x) {
        if (which == 1 && j >= i) continue;
        if (which == 2 && j <  i) continue;
        float const v = fabsf(d_lu[i + j * n]);
        if (v > 0.f) { hi = fmaxf(hi, v); lo = fminf(lo, v); }
    }
    s_max[threadIdx.x] = hi;
    s_min[threadIdx.x] = lo;
    __syncthreads();
    for (int q = launch::BLOCK_SIZE / 2; q > 0; q >>= 1) {
        if (static_cast<int>(threadIdx.x) < q) {
            s_max[threadIdx.x] = fmaxf(s_max[threadIdx.x], s_max[threadIdx.x + q]);
            s_min[threadIdx.x] = fminf(s_min[threadIdx.x], s_min[threadIdx.x + q]);
        }
        __syncthreads();
    }

    float const row_max = s_max[0];
    float const floor_v = row_max * exp2f(-static_cast<float>(covered_bits));

    double cnt = 0.;
    for (std::size_t j = threadIdx.x; j < n; j += blockDim.x) {
        if (which == 1 && j >= i) continue;
        if (which == 2 && j <  i) continue;
        float const v = fabsf(d_lu[i + j * n]);
        if (v > 0.f && v < floor_v) cnt += 1.;
    }
    s_cnt[threadIdx.x] = cnt;
    __syncthreads();
    for (int q = launch::BLOCK_SIZE / 2; q > 0; q >>= 1) {
        if (static_cast<int>(threadIdx.x) < q)
            s_cnt[threadIdx.x] += s_cnt[threadIdx.x + q];
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        d_range[i] = (s_min[0] > 0.f && row_max > 0.f)?
            log2(static_cast<double>(row_max / s_min[0])) : 0.;
        d_lost[i] = s_cnt[0];
    }
}

} /* namespace */

int main(int argc, char **argv) {

    std::cout.setf(std::ios::unitbuf);

    std::size_t const n = (argc > 1)?
        static_cast<std::size_t>(std::atoll(argv[1])) : 4096;

    std::cout << "R validation   n=" << n << "\n"
              << "R is built, then compared against PA - LU formed in fp64"
                 " from the same factor\n\n";

    std::cout << "  " << std::right
              << std::setw(9)  << "shift"
              << std::setw(14) << "||R||/||A||"
              << std::setw(16) << "R build error"
              << std::setw(12) << "bits left"
              << std::setw(12) << "growth"
              << std::setw(11) << "U rng med"
              << std::setw(11) << "U rng max"
              << std::setw(11) << "frac lost"
              << std::setw(12) << "R row sprd"
              << std::setw(12) << "Rhi vs ref"
              << std::setw(12) << "Rhi vs Rlo" << "\n";

    std::string const fam = (argc > 2)? argv[2] : "shift";
    bool const graded = fam == "graded";
    bool const spd    = fam == "spd";
    bool const wilk   = fam == "wilkinson";
    harness::matrix_kind const kind =
        graded? harness::matrix_kind::graded_diagonal
      : spd?    harness::matrix_kind::spd_graded
      : wilk?   harness::matrix_kind::wilkinson
      :         harness::matrix_kind::near_random;

    std::vector<double> shifts;
    if (graded || spd)
        for (double d = 0.; d <= 12.; d += 2.) shifts.push_back(d);
    else if (wilk)
        for (double m = 4.; m <= 40.; m += 8.) shifts.push_back(m);
    else {
        for (double s = static_cast<double>(n); s >= 0.5; s /= 4.)
            shifts.push_back(s);
        shifts.push_back(0.125);
    }

    solver::method const *m = nullptr;
    std::vector<solver::method> const &all = solver::registry();
    for (std::size_t i = 0; i != all.size(); ++i)
        if (std::string(all[i].name) == "R-IR")
            m = &all[i];

    for (std::size_t si = 0; si != shifts.size(); ++si) {

        problem prob(n, 1, kind, 7u, shifts[si]);

        /*  Build R TWICE with different Ozaki configurations and compare
            them against EACH OTHER as well as against the fp64 reference.

            Comparing R only against the reference cannot say which side is
            wrong — both compute PA - LU, a cancelling difference. But two R's
            built at very different precisions are independent of the
            reference: if they AGREE with each other and both disagree with the
            reference, the reference is the inaccurate side. If they disagree
            with each other, at least one R is genuinely wrong. */
        /*  Both configurations are settable so pairs can be swept without a
            rebuild. Defaults are two WITHIN-BOUND settings that differ in how
            they split — if two independent correct paths agree with each other
            but both differ from the fp64 reference, the reference is the
            inaccurate side. */
        char const *ab = std::getenv("LPS_RCHECK_A_BITS");
        char const *ap = std::getenv("LPS_RCHECK_A_PIECES");
        char const *ak = std::getenv("LPS_RCHECK_A_BLOCK");
        setenv("LPS_RIR_BUILD_OZAKI_BITS",   ab? ab : "6",  1);
        setenv("LPS_RIR_BUILD_OZAKI_PIECES", ap? ap : "9",  1);
        setenv("LPS_RIR_BUILD_OZAKI_BLOCK",  ak? ak : "16", 1);
        solver::state st;
        m->factor(st, prob);

        std::size_t const nn_r = n * n;
        float *d_r_a = static_cast<float *>(prob.acquire(nn_r * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(d_r_a, st.d_r, nn_r * sizeof(float),
                              cudaMemcpyDeviceToDevice));

        char const *bb = std::getenv("LPS_RCHECK_B_BITS");
        char const *bp = std::getenv("LPS_RCHECK_B_PIECES");
        char const *bk = std::getenv("LPS_RCHECK_B_BLOCK");
        setenv("LPS_RIR_BUILD_OZAKI_BITS",   bb? bb : "8",  1);
        setenv("LPS_RIR_BUILD_OZAKI_PIECES", bp? bp : "7",  1);
        setenv("LPS_RIR_BUILD_OZAKI_BLOCK",  bk? bk : "16", 1);
        solver::state st2;
        m->factor(st2, prob);
        unsetenv("LPS_RIR_BUILD_OZAKI_BITS");
        unsetenv("LPS_RIR_BUILD_OZAKI_PIECES");
        unsetenv("LPS_RIR_BUILD_OZAKI_BLOCK");

        std::size_t const nn = n * n;
        double *d_slu = static_cast<double *>(prob.acquire(nn * sizeof(double)));
        double *d_u   = static_cast<double *>(prob.acquire(nn * sizeof(double)));
        double *d_ref = static_cast<double *>(prob.acquire(nn * sizeof(double)));
        double *d_rd  = static_cast<double *>(prob.acquire(nn * sizeof(double)));

        promote_triangle_kernel<<<launch::grid_for(nn), launch::BLOCK_SIZE>>>(
            d_slu, st.d_lu, n, 1);
        promote_triangle_kernel<<<launch::grid_for(nn), launch::BLOCK_SIZE>>>(
            d_u, st.d_lu, n, 2);
        KERNEL_CHECK();

        /*  strict_L * U in fp64, overwriting d_slu with the product. */
        double const one = 1., zero = 0.;
        double *d_prod = static_cast<double *>(
            prob.acquire(nn * sizeof(double)));
        CUBLAS_CHECK(cublasDgemm(
            prob.blas, CUBLAS_OP_N, CUBLAS_OP_N,
            static_cast<int>(n), static_cast<int>(n), static_cast<int>(n),
            &one, d_slu, static_cast<int>(n), d_u, static_cast<int>(n),
            &zero, d_prod, static_cast<int>(n)));

        form_reference_kernel<<<launch::grid_for(nn), launch::BLOCK_SIZE>>>(
            d_ref, d_prod, d_u, prob.d_a, st.d_perm, n);
        KERNEL_CHECK();

        promote_r_kernel<<<launch::grid_for(nn), launch::BLOCK_SIZE>>>(
            d_rd, static_cast<float const *>(st.d_r), nn);
        KERNEL_CHECK();
        CUDA_CHECK(cudaDeviceSynchronize());

        /*  Growth: ||L|| ||U|| / ||A||. Forming R = PA - LU is a cancelling
            difference, and the cancellation is exactly this factor — both R
            and any reference for it lose log2(growth) bits. If growth is large
            the discrepancy between two independent computations of R says
            nothing about either one's correctness. */
        double const norm_l = metrics::norm(d_slu, nn, prob);
        double const norm_u = metrics::norm(d_u, nn, prob);

        /*  R_a (high precision, within bound) vs R_b (as shipped). */
        double *d_ra = static_cast<double *>(prob.acquire(nn * sizeof(double)));
        double *d_rb = static_cast<double *>(prob.acquire(nn * sizeof(double)));
        promote_r_kernel<<<launch::grid_for(nn), launch::BLOCK_SIZE>>>(
            d_ra, d_r_a, nn);
        promote_r_kernel<<<launch::grid_for(nn), launch::BLOCK_SIZE>>>(
            d_rb, static_cast<float const *>(st2.d_r), nn);
        KERNEL_CHECK();
        CUDA_CHECK(cudaDeviceSynchronize());

        double const norm_ref = metrics::norm(d_ref, nn, prob);
        double const norm_a   = metrics::norm(prob.d_a, nn, prob);
        double const err      = metrics::norm_difference(d_rd, d_ref, nn, prob);

        double const err_ab = metrics::norm_difference(d_ra, d_rb, nn, prob);
        double const err_a  = metrics::norm_difference(d_ra, d_ref, nn, prob);
        double const relative = (norm_ref > 0.)? err / norm_ref : 0.;
        double const rel_ab   = (norm_ref > 0.)? err_ab / norm_ref : 0.;
        double const rel_a    = (norm_ref > 0.)? err_a / norm_ref : 0.;
        double const size     = (norm_a > 0.)? norm_ref / norm_a : 0.;

        /*  Bits of A that R still resolves: R carries 24 bits of a quantity
            that is ||R||/||A|| of A, so the pair covers -log2(||R||/||A||) + 24
            bits. This is the storage identity, read off the measurement. */
        double const bits = (size > 0.)? -std::log2(size) + 24. : 0.;

        /*  Dynamic range of U's rows against the split's reach. */
        double *d_range = static_cast<double *>(prob.acquire(n * sizeof(double)));
        double *d_lost  = static_cast<double *>(prob.acquire(n * sizeof(double)));
        int const covered = 10 * 6;   /* bits * pieces, the build's default */
        row_range_kernel<<<static_cast<int>(n), launch::BLOCK_SIZE>>>(
            d_range, d_lost, st.d_lu, n, 2, covered);
        KERNEL_CHECK();
        CUDA_CHECK(cudaDeviceSynchronize());

        /*  Same question asked of R itself: how much does R vary ROW TO ROW?
            R is stored as plain fp32, so every entry shares one exponent
            range. If rows differ in magnitude by many bits, a per-row scale
            (n extra floats, essentially free) would let each row use its full
            24-bit mantissa instead of spending bits on the gap to the largest
            row. That is the cheapest available way to raise the floor. */
        double *d_rrange = static_cast<double *>(prob.acquire(n * sizeof(double)));
        double *d_rlost  = static_cast<double *>(prob.acquire(n * sizeof(double)));
        row_range_kernel<<<static_cast<int>(n), launch::BLOCK_SIZE>>>(
            d_rrange, d_rlost, static_cast<float const *>(st.d_r), n, 0, 24);
        KERNEL_CHECK();
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<double> rr(n);
        CUDA_CHECK(cudaMemcpy(rr.data(), d_rrange, n * sizeof(double),
                              cudaMemcpyDeviceToHost));

        /*  Row-to-row spread: how far the quietest row sits below the loudest,
            which is what a per-row scale would recover. */
        std::vector<double> rmax(n);
        {
            std::vector<float> hostr(n * n);
            CUDA_CHECK(cudaMemcpy(hostr.data(),
                                  static_cast<float const *>(st.d_r),
                                  nn * sizeof(float), cudaMemcpyDeviceToHost));
            for (std::size_t i = 0; i != n; ++i) {
                float m = 0.f;
                for (std::size_t j = 0; j != n; ++j)
                    m = std::max(m, std::fabs(hostr[i + j * n]));
                rmax[i] = (m > 0.f)? std::log2(static_cast<double>(m)) : 0.;
            }
            std::sort(rmax.begin(), rmax.end());
        }
        double const row_spread = rmax[n - 1] - rmax[n / 100];

        std::vector<double> hr(n), hl(n);
        CUDA_CHECK(cudaMemcpy(hr.data(), d_range, n * sizeof(double),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hl.data(), d_lost, n * sizeof(double),
                              cudaMemcpyDeviceToHost));
        std::sort(hr.begin(), hr.end());
        double lost = 0.;
        for (std::size_t i = 0; i != n; ++i) lost += hl[i];

        std::cout << "  " << std::right << std::scientific
                  << std::setprecision(1) << std::setw(9) << shifts[si]
                  << std::setw(14) << std::setprecision(2) << size
                  << std::setw(16) << relative
                  << std::fixed << std::setprecision(1)
                  << std::setw(12) << bits
                  << std::scientific << std::setprecision(2)
                  << std::setw(12)
                  << ((norm_a > 0.)? norm_l * norm_u / norm_a : 0.)
                  << std::fixed << std::setprecision(1)
                  << std::setw(11) << hr[n / 2]
                  << std::setw(11) << hr[n - 1]
                  << std::scientific << std::setprecision(2)
                  << std::setw(11) << lost / static_cast<double>(nn)
                  << std::fixed << std::setprecision(1)
                  << std::setw(12) << row_spread
                  << std::scientific << std::setprecision(2)
                  << std::setw(12) << rel_a
                  << std::setw(12) << rel_ab << "\n";
    }

    std::cout << "\n  R build error should sit near 6e-8 (R's own fp32 storage"
                 " limit) at every\n  shift. If it does, R is correct and any"
                 " accuracy floor comes from ||R||\n  growing, not from the"
                 " build.\n";

    return 0;
}
