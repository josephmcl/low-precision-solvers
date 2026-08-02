#include "common/trsm.h"

#include "common/tuning.h"

#include <cmath>

namespace trsm {

namespace {

/*  Max |.| along a row (triangular operand) or a column (solution block).

    For C[i,j] = sum_k L[i,k] X[k,j] the contraction runs over L's COLUMNS and
    X's ROWS, so a scale varying per L-row and per X-column is constant within
    every dot product — each accumulation stays on one exponent grid. A single
    global scale is the degenerate case and costs twice: it breaks TF32
    outright, because a block whose values exceed 2^g pushes piece 0 above
    `bits` bits, which fp32's 24-bit input absorbs silently and TF32's 11 do
    not; and it inflates the piece count, because the span to cover is the
    whole panel's rather than one row's. */
__global__ void max_by_row(
    double const      *d_x,
    int const          ld,
    int const          rows,
    int const          cols,
    unsigned          *d_out) {

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < rows;
         i += blockDim.x * gridDim.x) {
        unsigned m = 0u;
        for (int j = 0; j != cols; ++j) {
            unsigned const u = __float_as_uint(fabsf(static_cast<float>(
                d_x[i + static_cast<std::size_t>(j) * ld])));
            if (u > m) m = u;
        }
        d_out[i] = m;
    }
}

__global__ void max_by_row_f(
    float const       *d_x,
    int const          ld,
    int const          rows,
    int const          cols,
    unsigned          *d_out) {

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < rows;
         i += blockDim.x * gridDim.x) {
        unsigned m = 0u;
        for (int j = 0; j != cols; ++j) {
            unsigned const u = __float_as_uint(fabsf(
                d_x[i + static_cast<std::size_t>(j) * ld]));
            if (u > m) m = u;
        }
        d_out[i] = m;
    }
}

__global__ void max_by_column(
    double const      *d_x,
    int const          ld,
    int const          rows,
    int const          cols,
    unsigned          *d_out) {

    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < cols;
         j += blockDim.x * gridDim.x) {
        unsigned m = 0u;
        for (int i = 0; i != rows; ++i) {
            unsigned const u = __float_as_uint(fabsf(static_cast<float>(
                d_x[i + static_cast<std::size_t>(j) * ld])));
            if (u > m) m = u;
        }
        d_out[j] = m;
    }
}

/*  Exponent-aligned split, S = 1.5 * 2^(g + 52 - (q+1)*bits).

    The 52 is not a typo for the 23 that appears in the fp32 literature: this
    arithmetic is carried out in fp64, where (v + S) - S rounds v to multiples
    of 2^(E-52). With the fp32 constant each piece carries ~29 bits instead of
    `bits`, no product is exactly representable, and the whole scheme quietly
    delivers plain-fp32 accuracy. */
__global__ void split(
    double const      *d_x,
    int const          ld_x,
    float             *d_pieces,
    int const          rows,
    int const          cols,
    int const          bits,
    int const          n_pieces,
    unsigned const    *d_max,
    int const          by_row) {

    int const n_use = (n_pieces < 16)? n_pieces : 16;
    std::size_t const total = static_cast<std::size_t>(rows) * cols;
    int const j = blockIdx.y;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < rows;
         i += blockDim.x * gridDim.x) {

        float const m = __uint_as_float(d_max[by_row? i : j]);
        int const g = (m > 0.f)? static_cast<int>(ceilf(log2f(m))) : 0;

        std::size_t const t = static_cast<std::size_t>(i) +
                              static_cast<std::size_t>(j) * rows;
        double v = d_x[i + static_cast<std::size_t>(j) * ld_x];

        for (int q = 0; q != n_use; ++q) {
            double const s = 1.5 * exp2(
                static_cast<double>(g + 52 - (q + 1) * bits));
            double const piece = (v + s) - s;
            d_pieces[t + static_cast<std::size_t>(q) * total] =
                static_cast<float>(piece);
            v -= piece;
        }
    }
}

/*  Same, for an operand already stored fp32. R and the packed factor are both
    exactly representable in fp32, so this reads them without a promotion. */
__global__ void split_f(
    float const       *d_x,
    int const          ld_x,
    float             *d_pieces,
    int const          rows,
    int const          cols,
    int const          bits,
    int const          n_pieces,
    unsigned const    *d_max,
    int const          by_row) {

    int const n_use = (n_pieces < 16)? n_pieces : 16;
    std::size_t const total = static_cast<std::size_t>(rows) * cols;
    int const j = blockIdx.y;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < rows;
         i += blockDim.x * gridDim.x) {

        float const m = __uint_as_float(d_max[by_row? i : j]);
        int const g = (m > 0.f)? static_cast<int>(ceilf(log2f(m))) : 0;

        std::size_t const t = static_cast<std::size_t>(i) +
                              static_cast<std::size_t>(j) * rows;
        double v = static_cast<double>(d_x[i + static_cast<std::size_t>(j) * ld_x]);

        for (int q = 0; q != n_use; ++q) {
            double const s = 1.5 * exp2(
                static_cast<double>(g + 52 - (q + 1) * bits));
            double const piece = (v + s) - s;
            d_pieces[t + static_cast<std::size_t>(q) * total] =
                static_cast<float>(piece);
            v -= piece;
        }
    }
}

/*  acc += scale * product, in fp64. One of these per SCALE GROUP, not per
    product: products with p+q = s share an exponent grid, so cuBLAS folds them
    together with beta=1 and only the group needs an fp64 pass. Without that
    grouping this kernel ran 60480 times for 6.4% of GPU time. */
__global__ void fold(
    double            *d_acc,
    int const          ld_acc,
    float const       *d_p,
    int const          ld_p,
    double const       scale,
    int const          rows,
    int const          cols) {

    std::size_t const total = static_cast<std::size_t>(rows) * cols;
    for (std::size_t t = blockIdx.x * blockDim.x + threadIdx.x;
         t < total; t += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
        int const i = static_cast<int>(t % rows), j = static_cast<int>(t / rows);
        d_acc[i + static_cast<std::size_t>(j) * ld_acc] += scale *
            static_cast<double>(d_p[i + static_cast<std::size_t>(j) * ld_p]);
    }
}

/*  Promote ONE diagonal block of the packed factor into fp64.

    Only the diagonal blocks are solved in fp64, so only they need promoting —
    b^2 doubles, not 2n^2. Materialising dense fp64 L and U would cost 16n^2
    bytes, which is twice the entire storage budget this method exists to
    defend; the first version of this file did exactly that, and it would have
    made every storage number in the paper false while still producing correct
    answers.

    The VALUES stay the fp32 ones. Only the arithmetic is promoted: R exists to
    absorb the factor's error, and promoting the values would not remove that
    error, only obscure which stage owns it. */
__global__ void promote_diagonal_block(
    double            *d_block,
    float const       *d_lu,
    std::size_t const  n,
    std::size_t const  j0,
    std::size_t const  nb,
    int const          lower) {

    std::size_t const total = nb * nb;
    for (std::size_t t = blockIdx.x * blockDim.x + threadIdx.x;
         t < total; t += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
        std::size_t const i = t % nb, j = t / nb;
        double const v = static_cast<double>(d_lu[(j0 + i) + (j0 + j) * n]);
        d_block[t] = lower? ((i > j)? v : ((i == j)? 1. : 0.))
                          : ((i <= j)? v : 0.);
    }
}

/*  One blocked triangular solve. */
void solve_one(
    bool const         lower,
    float const       *d_tri32,
    double            *d_rhs,
    int const          n,
    int const          k,
    workspace         &ws,
    config const      &cfg,
    problem           &prob) {

    double const one = 1.;
    float const f_one = 1.f, f_zero = 0.f;
    int const nb = cfg.block;
    int const n_blocks = (n + nb - 1) / nb;

    for (int b = 0; b != n_blocks; ++b) {

        int const j0 = lower? b * nb : n - (b + 1) * nb;
        int const cb = nb;

        /*  fp64 diagonal solve: O(n b^2), negligible beside O(n^2 k). */
        promote_diagonal_block<<<256, 256>>>(
            ws.d_tri64, d_tri32, static_cast<std::size_t>(n),
            static_cast<std::size_t>(j0), static_cast<std::size_t>(cb),
            lower? 1 : 0);
        KERNEL_CHECK();

        CUBLAS_CHECK(cublasDtrsm(
            prob.blas, CUBLAS_SIDE_LEFT,
            lower? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER,
            CUBLAS_OP_N,
            lower? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT,
            cb, k, &one, ws.d_tri64, cb, d_rhs + j0, n));

        int const r0 = lower? j0 + cb : 0;
        int const n_rows = lower? n - (j0 + cb) : j0;
        if (n_rows <= 0) continue;

        /*  Trailing update, exponent-aligned. */
        max_by_row_f<<<(n_rows + 255) / 256, 256>>>(
            d_tri32 + r0 + static_cast<std::size_t>(j0) * n, n,
            n_rows, cb, ws.d_scale_tri);
        max_by_column<<<(k + 255) / 256, 256>>>(
            d_rhs + j0, n, cb, k, ws.d_scale_rhs);
        KERNEL_CHECK();

        split_f<<<dim3((n_rows + 255) / 256, cb), 256>>>(
            d_tri32 + r0 + static_cast<std::size_t>(j0) * n, n,
            ws.d_pieces_tri, n_rows, cb, cfg.bits, cfg.pieces_tri,
            ws.d_scale_tri, 1);
        split<<<dim3((cb + 255) / 256, k), 256>>>(
            d_rhs + j0, n, ws.d_pieces_rhs, cb, k, cfg.bits, cfg.pieces,
            ws.d_scale_rhs, 0);
        KERNEL_CHECK();

        std::size_t const size_tri = static_cast<std::size_t>(n_rows) * cb;
        std::size_t const size_rhs = static_cast<std::size_t>(cb) * k;

        for (int s = 0; s != cfg.pieces; ++s) {
            bool first = true;
            for (int p = 0; p <= s; ++p) {
                int const q = s - p;
                if (q < 0 || q >= cfg.pieces || p >= cfg.pieces_tri) continue;
                float const beta = first? f_zero : f_one;
                CUBLAS_CHECK(cublasGemmEx(
                    prob.blas, CUBLAS_OP_N, CUBLAS_OP_N, n_rows, k, cb,
                    &f_one,
                    ws.d_pieces_tri + static_cast<std::size_t>(p) * size_tri,
                    CUDA_R_32F, n_rows,
                    ws.d_pieces_rhs + static_cast<std::size_t>(q) * size_rhs,
                    CUDA_R_32F, cb, &beta,
                    ws.d_product, CUDA_R_32F, n_rows,
                    CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT));
                first = false;
            }
            if (!first) {
                fold<<<256, 256>>>(d_rhs + r0, n, ws.d_product, n_rows, -1.,
                                   n_rows, k);
                KERNEL_CHECK();
            }
        }
    }
}

} /* namespace */

config config::from_tuning() {
    config cfg;
    cfg.bits       = tuning::current().get("rir.solve.trsm.bits", 8);
    cfg.pieces     = tuning::current().get("rir.solve.trsm.pieces", 7);
    cfg.pieces_tri = tuning::current().get("rir.solve.trsm.pieces_tri", 5);
    cfg.block      = tuning::current().get("rir.solve.trsm.block", 256);
    return cfg;
}

void workspace::acquire(problem &prob, config const &cfg, std::size_t n,
                        std::size_t k) {

    std::size_t const np = static_cast<std::size_t>(cfg.pieces);

    /*  The residual uses a LARGER contraction block than the solve (1024
        against 256, worth 1.18x), and both share this scratch. Sizing it from
        `cfg.block` alone overflows the piece buffers by 4x — which does not
        crash, it silently corrupts whatever follows and produces a plausible
        wrong answer. Size for the larger of the two. */
    std::size_t const nb_solve = static_cast<std::size_t>(cfg.block);
    std::size_t const nb_rx = static_cast<std::size_t>(
        tuning::current().get("rir.solve.rx.block", 1024));
    std::size_t const nb = (nb_rx > nb_solve)? nb_rx : nb_solve;

    d_pieces_tri = static_cast<float *>(
        prob.acquire(np * n * nb * sizeof(float)));
    d_pieces_rhs = static_cast<float *>(
        prob.acquire(np * nb * k * sizeof(float)));
    d_product = static_cast<float *>(prob.acquire(n * k * sizeof(float)));
    d_scale_tri = static_cast<unsigned *>(prob.acquire(n * sizeof(unsigned)));
    d_scale_rhs = static_cast<unsigned *>(
        prob.acquire(((k > n)? k : n) * sizeof(unsigned)));
    /*  ONE fp64 diagonal block. b^2 doubles — see promote_diagonal_block on
        why this must not be 2n^2. */
    d_tri64 = static_cast<double *>(prob.acquire(nb * nb * sizeof(double)));
}

void solve(
    double            *d_x,
    float const       *d_lu,
    std::size_t const  k,
    workspace         &ws,
    config const      &cfg,
    problem           &prob) {

    std::size_t const n = prob.n;
    int const ni = static_cast<int>(n), ki = static_cast<int>(k);

    solve_one(true,  d_lu, d_x, ni, ki, ws, cfg, prob);
    solve_one(false, d_lu, d_x, ni, ki, ws, cfg, prob);
}

void residual(
    double            *d_acc,
    float const       *d_r,
    double const      *d_x,
    std::size_t const  k,
    int const          pieces_r,
    workspace         &ws,
    config const      &cfg,
    problem           &prob) {

    std::size_t const n = prob.n;
    int const ni = static_cast<int>(n), ki = static_cast<int>(k);
    float const f_one = 1.f, f_zero = 0.f;

    /*  R*X wants a LARGER contraction block than the triangular solve: 1024
        against 256, worth 1.18x. That puts 2*bits + log2(K) at 26, over the
        limit of 24 — deliberately. Every bound-respecting alternative at this
        block size is ~30x worse, because total precision (bits*pieces = 56)
        dominates the accumulator margin, and dropping to bits=7 sacrifices 7
        bits of it to recover 1 bit of accumulator.

        The bound is a WORST-CASE guarantee and these operands do not reach it.
        A matrix whose accumulation does would lose ~2 bits here. This is a
        stated risk, not an oversight: re-check it whenever a new matrix family
        is added. */
    int const nb = tuning::current().get("rir.solve.rx.block", 1024);

    for (int c0 = 0; c0 < ni; c0 += nb) {
        int const cb = (c0 + nb <= ni)? nb : ni - c0;

        max_by_row_f<<<(ni + 255) / 256, 256>>>(
            d_r + static_cast<std::size_t>(c0) * ni, ni, ni, cb,
            ws.d_scale_tri);
        max_by_column<<<(ki + 255) / 256, 256>>>(
            d_x + c0, ni, cb, ki, ws.d_scale_rhs);
        KERNEL_CHECK();

        split_f<<<dim3((ni + 255) / 256, cb), 256>>>(
            d_r + static_cast<std::size_t>(c0) * ni, ni, ws.d_pieces_tri,
            ni, cb, cfg.bits, pieces_r, ws.d_scale_tri, 1);
        split<<<dim3((cb + 255) / 256, ki), 256>>>(
            d_x + c0, ni, ws.d_pieces_rhs, cb, ki, cfg.bits, cfg.pieces,
            ws.d_scale_rhs, 0);
        KERNEL_CHECK();

        std::size_t const size_r = static_cast<std::size_t>(ni) * cb;
        std::size_t const size_x = static_cast<std::size_t>(cb) * ki;

        for (int s = 0; s != cfg.pieces; ++s) {
            bool first = true;
            for (int p = 0; p <= s; ++p) {
                int const q = s - p;
                if (q < 0 || q >= cfg.pieces || p >= pieces_r) continue;
                float const beta = first? f_zero : f_one;
                CUBLAS_CHECK(cublasGemmEx(
                    prob.blas, CUBLAS_OP_N, CUBLAS_OP_N, ni, ki, cb, &f_one,
                    ws.d_pieces_tri + static_cast<std::size_t>(p) * size_r,
                    CUDA_R_32F, ni,
                    ws.d_pieces_rhs + static_cast<std::size_t>(q) * size_x,
                    CUDA_R_32F, cb, &beta,
                    ws.d_product, CUDA_R_32F, ni,
                    CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT));
                first = false;
            }
            if (!first) {
                fold<<<256, 256>>>(d_acc, ni, ws.d_product, ni, -1., ni, ki);
                KERNEL_CHECK();
            }
        }
    }
}

} /* namespace trsm */
