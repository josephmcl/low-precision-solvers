#pragma once

#include "common/error.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>
#include <cstdlib>
#include <iostream>

#include "common/definitions.h"
#include "common/problem.h"

#include <cstddef>
#include <vector>

/*  Ozaki splitting: an accurate matrix product built from tensor-core GEMMs.

    Both fp64-free methods need one primitive — the product of an fp32 matrix
    with an fp64 one, accurate to far more bits than either GEMM provides — so
    it lives here rather than in either method, where two copies would drift.

    HOW IT WORKS. Round every value in a row to a common exponent grid with
    the classic (x + S) - S trick, S = 1.5 * 2^(g+23) for fp32, which rounds x
    to a multiple of 2^g. Then for pieces a_p, b_q carrying `bits` bits each:

        a_p       is a multiple of 2^(eA - bits(p+1))
        b_q       is a multiple of 2^(eB - bits(q+1))
        a_p * b_q is a multiple of 2^(eA+eB - bits(p+q+2)), <= 2*bits bits
        summed over K terms:                    <= 2*bits + log2(K) bits

    which is EXACT in an fp32 accumulator when that stays under 24 — but see
    validate(): the usable condition is <= 23, one bit tighter than the
    derivation, and with that margin the product comes out bit-identical to
    fp64.

    WHY THE GRID IS GLOBAL, NOT PER BLOCK. The grid has to be constant along
    the summed index, so the scale for A is one value per *full* row and for B
    one per *full* column, reused by every contraction block. Per-block maxima
    would be tighter — each block spans a narrower dynamic range, so fewer
    pieces would cover it — but partials from different blocks would then sit
    on different grids. Whether they could still be combined in the fp64
    accumulator is an open question and a live optimization: if it works, the
    piece count could drop from 6 to about 4 and products from 21 to 10. It is
    NOT attempted here. Anyone trying it should first measure the max/min ratio
    within a row against the same ratio within a row-block; if they are
    similar, the idea is void before any kernel is written.

    A DIFFERENCE FROM NAIVE SPLITTING, worth stating because getting it wrong
    silently costs everything. Dekker-style splitting gives each value a fixed
    *relative* precision, so products land on incompatible grids and the
    accumulator rounds anyway — the result then sits at plain TF32 accuracy, as
    if the split had bought nothing. Ozaki needs fixed *absolute* precision.
    That is what the row/column scaling above provides.

    Two consequences of the scaling that are easy to get backwards:

      - Pieces resolve bits below the ROW MAX, not below each value. Reaching
        2^-47 therefore needs ceil(47/bits) pieces, not three. An early
        implementation assumed three and was wrong by ten orders.
      - Wide dynamic range within a row is the *hard* case; uniformly graded
        rows are the easy one, because grading makes a row's entries share a
        scale, which is exactly what aligned splitting wants. */

namespace ozaki {

using harness::problem;

/*  Which triangle of the fp32 operand carries data. The triangular forms skip
    structurally-zero blocks, which is not a micro-optimization: it is worth
    ~1.4x on the residual-storage method's refinement, and it is the one
    arithmetic advantage that method has which the dense baselines cannot
    also take. */
/*  How the fp32-side operand is stored.

    R = PA - LU is the only array here worth compressing, and it is the whole
    point of the scheme: storage identity bits(eps) = bits(u_f) + bits(u_R)
    says the accuracy follows the TOTAL bits kept, so trading R's width against
    problem size is a dial rather than a compromise. fp32 R gives 8n^2 total,
    b24 gives 7n^2, bf16 gives 6n^2.

    b24 is sign + 8 exponent + 15 mantissa packed into three bytes. Note that
    is 15 mantissa bits, not the 20 an earlier projection assumed — 1+8+20 is
    29 bits and does not fit in three bytes. The accuracy has to be measured at
    15, not carried over from a 20-bit truncation experiment. */
enum class format {
    fp32,
    b24,
    bf16
};

std::size_t bytes_of(format const f);

/*  Round an fp32 array into `f`, in place-compatible layout (the destination
    may alias nothing; sizes differ). Used once, when R is formed. */
void compress(
    void              *d_out,
    float const       *d_in,
    std::size_t const  n_elements,
    format const       f,
    problem           &prob);

enum class shape {
    full,
    lower,   /* unit-diagonal L out of a packed factor */
    upper    /* U out of a packed factor              */
};

struct config {

    /*  Pieces per operand, and bits per piece. 9 x 6 = 54 bits, 45 products.

        Fewer, wider pieces would issue fewer products (6 x 9 bits is 21), but
        measured that is SLOWER, because the exactness bound then forces a
        narrow contraction block and the GEMMs stop filling the machine:

            bits np block  prods     ms   rel err
              9   6    64     21   11.3  4.37e-11   bound 24 -- inexact
              9   6    32     21   16.8  0.00e+00
              6   9   256     45    7.1  0.00e+00   <- default
              5  11   256     66   10.1  0.00e+00

        Product count is the wrong objective; GEMM width dominates. The fastest
        configuration is also an exact one. */
    int n_pieces = 9;
    int bits     = 6;

    /*  Contraction blocking. Sets the live piece footprint at
        n * block * n_pieces * 4 bytes, so it is the knob that keeps the
        transient storage bounded — pre-splitting the whole matrix instead
        costs 48n^2 and breaks any storage claim the method makes.

        Default 256, the largest value satisfying 2*bits + log2(block) <= 23
        at bits=6. Chosen on measured time, not guessed: see n_pieces. */
    int block = 256;

    /*  Scale groups kept. Defaults to n_pieces. Products a_p*b_q depend on (p,q) only through
        s = p+q, so those sharing s sit on one grid and can accumulate inside
        cuBLAS with beta=1, exactly — which turns 21 fp64 folds into
        n_groups. Keeping fewer than n_pieces groups truncates the
        lowest-magnitude terms; measured, dropping even one costs two orders
        of accuracy, so this is not a speed knob. */
    int n_groups = 9;

    /*  Scale groups from this index up share ONE fp64 fold instead of one
        each. Products in group s carry magnitude ~2^-(bits*s), so for large s
        the cross-grid rounding incurred by adding them on a common grid is far
        below the budget the split is trying to hold — earlier work validated
        merging everything from s=3 and measured it slightly MORE accurate, not
        less. n_groups disables merging.

        DEFAULTS TO OFF (-1 = keep every group separate), and must be opted
        into per site. Merging is only safe where the *consumer* of the product
        is coarser than the damage it does: R-IR tolerates it because R is then
        stored in fp32, which is coarser still. Measured on the raw L*U product
        it costs 15x at 6 pieces and 13000x at 9 — degradation that is real but
        invisible downstream, which is exactly the kind of default that ships a
        silent four-order regression to whoever builds a config by hand. Worth
        1-2% when it applies. */
    int merge_tail = -1;

    /*  Products actually issued: n_groups*(n_groups+1)/2. */
    int n_products() const {return n_groups * (n_groups + 1) / 2;}

    /*  Total bits of the product this configuration resolves. */
    int bits_resolved() const {return n_pieces * bits;}

    /*  For a product whose result is STORED and reused — R = PA - LU is the
        case — where the error is undamped and passes straight into everything
        downstream. Must satisfy the exactness bound; measured bit-identical to
        fp64. Costs ~3x the refinement configuration. */
    static config for_stored_product() {
        config c;
        c.bits = 6; c.n_pieces = 9; c.n_groups = 9; c.block = 256;
        return c;
    }

    /*  For a product consumed immediately by a CONVERGING REFINEMENT, where an
        inexact residual is self-corrected: it only has to give a good
        correction direction, and the next iteration cleans up what it got
        wrong.

        Measured on split-MPIR at n=8192, k=512, sweeping across and far above
        the exactness bound:

            bits/np/block  bound  solve ms  iters  backward
                 6/ 9/ 256   20.0     241.1      3  3.24e-17
                 6/ 9/ 512   21.0     194.4      3  3.24e-17
                 9/ 6/1536   28.6      87.3      3  3.11e-17
                 9/ 6/3072   29.6      81.7      3  3.11e-17
                 5/11/ 512   19.0     273.2      3  3.07e-17

        The converged answer does not move — 3.0e-17 either way, 3 iterations
        either way — while time varies 3.3x. So violating the bound here is
        free, and this configuration takes it.

        Do not carry that conclusion over to a stored product. The distinction
        is the whole point of having two named configurations: the same
        inexactness that vanishes inside a refinement loop is permanent in a
        matrix you keep. */
    static config for_refinement() {
        config c;
        c.bits = 9; c.n_pieces = 6; c.n_groups = 6; c.block = 3072;
        return c;
    }
};

/*  Check a configuration and report on std::cout. Returns false only for the
    hard constraint; the soft one warns.

    HARD: bits <= 11. TF32 carries 11 significant bits (10 explicit plus the
    implicit one), so a wider piece is not representable and the scheme
    silently degrades to TF32 accuracy. Measured, BITS=11 already collapses to
    2e-11, which locates the boundary exactly.

    BINDING, AND ONE BIT TIGHTER THAN DERIVED:

        2*bits + log2(block) <= 23

    Measured at n=4096 against an independent fp64 reference. Satisfying it
    with that one bit of margin makes the product **bit-identical to fp64**:

        bound   rel err
         24.0   4.37e-11
         23.0   0.00e+00
         22.0   0.00e+00
         20.0   0.00e+00
         18.0   0.00e+00

    fp32's 24-bit significand can hold a 24-bit value, but the intermediate
    sums during accumulation need the spare bit, so the derived <= 24 is off by
    one. With margin there is no error at all — the split is exact, not merely
    accurate.

    Above the bound the piece count goes INERT: error tracks `block` over four
    orders (2.9e-06 at block 3072) and adding pieces changes nothing. That is
    the diagnostic signature of pieces not landing on a common grid, and it is
    what to look for if this ever regresses.

    THIS CONTRADICTS a prior conclusion, recorded elsewhere, that the bound is
    "sufficient, not necessary" because a 4096-wide block was measured working
    at 9.6e-15. The two disagree by five orders and cannot both be right. This
    one has an independent fp64 reference and a mechanism that predicts the
    off-by-one. Do not assume the bound can be violated; if a configuration
    above it appears to work, find the structural difference first. */
bool validate(config const &cfg);

/*  Defaults, with LPS_OZ_BITS / LPS_OZ_PIECES / LPS_OZ_BLOCK applied if set.

    For tuning sweeps only. An override may place the configuration above the
    exactness bound, which validate() reports; callers proceed anyway so the
    inexact regime can be measured deliberately rather than only stumbled
    into. Whether that regime is acceptable depends on where the product sits:
    inside a converging refinement its error is corrected by the next
    iteration, but a product whose result is stored and reused — R = PA - LU —
    passes its error straight through. */
config from_environment();

/*  Piece buffers and scale factors, allocated once and reused.

    Owns its device allocations and frees them in the destructor — the same
    manual tracking the rest of the codebase uses. bytes() reports the live
    footprint so a storage claim can be checked against it rather than
    asserted. */
struct workspace {

    workspace(
        std::size_t const  n,
        std::size_t const  n_rhs,
        config const      &cfg,
        problem           &prob);

    ~workspace();

    workspace(workspace const &)            = delete;
    workspace &operator=(workspace const &) = delete;

    std::size_t bytes() const {return _bytes;}

    config const cfg;

    float *d_pieces_a = nullptr;   /* n     x block x n_pieces */
    float *d_pieces_x = nullptr;   /* block x n_rhs x n_pieces */
    float *d_partial  = nullptr;   /* one group's fp32 partial  */
    float *d_mu       = nullptr;   /* per-row scale of A        */
    float *d_nu       = nullptr;   /* per-column scale of X     */

private:

    std::size_t         _bytes = 0;
    std::vector<void *> _d_owned;

    void *_acquire(std::size_t const bytes);
};

/*  Scale factors, taken over full rows of A and full columns of X. Call once
    per factored operand; the result is reused by every contraction block, as
    the grid argument above requires. */
void row_max(
    float             *d_mu,
    void const        *d_a,
    format const       f,
    std::size_t const  n,
    std::size_t const  lda,
    shape const        which,
    problem           &prob);

void column_max(
    float             *d_nu,
    double const      *d_x,
    std::size_t const  n,
    std::size_t const  n_rhs,
    problem           &prob);

/*  d_acc += A * X, to cfg.bits_resolved() bits, accumulated in fp64.

    A is fp32, n x n, read through `which` so a packed factor needs no
    materialized copy. X is fp64, n x n_rhs. d_acc is fp64, n x n_rhs, and is
    added to rather than overwritten, so a caller forming PB - A*X seeds it
    with PB and passes a negated A or subtracts afterwards.

    Blocked over the contraction index: only one block of pieces is ever live,
    which is what keeps the footprint at workspace::bytes() instead of 48n^2.

    row_max and column_max must already have been called for this A and X.

    contraction_limit bounds the summed index: only c < contraction_limit is
    visited. Pass 0 for the full range.

    This is not a micro-optimization. Forming R = PA - LU one column block at a
    time, the block starting at column j receives nothing from contraction
    indices at or beyond j + block_width, because U is upper triangular and
    those rows of U are structurally zero in these columns. Running the full
    range anyway does about twice the necessary work — summed over all column
    blocks the useful part is n^2/2 against n^2. */
void accumulate_product(
    double            *d_acc,
    void const        *d_a,
    double const      *d_x,
    std::size_t const  lda,
    std::size_t const  n_rhs,
    shape const        which,
    workspace         &ws,
    problem           &prob,
    std::size_t const  contraction_limit = 0,
    format const       a_format = format::fp32);

} /* namespace ozaki */
