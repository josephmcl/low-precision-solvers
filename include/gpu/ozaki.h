#pragma once

#include "gpu/definitions.h"
#include "gpu/problem.h"

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

    which is EXACT in an fp32 accumulator when that stays under 24.

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
enum class shape {
    full,
    lower,   /* unit-diagonal L out of a packed factor */
    upper    /* U out of a packed factor              */
};

struct config {

    /*  Pieces per operand, and bits per piece. NP=6 / BITS=9 covers 54 bits
        with 21 products and is the measured operating point: it beats BITS=10
        at identical cost, and BITS=11 collapses entirely. */
    int n_pieces = 6;
    int bits     = 9;

    /*  Contraction blocking. Sets the live piece footprint at
        n * block * n_pieces * 4 bytes, so it is the knob that keeps the
        transient storage bounded — pre-splitting the whole matrix instead
        costs 48n^2 and breaks any storage claim the method makes.

        Default 64 because that is the largest value satisfying
        2*bits + log2(block) <= 24 at bits=9, and the sweep in validate()'s
        comment shows accuracy degrading immediately above it. This is a
        correctness default, not a performance one: it means many narrow GEMMs,
        and the cost has not yet been measured. Raising it requires either
        fewer bits per piece or an accumulator wider than fp32. */
    int block = 64;

    /*  Scale groups kept. Products a_p*b_q depend on (p,q) only through
        s = p+q, so those sharing s sit on one grid and can accumulate inside
        cuBLAS with beta=1, exactly — which turns 21 fp64 folds into
        n_groups. Keeping fewer than n_pieces groups truncates the
        lowest-magnitude terms; measured, dropping even one costs two orders
        of accuracy, so this is not a speed knob. */
    int n_groups = 6;

    /*  Products actually issued: n_groups*(n_groups+1)/2. */
    int n_products() const {return n_groups * (n_groups + 1) / 2;}

    /*  Total bits of the product this configuration resolves. */
    int bits_resolved() const {return n_pieces * bits;}
};

/*  Check a configuration and report on std::cout. Returns false only for the
    hard constraint; the soft one warns.

    HARD: bits <= 11. TF32 carries 11 significant bits (10 explicit plus the
    implicit one), so a wider piece is not representable and the scheme
    silently degrades to TF32 accuracy. Measured, BITS=11 already collapses to
    2e-11, which locates the boundary exactly.

    SOFT-LOOKING BUT BINDING: 2*bits + log2(block) <= 24, the exactness bound
    above. Measured on this implementation, at n=4096 against an independent
    fp64 reference:

        block    bound    3 pieces    6 pieces
           64     24.0    1.05e-09    4.37e-11
          256     26.0    2.03e-08    2.02e-08
         1024     28.0    5.47e-07    5.47e-07
         3072     29.6    2.90e-06    2.90e-06

    The bound binds. Error tracks `block` over four orders, and only where the
    bound holds does adding pieces buy anything — above it the piece count is
    inert, which is the signature of pieces not landing on a common grid.

    THIS CONTRADICTS a prior conclusion, recorded elsewhere, that the bound is
    "sufficient, not necessary" because a 4096-wide block was measured working
    at 9.6e-15. That was a different implementation and the two disagree by
    five orders. Do not assume the bound can be violated. If a configuration
    above the bound appears to work, treat it as an unexplained result and find
    the structural difference before relying on it — one of the two
    measurements is wrong, and this one has an independent reference.

    STILL UNRESOLVED: 4.37e-11 at block=64 is not the ~1e-15 that 54 resolved
    bits should give, so a second limit remains below the bound. That matters,
    because building R = PA - LU needs the product accurate to ~2^-48 relative
    for R's own fp32 storage to be the binding term. Diagnose that before using
    this for the residual-storage method. */
bool validate(config const &cfg);

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
    float const       *d_a,
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

    row_max and column_max must already have been called for this A and X. */
void accumulate_product(
    double            *d_acc,
    float const       *d_a,
    double const      *d_x,
    std::size_t const  lda,
    std::size_t const  n_rhs,
    shape const        which,
    workspace         &ws,
    problem           &prob);

} /* namespace ozaki */
