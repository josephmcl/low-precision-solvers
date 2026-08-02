#pragma once

#include "common/definitions.h"
#include "common/error.h"
#include "common/problem.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <cstddef>

/*  An accurate triangular solve built from tensor-core GEMMs.

    WHY THIS EXISTS. The fp32 triangular solve was measured to be R-IR's
    accuracy limiter by a factor of 8e6 — larger than R's build precision, R's
    storage, the residual product, the stopping rule, and the refinement count
    put together. Each of those was proposed and refuted in turn; the solve was
    found by replacing one stage at a time with an exact fp64 version, which is
    guaranteed to terminate where hypothesis-testing is not. With the solve
    fixed, R-IR tracks a direct fp64 solve to within 2.4x-2.9x across the whole
    conditioning range, and the "conditioning limit" attributed to the method
    for most of this project turns out not to exist.

    HOW. A blocked solve, not a triangular kernel:

      - diagonal blocks are solved in fp64. That is O(n b^2) against the
        O(n^2 k) trailing updates, so it is free at any sensible block size and
        removes the sequential dependency from the accurate path entirely.
      - trailing updates are exponent-aligned split products through TF32
        tensor cores with fp64 accumulation — the Ozaki scheme, same as the
        residual, applied to the update rather than to a residual.

    WHAT DOES NOT WORK, measured, so nobody repeats it:

      - Splitting the operands hi/lo without exponent alignment recovers 33x of
        an available 730000x. The binding constraint is the PRODUCT width:
        cublasSgemm truncates each product to fp32's 24 bits on output, so with
        24-bit operands 2*bits = 48 overflows the mantissa before a single
        addition happens. Shrinking the contraction length K to 1 — removing
        accumulation error entirely — buys only 5x. That is why `bits` is small
        here: it is a precondition, not a tuning knob.
      - fp16 pieces are exact (11 significand bits hold `bits` <= 8) and
        1.03x-1.45x SLOWER, because the GEMMs are not the bottleneck.
      - Caching the split pieces across passes has a measured ceiling of 1%:
        the triangular operand's split is not what dominates, the solution
        block's is, and that changes every pass by construction.

    THE SPLITTER CONSTANT IS THE EASY THING TO GET WRONG. (v + S) - S is
    evaluated in fp64 here, so its granularity is 2^(E-52), NOT the fp32
    2^(E-23). Using the fp32 constant leaves each piece carrying ~29 bits
    instead of `bits`, no product is exactly representable, and the scheme
    degrades to plain fp32 while looking entirely plausible. It cost two runs
    to find, and it will not announce itself. */

namespace trsm {

using harness::problem;

/*  Tunables. The bound `2*bits + log2(block) <= 24` is what keeps each product
    exact in the fp32 accumulator; see the note in trsm.cu on where it is
    deliberately exceeded and why. */
struct config {
    int  bits    = 8;    /*  bits per piece                                  */
    int  pieces  = 7;    /*  pieces for the fp64 operand (the solution block) */
    int  pieces_tri = 5; /*  pieces for the fp32 triangular factor            */
    int  block   = 256;  /*  contraction block; also the fp64 diagonal size   */

    static config from_tuning();
};

/*  Scratch, sized once and reused. Allocating inside a timed region charges
    the allocator to the arithmetic. */
struct workspace {
    float    *d_pieces_tri = nullptr;
    float    *d_pieces_rhs = nullptr;
    float    *d_product    = nullptr;
    unsigned *d_scale_tri  = nullptr;
    unsigned *d_scale_rhs  = nullptr;
    double   *d_tri64      = nullptr;

    void acquire(problem &prob, config const &cfg, std::size_t n,
                 std::size_t k);
};

/*  Solve L U X = B in place on d_x, both triangles, to fp64 accuracy.

    d_lu is the packed fp32 factor; L is unit lower, U is non-unit upper.
    d_x is fp64 and is overwritten with the solution. */
void solve(
    double            *d_x,
    float const       *d_lu,
    std::size_t const  k,
    workspace         &ws,
    config const      &cfg,
    problem           &prob);

/*  acc <- acc - R * X, compensated, with R stored fp32 and X fp64.

    Blocked over the contraction dimension so R's pieces are never materialised
    for the whole matrix: pieces * n^2 floats is 1.9 GB at n=8192 and would
    break the 8n^2 storage claim outright.

    This replaces the plain fp32 SGEMM. That SGEMM was recorded in this repo as
    an optimization "worth 1.27x, bit-identical to the Ozaki path" — measured
    when R-IR's floor was 1.11e-10, where nothing could have shown a
    difference. Against the current floor it costs up to 34x. */
void residual(
    double            *d_acc,
    float const       *d_r,
    double const      *d_x,
    std::size_t const  k,
    int const          pieces_r,
    workspace         &ws,
    config const      &cfg,
    problem           &prob);

} /* namespace trsm */
