#pragma once

#include "common/ozaki2.h"

#include <cstddef>
#include <vector>

/*  Ozaki-II GEMM, arXiv:2602.02549 Algorithm 1, on the device.

    Three stages, and only the middle one is cheap:

      Algorithm 2   scaling. Per-row shifts for A and per-column shifts for
                    B, both powers of two, chosen so the integer product
                    fits condition (5). Needs its OWN int8 GEMM to size the
                    result -- Cbar = Abar Bbar -- so the scheme costs N + 1
                    int8 GEMMs, not N.
      residues      A' mod p_l, B' mod p_l, N int8 GEMMs, int32 exact.
      Algorithm 3   CRT reconstruction, then undo the shifts.

    The scaled integer operands are the reason this is int32 and not
    narrower: at N = 8 they measure 31 bits on the reference cases, which is
    int32 with no headroom at all. A' and B' therefore live in int32, and
    only the residues are int8.

    Two reconstruction configurations, never to be conflated:

      config::fp64      Algorithm 3 verbatim, fp64 reduction against the
                        double-double constants. The accuracy reference and
                        the oracle's object. NOT fp64-free.
      config::ff        the fp32 float-expansion reconstruction of
                        `lem:oiiff`. The SASS gate applies to this one.

    Layouts follow the vendored int8 path so the same cuBLAS int8 call
    serves: A-side operands are ROW major m x k (ld = k, passed OP_T),
    B-side operands are COLUMN major k x n (ld = k, passed OP_N), and the
    int32 products come back column major m x n. */

namespace oii {

enum class config { fp64, ff };

struct state;

/*  Allocate for products up to m_max x n_max with contraction length k,
    using the first n_moduli of the source's list. Returns null on failure.

    m and n are CAPACITIES, passed again per call: the trailing update
    shrinks by a panel each step, and reallocating for every one of them
    would dominate a factorization. k is fixed because the trailing
    update's contraction is the panel width, which does not shrink -- the
    one panel narrower than b is the last, and it has no trailing update. */
state *create(
    std::size_t const m_max,
    std::size_t const k,
    std::size_t const n_max,
    int const         n_moduli);

void destroy(state *s);

/*  C = A B. A is ROW major m x k, B is ROW major k x n, C is written ROW
    major m x n. All fp64 on the device.

    Zero rows of A and zero columns of B are rejected rather than handled:
    Algorithm 2 divides by the row maximum, so it has no meaning there, and
    the oracle's zero-extension is a separate construction (`lem:zero`).
    Returns false if it sees one, or if condition (5) fails. */
bool gemm(
    state        *s,
    int const     m,
    int const     n,
    double const *d_a,
    double const *d_b,
    double       *d_c,
    config const  cfg);

/*  C = A B with DF32 operands and a DF32 result, and no fp64 instruction
    anywhere in the chain -- this is the configuration the paper's
    fp64-free claim is about, and tools/sass_gate.sh is what checks it.

    A is ROW major m x k as a pair of float arrays with leading dimension
    `lda`, B is ROW major k x n with leading dimension `ldb`, and C is m x n
    with leading dimension `ldc`. The strides are there so the trailing
    update can read L21 and U12 in place out of the factorization's
    carrier, which is one n-strided array, instead of gathering them.

    With `subtract`, C is updated in place as C -= A B in DF32 rather than
    overwritten -- the trailing update's own subtraction, kept in this
    kernel for the reason the vendored epilogue keeps it there: the product
    is already in registers, and writing it out to read it back doubles the
    traffic on the largest array in the factorization.

    Same restrictions as gemm(): no zero row of A, no zero column of B.

    There is no fp64 counterpart to select here. The fp64 configuration
    reads fp64 operands and is reached through gemm(); the two differ in
    their inputs as well as their reconstruction, which is why they are
    separate entry points rather than a flag. */
bool gemm_df32(
    state       *s,
    int const    m,
    int const    n,
    float const *d_ah,
    float const *d_al,
    int const    lda,
    float const *d_bh,
    float const *d_bl,
    int const    ldb,
    float       *d_ch,
    float       *d_cl,
    int const    ldc,
    bool const   subtract);

/*  The intermediates Algorithm 2 produces, copied to the host. For the
    gate against the oracle, which compares mu and nu and the scaled
    integer operands before it compares any product. */
struct scaling_view {
    std::vector<int> mu;      /* m */
    std::vector<int> nu;      /* n */
    std::vector<int> ap;      /* m x k, row major   */
    std::vector<int> bp;      /* k x n, COLUMN major */
};

void copy_scaling(state const *s, scaling_view &out);

} /* namespace oii */
