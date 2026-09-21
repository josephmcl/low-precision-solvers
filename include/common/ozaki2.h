#pragma once

#include <cmath>
#include <cstdint>

/*  Ozaki-II, accurate mode: arXiv:2602.02549 Algorithms 1-3.

    Ported to match the campaign's oracle, core_exp/ozaki2.py in the
    simulation repo, which is faithful to the source. Fidelity is the point
    rather than elegance: Theorem 2 bounds ALGORITHM 3, whose reduction runs
    in fp64 against double-double constants, and its R_64 term IS that path's
    rounding error. Reconstructing exactly instead computes a different and
    better object, and the envelope test then passes vacuously. An exact
    variant is kept, clearly labelled, in exact_crt.h.

    Where the sliced scheme splits each operand into k int8 pieces and pays
    k(k+1)/2 products, this reduces the operands modulo N pairwise-coprime
    moduli and pays N. Each residue product is exact in int32.

    Matched operating points against the sliced arm, from the campaign's
    phase9 data, paired on DELIVERED ACCURACY rather than cost or bits:
    (k=1, N=3), (k=2, N=5), (k=4, N=8). */

namespace ozaki2 {

constexpr int MAX_MODULI = 8;

/*  The source's list, largest first. Pairwise coprime but NOT prime —
    256 = 2^8 fills a uint8 exactly, and the source notes that the int32
    wrap it can cause is harmless because 2^31 = -2^31 (mod 256). */
constexpr std::int32_t MODULI[MAX_MODULI] =
    {256, 255, 253, 251, 247, 241, 239, 233};

/*  Section 3.1. P1 + P2 ~ P and s_l1 + s_l2 ~ (P/p_l) q_l are double-double
    because neither fits an fp64 word: at N = 8, log2(P) = 64 and P2 = 256
    exactly. At N <= 5 both corrections are zero and the pairs are inert —
    the dd machinery only earns its place at the top of the range. */
struct crt_constants {
    std::int32_t p[MAX_MODULI]  = {};
    double       s1[MAX_MODULI] = {};
    double       s2[MAX_MODULI] = {};
    double       P1             = 0.;
    double       P2             = 0.;
    double       Pinv           = 0.;
    long long    rho            = 0;
    int          n_moduli       = 0;
};

crt_constants make_crt_constants(int const n_moduli);

/*  log2(prod of the first n_moduli): 24.0 at N=3, 40.0 at N=5, 64.0 at N=8. */
double product_bits(int const n_moduli);

/*  The source's symmetric range, -floor(p/2) <= r <= floor(p/2). Not the
    non-negative representative: the residue products are signed and the
    envelope is stated against this convention. */
__host__ __device__ inline std::int64_t sym_mod(
    std::int64_t const x,
    std::int32_t const p) {

    std::int64_t r = x % p;
    if (r < 0)
        r += p;
    return (r > p / 2)? r - p : r;
}

/*  Algorithm 3, lines 9-12, for one output element.

    w[l] is sym_mod(sum_k A_l[i,k] B_l[k,j], p_l) — the residue product,
    exact in int32, carried here as a double.

    The two fmas are the source's, and they are load-bearing: the reduction
    subtracts Q P from a value of the same magnitude, so without the exact
    products the cancellation would take the answer with it. */
__host__ __device__ inline double crt_fp64(
    double const         *w,
    crt_constants const  &c) {

    double c1 = 0., c2 = 0.;
    for (int l = 0; l != c.n_moduli; ++l) {
        c1 += c.s1[l] * w[l];
        c2 += c.s2[l] * w[l];
    }

    double const q     = rint(c1 * c.Pinv);
    double const inner = fma(-q, c.P1, c1);
    return fma(-q, c.P2, inner + c2);
}

} /* namespace ozaki2 */
