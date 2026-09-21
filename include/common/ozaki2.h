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
constexpr int MAX_LIMB = 4;

struct crt_constants {
    std::int32_t p[MAX_MODULI]  = {};
    double       s1[MAX_MODULI] = {};
    double       s2[MAX_MODULI] = {};
    double       P1             = 0.;
    double       P2             = 0.;
    double       Pinv           = 0.;
    long long    rho            = 0;
    int          n_moduli       = 0;

    /*  fp32 limb splits for the fp64-free configuration, most significant
        first, each limb exact. Built on the host from the same exact
        integers the fp64 constants come from — setup, not solve arithmetic.
        s_l1 carries beta_l <= 43 significant bits, so two limbs hold it;
        P needs three. */
    float s1_limb[MAX_MODULI][MAX_LIMB] = {};
    float s2_limb[MAX_MODULI][MAX_LIMB] = {};
    float P_limb[MAX_LIMB]              = {};
    float Pinv_f                        = 0.f;
    float half_P_f                      = 0.f;

    /*  Algorithm 2's two scalars, both single_triangle_down of an fp64
        expression and therefore fp32 values carried in fp64 words:
        P' = log2(P-1)/2 - 0.5, and the step coefficient -0.5/(1-4u32).
        Directed rounding is load-bearing -- the shifts they produce must
        never overshoot condition (5) -- so they are computed once on the
        host and not re-derived per launch. */
    double Pprime                       = 0.;
    double c_step                       = 0.;
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

/*  ---- fp64-free configuration -----------------------------------------

    A float expansion: x[0] MOST significant, the value is the plain sum of
    the limbs. Kept non-overlapping by two_sum cascades rather than by a
    renormalization pass — the companion work measured that a naive 3-limb
    renormalization drops a significant limb under cancellation
    (FINDINGS_Rblock.md section 8), so there is none here.

    MAX_LIMB slots with the overflow reported rather than discarded: if
    `grow` returns nonzero the accumulator was too narrow and the caller
    must know, instead of silently losing the top. */
struct fexp {
    float x[MAX_LIMB];
};

__host__ __device__ inline fexp fexp_zero() {
    fexp e;
    for (int i = 0; i != MAX_LIMB; ++i)
        e.x[i] = 0.f;
    return e;
}

/*  Knuth TwoSum on fp32, host-callable (df32.cuh's is __device__ only). */
__host__ __device__ inline void two_sum32(
    float const a, float const b, float &s, float &err) {

    s = a + b;
    float const bb = s - a;
    err = (a - (s - bb)) + (b - bb);
}

/*  Cascaded add: each limb absorbs what it can and passes its rounding
    error down to the next. What falls off the end is the low-order
    remainder that did not fit, and is returned so a too-narrow accumulator
    is visible.

    Note the orientation. Shewchuk's grow-expansion carries the running SUM
    forward and emits the errors; written that way with a fixed limb count
    the carry out is the bulk of the value, not the tail — the first version
    here did exactly that and discarded it, which showed up as `lost` of
    2.6e14 against a result of the same order. Largest-first with the error
    cascading down is the form that truncates safely. */
__host__ __device__ inline float grow(fexp &e, float b, int const limbs) {

    for (int i = 0; i != limbs; ++i) {
        float s, err;
        two_sum32(e.x[i], b, s, err);
        e.x[i] = s;
        b = err;
    }
    return b;
}

/*  Exact product of an fp32 word with a small exact integer, added in. */
__host__ __device__ inline float grow_product(
    fexp &e, float const a, float const w, int const limbs) {

    float const p  = a * w;
    float const pe = fmaf(a, w, -p);      /* exact: TwoProd's error term */
    float lost = grow(e, pe, limbs);
    lost += grow(e, p, limbs);
    return lost;
}

/*  Sum of the limbs as a df32 pair, smallest first so the small terms are
    not swallowed by the leading one. */
__host__ __device__ inline void fexp_to_pair(
    fexp const &e, int const limbs, float &hi, float &lo) {

    float s = 0.f, c = 0.f;
    for (int i = limbs - 1; i >= 0; --i) {
        float t, err;
        two_sum32(s, e.x[i], t, err);
        s = t;
        c += err;
    }
    two_sum32(s, c, hi, lo);
}

/*  Algorithm 3's reconstruction with no fp64 instruction.

    C1 and C2 are accumulated EXACTLY — measured: the source's beta_l rule
    makes them exact in fp64 too, so computing them exactly is faithful, not
    a departure. The rounding Theorem 2's R_64 bounds lives in the reduction
    below, which is where the two configurations part company.

    `lost` reports any limb that did not fit, so a too-narrow accumulator is
    visible rather than silent. */
/*  w carries the residues, which satisfy |w_l| <= floor(p_l/2) <= 128 and
    are therefore exact in fp32. It is a float array and not a double one
    on purpose: the first version took `double const *` and the narrowing
    cast compiled to a single F2F.F32.F64, one fp64 instruction in 3472 and
    the only one in the kernel. The fp64 configuration keeps its double
    carrier; this path never sees one. */
__host__ __device__ inline void crt_fp32free(
    float const         *w,
    crt_constants const &c,
    int const            limbs,
    float               &hi,
    float               &lo,
    float               &lost) {

    fexp e1 = fexp_zero(), e2 = fexp_zero();
    lost = 0.f;

    for (int l = 0; l != c.n_moduli; ++l) {
        float const wl = w[l];
        for (int j = 0; j != limbs; ++j) {
            if (c.s1_limb[l][j] != 0.f)
                lost += grow_product(e1, c.s1_limb[l][j], wl, limbs);
            if (c.s2_limb[l][j] != 0.f)
                lost += grow_product(e2, c.s2_limb[l][j], wl, limbs);
        }
    }

    /*  Q = round(C1 / P), bounded by rho (~984 at N = 8) and hence a small
        exact integer. An fp32 estimate is not reliable to half a unit — the
        product alone rounds at ~6e-5 — so the estimate is CORRECTED against
        the range condition instead of trusted. Condition (5) guarantees
        |C''| < P/2 with margin, so at most a couple of steps are needed and
        the comparison never sits on the boundary. */
    float q1, q2;
    fexp_to_pair(e1, limbs, q1, q2);
    float const q = rintf((q1 + q2) * c.Pinv_f);

    fexp r = e1;
    for (int j = 0; j != limbs; ++j)
        if (e2.x[j] != 0.f)
            lost += grow(r, e2.x[j], limbs);
    for (int j = 0; j != limbs; ++j)
        if (c.P_limb[j] != 0.f)
            lost += grow_product(r, -c.P_limb[j], q, limbs);

    for (int step = 0; step != 4; ++step) {
        fexp_to_pair(r, limbs, hi, lo);
        float const sign = (hi > c.half_P_f)? -1.f
                         : (hi < -c.half_P_f)? 1.f : 0.f;
        if (sign == 0.f)
            break;
        for (int j = 0; j != limbs; ++j)
            if (c.P_limb[j] != 0.f)
                lost += grow_product(r, sign * c.P_limb[j], 1.f, limbs);
    }

    fexp_to_pair(r, limbs, hi, lo);
}

} /* namespace ozaki2 */
