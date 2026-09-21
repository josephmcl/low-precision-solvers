#pragma once

#include <cstddef>
#include <cstdint>

#include "df32.cuh"

/*  Ozaki-II: exact integer products through pairwise-coprime moduli.

    Where the sliced scheme splits each operand into k int8 pieces and pays
    k(k+1)/2 products, this represents the product modulo N small primes and
    pays N. Each residue GEMM is int8 in, int32 accumulate, exact; the true
    product is recovered from the residues by CRT provided it fits in
    P = prod p_r.

    Reconstruction is where fp64 does or does not enter, so it exists in two
    LABELLED configurations and they must never be conflated:

      crt_fp64  — accuracy reference. Uses fp64. NOT fp64-free.
      crt_df32  — the fp64-free combine. The SASS gate applies to this one.

    Garner's mixed-radix form is used rather than the textbook
    sum r_i M_i y_i mod P, because its digits are all < p_r and stay in int32:
    the entire modular part is exact integer arithmetic, and floating point
    appears only in the final weighted sum. That is what makes a fp64-free
    variant possible at all.

    RANGE. With the eight primes below and N = 8, log2(P) ~ 63.0, so the
    reconstruction is valid for |x| < P/2 and the fp64 combine cannot be
    exact there — 63 bits does not fit in 53. At N <= 5 it does. See
    .claude/ozaki2-crt.md. */

namespace ozaki2 {

constexpr int MAX_MODULI = 8;

/*  Primes just below 256, descending. Prime so pairwise coprime by
    construction; below 256 so a residue is a uint8 and feeds the int8 path
    directly. */
constexpr std::int32_t MODULI[MAX_MODULI] =
    {251, 241, 239, 233, 229, 227, 223, 211};

/*  log2(prod of the first n_moduli). N=3: 23.8, N=5: 39.5, N=8: 63.0. */
double product_bits(int const n_moduli);

/*  Inverses needed by Garner: table[j][i] = p_j^-1 mod p_i, for j < i.
    Filled once on the host, then read by the combine. */
struct garner_table {
    std::int32_t inverse[MAX_MODULI][MAX_MODULI] = {};
    std::int32_t p[MAX_MODULI]                   = {};
    /*  Mixed-radix digits of floor(P/2), the sign threshold. Held as digits
        so the comparison is exact integer work — see garner_signed. */
    std::int32_t half_digit[MAX_MODULI]          = {};
    int          n_moduli                        = 0;
};

garner_table make_garner_table(int const n_moduli);

/*  Mixed-radix digits of the value whose residues are `residue`.

    Exact: every intermediate is reduced mod p_i < 251, so products stay
    under 251^2 and int32 holds them without rounding. No floating point
    here, in either configuration. */
__host__ __device__ inline void garner_digits(
    std::int32_t const *residue,
    garner_table const &t,
    std::int32_t       *digit) {

    for (int i = 0; i != t.n_moduli; ++i) {

        std::int32_t const p = t.p[i];
        std::int32_t x = residue[i] % p;

        for (int j = 0; j != i; ++j) {
            x = (x - digit[j]) % p;
            if (x < 0)
                x += p;
            x = (x * t.inverse[j][i]) % p;
        }
        digit[i] = x;
    }
}

/*  Digits of |x| and its sign, decided ENTIRELY in integer arithmetic.

    The obvious alternative — combine the unsigned value in [0, P) and
    subtract P when it exceeds P/2 — is catastrophic and was measured so. A
    small negative reconstructs as P - |x|, which needs log2(P) bits; at
    N = 8 that is 63, so fp64 rounds it by ~P*2^-53 ~ 768 and the subtraction
    cancels everything but the rounding. Observed: relative error 1.0e3 on a
    true value of 1. No float representation narrower than log2(P) escapes
    it, which includes both configurations here.

    So the sign is settled by comparing mixed-radix digit vectors against
    floor(P/2) from the top down, and a negative value is re-derived from the
    complement residues (p_i - r_i) mod p_i, whose Garner digits are those of
    |x| directly. The combine then only ever sees a number of the true
    magnitude, and relative accuracy survives. */
__host__ __device__ inline bool garner_signed(
    std::int32_t const *residue,
    garner_table const &t,
    std::int32_t       *digit) {

    garner_digits(residue, t, digit);

    bool negative = false;
    for (int i = t.n_moduli - 1; i >= 0; --i) {
        if (digit[i] != t.half_digit[i]) {
            negative = digit[i] > t.half_digit[i];
            break;
        }
    }
    if (!negative)
        return false;

    std::int32_t complement[MAX_MODULI];
    for (int i = 0; i != t.n_moduli; ++i) {
        std::int32_t const p = t.p[i];
        complement[i] = (p - residue[i] % p) % p;
    }
    garner_digits(complement, t, digit);
    return true;
}

/*  Horner over the mixed radix: |x| = d0 + p0 (d1 + p1 (d2 + ...)).

    Evaluated innermost first, so the magnitude grows monotonically and each
    rounding is relative to the running value — total relative error ~N u. */
__host__ __device__ inline double crt_fp64(
    std::int32_t const *digit,
    garner_table const &t,
    bool const          negative) {

    double x = 0.;
    for (int i = t.n_moduli - 1; i >= 0; --i)
        x = x * static_cast<double>(t.p[i]) + static_cast<double>(digit[i]);

    return negative? -x : x;
}

/*  The same sum in double-float32. No fp64 instruction: the digits are
    int32, the radices are small integers exact in fp32, and every add and
    multiply is an error-free transformation on fp32 words.

    df_mul drops the lo*lo term, which is O(u_ff^2) relative and far below
    the floor this is reporting. */
__device__ inline df32 crt_df32(
    std::int32_t const *digit,
    garner_table const &t,
    bool const          negative) {

    df32 x = df_make(0.f, 0.f);
    for (int i = t.n_moduli - 1; i >= 0; --i) {
        x = df_mul(x, df_make(static_cast<float>(t.p[i]), 0.f));
        x = df_add(x, df_make(static_cast<float>(digit[i]), 0.f));
    }

    return negative? df_make(-x.hi, -x.lo) : x;
}

} /* namespace ozaki2 */
