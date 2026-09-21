#include "common/ozaki2.h"

#include <cmath>

namespace ozaki2 {

namespace {

using u128 = unsigned __int128;
using i128 = __int128;

int bit_length(u128 v) {

    int n = 0;
    while (v != 0) {
        v >>= 1;
        ++n;
    }
    return n;
}

double to_double(u128 const v) {

    /*  Split at 64 so the conversion goes through two exactly-representable
        halves rather than relying on a 128-bit conversion path. */
    double const hi = static_cast<double>(static_cast<std::uint64_t>(v >> 64));
    double const lo = static_cast<double>(static_cast<std::uint64_t>(v));
    return hi * 18446744073709551616. + lo;
}

/*  a^-1 mod m by extended Euclid. The moduli are NOT all prime, so Fermat
    does not apply; the inverse exists because the moduli are pairwise
    coprime, which makes gcd(P/p_l, p_l) = 1. */
std::int32_t mod_inverse(std::int32_t const a, std::int32_t const m) {

    std::int32_t t = 0, new_t = 1;
    std::int32_t r = m, new_r = a;

    while (new_r != 0) {
        std::int32_t const q  = r / new_r;
        std::int32_t const tt = t - q * new_t;
        t = new_t; new_t = tt;
        std::int32_t const rr = r - q * new_r;
        r = new_r; new_r = rr;
    }

    if (t < 0)
        t += m;
    return t;
}

} /* anonymous namespace */

double product_bits(int const n_moduli) {

    double bits = 0.;
    for (int i = 0; i != n_moduli && i != MAX_MODULI; ++i)
        bits += std::log2(static_cast<double>(MODULI[i]));

    return bits;
}

crt_constants make_crt_constants(int const n_moduli) {

    crt_constants c;
    c.n_moduli = (n_moduli < MAX_MODULI)? n_moduli : MAX_MODULI;

    u128 P = 1;
    for (int i = 0; i != c.n_moduli; ++i) {
        c.p[i] = MODULI[i];
        P *= static_cast<u128>(c.p[i]);
    }

    /*  rho = sum floor(p_l / 2), the source's bound on the residue sum. */
    for (int i = 0; i != c.n_moduli; ++i)
        c.rho += c.p[i] / 2;

    c.P1 = to_double(P);
    c.P2 = static_cast<double>(
        static_cast<i128>(P) - static_cast<i128>(static_cast<u128>(c.P1)));
    c.Pinv = 1. / to_double(P);

    /*  v_l = (P/p_l) q_l, with q_l = (P/p_l)^-1 mod p_l. */
    u128 v[MAX_MODULI];
    int  len[MAX_MODULI];
    int  top = 0;
    for (int i = 0; i != c.n_moduli; ++i) {
        u128 const m = P / static_cast<u128>(c.p[i]);
        std::int32_t const residue =
            static_cast<std::int32_t>(m % static_cast<u128>(c.p[i]));
        std::int32_t const q = mod_inverse(residue, c.p[i]);
        v[i]   = m * static_cast<u128>(q);
        len[i] = bit_length(v[i]);
        if (len[i] - 1 > top)
            top = len[i] - 1;
    }

    /*  s_l1 keeps the top beta_l bits of v_l; s_l2 is the exact remainder.
        beta_l is the source's, sized so the fp64 sum of rho terms cannot
        overflow the significand. */
    int ceil_log2_rho = bit_length(static_cast<u128>(c.rho - 1));

    for (int i = 0; i != c.n_moduli; ++i) {
        int beta = 53 - ceil_log2_rho + (len[i] - 1) - top;
        if (beta < 1)  beta = 1;
        if (beta > 53) beta = 53;

        int const shift = (len[i] - beta > 0)? len[i] - beta : 0;
        u128 const hi = (v[i] >> shift) << shift;

        c.s1[i] = to_double(hi);
        c.s2[i] = to_double(v[i] - hi);
    }

    return c;
}

} /* namespace ozaki2 */
