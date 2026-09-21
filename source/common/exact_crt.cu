#include "common/exact_crt.h"

#include <cmath>

namespace exact_crt {

namespace {

/*  a^-1 mod m by the extended Euclidean algorithm. m is prime and a < m, so
    the inverse always exists. */
std::int32_t mod_inverse(std::int32_t const a, std::int32_t const m) {

    std::int32_t t = 0, new_t = 1;
    std::int32_t r = m, new_r = a;

    while (new_r != 0) {
        std::int32_t const q = r / new_r;
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

garner_table make_garner_table(int const n_moduli) {

    garner_table t;
    t.n_moduli = (n_moduli < MAX_MODULI)? n_moduli : MAX_MODULI;

    for (int i = 0; i != t.n_moduli; ++i)
        t.p[i] = MODULI[i];

    for (int i = 0; i != t.n_moduli; ++i)
        for (int j = 0; j != i; ++j)
            t.inverse[j][i] = mod_inverse(t.p[j] % t.p[i], t.p[i]);

    /*  Digits of floor(P/2), the sign threshold. P fits in uint64 for
        N <= 8, so this is exact integer work and owes nothing to a double. */
    unsigned long long p_all = 1ull;
    for (int i = 0; i != t.n_moduli; ++i)
        p_all *= static_cast<unsigned long long>(t.p[i]);

    unsigned long long const half = p_all >> 1;
    std::int32_t residue[MAX_MODULI];
    for (int i = 0; i != t.n_moduli; ++i)
        residue[i] = static_cast<std::int32_t>(
            half % static_cast<unsigned long long>(t.p[i]));

    garner_digits(residue, t, t.half_digit);

    return t;
}

} /* namespace exact_crt */
