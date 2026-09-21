#pragma once

#include <cmath>
#include <cstddef>
#include <vector>

/*  Double-double reference arithmetic, out of band.

    An fp64 residual carries ~sqrt(n) * u_64 of its own error, which reaches
    the c * u_ff floors this project reports at n ~ 1600 — past that an
    fp64-referenced accuracy number is measuring the reference. Double-double
    gives u_dd ~ 1.2e-32 and settles it. Host only; never in the solve path.

    Requires IEEE round-to-nearest with no reassociation: do not build under
    -ffast-math. See .claude/reference-instrument.md. */

namespace reference {

/*  Unevaluated pair, value hi + lo, |lo| <= ulp(hi)/2. */
struct dd {
    double hi = 0.;
    double lo = 0.;
};

constexpr double U_64 = 1.1102230246251565e-16;   /* 2^-53  */
constexpr double U_DD = 1.232595164407831e-32;    /* 2^-106 */
constexpr double U_FF = 1.7763568394002505e-15;   /* 2^-49, the DF32 carrier */

inline dd make(double const hi, double const lo) {
    dd r;
    r.hi = hi;
    r.lo = lo;
    return r;
}

inline double to_double(dd const x) {return x.hi + x.lo;}

/*  ---- error-free transformations -------------------------------------- */

/*  Knuth TwoSum: s + e == a + b exactly, no ordering assumption. */
inline dd two_sum(double const a, double const b) {

    double const s  = a + b;
    double const bb = s - a;
    double const e  = (a - (s - bb)) + (b - bb);
    return make(s, e);
}

/*  Dekker FastTwoSum: same, 3 flops, requires |a| >= |b| or a == 0. */
inline dd fast_two_sum(double const a, double const b) {

    double const s = a + b;
    double const e = b - (s - a);
    return make(s, e);
}

/*  TwoProd via FMA: p + e == a * b exactly. */
inline dd two_prod(double const a, double const b) {

    double const p = a * b;
    double const e = std::fma(a, b, -p);
    return make(p, e);
}

/*  ---- composite double-double operations ------------------------------ */

/*  AccurateDWPlusDW (Joldes-Muller-Popescu 2017, Alg. 6), rel. err <= 3u^2.
    Accurate under cancellation, which b - A x is; the sloppy double-word sum
    is deliberately not provided here. */
inline dd add(dd const x, dd const y) {

    dd const s = two_sum(x.hi, y.hi);
    dd const t = two_sum(x.lo, y.lo);
    dd const v = fast_two_sum(s.hi, s.lo + t.hi);
    return fast_two_sum(v.hi, t.lo + v.lo);
}

inline dd negate(dd const x) {return make(-x.hi, -x.lo);}

inline dd sub(dd const x, dd const y) {return add(x, negate(y));}

/*  DWTimesDW (JMP 2017, Alg. 12), rel. err <= 4u^2. */
inline dd mul(dd const x, dd const y) {

    dd     const c  = two_prod(x.hi, y.hi);
    double const t0 = x.lo * y.lo;
    double const t1 = std::fma(x.hi, y.lo, t0);
    double const t2 = std::fma(x.lo, y.hi, t1);
    return fast_two_sum(c.hi, c.lo + t2);
}

/*  DWTimesFP (JMP 2017, Alg. 7), rel. err <= 2u^2. */
inline dd mul_fp64(dd const x, double const a) {

    dd     const c = two_prod(x.hi, a);
    double const t = std::fma(x.lo, a, c.lo);
    return fast_two_sum(c.hi, t);
}

/*  ---- reference quantities -------------------------------------------- */

/*  Per-column normwise Rigal-Gaches backward error,

        eta_j = ||b_j - A x_j||_2 / (||A||_F ||x_j||_2 + ||b_j||_2),

    the same quantity and normalization metrics::report emits, so the two are
    comparable column by column. */
struct column_error {
    std::size_t column = 0;
    double      eta    = 0.;
    double      norm_r = 0.;   /* ||b_j - A x_j||_2 */
    double      norm_x = 0.;   /* ||x_j||_2         */
    double      norm_b = 0.;   /* ||b_j||_2         */
};

struct report {
    double                    norm_a = 0.;   /* ||A||_F */
    std::vector<column_error> columns;
};

/*  ||M||_F for a host fp64 array, accumulated in double-double. */
double norm_frobenius(
    double const      *m,
    std::size_t const  n_elements);

/*  Backward errors for the listed right-hand sides.

    a is n x n column major, matching harness::problem's d_a; b and x are
    n x k column major. x_lo is the second word of a two-word solution, or
    nullptr for a plain fp64 one (float -> double widening is exact, so a
    DF32 caller may pass both).

    Sampled rather than whole-block: one column costs n^2 dd operations,
    ~2-3 s at n = 8192. */
report backward_error(
    double const                   *a,
    double const                   *b,
    double const                   *x,
    double const                   *x_lo,
    std::size_t const               n,
    std::size_t const               k,
    std::vector<std::size_t> const &columns);

double eta_max(report const &r);

double eta_median(report const &r);

/*  Size at which sqrt(n) u_64 reaches `floor`, i.e. where an fp64-formed
    residual stops resolving it. 1600 for the DF32 floor at c = 2.5. */
double contamination_size(double const floor);

/*  Smallest floor an fp64 reference still resolves at size n. */
double resolvable_floor(std::size_t const n);

} /* namespace reference */
