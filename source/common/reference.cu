#include "common/reference.h"

#include <algorithm>
#include <cmath>

namespace reference {

namespace {

/*  Each square is exact (two_prod on equal arguments), so only the summation
    rounds: ~n^2 u_dd, about 1e-24 relative at n = 8192. */
dd sum_squares(
    double const      *m,
    std::size_t const  n_elements) {

    dd acc = make(0., 0.);
    for (std::size_t i = 0; i != n_elements; ++i)
        acc = add(acc, two_prod(m[i], m[i]));

    return acc;
}

dd sum_squares_dd(std::vector<dd> const &v) {

    dd acc = make(0., 0.);
    for (std::size_t i = 0; i != v.size(); ++i)
        acc = add(acc, mul(v[i], v[i]));

    return acc;
}

/*  Narrowed on purpose: a sum of squares does not cancel, so the root is
    wanted only to fp64 relative accuracy. */
double root(dd const x) {return std::sqrt(to_double(x));}

} /* anonymous namespace */

double norm_frobenius(
    double const      *m,
    std::size_t const  n_elements) {

    return root(sum_squares(m, n_elements));
}

report backward_error(
    double const                   *a,
    double const                   *b,
    double const                   *x,
    double const                   *x_lo,
    std::size_t const               n,
    std::size_t const               k,
    std::vector<std::size_t> const &columns) {

    report out;
    out.norm_a = norm_frobenius(a, n * n);

    std::vector<dd> r(n);

    for (std::size_t c = 0; c != columns.size(); ++c) {

        std::size_t const j = columns[c];
        if (j >= k)
            continue;

        for (std::size_t i = 0; i != n; ++i)
            r[i] = make(b[i + j * n], 0.);

        /*  (q outer, i inner), not the dot-product order: a is column major,
            so A[i,q] = a[i + q*n] walks contiguous memory on the inner index.
            The dot order strides by n on every access. */
        for (std::size_t q = 0; q != n; ++q) {

            double const xq_hi = x[q + j * n];
            double const xq_lo = (x_lo != nullptr)? x_lo[q + j * n] : 0.;

            for (std::size_t i = 0; i != n; ++i) {

                /*  Both products exact, so the only rounding in the residual
                    is the dd accumulation. */
                double const aiq = a[i + q * n];
                dd p = two_prod(aiq, xq_hi);
                if (xq_lo != 0.)
                    p = add(p, two_prod(aiq, xq_lo));

                r[i] = sub(r[i], p);
            }
        }

        dd acc_x = make(0., 0.);
        dd acc_b = make(0., 0.);
        for (std::size_t i = 0; i != n; ++i) {
            dd const xi = make(x[i + j * n],
                               (x_lo != nullptr)? x_lo[i + j * n] : 0.);
            acc_x = add(acc_x, mul(xi, xi));
            acc_b = add(acc_b, two_prod(b[i + j * n], b[i + j * n]));
        }

        column_error e;
        e.column = j;
        e.norm_r = root(sum_squares_dd(r));
        e.norm_x = root(acc_x);
        e.norm_b = root(acc_b);

        double const denominator = out.norm_a * e.norm_x + e.norm_b;
        e.eta = (denominator > 0.)? e.norm_r / denominator : 0.;

        out.columns.push_back(e);
    }

    return out;
}

double eta_max(report const &r) {

    double m = 0.;
    for (std::size_t i = 0; i != r.columns.size(); ++i)
        m = std::max(m, r.columns[i].eta);

    return m;
}

double eta_median(report const &r) {

    if (r.columns.empty())
        return 0.;

    std::vector<double> e;
    e.reserve(r.columns.size());
    for (std::size_t i = 0; i != r.columns.size(); ++i)
        e.push_back(r.columns[i].eta);

    std::sort(e.begin(), e.end());
    std::size_t const m = e.size();

    return (m % 2 == 1)? e[m / 2] : 0.5 * (e[m / 2 - 1] + e[m / 2]);
}

double contamination_size(double const floor) {

    /*  sqrt(n) u_64 = floor  =>  n = (floor / u_64)^2. */
    double const ratio = floor / U_64;
    return ratio * ratio;
}

double resolvable_floor(std::size_t const n) {

    return std::sqrt(static_cast<double>(n)) * U_64;
}

} /* namespace reference */
