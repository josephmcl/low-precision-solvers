#pragma once

#include "gpu/definitions.h"
#include "gpu/problem.h"

#include <cstddef>

namespace metrics {

using type::real_t;
using harness::problem;

/*  Both error measures, with their normalizations stated.

    State the denominator or the number means nothing. The same solution,
    reported as a relative residual and as a backward error, differed by a
    factor of 2.3e4 in the work this harness replaces, and the gap was read
    as a precision bug for some time. Both are computed here from the same
    residual so the pair can never disagree about which solution it
    describes, and the reporter prints both. */
struct report {

    /*  ||PB - A*X||_F / (||A||_F ||X||_F). The scale-free measure of how
        far the computed X is from solving *some* nearby system, and the one
        to compare methods on. */
    real_t backward = 0.;

    /*  ||PB - A*X||_F / ||PB||_F. The same residual against the
        right-hand side. Larger than `backward` by ||A||_F ||X||_F / ||PB||_F
        and reported only so that a number quoted in this normalization
        elsewhere can be matched up. */
    real_t relative = 0.;

    /*  ||X - X_ref||_F / ||X_ref||_F against a supplied fp64 reference.
        Conditioning amplifies this differently for different methods (the
        forward-to-backward ratio varied 4.6x across two methods on one
        matrix), so forward error alone does not order methods. Zero when no
        reference was given. */
    real_t forward = 0.;

    real_t norm_a = 0.;
    real_t norm_x = 0.;
};

/*  Residual measures for a computed X, against prob.d_a in fp64.

    d_x_ref may be nullptr, in which case report.forward stays zero.

    A caveat worth carrying: a method that refines against its *own* fp64
    copy of A can report a backward error below u_64, because it is fitting
    X to that copy rather than to the exact matrix. Such a number is not a
    backward error against A and should not be compared with one. Values
    materially under 1e-16 here are that artifact, not accuracy. */
report evaluate(
    real_t const *d_x,
    real_t const *d_x_ref,
    problem      &prob);

/*  ||M||_F for an n x m fp64 device matrix, and the Frobenius norm of the
    difference of two. Exposed because a method validating its own
    intermediate (R against PA - LU, say) wants the same reduction the
    metrics use rather than a second one that rounds differently. */
real_t norm(
    real_t const      *d_m,
    std::size_t const  n_elements,
    problem           &prob);

real_t norm_difference(
    real_t const      *d_m,
    real_t const      *d_n,
    std::size_t const  n_elements,
    problem           &prob);

} /* namespace metrics */
