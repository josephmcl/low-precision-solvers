#pragma once

#include "common/error.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cmath>
#include <vector>

#include "common/definitions.h"
#include "common/problem.h"

#include <cstddef>

namespace metrics {

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
    double backward = 0.;

    /*  ||PB - A*X||_F / ||PB||_F. The same residual against the
        right-hand side. Larger than `backward` by ||A||_F ||X||_F / ||PB||_F
        and reported only so that a number quoted in this normalization
        elsewhere can be matched up. */
    double relative = 0.;

    /*  ||X - X_ref||_F / ||X_ref||_F against a supplied fp64 reference.
        Conditioning amplifies this differently for different methods (the
        forward-to-backward ratio varied 4.6x across two methods on one
        matrix), so forward error alone does not order methods. Zero when no
        reference was given. */
    double forward = 0.;

    /*  PER-RIGHT-HAND-SIDE Rigal-Gaches normwise backward error,

            eta_j = ||b_j - A x_j||_2 / (||A||_F ||x_j||_2 + ||b_j||_2),

        reported as the maximum and median over the k columns.

        WHY BOTH THIS AND `backward`. `backward` aggregates the whole block
        through Frobenius norms and omits the ||b|| term, so it is neither the
        standard metric nor able to expose a single badly-solved column inside
        a large multi-RHS batch -- at k=2048 one bad column moves an aggregate
        Frobenius norm by 2%. eta_max is the number a numerical-analysis
        reader expects, and eta_max/eta_median is the check that the aggregate
        is not hiding anything.

        Including ||b_j||_2 in the denominator is what makes this Rigal-Gaches
        rather than a residual ratio: it admits perturbations to b as well as
        A, which is why the quantity cannot fall below u regardless of how
        favourably ||A|| ||x|| happens to scale against ||b||. That property
        is why it resolves the sub-u_64 readings the aggregate metric
        produces on diagonally dominant systems.

        ||A|| is Frobenius, matching norm_a; state it when quoting. */
    double rg_max    = 0.;
    double rg_median = 0.;

    double norm_a = 0.;
    double norm_x = 0.;
};

/*  Residual measures for a computed X, against prob.d_a in fp64.

    d_x_ref may be nullptr, in which case report.forward stays zero.

    A caveat worth carrying: a method that refines against its *own* fp64
    copy of A can report a backward error below u_64, because it is fitting X
    to that copy rather than to the exact matrix. Such a number is not a
    backward error against A and should not be compared with one.

    But sub-u_64 values are not automatically that artifact. On a diagonally
    dominant matrix ||A||_F ||X||_F exceeds ||PB||_F by ~150x, so the same
    residual reads ~150x smaller in this normalization than in the relative
    one, and 1e-17 backward against 1e-15 relative is simply the scaling. The
    two together tell you which case you are in: an artifact shows up as a
    backward error far below what the relative column and the norms imply. */
report evaluate(
    double const *d_x,
    double const *d_x_ref,
    problem      &prob);

/*  ||M||_F for an n x m fp64 device matrix, and the Frobenius norm of the
    difference of two. Exposed because a method validating its own
    intermediate (R against PA - LU, say) wants the same reduction the
    metrics use rather than a second one that rounds differently. */
double norm(
    double const      *d_m,
    std::size_t const  n_elements,
    problem           &prob);

double norm_difference(
    double const      *d_m,
    double const      *d_n,
    std::size_t const  n_elements,
    problem           &prob);

} /* namespace metrics */
