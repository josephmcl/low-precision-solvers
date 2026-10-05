#pragma once

#include "common/problem.h"

/*  A known solution, an accurately formed right-hand side, a reference
    solution, and the measured conditioning, for the timing tables.

    The harness's own right-hand side is random and its forward error is taken
    against the FP64 direct solve, which is itself in error by about
    kappa u_64. Above kappa ~ 1e12 that reference is not accurate enough to
    compare methods at equal accuracy, so these replace it:

      x_true    uniform(-1, 1), fixed by the seed;
      b         A x_true accumulated in double-double, rounded to double;
      x_ref     the solution of A x = b (b as stored) by an FP64 LU and
                refinement with double-double residuals, the iterate kept in
                double-double.

    None of this is timed; it runs before the methods. */
namespace harness {

/*  Overwrites prob.d_b, every column, with fl(A x_true); d_xtrue receives
    x_true (n x k, column major). */
void make_xtrue_rhs(problem &prob, double *d_xtrue, unsigned seed);

struct reference_info {
    int    iterations      = 0;     /* refinement steps after the first solve      */
    double last_correction = 0.;    /* ||d||_inf / ||x||_inf of the last step       */
    double err_vs_xtrue    = 0.;    /* ||x_ref - x_true||_inf / ||x_true||_inf      */
    bool   converged       = false; /* last correction below 1e-24 (dd refinement stalls near 1e-27) */
};

/*  x_ref for every column of prob.d_b. */
reference_info reference_solution(problem &prob, double const *d_xtrue, double *d_xref);

struct conditioning {
    double sigma_max = 0., sigma_min = 0., kappa_2 = 0.;   /* power / inverse iteration */
    double kappa_inf = 0., cond_ax = 0.;                   /* explicit FP64 inverse     */
    int    iterations = 0;
};

/*  kappa_2 from sigma_max (power iteration on A^T A) and sigma_min (inverse
    iteration through the FP64 LU), kappa_inf = ||A||_inf ||A^-1||_inf and
    cond(A, x) = || |A^-1| |A| |x| ||_inf / ||x||_inf from the explicit FP64
    inverse (accurate to about kappa u_64). d_x is the first column used for
    cond(A, x). */
conditioning measure_conditioning(problem &prob, double const *d_x);

/*  Norms of b - A x for column `column` of prob.d_b, the residual accumulated
    in double-double; x = d_xh + d_xl (d_xl may be null). */
struct residual_norms { double r_inf = 0., r_2 = 0., x_inf = 0., x_2 = 0.; };
residual_norms dd_residual(problem &prob, double const *d_xh, double const *d_xl, std::size_t column);

} /* namespace harness */
