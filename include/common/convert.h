#pragma once

#include "common/definitions.h"
#include "common/error.h"

#include <cuda_runtime.h>
#include <vector>

#include "common/problem.h"

#include <cstddef>

/*  Precision conversion and elementwise updates on device arrays.

    Every refinement method needs the same handful: demote an fp64 residual to
    fp32 for a triangular solve, promote the result back, permute rows to match
    a factorization, accumulate a correction. They were written twice — once
    per method — with identical bodies and different names, which is how two
    copies drift and how a fix lands in one of them.

    All operate on flat element counts, so a caller passes n*n or n*k and the
    shape stays its own business. Each is a grid-stride loop; none synchronizes,
    so a caller batching several pays one sync at the end rather than one
    apiece. */

namespace convert {

using harness::problem;

/*  d_out = (float) d_in, and back. The demotion is where a residual loses
    2^-24, so it belongs immediately before a solve that is itself fp32 and
    nowhere earlier — demoting a quantity the refinement is trying to resolve
    below that precision throws away exactly what it is looking for. */
void demote(
    float             *d_out,
    double const      *d_in,
    std::size_t const  n_elements,
    problem           &prob);

void promote(
    double            *d_out,
    float const       *d_in,
    std::size_t const  n_elements,
    problem           &prob);

void negate(
    float             *d_m,
    std::size_t const  n_elements,
    problem           &prob);

void zero(
    double            *d_x,
    std::size_t const  n_elements,
    problem           &prob);

/*  d_x += (double) d_d. The iterate stays fp64 across this; the correction
    arrives in fp32 because that is what the triangular solve produced. */
void add_correction(
    double            *d_x,
    float const       *d_d,
    std::size_t const  n_elements,
    problem           &prob);

/*  d_x -= d_y, both fp64. */
void subtract(
    double            *d_x,
    double const      *d_y,
    std::size_t const  n_elements,
    problem           &prob);

/*  Row-permute an n x k matrix by the composed permutation from a
    factorization: out(i, j) = in(perm[i], j).

    getrf yields P*A = L*U, so a correction solve needs P*r rather than r.
    The demoting form exists because the result feeds an fp32 solve and doing
    both in one pass halves the traffic. */
void permute_rows(
    double            *d_out,
    double const      *d_in,
    int const         *d_perm,
    std::size_t const  n,
    std::size_t const  k,
    problem           &prob);

void permute_rows_demote(
    float             *d_out,
    double const      *d_in,
    int const         *d_perm,
    std::size_t const  n,
    std::size_t const  k,
    problem           &prob);

/*  Sum of squares of an fp32 array, and of the difference of two fp64 arrays.
    Both are convergence measures rather than metrics, so they return the
    SQUARE of the norm and skip the root: a caller comparing a ratio against a
    tolerance does not need it, and the tolerance is then stated in squared
    terms, which is easy to get wrong by a factor of two in the exponent if the
    convention is not written down. It is written down here. */
double sum_squares_f32(
    float const       *d_m,
    std::size_t const  n_elements,
    problem           &prob);

double sum_squares_difference(
    double const      *d_x,
    double const      *d_y,
    std::size_t const  n_elements,
    problem           &prob);

} /* namespace convert */
