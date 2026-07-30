#pragma once

#include "common/convert.h"
#include "common/definitions.h"
#include "common/error.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <algorithm>
#include <vector>

#include "common/problem.h"

#include <cstddef>

/*  The fp32 factorization every refinement method here is built on.

    All three low-precision schemes start the same way: demote A, factor it in
    fp32, and reduce getrf's sequential interchanges to a permutation vector.
    That was written out twice, including the interchange composition, which is
    the part easiest to get subtly wrong and hardest to notice — a permutation
    that is wrong in a few entries still produces a plausible-looking answer
    that refinement partially repairs. */

namespace factorize {

using harness::problem;

/*  Workspace size for lu_fp32, in floats. */
int buffer_size(problem &prob);

/*  d_lu <- L, U packed, from fl32(A). d_ipiv is getrf's raw interchange list;
    d_perm is that composed into a permutation, which is what a solve needs.

    Buffers are caller-owned so that nothing allocates inside a timed region:
    an allocator call there is charged to the arithmetic, and cudaMalloc of a
    few hundred MB is not cheap.

    d_a is the untouched fp64 reference and is only read. */
void lu_fp32(
    float             *d_lu,
    int               *d_ipiv,
    int               *d_perm,
    float             *d_work,
    int               *d_info,
    double const      *d_a,
    problem           &prob);

/*  Both triangular solves of the packed factor, in place: y <- U^-1 L^-1 y.

    tf32 lowers them to tensor-core precision. That is correct only where the
    solve's precision does not reach the answer — a fixed-point iteration whose
    outer passes merely converge toward it, never the solve that produces the
    delivered correction.

    The mode is set and restored around the calls rather than left on the
    handle. Setting it globally silently degraded every unrelated Sgemm and
    Strsm in earlier work; it is a property of the call site. */
void lu_solve(
    float             *d_y,
    float const       *d_lu,
    std::size_t const  k,
    problem           &prob,
    bool const         tf32 = false);

} /* namespace factorize */
