#pragma once

#include <cstddef>

/*  The INT-sliced factorization, vendored from nfp64gmresir and wrapped.

    Blocked right-looking LU with partial pivoting on a DF32 carrier; the
    trailing update slices L21 per row and U12 per column to `kfac` int8
    slices, multiplies them exactly in int32 on tensor cores, and folds the
    result back through a DF32 epilogue. Refinement is LU-IR with a DF32
    residual, so no fp64 arithmetic appears in the solve.

    EXACTLY ONE translation unit may include the vendor header: it defines
    __global__ kernels and file-scope mutable state in a header, so a second
    includer is a duplicate symbol at device link and two disconnected copies
    of the scratch pointers. That TU is source/gpu/factor_int8lu.cu, which is
    also compiled -c rather than -dc — see the Makefile. */

namespace int8lu_arm {

struct state;

/*  Resident matrix bytes per n^2: 8 for the captured residual operator plus
    8 for the factor carrier. The factorization consumes its carrier, so the
    operator the refinement needs cannot be the same array. */
constexpr double STORAGE_N2 = 16.;

/*  Allocate for an n x n solve at panel width b (256 is the tuned value) and
    slice depth kfac. Returns null on failure. */
state *create(
    std::size_t const n,
    int const         b,
    int const         kfac);

void destroy(state *s);

/*  Capture the residual operator and seed the carrier from a COLUMN-MAJOR
    fp64 A, as harness::problem holds it. Both are row-major DF32 pairs; the
    transpose and the split happen here. */
void prepare(
    state        *s,
    double const *d_a);

/*  Factor the carrier in place. Returns elapsed milliseconds, event-timed. */
double factor(state *s);

/*  LU-IR against the captured operator, to the DF32 floor or the cap.
    d_b and d_x are n x k fp64, column major, as the harness holds them;
    d_x is written. Reports the largest iteration count over the columns.

    MRHS is a COLUMN LOOP: the residual and both triangular solves are
    single-vector, so k right-hand sides cost k independent chains. Correct
    but not competitive past small k -- the vendored block residual and the
    SRHS apply exist for this and are not ported yet. */
double solve(
    state        *s,
    double       *d_x,
    double const *d_b,
    std::size_t const k,
    std::size_t  *n_iterations);

/*  Copy the factored carrier and the composed permutation to the host, for
    an out-of-band reconstruction check. Row major, n*n each. */
void copy_factor(
    state *s,
    float *hi,
    float *lo,
    int   *perm);

} /* namespace int8lu_arm */
