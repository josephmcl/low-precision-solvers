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

/*  Which kernel performs the trailing update. Everything else -- the
    panel, the row interchanges, the U12 block solve, the refinement -- is
    identical, which is the point: the two arms are then comparable on the
    one thing that differs.

    sliced  the vendored INT-sliced trailing update, k int8 slices per
            operand and the kept-pair set i + j < k.
    oii     Ozaki-II, N pairwise-coprime moduli, DF32 in and out and no
            fp64 instruction. `depth` is N rather than k.

    The oii arm runs the per-panel loop, not the vendored whole-factor
    CUDA graph: the graph is built around the sliced update and its
    schedule. It is therefore a correctness vehicle first; its timings are
    against the per-panel sliced path, not against the graph. */
enum class kernel { sliced, oii };

/*  Resident matrix bytes per n^2: 8 for the captured residual operator plus
    8 for the factor carrier. The factorization consumes its carrier, so the
    operator the refinement needs cannot be the same array. */
constexpr double STORAGE_N2 = 16.;

/*  Allocate for an n x n solve at panel width b (256 is the tuned value)
    and depth `kfac` -- the slice count k for the sliced kernel, the
    modulus count N for oii. Returns null on failure. */
state *create(
    std::size_t const n,
    int const         b,
    int const         kfac,
    kernel const      which = kernel::sliced);

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

/*  Phase-separated timing, milliseconds, accumulated over the last
    factor/solve pair.

    Collected from CUDA event markers laid at phase boundaries with a single
    synchronization at the end, NOT a sync per phase: the latter serializes
    the stream and inflates exactly the overlap being measured. The vendored
    RECONPROF=1 path does sync per launch and reports summed kernel duration
    — a different quantity, useful for slice-vs-trailing inside the factor,
    and not comparable with these. */
struct phase_times {
    double prepare  = 0.;   /* transpose + DF32 split of A           */
    double factor   = 0.;   /* int8lu_factor                         */
    double perm     = 0.;   /* compose interchanges, host round trip */
    double residual = 0.;   /* r = b - A x, DF32                     */
    double gather   = 0.;   /* P r                                   */
    double trsv     = 0.;   /* both triangular solves                */
    double update   = 0.;   /* split, correction, combine, norms     */
};

phase_times const &profile(state const *s);

/*  Copy the factored carrier and the composed permutation to the host, for
    an out-of-band reconstruction check. Row major, n*n each. */
void copy_factor(
    state *s,
    float *hi,
    float *lo,
    int   *perm);

} /* namespace int8lu_arm */
