#pragma once

#include <cstddef>

/*  Precision tags and the storage each method holds resident.

    Scalars are spelled out — `double` and `float`, never an alias. This
    harness exists to compare precisions against each other, so hiding a
    width behind a typedef would hide the variable under test. Where a
    quantity is fp32 because the method stores it that way, it is `float` at
    every level, and the low-precision arrays never silently widen.

    The storage identity that motivates the work,

        bits(eps) = bits(u_f) + bits(u_R),

    says a scheme's delivered accuracy is set by the total bits it keeps, not
    by the width of any one array. The constants below record what each
    method holds resident per n^2, which is the quantity that identity trades
    against accuracy. */

namespace precision {

/*  Arithmetic a method may use for its inner products. The harness reports
    the fp32:fp64 and tf32:fp64 rates of the device it runs on, because the
    ordering of the methods follows that ratio rather than the chip name:
    fp64-free schemes pay when it is large and cost when it is near one. */
enum class arithmetic {
    fp64,
    fp32,
    tf32,
    bf16
};

char const *name_of(arithmetic const a);

} /* namespace precision */

namespace storage {

/*  Matrix-resident bytes per n^2, excluding the O(nk) right-hand-side
    workspace every method needs alike. These are the numbers a capacity
    claim is made of, so they are stated once here rather than recomputed at
    each call site. */
constexpr double DIRECT_FP64 =  8.;   /* A fp64                        */
constexpr double VENDOR_IRS  = 12.;   /* A fp64 + LU fp32              */
constexpr double SPLIT_MPIR  = 12.;   /* A_hi + A_lo fp32 + LU fp32    */
constexpr double RIR_FP32_R  =  8.;   /* LU fp32 + R fp32              */
constexpr double RIR_BF16_R  =  6.;   /* LU fp32 + R bf16              */

} /* namespace storage */

namespace launch {

/*  Kernel launch geometry. One block owns BLOCK_SIZE consecutive elements of
    a grid-stride loop; reductions stage BLOCK_SIZE partials in shared
    memory, so it must stay a power of two. */
constexpr int BLOCK_SIZE = 256;
constexpr int MAX_BLOCKS = 1024;

/*  Grid for a flat, grid-stride loop over n_elements, clamped so a large
    problem does not ask for more blocks than the reduction path expects. */
int grid_for(std::size_t const n_elements);

} /* namespace launch */
