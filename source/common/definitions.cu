#include "common/definitions.h"

namespace precision {

char const *name_of(arithmetic const a) {

    switch (a) {
        case arithmetic::fp64: return "fp64";
        case arithmetic::fp32: return "fp32";
        case arithmetic::tf32: return "tf32";
        case arithmetic::bf16: return "bf16";
    }
    return "unknown";
}

} /* namespace precision */

namespace launch {

int grid_for(std::size_t const n_elements) {

    std::size_t const blocks =
        (n_elements + static_cast<std::size_t>(BLOCK_SIZE) - 1) /
        static_cast<std::size_t>(BLOCK_SIZE);

    if (blocks == 0)
        return 1;

    return static_cast<int>(
        std::min(blocks, static_cast<std::size_t>(MAX_BLOCKS)));
}

} /* namespace launch */
