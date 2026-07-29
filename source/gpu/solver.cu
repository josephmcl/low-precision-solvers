#include "gpu/solver.h"

#include "gpu/definitions.h"
#include "gpu/error.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <vector>

namespace solver {


void *state::acquire(std::size_t const bytes) {

    void *d_p = nullptr;
    if (!CUDA_CHECK(cudaMalloc(&d_p, bytes)))
        return nullptr;

    _d_owned.push_back(d_p);
    return d_p;
}

state::~state() {

    for (std::size_t i = 0; i != _d_owned.size(); ++i)
        CUDA_CHECK(cudaFree(_d_owned[i]));
}

/*  The scored methods.

    Ordering here is the ordering in the report, so it runs cheapest-storage
    first and the reference last. A new method is one entry plus one file
    pair; nothing else in the harness needs to know it exists. */
std::vector<method> const &registry() {

    static std::vector<method> const methods = {
        {"direct fp64",
         factor_direct,
         solve_direct,
         storage::DIRECT_FP64},

        {"vendor IRS fp32",
         factor_vendor_irs,
         solve_vendor_irs,
         storage::VENDOR_IRS}
    };

    return methods;
}

} /* namespace solver */
