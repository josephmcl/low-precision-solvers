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

void run(
    double       *d_x,
    double const *d_b,
    method const &m,
    state        &st,
    problem      &prob) {

    if (m.is_split()) {
        m.factor(st, prob);
        m.solve(d_x, d_b, st, prob);

        /*  total is the sum, not a third measurement of the same work: a
            separate stopwatch around both would also charge the host-side
            gap between them. */
        st.total_ms       = st.factor_ms + st.solve_ms;
        st.split_reported = true;
    }
    else {
        m.factor_solve(d_x, d_b, st, prob);
        st.factor_ms      = 0.;
        st.solve_ms       = 0.;
        st.split_reported = false;
    }
}

/*  The scored methods.

    Ordering here is the ordering in the report. A new method is one entry
    plus one file pair; nothing else in the harness needs to know it exists.
    Split methods pass {factor, solve, nullptr}; monolithic ones pass
    {nullptr, nullptr, factor_solve}. */
std::vector<method> const &registry() {

    static std::vector<method> const methods = {
        {"direct fp64",
         factor_direct,
         solve_direct,
         nullptr,
         storage::DIRECT_FP64},

        {"split-MPIR",
         factor_split_mpir,
         solve_split_mpir,
         nullptr,
         storage::SPLIT_MPIR},

        {"vendor IRS fp32",
         nullptr,
         nullptr,
         factor_solve_vendor_irs,
         storage::VENDOR_IRS}
    };

    return methods;
}

} /* namespace solver */
