#include "common/solver.h"

#include <cstdlib>
#include <cstring>

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

    /*  Report any error left pending before this method starts.

        A CUDA error is sticky: once one is raised, every subsequent call fails
        with it, so the first method to break makes the NEXT method look
        broken. Checking here attributes the fault to whoever actually caused
        it. Without this, a fault in method i is read as a fault in method i+1
        — which cost several debugging rounds. */
    if (cudaError_t const pending = cudaGetLastError(); pending != cudaSuccess)
        std::cout << "[solver] error pending BEFORE " << m.name << ": "
                  << cudaGetErrorString(pending)
                  << " (raised by an earlier method, not this one)\n";

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

    /*  LPS_ONLY=<substring> restricts the registry to matching methods.
        Exists for PROFILING: all four methods issue cuBLAS GEMMs, so an
        nsys/ncu trace of a full run cannot attribute a GEMM to a method, and
        the Ozaki kernels of R-IR and split-MPIR share names. Filtering at the
        registry means the trace contains one method's work and nothing else.

        Unset, this is a no-op and the registry is exactly as before, so it
        cannot perturb a measurement run that does not ask for it. */
    static std::vector<method> const all = {
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

        {"R-IR",
         factor_rir,
         solve_rir,
         nullptr,
         storage::RIR_FP32_R},

        {"vendor IRS fp32",
         nullptr,
         nullptr,
         factor_solve_vendor_irs,
         storage::VENDOR_IRS}
    };

    static std::vector<method> filtered;
    static bool built = false;
    if (!built) {
        built = true;
        char const *only = std::getenv("LPS_ONLY");
        if (only == nullptr || *only == '\0') filtered = all;
        else
            for (method const &m : all)
                if (std::strstr(m.name, only) != nullptr)
                    filtered.push_back(m);
    }
    return filtered;
}

} /* namespace solver */
