#include "common/solver.h"
#include "common/metrics.h"

#include <algorithm>

#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

/*  Measure each optimization by turning it off, one at a time and then all at
    once.

    Every entry below was adopted on a measurement, but those measurements were
    taken at different times against different baselines, and several of them
    interact — the contraction bound makes blocks partial, which is what killed
    the group-concatenation idea; accumulator merging looked worth 1.9% until
    profiling showed the folds were 5% of the build. A list of individually
    justified changes is not the same as a justified set. This runs the set.

    Each optimization is expressed as the environment override that DISABLES
    it, so the baseline is whatever the tuning file says and the ablation is a
    deviation from it. Nothing here hardcodes a tuned value; if the tuning file
    changes, the baseline moves with it and the deltas stay meaningful. */

namespace {

struct ablation {
    char const *name;
    char const *key;
    char const *off;
    char const *note;
};

/*  Disabling values, not enabling ones.

    The key strings must match what tuning::table::get() derives from its key
    name — "ozaki.triangular" becomes LPS_OZAKI_TRIANGULAR, not LPS_OZ_*. A
    mismatched name here does not error; it silently ablates nothing and the
    row reports 1.00x, which reads as "this optimization does not matter"
    rather than "this switch is not wired". That is exactly how this table
    reported the contraction bound as worthless on its first run, having
    measured it at 1.4x an hour earlier. */
ablation const ABLATIONS[] = {
    {"plain R*X",
     "LPS_RIR_SOLVE_PLAIN_RX", "0",
     "one fp32 SGEMM -> 21-product Ozaki cascade"},
    {"contraction bound",
     "LPS_OZAKI_CONTRACTION_BOUND", "0",
     "skip zero U rows past the column block -> full range"},
    {"triangular shapes",
     "LPS_OZAKI_TRIANGULAR", "0",
     "skip structurally-zero rows -> treat every block dense"},
    {"group merging",
     "LPS_RIR_BUILD_OZAKI_MERGE_TAIL", "9",
     "share one fp64 fold across the high-s tail -> one per group"},
    {"build bits=10",
     "LPS_RIR_BUILD_OZAKI_BITS", "9",
     "10 bits/piece on the triangular operand -> 9"},
    {"tuned blocking",
     "LPS_RIR_BUILD_OZAKI_BLOCK", "512",
     "device-tuned contraction block -> a deliberately poor one"}
};

constexpr int N_ABLATIONS = 6;

struct result {
    double factor = 0.;
    double solve  = 0.;
    double total  = 0.;
    double backward = 0.;
    std::size_t iterations = 0;
};

/*  R-IR only: it is the method every optimization here applies to. */
result run_rir(
    std::size_t const n,
    std::size_t const k,
    std::size_t const repeats) {

    harness::problem prob(n, k, harness::matrix_kind::diag_dominant);

    double *d_x = static_cast<double *>(
        prob.acquire(n * k * sizeof(double)));

    solver::method const *m = nullptr;
    std::vector<solver::method> const &all = solver::registry();
    for (std::size_t i = 0; i != all.size(); ++i)
        if (std::strcmp(all[i].name, "R-IR") == 0)
            m = &all[i];
    if (m == nullptr) {
        std::cout << "[ablate] R-IR not in the registry\n";
        return result();
    }

    /*  One discarded run, then the timed ones — the same discipline the sweep
        uses, because cuBLAS selects kernels per shape and the ablations change
        the shapes. */
    {
        solver::state warm;
        solver::run(d_x, prob.d_b, *m, warm, prob);
    }

    result out;
    std::vector<double> totals;
    for (std::size_t r = 0; r != repeats; ++r) {
        solver::state st;
        solver::run(d_x, prob.d_b, *m, st, prob);
        totals.push_back(st.total_ms);
        out.factor = st.factor_ms;
        out.solve  = st.solve_ms;
        out.iterations = st.n_iterations;
    }

    std::sort(totals.begin(), totals.end());
    out.total = totals[totals.size() / 2];

    metrics::report const err = metrics::evaluate(d_x, nullptr, prob);
    out.backward = err.backward;

    return out;
}

void set_off(ablation const &a) {
    setenv(a.key, a.off, 1);
}

void clear_all() {
    for (int i = 0; i != N_ABLATIONS; ++i)
        unsetenv(ABLATIONS[i].key);
}

} /* namespace */

int main(int argc, char **argv) {

    std::cout.setf(std::ios::unitbuf);

    /*  CSV mode for the results pipeline. One row per ablation, so a figure
        can plot cost-of-having against the optimization name without parsing
        the human-readable table. */
    bool csv = false;
    for (int i = 1; i < argc; ++i)
        if (std::strcmp(argv[i], "--csv") == 0) csv = true;

    std::size_t const n = (argc > 1)?
        static_cast<std::size_t>(std::atoll(argv[1])) : 8192;
    std::size_t const k = (argc > 2)?
        static_cast<std::size_t>(std::atoll(argv[2])) : 2048;
    std::size_t const repeats = (argc > 3)?
        static_cast<std::size_t>(std::atoll(argv[3])) : 2;

    if (csv)
        std::cout << "n,k,repeats,ablation,key,factor_ms,solve_ms,total_ms,"
                     "cost,iters,backward\n";
    else
    std::cout << "R-IR ablation   n=" << n << "  k=" << k
              << "  repeats=" << repeats << "\n"
              << "each row disables ONE optimization; the last disables all\n\n";

    clear_all();
    result const base = run_rir(n, k, repeats);

    if (!csv)
    std::cout << "  " << std::left << std::setw(22) << "configuration"
              << std::right
              << std::setw(9)  << "factor"
              << std::setw(9)  << "solve"
              << std::setw(9)  << "total"
              << std::setw(9)  << "cost"
              << std::setw(5)  << "it"
              << std::setw(12) << "backward" << "\n";

    if (csv)
        std::cout << n << ',' << k << ',' << repeats << ",baseline,--,"
                  << base.factor << ',' << base.solve << ',' << base.total
                  << ",1.00," << base.iterations << ',' << base.backward
                  << '\n';
    else
    std::cout << "  " << std::left << std::setw(22) << "all on (baseline)"
              << std::right << std::fixed << std::setprecision(1)
              << std::setw(9) << base.factor
              << std::setw(9) << base.solve
              << std::setw(9) << base.total
              << std::setw(9) << "--"
              << std::setw(5) << base.iterations
              << std::setw(12) << std::scientific << std::setprecision(2)
              << base.backward << "\n";

    for (int i = 0; i != N_ABLATIONS; ++i) {

        clear_all();
        set_off(ABLATIONS[i]);
        result const r = run_rir(n, k, repeats);

        /*  Cost of HAVING the optimization: how much slower without it. Below
            1.00 means the optimization is not paying at this size. */
        double const cost = (base.total > 0.)? r.total / base.total : 0.;

        if (csv) {
            std::cout << n << ',' << k << ',' << repeats << ",without "
                      << ABLATIONS[i].name << ',' << ABLATIONS[i].key << ','
                      << r.factor << ',' << r.solve << ',' << r.total << ','
                      << cost << ',' << r.iterations << ',' << r.backward
                      << '\n';
            continue;
        }
        std::cout << "  " << std::left << std::setw(22)
                  << (std::string("without ") + ABLATIONS[i].name)
                  << std::right << std::fixed << std::setprecision(1)
                  << std::setw(9) << r.factor
                  << std::setw(9) << r.solve
                  << std::setw(9) << r.total
                  << std::setw(8) << std::setprecision(2) << cost << "x"
                  << std::setw(5) << r.iterations
                  << std::setw(12) << std::scientific << std::setprecision(2)
                  << r.backward << "\n";
    }

    clear_all();
    for (int i = 0; i != N_ABLATIONS; ++i)
        set_off(ABLATIONS[i]);
    result const none = run_rir(n, k, repeats);
    double const cost = (base.total > 0.)? none.total / base.total : 0.;

    if (csv) {
        std::cout << n << ',' << k << ',' << repeats << ",all off,--,"
                  << none.factor << ',' << none.solve << ',' << none.total
                  << ',' << cost << ',' << none.iterations << ','
                  << none.backward << '\n';
        return 0;
    }
    std::cout << "  " << std::left << std::setw(22) << "all off"
              << std::right << std::fixed << std::setprecision(1)
              << std::setw(9) << none.factor
              << std::setw(9) << none.solve
              << std::setw(9) << none.total
              << std::setw(8) << std::setprecision(2) << cost << "x"
              << std::setw(5) << none.iterations
              << std::setw(12) << std::scientific << std::setprecision(2)
              << none.backward << "\n";

    clear_all();

    std::cout << "\n  Individual costs do not multiply to the combined one when"
                 " the optimizations\n  interact. A large gap between their"
                 " product and the 'all off' row is the\n  signal that they"
                 " do.\n";

    for (int i = 0; i != N_ABLATIONS; ++i)
        std::cout << "    " << std::left << std::setw(20) << ABLATIONS[i].name
                  << ABLATIONS[i].note << "\n";

    return 0;
}
