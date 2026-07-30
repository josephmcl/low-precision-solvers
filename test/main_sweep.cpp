#include "gpu/definitions.h"
#include "gpu/error.h"
#include "gpu/metrics.h"
#include "gpu/problem.h"
#include "gpu/solver.h"
#include "gpu/timing.h"

#include <cuda_runtime.h>

#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

/*  The four-way comparison at one n, swept over k.

    The repeat loop lives here and nowhere else. That is the point: a runner
    that wrapped some methods in its repeat loop and not others compared one
    solve against ten and reached a conclusion that had to be retracted. With
    the loop in the driver, the count is structurally the same for every
    method and no method can opt out.

    Each repeat re-runs the method end to end, factor included, because the
    factorization is part of what is being compared. */

namespace {

using harness::matrix_kind;

struct options {
    std::size_t              n        = 8192;
    std::vector<std::size_t> k        = {64, 512, 2048};
    std::size_t              repeats  = 3;
    matrix_kind              kind     = matrix_kind::diag_dominant;
    bool                     raw      = false;
};

std::vector<std::size_t> parse_list(char const *text) {

    std::vector<std::size_t> out;
    std::string const s(text);

    std::size_t begin = 0;
    while (begin <= s.size()) {
        std::size_t const comma = s.find(',', begin);
        std::string const item  = s.substr(
            begin, (comma == std::string::npos)? std::string::npos
                                               : comma - begin);
        if (!item.empty())
            out.push_back(static_cast<std::size_t>(std::atoll(item.c_str())));
        if (comma == std::string::npos)
            break;
        begin = comma + 1;
    }
    return out;
}

bool parse_kind(char const *text, matrix_kind &kind) {

    if (std::strcmp(text, "diag") == 0)   {kind = matrix_kind::diag_dominant; return true;}
    if (std::strcmp(text, "moderate") == 0) {kind = matrix_kind::moderate; return true;}
    if (std::strcmp(text, "random") == 0) {kind = matrix_kind::near_random; return true;}
    if (std::strcmp(text, "graded") == 0) {kind = matrix_kind::graded_rows; return true;}
    return false;
}

void usage() {

    std::cout
        << "usage: lps-sweep <n> [--k 64,512,2048] [--repeats 3]\n"
        << "                     [--matrix diag|moderate|random|graded] [--raw]\n\n"
        << "  Reports median of --repeats with spread. Compare any margin\n"
        << "  against the spread before believing it.\n";
}

/*  A method that cannot report a factor/solve split prints \"--\" rather than
    a zero, which would read as \"free\". */
void print_ms(double const ms, bool const known) {

    if (known)
        std::cout << std::setw(10) << std::fixed << std::setprecision(1) << ms;
    else
        std::cout << std::setw(10) << "--";
}

} /* namespace */

int main(int argc, char **argv) {

    options opt;

    if (argc > 1 && (std::strcmp(argv[1], "-h") == 0 ||
                     std::strcmp(argv[1], "--help") == 0)) {
        usage();
        return 0;
    }
    if (argc > 1)
        opt.n = static_cast<std::size_t>(std::atoll(argv[1]));

    for (int i = 2; i != argc; ++i) {
        if (std::strcmp(argv[i], "--k") == 0 && i + 1 != argc)
            opt.k = parse_list(argv[++i]);
        else if (std::strcmp(argv[i], "--repeats") == 0 && i + 1 != argc)
            opt.repeats = static_cast<std::size_t>(std::atoll(argv[++i]));
        else if (std::strcmp(argv[i], "--raw") == 0)
            opt.raw = true;
        else if (std::strcmp(argv[i], "--matrix") == 0 && i + 1 != argc) {
            if (!parse_kind(argv[++i], opt.kind)) {
                usage();
                return 1;
            }
        }
        else {
            usage();
            return 1;
        }
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    std::cout << "device   " << prop.name << "  ("
              << prop.totalGlobalMem / 1000000000. << " GB)\n"
              << "matrix   " << harness::name_of(opt.kind) << "\n"
              << "n        " << opt.n << "\n"
              << "repeats  " << opt.repeats << "  (median reported)\n\n";

    std::vector<solver::method> const &methods = solver::registry();

    for (std::size_t ki = 0; ki != opt.k.size(); ++ki) {

        std::size_t const k = opt.k[ki];

        /*  One problem per (n, k): every method below is scored on
            bit-identical A and B. */
        harness::problem prob(opt.n, k, opt.kind);

        double *d_x = static_cast<double *>(
            prob.acquire(opt.n * k * sizeof(double)));
        double *d_x_ref = static_cast<double *>(
            prob.acquire(opt.n * k * sizeof(double)));

        std::cout << "k = " << k << "\n"
                  << "  " << std::left << std::setw(18) << "method"
                  << std::right
                  << std::setw(10) << "total"
                  << std::setw(9)  << "spread"
                  << std::setw(10) << "factor"
                  << std::setw(10) << "solve"
                  << std::setw(6)  << "it"
                  << std::setw(12) << "backward"
                  << std::setw(12) << "relative"
                  << std::setw(12) << "forward"
                  << std::setw(9)  << "storage" << "\n";

        bool have_reference = false;

        for (std::size_t mi = 0; mi != methods.size(); ++mi) {

            solver::method const &m = methods[mi];

            std::vector<double> total;
            std::vector<double> factor;
            std::vector<double> solve;
            std::size_t         n_iterations = 0;
            bool                split        = false;

            /*  One untimed run first, discarded.

                A library warmup on a small problem is not sufficient: cuSOLVER
                selects kernels and sizes workspace per shape, so warming
                IRSXgesv at (64, 1) left it cold at (8192, 64) and the first
                timed repeat carried a ~32 ms penalty — a 62% spread against
                0.6% on the direct solve. Discarding one run per (method, n, k)
                is applied to every method identically, so it cannot favour
                one, and it is the only form of warmup that covers
                shape-dependent setup.

                Then uniform repeats, fresh state each time so the
                factorization is re-paid and nothing carries over. */
            {
                solver::state warm;
                solver::run(d_x, prob.d_b, m, warm, prob);
            }

            for (std::size_t r = 0; r != opt.repeats; ++r) {
                solver::state st;
                solver::run(d_x, prob.d_b, m, st, prob);
                total.push_back(st.total_ms);
                factor.push_back(st.factor_ms);
                solve.push_back(st.solve_ms);

                /*  Max, not last. Recording only the final repeat would hide
                    a method whose iteration count varies between runs, which
                    is itself a result worth seeing. */
                if (st.n_iterations > n_iterations)
                    n_iterations = st.n_iterations;
                split = st.split_reported;
            }

            if (opt.raw) {
                std::cout << "    raw " << m.name << ":";
                for (std::size_t r = 0; r != total.size(); ++r)
                    std::cout << " " << std::fixed << std::setprecision(1)
                              << total[r];
                std::cout << "\n";
            }

            /*  The first method in the registry is the fp64 reference, so its
                X becomes the forward-error reference for the rest. */
            if (!have_reference) {
                CUDA_CHECK(cudaMemcpy(
                    d_x_ref,
                    d_x,
                    opt.n * k * sizeof(double),
                    cudaMemcpyDeviceToDevice));
                have_reference = true;
            }

            metrics::report const err = metrics::evaluate(
                d_x,
                (mi == 0)? nullptr : d_x_ref,
                prob);

            timing::sample const t = timing::summarize(total);
            timing::sample const f = timing::summarize(factor);
            timing::sample const s = timing::summarize(solve);

            std::cout << "  " << std::left << std::setw(18) << m.name
                      << std::right
                      << std::setw(10) << std::fixed << std::setprecision(1)
                      << t.median
                      << std::setw(8) << std::setprecision(1)
                      << t.spread() * 100. << "%";
            print_ms(f.median, split);
            print_ms(s.median, split);
            std::cout << std::setw(6) << n_iterations
                      << std::setw(12) << std::scientific << std::setprecision(2)
                      << err.backward
                      << std::setw(12) << err.relative
                      << std::setw(12) << err.forward
                      << std::setw(7) << std::fixed << std::setprecision(0)
                      << m.storage_n2 << "n2"
                      << "\n";
        }

        std::cout << "\n";
    }

    return 0;
}
