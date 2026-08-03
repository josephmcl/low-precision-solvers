#include "common/solver.h"
#include "common/metrics.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

/*  CSV generation for the results tables.

    WHY A SEPARATE DRIVER. `lps-sweep` prints a table for a human reading one
    configuration; everything in the paper needs many configurations joined on
    common keys. Emitting one row per (matrix, method, setting) and doing the
    joins downstream keeps the harness from growing a reporting language, and
    it means a plot can be regenerated without re-running the GPU.

    WHAT IS SWEPT. Four axes, independently:

      --sizes n[:k],...      problem shape
      --families f,...       synthetic matrix families and their parameter
      --matrices path,...    Matrix Market files (SuiteSparse)
      --iters                per-iteration error, by capping max_outer and
                             re-running; needs no solver instrumentation and
                             therefore cannot perturb what it measures

    WHAT IS REPORTED. Four error metrics, because they answer different
    questions and this project has repeatedly confused them:

      backward  ||b - Ax|| / (||A|| ||x||)   — did we solve a nearby problem
      relative  ||b - Ax|| / ||b||           — residual as the caller sees it
      forward   ||x - x_ref|| / ||x_ref||    — distance to the fp64 answer
      rho       ||dX_m|| / ||dX_{m-1}||      — the fixed point's contraction

    Backward error is the one to compare methods on; forward error tracks it
    scaled by the condition number, so a method can look bad in forward terms
    purely because the problem is hard. Both are emitted so that claim can be
    checked rather than asserted.

    Build:  make bin/lps-profile
    Run:    bin/lps-profile --sizes 4096:512,8192:2048 --families shift,graded
            bin/lps-profile --matrices ss/*.mtx --iters
    Output: CSV on stdout; redirect to a file. */

namespace {

using harness::problem;

struct row {
    std::string matrix;      /*  family name or file stem                   */
    double      param = 0.;  /*  shift / decades / block                    */
    std::size_t n = 0, k = 0;
    std::string method;
    int         iter_cap = -1;   /*  -1 = run to convergence                */
    std::size_t iters = 0;
    double factor_ms = 0., solve_ms = 0., total_ms = 0.;
    double backward = 0., relative = 0., forward = 0., rho = 0.;
    double storage_n2 = 0.;
    double r_norm = 0.;      /*  ||R||/||A||, R-IR only                     */
};

void header() {
    std::cout << "matrix,param,n,k,method,iter_cap,iters,"
                 "factor_ms,solve_ms,total_ms,"
                 "backward,relative,forward,rho,storage_n2,r_norm\n";
}

void emit(row const &r) {
    std::cout << r.matrix << ',' << r.param << ',' << r.n << ',' << r.k << ','
              << r.method << ',' << r.iter_cap << ',' << r.iters << ','
              << r.factor_ms << ',' << r.solve_ms << ',' << r.total_ms << ','
              << r.backward << ',' << r.relative << ',' << r.forward << ','
              << r.rho << ',' << r.storage_n2 << ',' << r.r_norm << '\n';
}

/*  Matrix Market reader, dense-ified.

    SuiteSparse matrices arrive sparse and every method here is dense, so this
    materialises them. That bounds what can be read to whatever fits n^2 fp64 —
    around n = 30000 on a 33 GB card — which is the honest limit of using this
    harness on real matrices, not a property of the methods.

    Handles coordinate real/integer, general and symmetric. Pattern-only files
    are rejected rather than filled with ones, which would silently change the
    problem. */
bool read_matrix_market(std::string const &path, std::vector<double> &a,
                        std::size_t &n) {

    std::ifstream in(path);
    if (!in) { std::cerr << "[profile] cannot open " << path << "\n"; return false; }

    std::string line;
    if (!std::getline(in, line)) return false;

    bool symmetric = line.find("symmetric") != std::string::npos;
    bool skew      = line.find("skew") != std::string::npos;
    if (line.find("pattern") != std::string::npos) {
        std::cerr << "[profile] " << path << ": pattern-only, skipped "
                     "(filling with ones would change the problem)\n";
        return false;
    }
    if (line.find("coordinate") == std::string::npos) {
        std::cerr << "[profile] " << path << ": only coordinate format\n";
        return false;
    }

    while (std::getline(in, line))
        if (!line.empty() && line[0] != '%') break;

    std::size_t rows = 0, cols = 0, nnz = 0;
    { std::istringstream hs(line); hs >> rows >> cols >> nnz; }
    if (rows != cols) {
        std::cerr << "[profile] " << path << ": not square (" << rows
                  << "x" << cols << ")\n";
        return false;
    }

    n = rows;
    a.assign(n * n, 0.);
    for (std::size_t e = 0; e != nnz; ++e) {
        std::size_t i = 0, j = 0; double v = 0.;
        if (!(in >> i >> j >> v)) break;
        --i; --j;
        a[i + j * n] = v;
        if (symmetric && i != j) a[j + i * n] = v;
        if (skew && i != j)      a[j + i * n] = -v;
    }
    return true;
}

std::string stem(std::string const &path) {
    std::size_t const s = path.find_last_of("/\\");
    std::string base = (s == std::string::npos)? path : path.substr(s + 1);
    std::size_t const d = base.find_last_of('.');
    return (d == std::string::npos)? base : base.substr(0, d);
}

std::vector<std::string> split_list(std::string const &s) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) if (!item.empty()) out.push_back(item);
    return out;
}

harness::matrix_kind kind_of(std::string const &name) {
    if (name == "graded")    return harness::matrix_kind::graded_diagonal;
    if (name == "spd")       return harness::matrix_kind::spd_graded;
    if (name == "wilkinson") return harness::matrix_kind::wilkinson;
    if (name == "diag")      return harness::matrix_kind::diag_dominant;
    if (name == "randsvd")   return harness::matrix_kind::randsvd_log;
    if (name == "clust_pos") return harness::matrix_kind::eig_clustered_pos;
    if (name == "clust_sgn") return harness::matrix_kind::eig_clustered_signed;
    if (name == "arith_pos") return harness::matrix_kind::eig_arith_pos;
    if (name == "arith_sgn") return harness::matrix_kind::eig_arith_signed;
    return harness::matrix_kind::near_random;
}

/*  Parameter sweep for each family, matching what lps-kappa uses so the two
    tools are comparable. */
std::vector<double> params_for(std::string const &fam) {
    std::vector<double> p;
    if (fam == "graded" || fam == "spd")
        for (double d = 0.; d <= 12.; d += 2.) p.push_back(d);
    else if (fam == "wilkinson")
        for (double m = 4.; m <= 40.; m += 8.) p.push_back(m);
    else if (fam == "randsvd" || fam == "clust_pos" || fam == "clust_sgn" ||
             fam == "arith_pos" || fam == "arith_sgn")
        /*  log10(kappa): 1 to 12 decades, the range these families are used
            over in the refinement literature. */
        for (double d = 1.; d <= 12.; d += 1.) p.push_back(d);
    else if (fam == "diag")
        /*  -1 means "use the family's own shift". Passing 0. here overrides it
            to a zero diagonal, which makes the matrix near-singular and was
            silently producing backward errors ~50x worse than lps-sweep on
            what is nominally the easiest family. */
        p.push_back(-1.);
    else
        for (double s = 4096.; s >= 0.25; s /= 4.) p.push_back(s);
    return p;
}

/*  One measurement. `cap` overrides the method's own stopping rule when
    non-negative, which is how the per-iteration curve is produced: the same
    problem is solved repeatedly with the cap walked upward, and the error
    after each pass falls out. It measures the solver as shipped rather than an
    instrumented copy of it. */
void measure(row &r, solver::method const &m, problem &prob,
             std::size_t repeats) {

    double *d_x = static_cast<double *>(
        prob.acquire(prob.n * prob.k * sizeof(double)));

    if (r.iter_cap >= 0) {
        std::string const v = std::to_string(r.iter_cap);
        setenv("LPS_RIR_SOLVE_MAX_OUTER", v.c_str(), 1);
        setenv("LPS_MPIR_MAX_OUTER", v.c_str(), 1);
        /*  No reload needed: table::get() reads the environment at call time,
            so setenv takes effect on the next query. */
    }

    { solver::state warm; solver::run(d_x, prob.d_b, m, warm, prob); }

    /*  state owns device memory, so it is neither copyable nor assignable —
        construct a fresh one per repeat inside its own scope and carry out
        only the scalars. */
    std::vector<double> totals;
    std::size_t iters = 0;
    double factor_ms = 0., solve_ms = 0., rho = 0., storage = 0.;
    for (std::size_t i = 0; i != repeats; ++i) {
        solver::state st;
        solver::run(d_x, prob.d_b, m, st, prob);
        totals.push_back(st.total_ms);
        iters = st.n_iterations; factor_ms = st.factor_ms;
        solve_ms = st.solve_ms;  rho = st.rho;
        storage = st.storage_n2;
    }
    std::sort(totals.begin(), totals.end());

    metrics::report const err = metrics::evaluate(d_x, nullptr, prob);

    r.method     = m.name;
    r.iters      = iters;
    r.factor_ms  = factor_ms;
    r.solve_ms   = solve_ms;
    r.total_ms   = totals[totals.size() / 2];
    r.backward   = err.backward;
    r.relative   = err.relative;
    r.forward    = err.forward;
    r.rho        = rho;
    r.storage_n2 = (storage > 0.)? storage : m.storage_n2;

    if (r.iter_cap >= 0) {
        unsetenv("LPS_RIR_SOLVE_MAX_OUTER");
        unsetenv("LPS_MPIR_MAX_OUTER");
    }
}

} /* namespace */

int main(int argc, char **argv) {

    std::cout.setf(std::ios::unitbuf);

    std::vector<std::string> sizes    = {"4096:512"};
    std::vector<std::string> families = {"diag"};
    std::vector<std::string> files;
    std::size_t repeats  = 2;
    bool per_iteration   = false;
    int  max_cap         = 6;
    bool families_explicit = false;

    for (int i = 1; i < argc; ++i) {
        std::string const a = argv[i];
        auto next = [&]() { return (i + 1 < argc)? argv[++i] : ""; };
        if      (a == "--sizes")    sizes    = split_list(next());
        else if (a == "--families") { families = split_list(next());
                                      families_explicit = true; }
        else if (a == "--matrices") files    = split_list(next());
        else if (a == "--repeats")  repeats  = std::atoll(next());
        else if (a == "--iters")    per_iteration = true;
        else if (a == "--max-cap")  max_cap  = std::atoi(next());
        else if (a == "--help") {
            std::cerr <<
              "usage: lps-profile [options]  > out.csv\n"
              "  --sizes n[:k],...      default 4096:512\n"
              "  --families f,...       shift|graded|spd|wilkinson|diag\n"
              "  --matrices f.mtx,...   Matrix Market (SuiteSparse)\n"
              "  --iters                per-iteration error curve\n"
              "  --max-cap N            cap ceiling for --iters (default 6)\n"
              "  --repeats N            timing medians (default 2)\n";
            return 0;
        }
    }

    header();
    std::vector<solver::method> const &methods = solver::registry();

    /*  Synthetic families.

        Skipped when --matrices is given without an explicit --families: the
        SuiteSparse run carries k in --sizes but takes n from the file, so the
        n it would sweep here is 0, and a size-0 problem drives cuBLAS into
        "illegal value" and emits rows of zeros that parse as real data. Pass
        --families explicitly to run both in one invocation. */
    if (!files.empty() && !families_explicit) families.clear();

    for (std::string const &fam : families) {
        for (double const param : params_for(fam)) {
            for (std::string const &sz : sizes) {

                std::size_t n = 0, k = 512;
                { std::size_t const c = sz.find(':');
                  n = std::atoll(sz.substr(0, c).c_str());
                  if (c != std::string::npos) k = std::atoll(sz.substr(c + 1).c_str()); }

                problem prob(n, k, kind_of(fam), 7u, param);

                for (std::size_t mi = 0; mi != methods.size(); ++mi) {
                    std::vector<int> caps;
                    if (per_iteration)
                        for (int c = 1; c <= max_cap; ++c) caps.push_back(c);
                    else
                        caps.push_back(-1);

                    for (int const c : caps) {
                        row r;
                        r.matrix = fam; r.param = param;
                        r.n = n; r.k = k; r.iter_cap = c;
                        measure(r, methods[mi], prob, repeats);
                        emit(r);
                    }
                }
            }
        }
    }

    /*  SuiteSparse. k is taken from the first --sizes entry; n comes from the
        file. */
    for (std::string const &path : files) {

        std::vector<double> a; std::size_t n = 0;
        if (!read_matrix_market(path, a, n)) continue;

        std::size_t k = 512;
        { std::string const &sz = sizes.front();
          std::size_t const c = sz.find(':');
          if (c != std::string::npos) k = std::atoll(sz.substr(c + 1).c_str()); }

        problem prob(n, k, harness::matrix_kind::external, 7u, 0., a.data());

        for (std::size_t mi = 0; mi != methods.size(); ++mi) {
            row r;
            r.matrix = stem(path); r.param = 0.;
            r.n = n; r.k = k; r.iter_cap = -1;
            measure(r, methods[mi], prob, repeats);
            emit(r);
        }
    }

    return 0;
}
