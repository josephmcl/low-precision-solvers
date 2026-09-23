#include "common/matrix_market.h"

#include <fstream>
#include <iostream>
#include <sstream>

namespace harness {

bool read_matrix_market(std::string const &path, std::vector<double> &a,
                        std::size_t &n) {

    std::ifstream in(path);
    if (!in) { std::cerr << "[mtx] cannot open " << path << "\n"; return false; }

    std::string line;
    if (!std::getline(in, line)) return false;

    bool symmetric = line.find("symmetric") != std::string::npos;
    bool skew      = line.find("skew") != std::string::npos;
    if (line.find("pattern") != std::string::npos) {
        std::cerr << "[mtx] " << path << ": pattern-only, skipped "
                     "(filling with ones would change the problem)\n";
        return false;
    }
    if (line.find("coordinate") == std::string::npos) {
        std::cerr << "[mtx] " << path << ": only coordinate format\n";
        return false;
    }

    while (std::getline(in, line))
        if (!line.empty() && line[0] != '%') break;

    std::size_t rows = 0, cols = 0, nnz = 0;
    { std::istringstream hs(line); hs >> rows >> cols >> nnz; }
    if (rows != cols) {
        std::cerr << "[mtx] " << path << ": not square (" << rows
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

} /* namespace harness */
