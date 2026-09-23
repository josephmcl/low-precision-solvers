#pragma once

#include <cstddef>
#include <string>
#include <vector>

/*  Matrix Market reader, shared.

    Extracted from profile/main_profile.cpp when the campaign driver
    needed it too. A second copy is not an option: two readers would be
    turning the same file into different matrices the first time one of
    them was "improved", and nothing downstream would say so. */

namespace harness {

/*  Densify a Matrix Market file into a column-major n x n. Returns
    false, with a reason on stderr, for anything it will not read.

    SuiteSparse matrices arrive sparse and every method here is dense,
    so this materialises them -- which bounds what can be read to
    whatever fits n^2 fp64, around n = 30000 on a 33 GB card. That is
    the honest limit of using this harness on real matrices, not a
    property of the methods.

    Handles coordinate real/integer, general and symmetric. Pattern-only
    files are rejected rather than filled with ones, which would
    silently change the problem. */
bool read_matrix_market(
    std::string const   &path,
    std::vector<double> &a,
    std::size_t         &n);

} /* namespace harness */
