#include "common/error.h"

namespace error {

bool cuda_status(
    cudaError_t const  err,
    char const        *file,
    int const          line) {

    if (err == cudaSuccess)
        return true;

    /*  DIAGNOSTICS GO TO STDERR, NEVER STDOUT.

    stdout is the CSV stream for every profile tool. When vendor IRS ran out of
    memory at n=49152 the resulting cascade of "illegal memory access" reports
    was written INTO srhs_xl.csv -- 43,312 error lines around 16 data rows. A
    failure that should have cost one row cost the whole file, and the stderr
    log it should have gone to was empty. */
std::cerr << "[cuda] " << file << ":" << line << " "
              << cudaGetErrorString(err) << "\n";
    return false;
}

bool cublas_status(
    cublasStatus_t const  status,
    char const           *file,
    int const             line) {

    if (status == CUBLAS_STATUS_SUCCESS)
        return true;

    std::cerr << "[cublas] " << file << ":" << line << " status "
              << static_cast<int>(status) << "\n";
    return false;
}

bool cusolver_status(
    cusolverStatus_t const  status,
    char const             *file,
    int const               line) {

    if (status == CUSOLVER_STATUS_SUCCESS)
        return true;

    std::cerr << "[cusolver] " << file << ":" << line << " status "
              << static_cast<int>(status) << "\n";
    return false;
}

} /* namespace error */
