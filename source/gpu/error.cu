#include "gpu/error.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#include <iostream>

namespace error {

bool cuda_status(
    cudaError_t const  err,
    char const        *file,
    int const          line) {

    if (err == cudaSuccess)
        return true;

    std::cout << "[cuda] " << file << ":" << line << " "
              << cudaGetErrorString(err) << "\n";
    return false;
}

bool cublas_status(
    cublasStatus_t const  status,
    char const           *file,
    int const             line) {

    if (status == CUBLAS_STATUS_SUCCESS)
        return true;

    std::cout << "[cublas] " << file << ":" << line << " status "
              << static_cast<int>(status) << "\n";
    return false;
}

bool cusolver_status(
    cusolverStatus_t const  status,
    char const             *file,
    int const               line) {

    if (status == CUSOLVER_STATUS_SUCCESS)
        return true;

    std::cout << "[cusolver] " << file << ":" << line << " status "
              << static_cast<int>(status) << "\n";
    return false;
}

} /* namespace error */
