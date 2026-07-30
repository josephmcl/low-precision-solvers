#pragma once

#include <iostream>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

/*  One status helper per vendor status type. Each prints a tagged message
    naming the file and line and returns; the harness reports and keeps
    going rather than unwinding. Call one immediately after every vendor
    call, and cuda_status(cudaGetLastError()) after every kernel launch. */

namespace error {

bool cuda_status(
    cudaError_t const  err,
    char const        *file,
    int const          line);

bool cublas_status(
    cublasStatus_t const  status,
    char const           *file,
    int const             line);

bool cusolver_status(
    cusolverStatus_t const  status,
    char const             *file,
    int const               line);

} /* namespace error */

/*  The line-reporting wrappers. These are macros only so that __FILE__ and
    __LINE__ resolve at the call site; they evaluate their argument once. */
#define CUDA_CHECK(x)     error::cuda_status((x), __FILE__, __LINE__)
#define CUBLAS_CHECK(x)   error::cublas_status((x), __FILE__, __LINE__)
#define CUSOLVER_CHECK(x) error::cusolver_status((x), __FILE__, __LINE__)

/*  Follow every kernel launch with this. A launch failure is otherwise
    reported at the next unrelated synchronizing call, which sends the
    reader to the wrong line. */
#define KERNEL_CHECK() error::cuda_status(cudaGetLastError(), __FILE__, __LINE__)
