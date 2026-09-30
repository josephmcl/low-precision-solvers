#pragma once

/*  Compatibility shims for cuSOLVER's emulation API.

    The math-mode and emulation-strategy entry points do not exist before
    CUDA 12.8, and `CUSOLVER_FP64_EMULATED_FIXEDPOINT_MATH` needs 13.2.
    A tree that calls them unguarded will not compile on an older
    toolkit, which is how the university host (CUDA 12.5) refused a build
    that is fine on the rented boxes (13.2, 13.3).

    Guarded here rather than at each of the fifteen call sites, so the
    call sites stay readable and the version logic lives in one place.
    Where the API is absent the shims report failure, which the callers
    already handle -- they treat a non-zero status as "emulation
    unavailable" and fall back to the native path.

    IMPORTANT for anything quoted from a build where LPS_HAVE_CUSOLVER_EMU
    is 0: the emulated-fp64 baseline does not exist there. It is not a
    slow emulated run, it is no run at all. */

#include <cublas_v2.h>
#include <cusolverDn.h>

#if defined(CUDART_VERSION) && CUDART_VERSION >= 12080
#define LPS_HAVE_CUSOLVER_EMU 1
#else
#define LPS_HAVE_CUSOLVER_EMU 0
#endif

#if CUDART_VERSION >= 13020
#define LPS_HAVE_FP64_EMU 1
#else
#define LPS_HAVE_FP64_EMU 0
#endif

#if !LPS_HAVE_CUSOLVER_EMU

typedef int cusolverMathMode_t;
#define CUSOLVER_DEFAULT_MATH               0
#define CUSOLVER_FP32_EMULATED_BF16X9_MATH  1
#define CUSOLVER_FP64_EMULATED_FIXEDPOINT_MATH 2

typedef int cusolverEmulationStrategy_t;
#define CUDA_EMULATION_STRATEGY_EAGER       0
#define CUDA_EMULATION_STRATEGY_PERFORMANT  1

inline cusolverStatus_t cusolverDnGetMathMode(cusolverDnHandle_t,
                                             cusolverMathMode_t *m) {
    if (m != nullptr) *m = CUSOLVER_DEFAULT_MATH;
    return CUSOLVER_STATUS_NOT_SUPPORTED;
}
inline cusolverStatus_t cusolverDnSetMathMode(cusolverDnHandle_t,
                                              cusolverMathMode_t) {
    return CUSOLVER_STATUS_NOT_SUPPORTED;
}
inline cusolverStatus_t cusolverDnGetEmulationStrategy(
        cusolverDnHandle_t, cusolverEmulationStrategy_t *s) {
    if (s != nullptr) *s = CUDA_EMULATION_STRATEGY_PERFORMANT;
    return CUSOLVER_STATUS_NOT_SUPPORTED;
}
inline cusolverStatus_t cusolverDnSetEmulationStrategy(
        cusolverDnHandle_t, cusolverEmulationStrategy_t) {
    return CUSOLVER_STATUS_NOT_SUPPORTED;
}

#elif !LPS_HAVE_FP64_EMU
/*  12.8 <= CUDA < 13.2: the API exists, the fp64 enumerator does not. */
#define CUSOLVER_FP64_EMULATED_FIXEDPOINT_MATH CUSOLVER_DEFAULT_MATH
#endif

/*  cuBLAS's emulated compute type, same story: added in 12.8. Guarded
    here too so the one header covers both libraries. */
#if !LPS_HAVE_CUSOLVER_EMU
#define CUBLAS_COMPUTE_32F_EMULATED_16BFX9 CUBLAS_COMPUTE_32F
#endif
