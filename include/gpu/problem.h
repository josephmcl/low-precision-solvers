#pragma once

#include "gpu/definitions.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#include <cstddef>
#include <vector>

namespace harness {


/*  The matrix classes the methods are scored on. Conditioning is what
    decides whether a residual-storage scheme holds up: R = PA - LU grows
    with a poor fp32 factorization, and once ||R|| is large the bits kept
    in R cover proportionally less of A. NEAR_RANDOM is therefore not an
    optional extra — it is the case that separates the methods, and a
    result quoted only on DIAG_DOMINANT says little. */
enum class matrix_kind {
    diag_dominant,   /* A = rand(-1,1) + n*I        -- easy               */
    moderate,        /* A = rand(-1,1) + sqrt(n)*I                        */
    near_random,     /* A = rand(-1,1) + I          -- the stress case    */
    graded_rows      /* rows scaled over 6 decades                        */
};

char const *name_of(matrix_kind const kind);

/*  Shared state: device handles, the fp64 reference matrix, the
    right-hand side. Constructed once per (n, k, kind) and passed to every
    method, so all methods are scored on bit-identical inputs. Members are
    public; the constructor exists to establish the invariant that the
    handles are live and d_a holds the generated matrix.

    d_a is the reference A in fp64 and is never written by a method. Methods
    that need A in another form derive it into their own state. */
class problem {

public:

    problem(
        std::size_t const  n,
        std::size_t const  k,
        matrix_kind const  kind,
        unsigned const     seed = 7u);

    ~problem();

    problem(problem const &)            = delete;
    problem &operator=(problem const &) = delete;

    std::size_t const n;
    std::size_t const k;
    matrix_kind const kind;

    cublasHandle_t     blas   = nullptr;
    cusolverDnHandle_t solver = nullptr;

    double *d_a = nullptr;   /* n x n, column major, fp64 reference */
    double *d_b = nullptr;   /* n x k, the right-hand sides         */

    /*  Device scratch the metrics need; owned here so a method's timed
        region never includes an allocation the harness could have hoisted. */
    double *d_partial = nullptr;   /* MAX_BLOCKS reduction partials */

    std::size_t n_elements() const {return n * n;}
    std::size_t n_rhs_elements() const {return n * k;}

    /*  cudaMalloc the bytes, record the pointer for the destructor, and
        return it. Every device allocation the harness makes goes through
        here so that nothing leaks across a sweep of many (n, k). */
    void *acquire(std::size_t const bytes);

private:

    std::vector<void *> _d_owned;

    void _generate(unsigned const seed);
    void _warm_libraries();
};

/*  Warm every vendor library before the first timed region.

    This is not optional hygiene. A cold cuSOLVER call costs on the order of
    100 ms of lazy initialization, and attributing that to whichever phase
    happens to run first has produced false findings three separate times in
    the work this harness replaces. The problem constructor calls this; it is
    exposed so a driver adding new vendor calls can warm those too. */
void warm_libraries(problem &prob);

} /* namespace harness */
