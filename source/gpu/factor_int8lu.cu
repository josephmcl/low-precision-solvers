#include "common/int8lu_arm.h"

#include "common/timing.h"

/*  int8lu.cuh calls std::sort without including <algorithm>; upstream never
    saw it because its one driver includes <algorithm> first. Vendor files are
    kept verbatim, so the include goes here. */
#include <algorithm>

#include "int8lu.cuh"

namespace int8lu_arm {

struct state {
    std::size_t    n       = 0;
    int            b       = 0;
    int            kfac    = 0;
    cublasHandle_t blas    = nullptr;
    Int8LUScratch  scratch = {};
};

state *create(
    std::size_t const n,
    int const         b,
    int const         kfac) {

    state *s = new state;
    s->n    = n;
    s->b    = b;
    s->kfac = kfac;

    if (!CUBLAS_CHECK(cublasCreate(&s->blas))) {
        delete s;
        return nullptr;
    }

    int8lu_scratch_alloc(s->scratch, static_cast<int>(n), b, kfac);
    return s;
}

void destroy(state *s) {

    if (s == nullptr)
        return;

    int8lu_scratch_free(s->scratch);
    if (s->blas != nullptr)
        CUBLAS_CHECK(cublasDestroy(s->blas));

    delete s;
}

double factor(
    state *s,
    float *d_hi,
    float *d_lo,
    int   *d_piv) {

    if (s == nullptr)
        return 0.;

    timing::stopwatch watch;
    watch.start();

    int8lu_factor(s->blas, static_cast<int>(s->n), s->b, s->kfac,
                  UPD_INT8, d_hi, d_lo, d_piv, s->scratch);

    return watch.stop();
}

} /* namespace int8lu_arm */
