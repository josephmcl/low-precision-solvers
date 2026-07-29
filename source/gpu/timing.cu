#include "gpu/timing.h"

#include "gpu/error.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <vector>

namespace timing {

using type::real_t;

sample summarize(std::vector<real_t> const &ms) {

    sample out;
    if (ms.empty())
        return out;

    std::vector<real_t> sorted(ms);
    std::sort(sorted.begin(), sorted.end());

    std::size_t const n = sorted.size();
    out.low    = sorted.front();
    out.high   = sorted.back();
    out.median = (n % 2 == 1)?
        sorted[n / 2] : 0.5 * (sorted[n / 2 - 1] + sorted[n / 2]);

    return out;
}

stopwatch::stopwatch() {

    CUDA_CHECK(cudaEventCreate(&_begin));
    CUDA_CHECK(cudaEventCreate(&_end));
}

stopwatch::~stopwatch() {

    if (_begin != nullptr)
        CUDA_CHECK(cudaEventDestroy(_begin));
    if (_end != nullptr)
        CUDA_CHECK(cudaEventDestroy(_end));
}

void stopwatch::start() {

    CUDA_CHECK(cudaEventRecord(_begin));
}

real_t stopwatch::stop() {

    CUDA_CHECK(cudaEventRecord(_end));
    CUDA_CHECK(cudaEventSynchronize(_end));

    float elapsed = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed, _begin, _end));

    return static_cast<real_t>(elapsed);
}

} /* namespace timing */
