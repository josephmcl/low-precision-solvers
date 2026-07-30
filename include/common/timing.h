#pragma once

#include "common/error.h"

#include <algorithm>

#include "common/definitions.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <vector>

namespace timing {


/*  Event-timed regions, repeated, reported as median and spread.

    Three rules are enforced here rather than left to the caller, because
    each of them has silently produced a wrong result in the work this
    harness replaces:

    1.  CUDA events, not wall clock. A wall-clock delta around a device call
        also charges host-side setup and whatever the driver deferred into
        it. One cold library call, measured this way, was read as an
        algorithmic cost for five rounds of "optimization".

    2.  Repeat and take the median with the spread. Single runs of one
        method varied 13% here, which is wider than several margins that
        were reported as wins.

    3.  The same repeat count for every method. A runner that wrapped only
        two of four methods in its repeat loop compared one solve against
        ten and reached a published-looking energy conclusion that had to be
        retracted. n_repeats lives in the sweep configuration, not in any
        method, so it cannot be applied unevenly. */

struct sample {
    double median = 0.;
    double low    = 0.;
    double high   = 0.;

    /*  (high - low) / median. Compare against any margin before believing
        it: a 2% margin under a 13% spread is not a result. */
    double spread() const {
        return (median > 0.)? (high - low) / median : 0.;
    }
};

sample summarize(std::vector<double> const &ms);

/*  A pair of CUDA events with the elapsed-time call folded in, so a timed
    region reads as start / stop / elapsed and cannot forget to
    synchronize. */
class stopwatch {

public:

    stopwatch();
    ~stopwatch();

    stopwatch(stopwatch const &)            = delete;
    stopwatch &operator=(stopwatch const &) = delete;

    void start();

    /*  Records the stop event, synchronizes on it, and returns
        milliseconds. */
    double stop();

private:

    cudaEvent_t _begin = nullptr;
    cudaEvent_t _end   = nullptr;
};

} /* namespace timing */
