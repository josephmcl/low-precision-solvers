#pragma once

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>

#include <cstddef>
#include <string>
#include <vector>

/*  Device-specific tuning, loaded from a file rather than compiled in.

    The parameters that matter here are not portable. Between the two devices
    measured so far the best Ozaki blocking differed by 2x and the resulting R
    build time by 1.6x, tracking each device's tensor-core rate rather than
    anything about the algorithm. A default compiled into the source is
    therefore right for at most one machine, and silently wrong elsewhere —
    which is worse than being absent, because nothing in the output says the
    numbers came from a configuration tuned for a different chip.

    Resolution order, first hit wins:

      1. environment variable, if set (for sweeps: LPS_<KEY> with dots as
         underscores, e.g. LPS_RIR_BUILD_OZAKI_BLOCK)
      2. the file named by LPS_TUNING, if set
      3. tuning/<device>.conf, with the device name lowercased and
         non-alphanumerics collapsed to underscores
      4. the built-in default

    The file is `key = value`, one per line, `#` to end of line for comments.
    Unknown keys are reported, not ignored: a typo in a tuning file otherwise
    presents as "the tuning did nothing", which is indistinguishable from the
    parameter not mattering. */

namespace tuning {

class table {

public:

    /*  Default-constructed empty, then filled by load().

        Deliberately NOT a constructor that queries the device: as a member it
        would then run a CUDA call during the owner's member-initialization
        phase, before the owner's constructor body has created its handles.
        That ordering is easy to get wrong, invisible at the call site, and
        surfaced here as cuSOLVER returning NOT_INITIALIZED from an unrelated
        method several hundred lines away. */
    table() = default;

    /*  Loads by the resolution order above. device_name is what
        cudaDeviceProp.name reports; passed in so this needs no CUDA
        dependency. */
    void load(std::string const &device_name);

    int get(std::string const &key, int const fallback) const;

    /*  Where the values actually came from, for the run header. A report that
        does not say which tuning it used cannot be reproduced. */
    std::string const &source() const {return _source;}

    /*  Keys that were read but never requested by any caller — a typo, a stale
        key from an older version, or a parameter that no longer exists. */
    std::vector<std::string> unused() const;

private:

    struct entry {
        std::string key;
        int         value = 0;
        mutable bool read = false;
    };

    std::vector<entry> _entries;
    std::string        _source = "built-in defaults";

    bool _load(std::string const &path);
};

/*  The process-wide tuning table.

    A free function rather than a member of the problem bundle: making it a
    member forced every translation unit that touches a problem to also see
    <string> and <vector>, changing that struct's layout in files compiled as
    device code. Tuning is a property of the machine, not of any one problem,
    so it belongs here. */
table &current();


/*  Canonical file name for a device: lowercase, non-alphanumerics to
    underscores, collapsed. "NVIDIA GeForce RTX 5090" -> "nvidia_geforce_rtx_5090". */
std::string file_name_for(std::string const &device_name);

} /* namespace tuning */
