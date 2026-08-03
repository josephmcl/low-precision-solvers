#include "common/solver.h"
#include "common/metrics.h"

#include <nvml.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

/*  Per-method energy, measured through NVML.

    WHY THIS EXISTS AND WHY IT IS WRITTEN THIS WAY. An earlier version of this
    measurement was RETRACTED: its runner applied the repeat count to only two
    of four methods, so it compared the energy of one direct solve against ten
    R-IR solves and reported the ratio as an efficiency result. Nothing about
    the power sampling was wrong; the experiment design was.

    The fix is structural rather than careful. `repeats` is a property of the
    LOOP here, not of any method, and no method can see it or change it — the
    same count is applied to every entry in the registry by construction, so
    the failure mode cannot recur without deleting the loop. That is the only
    reason to trust this file over the one it replaces.

    WHAT IS REPORTED, per method:

      joules            integrated power over the timed region
      joules_idle_adj   the same, minus the measured idle draw
      watts_mean        joules / seconds
      joules_per_solve  joules / repeats — THE comparable number
      j_per_solve_adj   idle-adjusted, per solve

    SETTLING, AND WHY IT IS THE WHOLE BALLGAME. An earlier version sampled the
    idle baseline after a 500 ms sleep. That is far too short: measured on a
    5090, board power after a workload decays 167 W (P0) -> 23 W (P5) at 5 s ->
    2.6 W (P8) at 10 s. A 500 ms sample therefore catches a TRANSIENT whose
    value depends entirely on how recently the card was used — the same command
    read 3.27 W on a cold box and 104.4 W when run straight after another
    experiment, a 30x spread that propagated into every *_adj column.

    So the IDLE BASELINE is taken from a settled card: `settle()` polls NVML
    until power has been low and flat for a fixed window. Idle then reads ~3 W
    reproducibly, and the adjustment it produces is ~0.4% — not the 2x the bad
    baseline implied.

    SETTLING BEFORE EACH METHOD IS THE OPPOSITE OF CORRECT, and was tried here
    before being backed out. A settled card restarts at 195 MHz and ramps up
    INSIDE the measured window, diluting mean power — and diluting it more for
    a short method than a long one. Measured: R-IR (5 s) fell 13% and
    split-MPIR (8.9 s) fell 5%, a duration-dependent bias pointing exactly the
    way that flatters the method under test. Each method instead gets a fixed
    COMMON warmup (`--warmup`, default 3 s of methods[0]) so it enters its
    measured region at a consistent P0 clock and thermal state that does not
    depend on which method is about to run.

    ORDER IS A CONFOUND. Methods used to be measured in registry order, so the
    first absorbed whatever heat and boost state the previous process left
    behind — `direct fp64` read 80.5 J after another experiment against 76.7 J
    otherwise. The method order now ROTATES each round, so no method owns the
    first slot.

    CLOCKS CANNOT BE PINNED HERE. `nvidia-smi -lgc` needs permissions a
    container does not have, so boost variance is handled statistically instead
    of by control: `rounds` independent measurements per method, reported as
    median with min/max. Measured residual spread is 0.3-1.9%.

    WHAT IS REPORTED, per method: the median over rounds, plus min and max so
    the spread is visible in the figure rather than asserted in prose.

    HONEST LIMITS. NVML reports board power at roughly 10-100 ms granularity
    with unspecified filtering, so a single 40 ms solve is at or below the
    sampling resolution — that is why the loop runs `repeats` of them and
    divides. Energy below ~100 ms of work should not be quoted from this tool.

    Build:  make bin/lps-energy
    Run:    bin/lps-energy 8192 --k 2048 --repeats 20 --rounds 3 --csv       */

namespace {

using harness::problem;

/*  Background power sampler. Integrates milliwatt readings over wall time by
    trapezoid; the sampling interval is not assumed constant because NVML
    queries block for an unspecified time. */
class sampler {
public:
    explicit sampler(nvmlDevice_t dev) : _dev(dev) {}

    void start() {
        _joules = 0.; _n = 0; _peak_w = 0.; _run = true;
        _thread = std::thread([this] { _loop(); });
    }

    void stop() {
        _run = false;
        if (_thread.joinable()) _thread.join();
    }

    double joules()  const { return _joules; }
    double peak_w()  const { return _peak_w; }
    std::size_t samples() const { return _n; }

    /*  One instantaneous reading, for the idle baseline. */
    double watts_now() const {
        unsigned mw = 0;
        return (nvmlDeviceGetPowerUsage(_dev, &mw) == NVML_SUCCESS)?
               mw / 1000. : 0.;
    }

    unsigned temp_now() const {
        unsigned t = 0;
        return (nvmlDeviceGetTemperature(_dev, NVML_TEMPERATURE_GPU, &t)
                == NVML_SUCCESS)? t : 0u;
    }

private:
    void _loop() {
        auto last = std::chrono::steady_clock::now();
        double last_w = watts_now();
        while (_run) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            auto const now = std::chrono::steady_clock::now();
            double const w = watts_now();
            double const dt =
                std::chrono::duration<double>(now - last).count();
            _joules += 0.5 * (w + last_w) * dt;   /* trapezoid */
            if (w > _peak_w) _peak_w = w;
            last = now; last_w = w; ++_n;
        }
    }

    nvmlDevice_t      _dev;
    std::thread       _thread;
    std::atomic<bool> _run{false};
    double            _joules = 0., _peak_w = 0.;
    std::size_t       _n = 0;
};

/*  Wait until the card is quiescent: power low and flat for `window` samples.

    Measured decay on a 5090 after a real workload is 167 W -> 23 W (5 s) ->
    2.6 W (10 s), so a fixed short sleep samples a moving target. Polling for
    flatness rather than sleeping a fixed time is what makes the idle baseline
    reproducible, and it is also what puts every method's measurement at the
    same starting card state.

    FLATNESS ALONE IS NOT ENOUGH, and an earlier version that used it returned
    25 W on one invocation against 3.5 W on two others. The decay is a
    STAIRCASE, not a slope — P0 167 W, P5 ~23 W, P8 ~2.6 W — and the P5 tread
    is perfectly flat while it lasts. A flatness test lands on it.

    Two defences: poll for at least `min_s` (longer than the observed ~10 s
    fall to P8, so the plateau is walked through rather than stopped on), and
    return the MINIMUM observed rather than the last window. The minimum is the
    right estimator regardless: transients can only push power above the floor,
    never below it, so the minimum converges on the floor from one side.

    Returns the settled power. Gives up after `timeout_s` and reports what it
    reached, because a box with another tenant on the GPU may never settle and
    a hang is worse than a documented approximation. */
double settle(sampler const &smp, double timeout_s = 90., double min_s = 12.,
              int window = 4, double flat_w = 1.5) {

    auto const t0 = std::chrono::steady_clock::now();
    std::vector<double> recent;
    double seen_min = 1e30;

    auto const elapsed = [&] {
        return std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count();
    };

    for (;;) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        double const w = smp.watts_now();
        if (w > 0. && w < seen_min) seen_min = w;

        recent.push_back(w);
        if (static_cast<int>(recent.size()) > window)
            recent.erase(recent.begin());

        if (elapsed() >= min_s &&
            static_cast<int>(recent.size()) == window) {
            auto const lo = *std::min_element(recent.begin(), recent.end());
            auto const hi = *std::max_element(recent.begin(), recent.end());
            if (hi - lo <= flat_w) return seen_min;
        }
        if (elapsed() > timeout_s) {
            std::cerr << "[energy] settle timed out after " << timeout_s
                      << " s, floor seen " << seen_min
                      << " W - treating as settled\n";
            return (seen_min < 1e29)? seen_min : 0.;
        }
    }
}

double median_of(std::vector<double> v) {
    if (v.empty()) return 0.;
    std::sort(v.begin(), v.end());
    std::size_t const m = v.size() / 2;
    return (v.size() % 2)? v[m] : 0.5 * (v[m - 1] + v[m]);
}

struct energy_row {
    std::string method;
    std::size_t n = 0, k = 0, repeats = 0;
    double seconds = 0., joules = 0., joules_adj = 0.;
    double watts_mean = 0., watts_peak = 0.;
    double j_per_solve = 0., j_per_solve_adj = 0.;
    double backward = 0.;
    unsigned temp_c = 0;
    std::size_t samples = 0;
};

} /* namespace */

int main(int argc, char **argv) {

    std::cout.setf(std::ios::unitbuf);

    std::size_t n = 8192, k = 2048, repeats = 20, rounds = 3;
    double warmup_s = 3., soak_s = 30.;
    bool csv = false, do_settle = true;

    for (int i = 1; i < argc; ++i) {
        std::string const a = argv[i];
        auto next = [&]() { return (i + 1 < argc)? argv[++i] : "0"; };
        if      (a == "--k")         k = std::atoll(next());
        else if (a == "--repeats")   repeats = std::atoll(next());
        else if (a == "--rounds")    rounds = std::atoll(next());
        else if (a == "--warmup")    warmup_s = std::atof(next());
        else if (a == "--soak")      soak_s = std::atof(next());
        else if (a == "--no-settle") do_settle = false;
        else if (a == "--csv")       csv = true;
        else if (a[0] != '-')        n = std::atoll(a.c_str());
    }
    if (rounds == 0) rounds = 1;

    if (nvmlInit() != NVML_SUCCESS) {
        std::cerr << "[energy] nvmlInit failed - is NVML available?\n";
        return 1;
    }
    nvmlDevice_t dev;
    if (nvmlDeviceGetHandleByIndex(0, &dev) != NVML_SUCCESS) {
        std::cerr << "[energy] no device 0\n";
        return 1;
    }

    sampler smp(dev);

    /*  Idle baseline, measured from a SETTLED card rather than after a fixed
        sleep. The fixed sleep is what made this number swing 30x between runs;
        see the header. */
    if (!csv && do_settle)
        std::cerr << "[energy] settling before idle baseline...\n";
    double const idle_w = do_settle? settle(smp) : smp.watts_now();

    if (!csv) {
        std::cout << "energy   n=" << n << "  k=" << k
                  << "  repeats=" << repeats << "  rounds=" << rounds << "\n"
                  << "idle draw  " << idle_w << " W, measured settled"
                     "  (subtracted in the *_adj columns)\n\n";
        std::cout << "  method              J/solve      min      max"
                     "  spread   W_mean   backward\n";
    }
    else {
        std::cout << "n,k,repeats,rounds,method,idle_w,seconds,joules,"
                     "joules_adj,watts_mean,watts_peak,j_per_solve,"
                     "j_per_solve_min,j_per_solve_max,j_per_solve_spread,"
                     "j_per_solve_adj,backward,temp_c,samples\n";
    }

    problem prob(n, k, harness::matrix_kind::diag_dominant);
    double *d_x = static_cast<double *>(prob.acquire(n * k * sizeof(double)));

    std::vector<solver::method> const &methods = solver::registry();

    /*  THERMAL SOAK. The per-method warmup fixes the clock state but not the
        temperature: the card is coldest at process start, so whichever method
        happens to run first in round 1 is measured on a cold die and reads
        low. That showed as a 3.3% spread on `direct fp64` whose minimum was
        always round 1 — a drift, not noise, and rotation cannot remove it
        because it is a property of the clock rather than of the order.

        Running a fixed load to thermal steady state before ANY measurement
        removes it at the source. */
    if (soak_s > 0.) {
        if (!csv)
            std::cerr << "[energy] thermal soak " << soak_s << " s ("
                      << smp.temp_now() << " C)\n";
        auto const s0 = std::chrono::steady_clock::now();
        while (std::chrono::duration<double>(
                   std::chrono::steady_clock::now() - s0).count() < soak_s) {
            solver::state ss_;
            solver::run(d_x, prob.d_b, methods[0], ss_, prob);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        if (!csv)
            std::cerr << "[energy] soaked to " << smp.temp_now() << " C\n";
    }

    /*  One accumulated result per method; rounds append into these. */
    std::vector<energy_row>          rows(methods.size());
    std::vector<std::vector<double>> per_round(methods.size());

    for (std::size_t round = 0; round != rounds; ++round) {
        for (std::size_t j = 0; j != methods.size(); ++j) {

            /*  ROTATED ORDER. Without this the same method always runs first
                and always inherits whatever card state the previous process
                left; that alone moved `direct fp64` from 76.7 J to 80.5 J.
                Rotating means each method visits each slot across rounds. */
            std::size_t const mi = (j + round) % methods.size();

            /*  Warm up outside the measured region: first-call kernel
                selection and any lazy module load would otherwise be charged
                as energy. */
            { solver::state warm;
              solver::run(d_x, prob.d_b, methods[mi], warm, prob); }
            CUDA_CHECK(cudaDeviceSynchronize());

            /*  COMMON WARMUP, not a settle. Every method must enter its
                measured region from the same card state, and that state has to
                be the STEADY one.

                Settling to P8 first was tried and is wrong: the card restarts
                at 195 MHz and ramps up inside the measured window, so the mean
                power is diluted by the ramp — and diluted MORE for a short
                method than a long one. It moved R-IR (5 s) down 13% and
                split-MPIR (8.9 s) down 5%, a duration-dependent bias that
                flatters exactly the method under test.

                Running a FIXED workload for a fixed time instead puts the card
                at a consistent P0 clock and thermal state before every
                measurement, identical across methods because the warmup does
                not depend on which method follows it. */
            {
                auto const w0 = std::chrono::steady_clock::now();
                while (std::chrono::duration<double>(
                           std::chrono::steady_clock::now() - w0).count()
                       < warmup_s) {
                    solver::state ws_;
                    solver::run(d_x, prob.d_b, methods[0], ws_, prob);
                }
                CUDA_CHECK(cudaDeviceSynchronize());
            }

            auto const t0 = std::chrono::steady_clock::now();
            smp.start();

            /*  THE UNIFORM LOOP. `repeats` is fixed above and is not visible
                to any method — this is the invariant whose absence retracted
                the previous energy results. */
            for (std::size_t r = 0; r != repeats; ++r) {
                solver::state st;
                solver::run(d_x, prob.d_b, methods[mi], st, prob);
            }
            CUDA_CHECK(cudaDeviceSynchronize());

            smp.stop();
            auto const t1 = std::chrono::steady_clock::now();

            double const seconds =
                std::chrono::duration<double>(t1 - t0).count();
            double const joules = smp.joules();
            double const jps    = joules / static_cast<double>(repeats);

            per_round[mi].push_back(jps);

            /*  Keep the last round's detail; the headline is the median. */
            energy_row &row = rows[mi];
            row.method  = methods[mi].name;
            row.n = n; row.k = k; row.repeats = repeats;
            row.seconds = seconds;
            row.joules  = joules;
            row.joules_adj = joules - idle_w * seconds;
            row.watts_mean = (seconds > 0.)? joules / seconds : 0.;
            row.watts_peak = smp.peak_w();
            row.j_per_solve = jps;
            row.j_per_solve_adj = row.joules_adj
                                / static_cast<double>(repeats);
            row.temp_c  = smp.temp_now();
            row.samples = smp.samples();

            metrics::report const err = metrics::evaluate(d_x, nullptr, prob);
            row.backward = err.backward;

            if (!csv)
                std::cerr << "[energy] round " << (round + 1) << '/' << rounds
                          << "  " << row.method << "  " << jps << " J/solve\n";
        }
    }

    for (std::size_t mi = 0; mi != methods.size(); ++mi) {

        energy_row &row = rows[mi];
        auto const &v = per_round[mi];
        double const med = median_of(v);
        double const lo  = *std::min_element(v.begin(), v.end());
        double const hi  = *std::max_element(v.begin(), v.end());
        double const spread = (med > 0.)? (hi - lo) / med : 0.;

        /*  The median is the reported figure; min/max travel with it so a
            figure can carry error bars instead of a prose claim about
            stability. */
        row.j_per_solve = med;

        if (csv) {
            std::cout << n << ',' << k << ',' << repeats << ',' << rounds
                      << ',' << row.method << ',' << idle_w << ','
                      << row.seconds << ',' << row.joules << ','
                      << row.joules_adj << ',' << row.watts_mean << ','
                      << row.watts_peak << ',' << med << ','
                      << lo << ',' << hi << ',' << spread << ','
                      << row.j_per_solve_adj << ',' << row.backward << ','
                      << row.temp_c << ',' << row.samples << '\n';
        }
        else {
            std::printf("  %-18s %8.2f %8.2f %8.2f %6.2f%% %8.1f   %.2e\n",
                        row.method.c_str(), med, lo, hi, 100. * spread,
                        row.watts_mean, row.backward);
        }
    }

    if (!csv)
        std::cout << "\n  Median of " << rounds << " rounds. J/solve is the"
                     " comparable number; spread is the\n  observed"
                     " run-to-run variation - quote it rather than asserting"
                     " stability.\n\n  Every method ran exactly "
                  << repeats << " solves, entered its measured region from"
                     "\n  the same thermally-soaked, commonly-warmed card"
                     " state, and took each\n  slot in a rotated order. Those"
                     " are the fixes for the retracted and the\n  unstable"
                     " measurements, and each is enforced by the loop, not by"
                     " convention.\n";

    nvmlShutdown();
    return 0;
}
