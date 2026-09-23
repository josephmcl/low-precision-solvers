#pragma once

#include <cstdint>

/*  Keyed stochastic rounding for the slicer.

    A port of the oracle's ONE dither spec (`sr_slicing.py`, R1). The
    rounding decision for a position is a pure function of the key

        (i, j, level, step, stream salt)

    and nothing else: no RNG state, no call order, no data dependence.
    That is what makes SR usable here at all. Every gate in this project
    rests on bit-identity -- `lps-absorb` sweeps the depth dial and
    compares factors bitwise, `lps-consistency` compares batched against
    column -- and a stateful generator would make all of them
    untestable. Counter-based keying keeps SR reproducible across
    backends, block decompositions and visit order, so the existing
    gates keep their meaning.

    `step` is the ancestor field, and it is part of the spec rather than
    an option: it must advance with the update or panel index, or the
    frozen field's (1/2 - u) conditional bias accumulates coherently
    across steps. The oracle's REPORT.md records that disagreement; the
    old frozen-key-only spelling is gone there and is not reproduced
    here.

    The row operand and the column operand of a product are separate
    streams, so a shared position never means a shared draw.

    Deliberate violations live in the oracle's `keyed_sr.py` as
    schedules S3 (ancestor-sharing) and S3b (sharing one draw across a
    contraction). They exist to be distinguishable from the canonical
    schedule; `sr_share_j` below reproduces S3b so the hardware side can
    be checked against the same negative control rather than only
    against the case that should work. */

namespace sr {

/*  Stream salts, matching the oracle byte for byte. */
std::uint64_t constexpr SALT_ROWS = 0x9E3779B97F4A7C15ull;
std::uint64_t constexpr SALT_COLS = 0x51ED270100000001ull;

std::uint64_t constexpr MIX_I = 0x9E3779B97F4A7C15ull;
std::uint64_t constexpr MIX_J = 0xC2B2AE3D27D4EB4Full;

#if defined(__CUDACC__)
#define SR_HD __host__ __device__ __forceinline__
#else
#define SR_HD inline
#endif

/*  splitmix64 finalizer. The oracle keeps two spellings, one masking
    first for python-int hygiene and one relying on array wraparound;
    both are the same function in uint64 arithmetic, which is what this
    is. */
SR_HD std::uint64_t mix64(std::uint64_t h) {
    h ^= h >> 30;
    h *= 0xBF58476D1CE4E5B9ull;
    h ^= h >> 27;
    h *= 0x94D049BB133111EBull;
    return h ^ (h >> 31);
}

/*  The non-positional half of the key. */
SR_HD std::uint64_t base_key(int const level, int const step,
                             std::uint64_t const salt) {
    std::uint64_t const l = static_cast<std::uint64_t>(level);
    std::uint64_t const s = static_cast<std::uint64_t>(step);
    return mix64(salt
                 + mix64(s * 0x27D4EB2F165667C5ull
                         + mix64(l * 0x165667B19E3779F9ull
                                 + 0x94D049BB133111EBull)));
}

/*  u in [0,1) at position (i,j). 53 bits, exactly the oracle's
    (h >> 11) / 2^53, so the comparison can be bitwise rather than
    statistical. */
SR_HD double dither(std::uint64_t const key,
                    std::uint64_t const i, std::uint64_t const j) {
    std::uint64_t const u = mix64(key + i * MIX_I + j * MIX_J);
    return static_cast<double>(u >> 11) * (1.0 / 9007199254740992.0);
}

/*  One stochastically rounded quantum: floor plus a Bernoulli on the
    fraction. Mean-zero per entry with variance frac(1-frac), which is
    what makes the variance-field identity exact instead of approximate
    -- the property lem:mart and thm:conc are stated over. */
SR_HD int quantize(double const q, double const u) {
    double const f = ::floor(q);
    return static_cast<int>(f) + ((u < (q - f))? 1 : 0);
}

}
