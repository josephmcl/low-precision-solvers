#include "common/oii_gemm.h"

#include "common/error.h"

#include <cublas_v2.h>

#include <cstdio>

/*  Inline DF32 only -- df32.cuh declares no kernels and no file-scope
    state, so unlike int8lu.cuh it may be included by any number of TUs.
    The subtraction uses the vendored df_sub_acc (df_ext.cuh, also
    header-only) so the two arms subtract identically and the comparison
    is of the products. That file's own usage map puts the accurate
    variant on exactly this path: "carrier subtraction in the trailing
    update". The sloppy df_add has no relative accuracy under
    cancellation, which a Schur subtraction does. */
#include "df32.cuh"
#include "df_ext.cuh"

namespace oii {

namespace {

int const TPB = 256;

/*  cuBLAS int8 GEMM requires the contraction dimension -- which is the
    leading dimension of both operands in this layout -- to be a multiple
    of 4, and returns CUBLAS_STATUS_NOT_SUPPORTED otherwise. Every k-strided
    buffer is therefore allocated at the padded stride and zero filled once;
    the kernels only ever write h < k, so the pad stays zero and contributes
    nothing to any product. */
inline std::size_t pad4(std::size_t const k) {
    return (k + 3) & ~static_cast<std::size_t>(3);
}

/*  Reasons a call cannot produce a meaningful answer, raised by the kernels
    and read back once. Both were previously unchecked, and neither is
    hypothetical: Algorithm 2 divides by a row maximum, so on a zero row
    log2(0) is -inf and `(int)floor(-inf)` is undefined behaviour. It
    happens to come out right today, because a zero row of A only poisons
    its own shift and that row's product is zero anyway -- but a zero
    COLUMN of B poisons nu[j] for every row, and undefined behaviour that
    currently works is not a property to rely on. */
/*  Only one way left to fail: the scaled operands leaving the range the
    source permits them, which is |A'| < 2^53. Zero rows of A and zero
    columns of B are HANDLED -- see the zero-extension note at gemm(). */
int const FLAG_OVERFLOW  = 2;   /* A' or B' outside 2^53                 */

/*  The scaled operands are held in int64, not int32.

    They measure 31 bits on random operands at N = 8 -- int32 with no
    headroom -- and an operand whose line maximum is an exact power of
    two tips them over: B = I at N = 8 gives nu = 31 and B' = 2^31,
    one past int32, which is what the guard caught. The source's own
    bound is |A'| < 2^53 and the oracle holds these in fp64, so int32
    was never the right container; it was the one that happened to fit
    the cases measured first.

    Only the RESIDUES are int8, and they are unaffected: sym_mod already
    takes an int64. What this costs is 8 bytes per scaled entry instead
    of 4, on two m x k and k x n arrays. */
__device__ inline long long clamp_i53(long long const v, int *flags) {

    long long const lim = 1LL << 53;
    if (v >= lim || v <= -lim) {
        atomicOr(flags, FLAG_OVERFLOW);
        return (v > 0)? lim - 1 : -(lim - 1);
    }
    return v;
}

/*  Algorithm 2 line 3: row maxima of |A|, A row major m x k. One block per
    row; the reduction is over k, which is the panel width in the arm and
    so never large enough to want a two-stage reduction. */
__global__ void k_absmax_rows(
    int const     m,
    int const     k,
    double const *a,
    double       *out) {

    __shared__ double red[TPB];
    int const i = blockIdx.x;
    if (i >= m)
        return;

    double v = 0.;
    for (int h = threadIdx.x; h < k; h += blockDim.x)
        v = fmax(v, fabs(a[static_cast<std::size_t>(i) * k + h]));
    red[threadIdx.x] = v;
    __syncthreads();
    for (int o = blockDim.x >> 1; o > 0; o >>= 1) {
        if (threadIdx.x < o)
            red[threadIdx.x] = fmax(red[threadIdx.x], red[threadIdx.x + o]);
        __syncthreads();
    }
    if (threadIdx.x == 0)
        out[i] = red[0];
}

/*  Column maxima of |B|, B row major k x n. */
__global__ void k_absmax_cols(
    int const     k,
    int const     n,
    double const *b,
    double       *out) {

    __shared__ double red[TPB];
    int const j = blockIdx.x;
    if (j >= n)
        return;

    double v = 0.;
    for (int h = threadIdx.x; h < k; h += blockDim.x)
        v = fmax(v, fabs(b[static_cast<std::size_t>(h) * n + j]));
    red[threadIdx.x] = v;
    __syncthreads();
    for (int o = blockDim.x >> 1; o > 0; o >>= 1) {
        if (threadIdx.x < o)
            red[threadIdx.x] = fmax(red[threadIdx.x], red[threadIdx.x + o]);
        __syncthreads();
    }
    if (threadIdx.x == 0)
        out[j] = red[0];
}

/*  Lines 5 and 6. mup = 5 - floor(log2(max)), Abar = ceil(|A| 2^mup), which
    the source bounds by 2^6 so it is an int8. Abar stays ROW major. */
__global__ void k_prescale_a(
    int const     m,
    int const     k,
    int const     kp,
    double const *a,
    double const *amax,
    int          *mup,
    signed char  *abar,
    int          *flags) {

    int const i = blockIdx.x;
    if (i >= m)
        return;

    /*  lem:zero. A zero row of A contributes an identically zero output
        row, so it needs no shift and no slices; giving it mup = 0 and
        abar = 0 propagates exactly that. See the note at gemm(). */
    if (amax[i] == 0.) {
        if (threadIdx.x == 0)
            mup[i] = 0;
        for (int h = threadIdx.x; h < k; h += blockDim.x)
            abar[static_cast<std::size_t>(i) * kp + h] = 0;
        return;
    }

    /*  ilogb, NOT floor(log2(.)). CUDA's log2 is not correctly
        rounded, and at an exact power of two it can return just
        under the integer: log2(8.0) floored to 2, which made this
        shift one too large and pushed the scaled operand to exactly
        2^31. Random reference operands never land on a power of
        two, so the gate passed; the DF32 path was already right
        because it uses ilogbf. ilogb IS the exponent, exactly. */
    int const s = 5 - ilogb(amax[i]);
    if (threadIdx.x == 0)
        mup[i] = s;
    double const f = exp2(static_cast<double>(s));
    for (int h = threadIdx.x; h < k; h += blockDim.x)
        abar[static_cast<std::size_t>(i) * kp + h] =
            static_cast<signed char>(
                ceil(fabs(a[static_cast<std::size_t>(i) * k + h]) * f));
}

/*  Same for B, but Bbar comes out COLUMN major so cuBLAS can take it OP_N
    with ld = k against the OP_T A side. */
__global__ void k_prescale_b(
    int const     k,
    int const     kp,
    int const     n,
    double const *b,
    double const *bmax,
    int          *nup,
    signed char  *bbar,
    int          *flags) {

    int const j = blockIdx.x;
    if (j >= n)
        return;

    if (bmax[j] == 0.) {
        if (threadIdx.x == 0)
            nup[j] = 0;
        for (int h = threadIdx.x; h < k; h += blockDim.x)
            bbar[static_cast<std::size_t>(j) * kp + h] = 0;
        return;
    }

    /*  ilogb, NOT floor(log2(.)). CUDA's log2 is not correctly
        rounded, and at an exact power of two it can return just
        under the integer: log2(8.0) floored to 2, which made this
        shift one too large and pushed the scaled operand to exactly
        2^31. Random reference operands never land on a power of
        two, so the gate passed; the DF32 path was already right
        because it uses ilogbf. ilogb IS the exponent, exactly. */
    int const s = 5 - ilogb(bmax[j]);
    if (threadIdx.x == 0)
        nup[j] = s;
    double const f = exp2(static_cast<double>(s));
    for (int h = threadIdx.x; h < k; h += blockDim.x)
        bbar[static_cast<std::size_t>(j) * kp + h] =
            static_cast<signed char>(
                ceil(fabs(b[static_cast<std::size_t>(h) * n + j]) * f));
}

/*  Lines 8 to 12. Dbar = single_triangle_up(Cbar) -- the directed rounding
    is the point, an int-to-float conversion that rounded to nearest could
    understate the magnitude and put condition (5) at risk. Cbar is column
    major m x n.

    `rows` selects which margin is reduced: the row maxima give e and hence
    mu, the column maxima give f and hence nu. */
__global__ void k_shifts(
    int const     m,
    int const     n,
    int const    *cbar,
    int const    *pre,
    double const  pprime,
    double const  c_step,
    bool const    rows,
    int          *shift,
    int          *flags) {

    __shared__ float red[TPB];
    int const idx = blockIdx.x;
    int const len = rows? n : m;
    if (idx >= (rows? m : n))
        return;

    float v = 0.f;
    for (int t = threadIdx.x; t < len; t += blockDim.x) {
        int const c = rows? cbar[static_cast<std::size_t>(t) * m + idx]
                          : cbar[static_cast<std::size_t>(idx) * m + t];
        v = fmaxf(v, __int2float_ru(c));
    }
    red[threadIdx.x] = v;
    __syncthreads();
    for (int o = blockDim.x >> 1; o > 0; o >>= 1) {
        if (threadIdx.x < o)
            red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + o]);
        __syncthreads();
    }
    if (threadIdx.x != 0)
        return;

    /*  A zero maximum means the line contributes nothing to the
        product. log2f(0) is -inf and the int conversion after it is
        undefined, so the shift is left at the prescale's value; the
        operand is zero either way and the output line comes out an
        exact zero. */
    if (!(red[0] > 0.f)) {
        shift[idx] = pre[idx];
        return;
    }

    float const e = log2f(red[0]);
    /*  The fma is fp64 on fp32-valued arguments, as the source specifies;
        rounding it in fp32 would move the floor by one on ties. */
    double const step = fma(c_step, static_cast<double>(e), pprime);
    shift[idx] = pre[idx] + static_cast<int>(floor(step));
}

/*  Lines 13 and 14: A' = trunc(2^mu A) and B' = trunc(B 2^nu), as int32.
    31 bits at N = 8, so this is the narrowest container that holds them. */
__global__ void k_scale_trunc_a(
    int const     m,
    int const     k,
    int const     kp,
    double const *a,
    int const    *mu,
    long long    *ap,
    int          *flags) {

    int const i = blockIdx.x;
    if (i >= m)
        return;
    double const f = exp2(static_cast<double>(mu[i]));
    for (int h = threadIdx.x; h < k; h += blockDim.x)
        ap[static_cast<std::size_t>(i) * kp + h] = clamp_i53(
            static_cast<long long>(
                trunc(a[static_cast<std::size_t>(i) * k + h] * f)), flags);
}

__global__ void k_scale_trunc_b(
    int const     k,
    int const     kp,
    int const     n,
    double const *b,
    int const    *nu,
    long long    *bp,
    int          *flags) {

    int const j = blockIdx.x;
    if (j >= n)
        return;
    double const f = exp2(static_cast<double>(nu[j]));
    for (int h = threadIdx.x; h < k; h += blockDim.x)
        bp[static_cast<std::size_t>(j) * kp + h] = clamp_i53(
            static_cast<long long>(
                trunc(b[static_cast<std::size_t>(h) * n + j] * f)), flags);
}

/*  Residues for every modulus at once. int8 with the natural wrap: the
    symmetric range is [-floor(p/2), floor(p/2)], which is inside int8 for
    every modulus but 256, where the single value 128 wraps to -128. That
    is harmless and the source says so -- -128 == 128 (mod 256) -- and the
    int32 accumulation that follows is reduced mod p again anyway. */
__global__ void k_residues(
    int const        len,
    int const        n_moduli,
    int const       *p,
    long long const *src,
    signed char     *dst) {

    int const t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= len)
        return;
    long long const v = src[t];
    for (int l = 0; l != n_moduli; ++l) {
        long long r = v % p[l];
        if (r < 0)
            r += p[l];
        if (r > p[l] / 2)
            r -= p[l];
        dst[static_cast<std::size_t>(l) * len + t] =
            static_cast<signed char>(r);
    }
}

/*  Algorithm 3 plus line 15 of Algorithm 1: reduce each modulus' int32
    product into the symmetric range, reconstruct, and undo the shifts.
    Products are column major m x n; C is written ROW major. */
__global__ void k_crt_unscale(
    int const                   m,
    int const                   n,
    int const                   n_moduli,
    int const                  *prod,
    ozaki2::crt_constants const c,
    int const                  *mu,
    int const                  *nu,
    bool const                  free_cfg,
    double                     *out) {

    std::size_t const g = static_cast<std::size_t>(blockIdx.x) * blockDim.x
                        + threadIdx.x;
    if (g >= static_cast<std::size_t>(m) * n)
        return;
    int const i = static_cast<int>(g / n), j = static_cast<int>(g % n);

    /*  The residues are small integers. The fp64 configuration wants them
        as doubles because Algorithm 3's reduction is fp64; the fp64-free
        one wants floats and must not be handed a double to narrow. */
    double w[ozaki2::MAX_MODULI];
    float  wf[ozaki2::MAX_MODULI];
    for (int l = 0; l != n_moduli; ++l) {
        int const v = prod[static_cast<std::size_t>(l) * m * n
                           + static_cast<std::size_t>(j) * m + i];
        int const r = static_cast<int>(ozaki2::sym_mod(v, c.p[l]));
        w[l]  = static_cast<double>(r);
        wf[l] = static_cast<float>(r);
    }

    double cpp;
    if (free_cfg) {
        float hi, lo, lost;
        ozaki2::crt_fp32free(wf, c, ozaki2::MAX_LIMB, hi, lo, lost);
        cpp = static_cast<double>(hi) + static_cast<double>(lo);
    } else {
        cpp = ozaki2::crt_fp64(w, c);
    }

    out[g] = cpp * exp2(-static_cast<double>(mu[i]))
                 * exp2(-static_cast<double>(nu[j]));
}


/*  ---- fp64-free Algorithm 2 -------------------------------------------

    The operands arrive as DF32 pairs off the carrier, so every quantity
    below is derived from (hi, lo) with fp32 arithmetic and integers only.
    Three places need care, and each is exact rather than nearly so:

      the exponent      floor(log2|x|) for x = hi + lo. This is ilogb(hi)
                        except when |hi| is an exact power of two and lo
                        pulls the magnitude below it, where it is one less.

      ceil and trunc    of a scaled pair. Scaling by 2^s is exact on both
                        words, but the SUM is not representable, so the
                        rounding of hi+lo is recovered with two_sum32 and
                        the integer is corrected against it.

      line 11's floor   floor(fma(c, e, P')). The product c*e is split
                        exactly by fmaf and the addition by two_sum32, so
                        the floor is taken against the exact value.

    That last one is the one departure worth stating: the source rounds the
    fma to fp64 and floors THAT, while this floors the exact value. They
    differ only when the exact value lies within u_64 of an integer. The
    bitwise gate against the oracle is what says whether it ever happens. */

/*  Exponent of a DF32 pair. EXP_ZERO marks an exactly-zero entry, which
    has no exponent: returning -126 for it (as the first version did)
    makes an all-zero line indistinguishable from a line of denormals
    and defeats the zero-line detection downstream. */
int const EXP_ZERO = -30000;

__device__ inline int df_ilogb(float const hi, float const lo) {

    if (hi == 0.f)
        return (lo == 0.f)? EXP_ZERO : ilogbf(lo);
    int e = ilogbf(hi);
    float const m = ldexpf(fabsf(hi), -e);          /* in [1, 2) */
    if (m == 1.0f && ((hi > 0.f)? (lo < 0.f) : (lo > 0.f)))
        --e;
    return e;
}

/*  ceil(|hi + lo| * 2^s), which Algorithm 2 bounds by 2^6. */
__device__ inline int df_ceil_scaled(
    float const hi, float const lo, int const s) {

    /*  |hi + lo| = |hi| + sign(hi) lo, the pair being non-overlapping. */
    float const th = ldexpf(fabsf(hi), s);
    float const tl = ldexpf((hi < 0.f)? -lo : lo, s);
    float su, er;
    ozaki2::two_sum32(th, tl, su, er);

    float c = ceilf(su);
    if (er > c - su)
        c += 1.f;
    else if (er < (c - 1.f) - su)
        c -= 1.f;
    return static_cast<int>(c);
}

/*  trunc((hi + lo) * 2^s) as an int32. 31 bits at N = 8, which is why this
    goes through a 64-bit integer and never an fp32 intermediate.

    Do NOT two_sum the two scaled words and treat the remainder as a
    fraction. At N = 8 the shift reaches 31, so the high word lands near
    2^31 where ulp is 2^7 and the two_sum error is an integer as large as
    64 -- not a fractional correction at all. The first version assumed
    |r| < 1 and was wrong on 91 of 108 entries at N = 8 while staying
    exact at N <= 5, which is what the staged gate showed.

    Instead each scaled word is split into its own integer and fractional
    parts, which is exact because ldexpf is exact and truncf of a float is
    a float, and only the two fractions are combined. */
__device__ inline long long df_trunc_scaled(
    float const hi, float const lo, int const s) {

    float const th = ldexpf(hi, s);
    float const tl = ldexpf(lo, s);

    float const nh = truncf(th), nl = truncf(tl);
    long long   v  = static_cast<long long>(nh)
                   + static_cast<long long>(nl);
    float const fh = th - nh;                       /* exact, |fh| < 1 */
    float const fl = tl - nl;                       /* exact, |fl| < 1 */

    /*  |fh + fl| < 2, so one carry at most, taken exactly. */
    float sf, ef;
    ozaki2::two_sum32(fh, fl, sf, ef);
    float const c = truncf(sf);
    v += static_cast<long long>(c);
    float const r = (sf - c) + ef;                  /* |r| < 1 + eps */

    if (v >= 0) {
        if (r < 0.f) { if (v > 0) --v; }
        else if (r >= 1.f) ++v;
    } else {
        if (r > 0.f) ++v;
        else if (r <= -1.f) --v;
    }
    return v;
}

/*  floor(fma(c_step, e, Pprime)), exactly, in fp32. */
__device__ inline int ff_floor_step(
    float const e, float const c_step, float const pprime) {

    float const p  = c_step * e;
    float const pe = fmaf(c_step, e, -p);           /* exact */
    float s1, e1;
    ozaki2::two_sum32(p, pprime, s1, e1);

    float f = floorf(s1);
    float const r = (s1 - f) + e1 + pe;
    if (r < 0.f)
        f -= 1.f;
    else if (r >= 1.f)
        f += 1.f;
    return static_cast<int>(f);
}

/*  Row exponents of A. floor(log2(.)) is monotone, so the maximum of the
    per-entry exponents IS the exponent of the row maximum -- no reduction
    over magnitudes is needed, and the reduction stays in integers. */
__global__ void k_ff_prescale_a(
    int const     m,
    int const     k,
    int const     kp,
    int const     lda,
    float const  *ah,
    float const  *al,
    int          *mup,
    signed char  *abar,
    int          *flags) {

    __shared__ int red[TPB];
    int const i = blockIdx.x;
    if (i >= m)
        return;

    int e = EXP_ZERO;
    for (int h = threadIdx.x; h < k; h += blockDim.x) {
        std::size_t const t = static_cast<std::size_t>(i) * lda + h;
        int const q = df_ilogb(ah[t], al[t]);
        e = (q > e)? q : e;
    }
    red[threadIdx.x] = e;
    __syncthreads();
    for (int o = blockDim.x >> 1; o > 0; o >>= 1) {
        if (threadIdx.x < o)
            red[threadIdx.x] = (red[threadIdx.x] > red[threadIdx.x + o])?
                red[threadIdx.x] : red[threadIdx.x + o];
        __syncthreads();
    }

    /*  lem:zero, as in the fp64 path. */
    if (red[0] == EXP_ZERO) {
        if (threadIdx.x == 0)
            mup[i] = 0;
        for (int h = threadIdx.x; h < k; h += blockDim.x)
            abar[static_cast<std::size_t>(i) * kp + h] = 0;
        return;
    }

    int const s = 5 - red[0];
    if (threadIdx.x == 0)
        mup[i] = s;
    for (int h = threadIdx.x; h < k; h += blockDim.x) {
        std::size_t const t = static_cast<std::size_t>(i) * lda + h;
        abar[static_cast<std::size_t>(i) * kp + h] =
            static_cast<signed char>(df_ceil_scaled(ah[t], al[t], s));
    }
}

__global__ void k_ff_prescale_b(
    int const     k,
    int const     kp,
    int const     n,
    int const     ldb,
    float const  *bh,
    float const  *bl,
    int          *nup,
    signed char  *bbar,
    int          *flags) {

    __shared__ int red[TPB];
    int const j = blockIdx.x;
    if (j >= n)
        return;

    int e = EXP_ZERO;
    for (int h = threadIdx.x; h < k; h += blockDim.x) {
        std::size_t const t = static_cast<std::size_t>(h) * ldb + j;
        int const q = df_ilogb(bh[t], bl[t]);
        e = (q > e)? q : e;
    }
    red[threadIdx.x] = e;
    __syncthreads();
    for (int o = blockDim.x >> 1; o > 0; o >>= 1) {
        if (threadIdx.x < o)
            red[threadIdx.x] = (red[threadIdx.x] > red[threadIdx.x + o])?
                red[threadIdx.x] : red[threadIdx.x + o];
        __syncthreads();
    }

    /*  lem:zero, as in the fp64 path. */
    if (red[0] == EXP_ZERO) {
        if (threadIdx.x == 0)
            nup[j] = 0;
        for (int h = threadIdx.x; h < k; h += blockDim.x)
            bbar[static_cast<std::size_t>(j) * kp + h] = 0;
        return;
    }

    int const s = 5 - red[0];
    if (threadIdx.x == 0)
        nup[j] = s;
    for (int h = threadIdx.x; h < k; h += blockDim.x) {
        std::size_t const t = static_cast<std::size_t>(h) * ldb + j;
        bbar[static_cast<std::size_t>(j) * kp + h] =
            static_cast<signed char>(df_ceil_scaled(bh[t], bl[t], s));
    }
}

/*  Lines 8 to 12, fp32. Same shape as k_shifts but the fma is the exact
    fp32 one above rather than an fp64 instruction. */
__global__ void k_ff_shifts(
    int const     m,
    int const     n,
    int const    *cbar,
    int const    *pre,
    float const   pprime,
    float const   c_step,
    bool const    rows,
    int          *shift,
    int          *flags) {

    __shared__ float red[TPB];
    int const idx = blockIdx.x;
    int const len = rows? n : m;
    if (idx >= (rows? m : n))
        return;

    float v = 0.f;
    for (int t = threadIdx.x; t < len; t += blockDim.x) {
        int const c = rows? cbar[static_cast<std::size_t>(t) * m + idx]
                          : cbar[static_cast<std::size_t>(idx) * m + t];
        v = fmaxf(v, __int2float_ru(c));
    }
    red[threadIdx.x] = v;
    __syncthreads();
    for (int o = blockDim.x >> 1; o > 0; o >>= 1) {
        if (threadIdx.x < o)
            red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + o]);
        __syncthreads();
    }
    if (threadIdx.x != 0)
        return;
    if (!(red[0] > 0.f)) {
        shift[idx] = pre[idx];
        return;
    }
    shift[idx] = pre[idx] + ff_floor_step(log2f(red[0]), c_step, pprime);
}

__global__ void k_ff_scale_trunc_a(
    int const    m,
    int const    k,
    int const    kp,
    int const    lda,
    float const *ah,
    float const *al,
    int const   *mu,
    long long   *ap,
    int         *flags) {

    int const i = blockIdx.x;
    if (i >= m)
        return;
    for (int h = threadIdx.x; h < k; h += blockDim.x) {
        std::size_t const t = static_cast<std::size_t>(i) * lda + h;
        ap[static_cast<std::size_t>(i) * kp + h] =
            clamp_i53(df_trunc_scaled(ah[t], al[t], mu[i]), flags);
    }
}

__global__ void k_ff_scale_trunc_b(
    int const    k,
    int const    kp,
    int const    n,
    int const    ldb,
    float const *bh,
    float const *bl,
    int const   *nu,
    long long   *bp,
    int         *flags) {

    int const j = blockIdx.x;
    if (j >= n)
        return;
    for (int h = threadIdx.x; h < k; h += blockDim.x) {
        std::size_t const t = static_cast<std::size_t>(h) * ldb + j;
        bp[static_cast<std::size_t>(j) * kp + h] =
            clamp_i53(df_trunc_scaled(bh[t], bl[t], nu[j]), flags);
    }
}

/*  Algorithm 3 and line 15, entirely in fp32. Undoing the shifts is an
    exponent move on both words of the pair, so it is exact. */
__global__ void k_ff_crt_unscale(
    int const                   m,
    int const                   n,
    int const                   ldc,
    int const                   n_moduli,
    int const                  *prod,
    ozaki2::crt_constants const c,
    int const                  *mu,
    int const                  *nu,
    bool const                  subtract,
    float                      *out_hi,
    float                      *out_lo) {

    std::size_t const g = static_cast<std::size_t>(blockIdx.x) * blockDim.x
                        + threadIdx.x;
    if (g >= static_cast<std::size_t>(m) * n)
        return;
    int const i = static_cast<int>(g / n), j = static_cast<int>(g % n);

    float w[ozaki2::MAX_MODULI];
    for (int l = 0; l != n_moduli; ++l) {
        int const v = prod[static_cast<std::size_t>(l) * m * n
                           + static_cast<std::size_t>(j) * m + i];
        w[l] = static_cast<float>(ozaki2::sym_mod(v, c.p[l]));
    }

    float hi, lo, lost;
    ozaki2::crt_fp32free(w, c, ozaki2::MAX_LIMB, hi, lo, lost);

    /*  Undoing the shift is an exponent move on both words, so it is
        exact. */
    int const sh = -(mu[i] + nu[j]);
    float const rh = ldexpf(hi, sh), rl = ldexpf(lo, sh);

    std::size_t const o = static_cast<std::size_t>(i) * ldc + j;
    if (!subtract) {
        out_hi[o] = rh;
        out_lo[o] = rl;
        return;
    }

    /*  S22 -= C, the trailing update's own subtraction, in DF32. Kept here
        rather than in a separate pass for the reason the vendored epilogue
        keeps it: the product is already in registers and writing it out
        only to read it back doubles the traffic on the largest array in
        the factorization. */
    df32 const r = df_sub_acc(df_make(out_hi[o], out_lo[o]),
                              df_make(rh, rl));
    out_hi[o] = r.hi;
    out_lo[o] = r.lo;
}

} /* anonymous namespace */

struct state {
    std::size_t m = 0, k = 0, n = 0, kp = 0;
    int n_moduli = 0;
    ozaki2::crt_constants c;
    cublasHandle_t blas = nullptr;

    double *amax = nullptr, *bmax = nullptr;
    int    *mup = nullptr, *nup = nullptr, *mu = nullptr, *nu = nullptr;
    int    *dp = nullptr;                 /* the moduli, on the device */
    signed char *abar = nullptr, *bbar = nullptr;
    int    *cbar = nullptr;
    long long *ap = nullptr, *bp = nullptr;
    signed char *ares = nullptr, *bres = nullptr;
    int    *prod = nullptr;
    int    *flags = nullptr;
};

namespace {

template <typename T>
bool grab(T *&p, std::size_t const count) {
    return CUDA_CHECK(cudaMalloc(&p, count * sizeof(T)));
}

} /* anonymous namespace */

state *create(
    std::size_t const m,
    std::size_t const k,
    std::size_t const n,
    int const         n_moduli) {


    state *s = new state;
    s->m = m; s->k = k; s->n = n; s->kp = pad4(k);
    s->n_moduli = (n_moduli < ozaki2::MAX_MODULI)? n_moduli
                                                 : ozaki2::MAX_MODULI;
    s->c = ozaki2::make_crt_constants(s->n_moduli);

    if (!CUBLAS_CHECK(cublasCreate(&s->blas))) {
        destroy(s);
        return nullptr;
    }

    std::size_t const nm = static_cast<std::size_t>(s->n_moduli);
    bool ok = grab(s->amax, m) && grab(s->bmax, n)
           && grab(s->mup, m)  && grab(s->nup, n)
           && grab(s->mu, m)   && grab(s->nu, n)
           && grab(s->dp, ozaki2::MAX_MODULI)
           && grab(s->abar, m * s->kp) && grab(s->bbar, s->kp * n)
           && grab(s->cbar, m * n)
           && grab(s->ap, m * s->kp)   && grab(s->bp, s->kp * n)
           && grab(s->ares, nm * m * s->kp)
           && grab(s->bres, nm * s->kp * n)
           && grab(s->prod, nm * m * n)
           && grab(s->flags, 1);
    if (!ok) {
        destroy(s);
        return nullptr;
    }

    /*  Zero once: the pad columns are never written again. */
    CUDA_CHECK(cudaMemset(s->abar, 0, m * s->kp));
    CUDA_CHECK(cudaMemset(s->bbar, 0, s->kp * n));
    CUDA_CHECK(cudaMemset(s->ap, 0, m * s->kp * sizeof(long long)));
    CUDA_CHECK(cudaMemset(s->bp, 0, s->kp * n * sizeof(long long)));

    CUDA_CHECK(cudaMemcpy(s->dp, s->c.p,
                          ozaki2::MAX_MODULI * sizeof(int),
                          cudaMemcpyHostToDevice));
    return s;
}

void destroy(state *s) {

    if (s == nullptr)
        return;
    cudaFree(s->amax); cudaFree(s->bmax);
    cudaFree(s->mup);  cudaFree(s->nup);
    cudaFree(s->mu);   cudaFree(s->nu);   cudaFree(s->dp);
    cudaFree(s->abar); cudaFree(s->bbar); cudaFree(s->cbar);
    cudaFree(s->ap);   cudaFree(s->bp);
    cudaFree(s->ares); cudaFree(s->bres); cudaFree(s->prod);
    cudaFree(s->flags);
    if (s->blas != nullptr)
        cublasDestroy(s->blas);
    delete s;
}

namespace {

/*  One 4-byte read back per call, which also synchronizes. That is once
    per trailing update in the arm, against N + 1 GEMMs, so it is not a
    cost worth trading correctness for. */
bool report_flags(state const *s) {

    int f = 0;
    CUDA_CHECK(cudaMemcpy(&f, s->flags, sizeof f, cudaMemcpyDeviceToHost));
    if (f == 0)
        return true;
    if (f & FLAG_OVERFLOW)
        std::fprintf(stderr, "[oii] scaled operand outside int32: "
                             "condition (5) does not hold for these "
                             "operands at this modulus count\n");
    return false;
}

} /* anonymous namespace */

bool gemm(
    state        *s,
    int const     m,
    int const     n,
    double const *d_a,
    double const *d_b,
    double       *d_c,
    config const  cfg) {

    if (s == nullptr)
        return false;
    if (m > static_cast<int>(s->m) || n > static_cast<int>(s->n)) {
        std::fprintf(stderr, "[oii] %dx%d exceeds the capacity %zux%zu\n",
                     m, n, s->m, s->n);
        return false;
    }

    int const k = static_cast<int>(s->k);
    int const kp = static_cast<int>(s->kp);

    CUDA_CHECK(cudaMemset(s->flags, 0, sizeof(int)));

    k_absmax_rows<<<m, TPB>>>(m, k, d_a, s->amax);
    KERNEL_CHECK();
    k_absmax_cols<<<n, TPB>>>(k, n, d_b, s->bmax);
    KERNEL_CHECK();
    k_prescale_a<<<m, TPB>>>(m, k, kp, d_a, s->amax, s->mup, s->abar,
                                       s->flags);
    KERNEL_CHECK();
    k_prescale_b<<<n, TPB>>>(k, kp, n, d_b, s->bmax, s->nup, s->bbar,
                                       s->flags);
    KERNEL_CHECK();

    int const one = 1, zero = 0;
    if (!CUBLAS_CHECK(cublasGemmEx(
            s->blas, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &one,
            s->abar, CUDA_R_8I, kp, s->bbar, CUDA_R_8I, kp, &zero,
            s->cbar, CUDA_R_32I, m, CUBLAS_COMPUTE_32I,
            CUBLAS_GEMM_DEFAULT)))
        return false;

    k_shifts<<<m, TPB>>>(m, n, s->cbar, s->mup,
                                 s->c.Pprime, s->c.c_step, true, s->mu,
                                 s->flags);
    KERNEL_CHECK();
    k_shifts<<<n, TPB>>>(m, n, s->cbar, s->nup,
                                 s->c.Pprime, s->c.c_step, false, s->nu,
                                 s->flags);
    KERNEL_CHECK();

    k_scale_trunc_a<<<m, TPB>>>(m, k, kp, d_a, s->mu, s->ap, s->flags);
    KERNEL_CHECK();
    k_scale_trunc_b<<<n, TPB>>>(k, kp, n, d_b, s->nu, s->bp, s->flags);
    KERNEL_CHECK();

    int const la = m * kp, lb = kp * n;
    k_residues<<<(la + TPB - 1) / TPB, TPB>>>(
                la, s->n_moduli, s->dp, s->ap, s->ares);
    KERNEL_CHECK();
    k_residues<<<(lb + TPB - 1) / TPB, TPB>>>(
                lb, s->n_moduli, s->dp, s->bp, s->bres);
    KERNEL_CHECK();

    for (int l = 0; l != s->n_moduli; ++l)
        if (!CUBLAS_CHECK(cublasGemmEx(
                s->blas, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &one,
                s->ares + static_cast<std::size_t>(l) * la, CUDA_R_8I, kp,
                s->bres + static_cast<std::size_t>(l) * lb, CUDA_R_8I, kp,
                &zero,
                s->prod + static_cast<std::size_t>(l) * m * n,
                CUDA_R_32I, m, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT)))
            return false;

    std::size_t const out = static_cast<std::size_t>(m) * n;
    k_crt_unscale<<<(out + TPB - 1) / TPB, TPB>>>(
                m, n, s->n_moduli, s->prod, s->c, s->mu, s->nu,
                cfg == config::ff, d_c);
    KERNEL_CHECK();

    return CUDA_CHECK(cudaGetLastError()) && report_flags(s);
}

bool gemm_df32(
    state       *s,
    int const    m,
    int const    n,
    float const *d_ah,
    float const *d_al,
    int const    lda,
    float const *d_bh,
    float const *d_bl,
    int const    ldb,
    float       *d_ch,
    float       *d_cl,
    int const    ldc,
    bool const   subtract) {

    if (s == nullptr)
        return false;
    if (m > static_cast<int>(s->m) || n > static_cast<int>(s->n)) {
        std::fprintf(stderr, "[oii] %dx%d exceeds the capacity %zux%zu\n",
                     m, n, s->m, s->n);
        return false;
    }

    int const k  = static_cast<int>(s->k);
    int const kp = static_cast<int>(s->kp);

    CUDA_CHECK(cudaMemset(s->flags, 0, sizeof(int)));

    float const pp = static_cast<float>(s->c.Pprime);
    float const cs = static_cast<float>(s->c.c_step);

    k_ff_prescale_a<<<m, TPB>>>(m, k, kp, lda, d_ah, d_al, s->mup, s->abar,
                                s->flags);
    KERNEL_CHECK();
    k_ff_prescale_b<<<n, TPB>>>(k, kp, n, ldb, d_bh, d_bl, s->nup, s->bbar,
                                s->flags);
    KERNEL_CHECK();

    int const one = 1, zero = 0;
    if (!CUBLAS_CHECK(cublasGemmEx(
            s->blas, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &one,
            s->abar, CUDA_R_8I, kp, s->bbar, CUDA_R_8I, kp, &zero,
            s->cbar, CUDA_R_32I, m, CUBLAS_COMPUTE_32I,
            CUBLAS_GEMM_DEFAULT)))
        return false;

    k_ff_shifts<<<m, TPB>>>(m, n, s->cbar, s->mup, pp, cs, true, s->mu,
                            s->flags);
    KERNEL_CHECK();
    k_ff_shifts<<<n, TPB>>>(m, n, s->cbar, s->nup, pp, cs, false, s->nu,
                            s->flags);
    KERNEL_CHECK();

    k_ff_scale_trunc_a<<<m, TPB>>>(m, k, kp, lda, d_ah, d_al, s->mu, s->ap,
                                   s->flags);
    KERNEL_CHECK();
    k_ff_scale_trunc_b<<<n, TPB>>>(k, kp, n, ldb, d_bh, d_bl, s->nu, s->bp,
                                   s->flags);
    KERNEL_CHECK();

    int const la = m * kp, lb = kp * n;
    k_residues<<<(la + TPB - 1) / TPB, TPB>>>(
        la, s->n_moduli, s->dp, s->ap, s->ares);
    KERNEL_CHECK();
    k_residues<<<(lb + TPB - 1) / TPB, TPB>>>(
        lb, s->n_moduli, s->dp, s->bp, s->bres);
    KERNEL_CHECK();

    for (int l = 0; l != s->n_moduli; ++l)
        if (!CUBLAS_CHECK(cublasGemmEx(
                s->blas, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &one,
                s->ares + static_cast<std::size_t>(l) * la, CUDA_R_8I, kp,
                s->bres + static_cast<std::size_t>(l) * lb, CUDA_R_8I, kp,
                &zero,
                s->prod + static_cast<std::size_t>(l) * m * n,
                CUDA_R_32I, m, CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT)))
            return false;

    std::size_t const out = static_cast<std::size_t>(m) * n;
    k_ff_crt_unscale<<<(out + TPB - 1) / TPB, TPB>>>(
        m, n, ldc, s->n_moduli, s->prod, s->c, s->mu, s->nu,
        subtract, d_ch, d_cl);
    KERNEL_CHECK();

    return CUDA_CHECK(cudaGetLastError()) && report_flags(s);
}

/*  Uses the capacities, so it is only meaningful right after a call made
    at full size -- which is what the gate does. */
void copy_scaling(state const *s, scaling_view &out) {

    if (s == nullptr)
        return;
    out.mu.resize(s->m);
    out.nu.resize(s->n);
    out.ap.resize(s->m * s->k);
    out.bp.resize(s->k * s->n);
    std::vector<long long> pa(s->m * s->kp), pb(s->kp * s->n);
    CUDA_CHECK(cudaMemcpy(out.mu.data(), s->mu, s->m * sizeof(int),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(out.nu.data(), s->nu, s->n * sizeof(int),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(pa.data(), s->ap, pa.size() * sizeof(long long),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(pb.data(), s->bp, pb.size() * sizeof(long long),
                          cudaMemcpyDeviceToHost));
    for (std::size_t i = 0; i != s->m; ++i)
        for (std::size_t h = 0; h != s->k; ++h)
            out.ap[i * s->k + h] = pa[i * s->kp + h];
    for (std::size_t j = 0; j != s->n; ++j)
        for (std::size_t h = 0; h != s->k; ++h)
            out.bp[j * s->k + h] = pb[j * s->kp + h];
}

} /* namespace oii */
