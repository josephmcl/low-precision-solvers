#include "common/definitions.h"
#include "common/error.h"
#include "common/timing.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

/*  Device rates and per-method capacity. Run this before the sweep.

    WHY THE RATES COME FIRST. Which method wins is not a property of the
    algorithm, it is a property of the device's fp32:fp64 ratio — and that
    ratio varies by more than two orders across parts of the same generation.
    Measured on four devices, the fp64-free schemes beat a direct fp64 solve
    where tf32:fp64 was 68x and above, and lost to it where the ratio was 29x.
    A speedup number is not portable; the ratio is, including to hardware that
    does not exist yet. So the harness reports it rather than naming chips.

    The rule that fell out: fp64-free solvers pay when fp32:fp64 is roughly 10
    or more, and are counterproductive near 2.

    WHY CAPACITY IS SEPARATE. Speed is contingent on that ratio; the storage
    ratios are not. They reproduced identically on two devices with a 34x
    difference in fp64 throughput, which is what makes capacity the durable
    claim and speed the situational one. */

namespace {

/*  Sustained GEMM rate. Repeats until at least a fixed wall time has elapsed,
    so a fast device is not measured on one launch, and discards the first
    pass because cuBLAS selects kernels per shape. */
double gemm_rate(
    cublasHandle_t     blas,
    std::size_t const  n,
    cublasComputeType_t const compute,
    cudaDataType const type,
    void              *d_a,
    void              *d_b,
    void              *d_c) {

    float  const one_f = 1.f, zero_f = 0.f;
    double const one_d = 1., zero_d = 0.;

    bool const is_double = (type == CUDA_R_64F);
    void const *alpha = is_double? static_cast<void const *>(&one_d)
                                 : static_cast<void const *>(&one_f);
    void const *beta  = is_double? static_cast<void const *>(&zero_d)
                                 : static_cast<void const *>(&zero_f);

    auto once = [&]() {
        CUBLAS_CHECK(cublasGemmEx(
            blas,
            CUBLAS_OP_N, CUBLAS_OP_N,
            static_cast<int>(n), static_cast<int>(n), static_cast<int>(n),
            alpha,
            d_a, type, static_cast<int>(n),
            d_b, type, static_cast<int>(n),
            beta,
            d_c, type, static_cast<int>(n),
            compute, CUBLAS_GEMM_DEFAULT));
    };

    once();
    CUDA_CHECK(cudaDeviceSynchronize());

    int const n_repeats = 5;
    timing::stopwatch watch;
    watch.start();
    for (int r = 0; r != n_repeats; ++r)
        once();
    double const ms = watch.stop();

    /*  2n^3 flops per GEMM: one multiply and one add per MAC. */
    double const flops = 2. * static_cast<double>(n) * static_cast<double>(n) *
                         static_cast<double>(n) * n_repeats;
    return flops / (ms * 1.e-3) / 1.e12;
}

/*  Largest n that actually allocates, by binary search, so the number reflects
    the allocator rather than arithmetic on paper. Ozaki piece buffers are
    included: they are live during the build and the solve, and leaving them
    out was how an earlier storage claim came to be overstated by 8x. */
long long max_n(
    double const  bytes_per_n2,
    bool const    with_pieces,
    int const     block,
    int const     n_pieces) {

    long long low = 1024, high = 400000, best = 0;

    while (low <= high) {

        long long mid = (low + high) / 2;
        mid = (mid / 1024) * 1024;
        if (mid < low)
            mid = low;

        std::size_t need = static_cast<std::size_t>(
            bytes_per_n2 * static_cast<double>(mid) * static_cast<double>(mid));
        if (with_pieces)
            need += static_cast<std::size_t>(mid) *
                    static_cast<std::size_t>(block) *
                    static_cast<std::size_t>(n_pieces) * 4u;

        void *d_p = nullptr;
        if (cudaMalloc(&d_p, need) == cudaSuccess) {
            CUDA_CHECK(cudaFree(d_p));
            best = mid;
            low  = mid + 1024;
        }
        else {
            cudaGetLastError();
            high = mid - 1024;
        }
    }

    return best;
}

} /* namespace */

int main(int argc, char **argv) {

    /*  Unbuffered. A harness that loses its output when it crashes turns a
        one-line failure into a bisection: three separate debugging rounds here
        were spent re-running a segfault whose diagnostic had already been
        printed into a buffer that was then discarded. The cost is negligible —
        this program prints tens of lines, not millions. */
    std::cout.setf(std::ios::unitbuf);

    std::size_t const n = (argc > 1)?
        static_cast<std::size_t>(std::atoll(argv[1])) : 8192;

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    std::size_t free_bytes = 0, total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));

    std::cout << "device  " << prop.name << "\n"
              << "memory  " << std::fixed << std::setprecision(1)
              << total_bytes / 1.e9 << " GB total, "
              << free_bytes / 1.e9 << " GB free\n\n";

    cublasHandle_t blas = nullptr;
    CUBLAS_CHECK(cublasCreate(&blas));

    {
        void *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
        CUDA_CHECK(cudaMalloc(&d_a, n * n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_b, n * n * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_c, n * n * sizeof(double)));
        /*  REAL DATA, NOT ZEROS. These buffers were once cudaMemset to 0, and
            that silently corrupted every data-dependent rate: the fp64
            fixed-point emulation short-circuits on all-zero input and read
            3.17x native, where on real operands it is 1.79-2.23x. A rate
            measured on zeros is not a rate. */
        {
            std::vector<double> host(n * n);
            for (std::size_t i = 0; i != n * n; ++i)
                host[i] = 2. * (double(i % 1009) / 1009.) - 1.;
            CUDA_CHECK(cudaMemcpy(d_a, host.data(), n * n * sizeof(double),
                                  cudaMemcpyHostToDevice));
            for (std::size_t i = 0; i != n * n; ++i)
                host[i] = 2. * (double(i % 1013) / 1013.) - 1.;
            CUDA_CHECK(cudaMemcpy(d_b, host.data(), n * n * sizeof(double),
                                  cudaMemcpyHostToDevice));
        }

        double const fp64 = gemm_rate(
            blas, n, CUBLAS_COMPUTE_64F, CUDA_R_64F, d_a, d_b, d_c);
        double const fp32 = gemm_rate(
            blas, n, CUBLAS_COMPUTE_32F, CUDA_R_32F, d_a, d_b, d_c);
        double const tf32 = gemm_rate(
            blas, n, CUBLAS_COMPUTE_32F_FAST_TF32, CUDA_R_32F, d_a, d_b, d_c);

        /*  fp16 and bf16 WITH AN FP32 ACCUMULATOR — the only variant relevant
            to Ozaki. The scheme's exactness bound is 2*bits + log2(K) <= 24,
            and the 24 is the fp32 accumulator's mantissa, so accumulating in
            fp32 leaves the bound unchanged and makes this a pure throughput
            question.

            Why it is worth measuring at all: fp16 carries 11 significand bits,
            exactly as many as TF32, so a piece that fits one fits the other and
            the piece count does not move. On consumer Blackwell the two run at
            the SAME tensor rate, which is why fp16 was measured as 1.03-1.45x
            slower there and dropped. That is a property of those parts, not of
            the method, and it has to be re-asked on a datacenter part. */
        double const fp16 = gemm_rate(
            blas, n, CUBLAS_COMPUTE_32F, CUDA_R_16F, d_a, d_b, d_c);
        double const bf16 = gemm_rate(
            blas, n, CUBLAS_COMPUTE_32F, CUDA_R_16BF, d_a, d_b, d_c);

        /*  EMULATED fp32 and fp64 (Blackwell, CUDA 12.9+).
            32F_EMULATED_16BFX9 splits each fp32 operand into 3 bf16 values and
            runs 9 tensor-core products — cuBLAS's own Ozaki. It matters here
            because R-IR's solve is dominated by a plain SIMT fp32 GEMM (R*X)
            running at the fp32 rate while the tensor cores idle.
            64F_EMULATED_FIXEDPOINT matters for BASELINE FAIRNESS: if fp64 can
            be emulated fast on this part, then "direct fp64" measured against
            the native 1.05 TFLOP/s rate is an artificially weak baseline, and
            any speedup quoted against it is inflated.

            Emulation may be unsupported for a given shape/type, in which case
            cuBLAS reports an error and the rate reads 0 — which is the honest
            answer, not a failure to handle. */
        double const fp32_emu = gemm_rate(
            blas, n, CUBLAS_COMPUTE_32F_EMULATED_16BFX9, CUDA_R_32F,
            d_a, d_b, d_c);
        double const fp64_emu = gemm_rate(
            blas, n, CUBLAS_COMPUTE_64F_EMULATED_FIXEDPOINT, CUDA_R_64F,
            d_a, d_b, d_c);

        std::cout << "GEMM rates at n = " << n << "\n"
                  << "  " << std::left << std::setw(10) << "fp64"
                  << std::right << std::setw(10) << std::setprecision(2)
                  << fp64 << " TFLOP/s\n"
                  << "  " << std::left << std::setw(10) << "fp32"
                  << std::right << std::setw(10) << fp32 << " TFLOP/s\n"
                  << "  " << std::left << std::setw(10) << "tf32"
                  << std::right << std::setw(10) << tf32 << " TFLOP/s\n"
                  << "  " << std::left << std::setw(10) << "fp16/fp32acc"
                  << std::right << std::setw(10) << fp16 << " TFLOP/s\n"
                  << "  " << std::left << std::setw(10) << "bf16/fp32acc"
                  << std::right << std::setw(10) << bf16 << " TFLOP/s\n\n"
                  << "  fp16 : tf32  " << std::setprecision(2)
                  << (tf32 > 0.? fp16 / tf32 : 0.) << "x"
                  << "   (>1 means an fp16 Ozaki path could pay;"
                     " ~1 means it cannot)\n"
                  << "  bf16 : tf32  " << (tf32 > 0.? bf16 / tf32 : 0.)
                  << "x   (bf16 has 8 significand bits against tf32's 11,"
                     " so it needs MORE pieces)\n\n"
                  << "emulated (Blackwell)\n"
                  << "  " << std::left << std::setw(14) << "fp32 emu bf16x9"
                  << std::right << std::setw(10) << fp32_emu << " TFLOP/s"
                  << "   vs native fp32 "
                  << (fp32 > 0.? fp32_emu / fp32 : 0.) << "x\n"
                  << "  " << std::left << std::setw(14) << "fp64 emu fixed"
                  << std::right << std::setw(10) << fp64_emu << " TFLOP/s"
                  << "   vs native fp64 "
                  << (fp64 > 0.? fp64_emu / fp64 : 0.) << "x\n\n";

        if (fp64 > 0.) {
            double const ratio_32 = fp32 / fp64;
            double const ratio_tf = tf32 / fp64;
            std::cout << "  fp32 : fp64  " << std::setprecision(1)
                      << ratio_32 << "x\n"
                      << "  tf32 : fp64  " << ratio_tf << "x\n\n"
                      << "  verdict: fp64-free solvers "
                      << ((ratio_32 >= 10.)? "should pay here"
                                           : "are likely counterproductive here")
                      << " (rule: fp32:fp64 >~ 10)\n\n";
        }

        CUDA_CHECK(cudaFree(d_a));
        CUDA_CHECK(cudaFree(d_b));
        CUDA_CHECK(cudaFree(d_c));
    }

    CUBLAS_CHECK(cublasDestroy(blas));

    /*  Piece buffers sized as the stored-product configuration, the larger of
        the two, so the capacity numbers are the conservative ones. */
    int const block    = 256;
    int const n_pieces = 9;

    struct scheme {char const *name; double bytes_per_n2; bool pieces;};
    scheme const schemes[] = {
        {"direct fp64    (8n2)",  storage::DIRECT_FP64, false},
        {"vendor IRS    (12n2)",  storage::VENDOR_IRS,  false},
        {"split-MPIR    (12n2)",  storage::SPLIT_MPIR,  true },
        {"R-IR fp32 R    (8n2)",  storage::RIR_FP32_R,  true },
        {"R-IR bf16 R    (6n2)",  storage::RIR_BF16_R,  true }
    };

    std::cout << "capacity: largest n that allocates\n"
              << "  " << std::left << std::setw(22) << "scheme"
              << std::right << std::setw(10) << "max n"
              << std::setw(14) << "elements" << "\n";

    long long reference = 0;
    std::vector<long long> best;

    for (int i = 0; i != 5; ++i) {
        long long const m = max_n(
            schemes[i].bytes_per_n2, schemes[i].pieces, block, n_pieces);
        best.push_back(m);
        if (schemes[i].bytes_per_n2 == storage::SPLIT_MPIR && schemes[i].pieces)
            reference = m;

        std::cout << "  " << std::left << std::setw(22) << schemes[i].name
                  << std::right << std::setw(10) << m
                  << std::setw(12) << std::setprecision(2)
                  << static_cast<double>(m) * static_cast<double>(m) / 1.e9
                  << "e9" << "\n";
    }

    if (reference > 0) {
        std::cout << "\n  relative to split-MPIR (the 12n2 schemes are the"
                     " ones a residual-storage method excludes):\n";
        for (int i = 0; i != 5; ++i)
            std::cout << "    " << std::left << std::setw(22) << schemes[i].name
                      << std::right << std::setprecision(3)
                      << "  n x" << static_cast<double>(best[i]) /
                                    static_cast<double>(reference)
                      << "   elements x"
                      << (static_cast<double>(best[i]) *
                          static_cast<double>(best[i])) /
                         (static_cast<double>(reference) *
                          static_cast<double>(reference))
                      << "\n";

        std::cout << "\n  Note the exclusion is against the 12n2 schemes only."
                     " A direct fp64 solve is also 8n2, so at these sizes the"
                     " residual-storage advantage over *it* is not capacity --"
                     " it is that A need not stay resident.\n";
    }

    return 0;
}
