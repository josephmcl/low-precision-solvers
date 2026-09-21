# GPU backend.
#
# Library objects (source/gpu, each paired with a header in include/gpu) are
# listed once in COMMON. Entry points live in test/ and are deliberately
# NOT paired with headers: nothing includes them, so a header would only be
# a place for their internals to leak out of. Each gets its own *_MAIN
# variable and link rule, so a new binary is an entry here rather than a new
# build system.
#
# test/ is .cpp where the file has no device code and .cu where it does;
# main_ozaki_test needs kernels to build its reference operands.

NVCC      := nvcc
CUDA_ARCH ?= native

# The vendored headers are part of this build: ozaki2.h uses df32.cuh for the
# fp64-free combine, and the arm's TU needs the whole set.
INCLUDE   := -Iinclude -Ivendor/nfp64gmresir/cuda

# wgmma and TMA exist only under the arch-specific targets (compute_90a,
# sm_100a); -arch=sm_90 drops them silently and takes a slower path. The
# -gencode pair carries the 'a' suffix through both halves. NOTE native
# resolves to sm_90 on Hopper, not sm_90a, so the wgmma path needs
# an explicit  make CUDA_ARCH=sm_90a
ifeq ($(CUDA_ARCH),native)
ARCH_FLAGS := -arch=native
else
ARCH_FLAGS := -gencode arch=compute_$(patsubst sm_%,%,$(CUDA_ARCH)),code=$(CUDA_ARCH)
endif

# C++17 is the CUDA floor; host flags carry the same warning posture as the
# rest of the project. New code must compile clean under these.
# -MMD -MP emit a .d per object listing the headers it used, which is what
# makes a header edit rebuild its dependents. Without this, editing a header
# leaves stale objects compiled against the OLD struct layout linked against
# fresh ones — an ODR violation that presents as a vendor handle read from the
# wrong offset, i.e. a library call failing with NOT_INITIALIZED in a method
# that has nothing wrong with it. That cost most of a debugging session.
NVCCFLAGS := -std=c++17 -O3 $(INCLUDE) \
             $(ARCH_FLAGS) \
             -MMD -MP \
             -Xcompiler -Wall,-Wpedantic

# fast-math reassociates and breaks the error-free transformations (TwoSum,
# TwoProd) the DF32 carrier and reference.h are built on. It emits no fp64
# and raises no error, so neither the SASS gate nor the status checks can
# see it — the build has to refuse.
FASTMATH := $(strip $(findstring use_fast_math,$(NVCCFLAGS)) \
                    $(findstring ffast-math,$(NVCCFLAGS)))
ifneq ($(FASTMATH),)
$(error refusing to build: '$(FASTMATH)' in NVCCFLAGS breaks the error-free \
        transformations. --fmad=true is fine; TwoProd pins it with __fmaf_rn.)
endif

LDLIBS    := -lcublas -lcusolver -lcuda

INT8LU_OBJ  := build/gpu/factor_int8lu.o
VENDOR_DIR  := vendor/nfp64gmresir/cuda
VENDOR_INC  := -I$(VENDOR_DIR)
VENDOR_OBJ  := build/gpu/panel_persist.o

SRC_DIR   := source/gpu
COM_DIR   := source/common
COM_OBJ   := build/common
TEST_DIR  := test
OBJ_DIR   := build/gpu
TOBJ_DIR  := build/test
PROF_DIR  := profile
POBJ_DIR  := build/profile
BIN_DIR   := bin

COMMON    := \
	$(COM_OBJ)/definitions.o \
	$(COM_OBJ)/error.o       \
	$(COM_OBJ)/timing.o      \
	$(COM_OBJ)/convert.o     \
	$(COM_OBJ)/factorize.o   \
	$(COM_OBJ)/tuning.o      \
	$(COM_OBJ)/problem.o     \
	$(COM_OBJ)/metrics.o     \
	$(COM_OBJ)/exact_crt.o   \
	$(COM_OBJ)/ozaki2.o      \
	$(COM_OBJ)/ozaki.o       \
	$(COM_OBJ)/trsm.o        \
	$(COM_OBJ)/solver.o      \
	$(OBJ_DIR)/solve_direct.o \
	$(OBJ_DIR)/solve_split_mpir.o \
	$(OBJ_DIR)/solve_rir.o \
	$(OBJ_DIR)/solve_int8lu.o \
	$(OBJ_DIR)/factor_solve_vendor_irs.o \
	$(INT8LU_OBJ) \
	$(VENDOR_OBJ)

SWEEP_MAIN  := $(TOBJ_DIR)/main_sweep.o
PROBE_MAIN  := $(TOBJ_DIR)/main_probe.o
OZTEST_MAIN := $(TOBJ_DIR)/main_ozaki_test.o
ABLATE_MAIN := $(TOBJ_DIR)/main_ablate.o
KAPPA_MAIN  := $(TOBJ_DIR)/main_kappa.o
RCHECK_MAIN := $(TOBJ_DIR)/main_rcheck.o
PROF_MAIN   := $(POBJ_DIR)/main_profile.o
OOM_MAIN    := $(POBJ_DIR)/main_oom.o
ENERGY_MAIN := $(POBJ_DIR)/main_energy.o
KCHECK_MAIN := $(POBJ_DIR)/main_kappacheck.o
DDCHECK_MAIN:= $(TOBJ_DIR)/main_ddcheck.o
I8PROBE_MAIN:= $(TOBJ_DIR)/main_int8lu_probe.o
EXCRT_MAIN  := $(TOBJ_DIR)/main_exact_crt.o
OIICRT_MAIN := $(TOBJ_DIR)/main_oii_crt.o
LIFTD_MAIN  := $(TOBJ_DIR)/main_lift_dump.o
OIIGEMM_MAIN:= $(TOBJ_DIR)/main_oii_gemm.o
INSTR_DIR   := build/instrumented
# panel_persist.cu is a cooperative-launch TU and must stay relocatable
# (-dc); int8lu_factor references its launcher unconditionally, so it links
# even when PANELPERSIST is never set.

.PHONY: all clean

all: $(BIN_DIR)/lps-sweep $(BIN_DIR)/lps-probe $(BIN_DIR)/lps-ozaki-test \
     $(BIN_DIR)/lps-ablate $(BIN_DIR)/lps-kappa \
     $(BIN_DIR)/lps-rcheck $(BIN_DIR)/lps-profile \
     $(BIN_DIR)/lps-oom $(BIN_DIR)/lps-energy \
     $(BIN_DIR)/lps-kappacheck $(BIN_DIR)/lps-ddcheck \
     $(BIN_DIR)/lps-int8lu-probe $(BIN_DIR)/lps-exact-crt $(BIN_DIR)/lps-oii-crt \
     $(BIN_DIR)/lps-lift-dump $(BIN_DIR)/lps-oii-gemm

$(BIN_DIR)/lps-sweep: $(COMMON) $(SWEEP_MAIN) | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LDLIBS)

$(BIN_DIR)/lps-probe: $(COMMON) $(PROBE_MAIN) | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LDLIBS)

$(BIN_DIR)/lps-ablate: $(COMMON) $(ABLATE_MAIN) | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LDLIBS)

$(BIN_DIR)/lps-kappa: $(COMMON) $(KAPPA_MAIN) | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LDLIBS)

$(BIN_DIR)/lps-rcheck: $(COMMON) $(RCHECK_MAIN) | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LDLIBS)

$(BIN_DIR)/lps-ozaki-test: $(COMMON) $(OZTEST_MAIN) | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LDLIBS)

# Whole-program (-c, NOT -dc): the vendored wgmma trailing kernel serializes
# its mma.async instructions under relocatable device code -- ptxas C7509,
# "Extern calls in the function". Measured here: the remark appears with -dc
# and is absent with -c. Upstream compiled one TU per binary and never hit it.
# This is also the only TU permitted to include int8lu.cuh.
$(INT8LU_OBJ): $(SRC_DIR)/factor_int8lu.cu | $(OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) $(VENDOR_INC) -c $< -o $@

$(OBJ_DIR)/panel_persist.o: $(VENDOR_DIR)/panel_persist.cu | $(OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) $(VENDOR_INC) -dc $< -o $@

$(BIN_DIR)/lps-int8lu-probe: $(COMMON) $(COM_OBJ)/reference.o $(I8PROBE_MAIN) | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LDLIBS)

# Export harness. It includes an INSTRUMENTED copy of int8lu.cuh generated
# from the pristine vendor header, so the import stays byte-identical and the
# md5 manifest keeps checking. It deliberately does not link the arm: that TU
# includes the same header, and two includers is a duplicate symbol.
$(INSTR_DIR)/int8lu_instrumented.cuh: $(VENDOR_DIR)/int8lu.cuh tools/instrument_int8lu.py
	python3 tools/instrument_int8lu.py $@

$(LIFTD_MAIN): $(TEST_DIR)/main_lift_dump.cu $(INSTR_DIR)/int8lu_instrumented.cuh | $(TOBJ_DIR)
	$(NVCC) $(NVCCFLAGS) -I$(INSTR_DIR) $(VENDOR_INC) -c $< -o $@

$(BIN_DIR)/lps-lift-dump: $(COM_OBJ)/definitions.o $(COM_OBJ)/error.o \
                          $(COM_OBJ)/timing.o $(COM_OBJ)/convert.o \
                          $(COM_OBJ)/problem.o $(COM_OBJ)/tuning.o \
                          $(VENDOR_OBJ) $(LIFTD_MAIN) \
                          | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LDLIBS)

$(BIN_DIR)/lps-oii-gemm: $(COM_OBJ)/ozaki2.o $(COM_OBJ)/error.o \
                         $(COM_OBJ)/definitions.o $(OBJ_DIR)/oii_gemm.o \
                         $(OIIGEMM_MAIN) | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LDLIBS)

$(BIN_DIR)/lps-oii-crt: $(COM_OBJ)/ozaki2.o $(OIICRT_MAIN) | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ -lcudart

$(BIN_DIR)/lps-exact-crt: $(COM_OBJ)/exact_crt.o $(COM_OBJ)/error.o $(COM_OBJ)/definitions.o $(EXCRT_MAIN) | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ -lcudart

# Host-only; does not link COMMON so it runs on a box with no GPU.
$(BIN_DIR)/lps-ddcheck: $(COM_OBJ)/reference.o $(DDCHECK_MAIN) | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@

$(COM_OBJ)/%.o: $(COM_DIR)/%.cu | $(COM_OBJ)
	$(NVCC) $(NVCCFLAGS) -dc $< -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu | $(OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) -dc $< -o $@

$(TOBJ_DIR)/%.o: $(TEST_DIR)/%.cu | $(TOBJ_DIR)
	$(NVCC) $(NVCCFLAGS) -dc $< -o $@

$(TOBJ_DIR)/%.o: $(TEST_DIR)/%.cpp | $(TOBJ_DIR)
	$(NVCC) $(NVCCFLAGS) -dc -x cu $< -o $@

$(OBJ_DIR) $(COM_OBJ) $(TOBJ_DIR) $(POBJ_DIR) $(BIN_DIR):
	mkdir -p $@

# Assert named kernels emit zero fp64 SASS. The conversions are in the list
# too -- F2F.F64 is how fp64 usually re-enters. Name in-path kernels only;
# out-of-path fp64 (metrics, reference) is legitimate.
#
#   make sass-gate GATE_BIN=bin/lps-sweep GATE_SYMS='k_a k_b'
FP64_RE   := DADD|DMUL|DFMA|DSETP|DMNMX|DMMA|F2F\.F64|F2F\.F32\.F64|F2F\.F64\.F32|D2F|F2D
GATE_BIN  ?=
GATE_SYMS ?=

.PHONY: sass-gate
sass-gate:
	@test -n "$(GATE_BIN)" || { echo "usage: make sass-gate GATE_BIN=<binary> GATE_SYMS='sym ...'"; exit 2; }
	@test -n "$(GATE_SYMS)" || { echo "sass-gate: no GATE_SYMS given, nothing asserted"; exit 2; }
	@tools/sass_gate.sh $(GATE_BIN) $(GATE_SYMS)


clean:
	rm -rf build $(BIN_DIR)

# Wildcard, not an enumeration of targets. Listing them by name means every
# new binary silently loses header dependency tracking until someone remembers
# to add it here — which happened, and presented as a link error against a
# constructor signature that had changed three files away. Any .d that exists
# gets included, so a new target is covered the moment it first compiles.
$(POBJ_DIR)/%.o: $(PROF_DIR)/%.cpp | $(POBJ_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(POBJ_DIR)/%.o: $(PROF_DIR)/%.cu | $(POBJ_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(BIN_DIR)/lps-oom: $(OOM_MAIN) | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LDLIBS)

$(BIN_DIR)/lps-energy: $(COMMON) $(ENERGY_MAIN) | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LDLIBS) -lnvidia-ml

$(BIN_DIR)/lps-profile: $(COMMON) $(PROF_MAIN) | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LDLIBS)

$(BIN_DIR)/lps-kappacheck: $(COMMON) $(KCHECK_MAIN) | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LDLIBS)

-include $(wildcard $(COM_OBJ)/*.d $(OBJ_DIR)/*.d $(TOBJ_DIR)/*.d $(POBJ_DIR)/*.d)
