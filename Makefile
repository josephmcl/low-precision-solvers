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

INCLUDE   := -Iinclude

# C++17 is the CUDA floor; host flags carry the same warning posture as the
# rest of the project. New code must compile clean under these.
# -MMD -MP emit a .d per object listing the headers it used, which is what
# makes a header edit rebuild its dependents. Without this, editing a header
# leaves stale objects compiled against the OLD struct layout linked against
# fresh ones — an ODR violation that presents as a vendor handle read from the
# wrong offset, i.e. a library call failing with NOT_INITIALIZED in a method
# that has nothing wrong with it. That cost most of a debugging session.
NVCCFLAGS := -std=c++17 -O3 $(INCLUDE) \
             -arch=$(CUDA_ARCH) \
             -MMD -MP \
             -Xcompiler -Wall,-Wpedantic

LDLIBS    := -lcublas -lcusolver

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
	$(COM_OBJ)/ozaki.o       \
	$(COM_OBJ)/trsm.o        \
	$(COM_OBJ)/solver.o      \
	$(OBJ_DIR)/solve_direct.o \
	$(OBJ_DIR)/solve_split_mpir.o \
	$(OBJ_DIR)/solve_rir.o \
	$(OBJ_DIR)/factor_solve_vendor_irs.o

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

.PHONY: all clean

all: $(BIN_DIR)/lps-sweep $(BIN_DIR)/lps-probe $(BIN_DIR)/lps-ozaki-test \
     $(BIN_DIR)/lps-ablate $(BIN_DIR)/lps-kappa \
     $(BIN_DIR)/lps-rcheck $(BIN_DIR)/lps-profile \
     $(BIN_DIR)/lps-oom $(BIN_DIR)/lps-energy \
     $(BIN_DIR)/lps-kappacheck

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
