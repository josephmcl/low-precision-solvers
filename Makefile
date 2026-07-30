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
NVCCFLAGS := -std=c++17 -O3 $(INCLUDE) \
             -arch=$(CUDA_ARCH) \
             -Xcompiler -Wall,-Wpedantic

LDLIBS    := -lcublas -lcusolver

SRC_DIR   := source/gpu
TEST_DIR  := test
OBJ_DIR   := build/gpu
TOBJ_DIR  := build/test
BIN_DIR   := bin

COMMON    := \
	$(OBJ_DIR)/definitions.o \
	$(OBJ_DIR)/error.o       \
	$(OBJ_DIR)/timing.o      \
	$(OBJ_DIR)/problem.o     \
	$(OBJ_DIR)/metrics.o     \
	$(OBJ_DIR)/ozaki.o       \
	$(OBJ_DIR)/solver.o      \
	$(OBJ_DIR)/solve_direct.o \
	$(OBJ_DIR)/solve_split_mpir.o \
	$(OBJ_DIR)/solve_rir.o \
	$(OBJ_DIR)/factor_solve_vendor_irs.o

SWEEP_MAIN  := $(TOBJ_DIR)/main_sweep.o
PROBE_MAIN  := $(TOBJ_DIR)/main_probe.o
OZTEST_MAIN := $(TOBJ_DIR)/main_ozaki_test.o

.PHONY: all clean

all: $(BIN_DIR)/lps-sweep $(BIN_DIR)/lps-probe $(BIN_DIR)/lps-ozaki-test

$(BIN_DIR)/lps-sweep: $(COMMON) $(SWEEP_MAIN) | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LDLIBS)

$(BIN_DIR)/lps-probe: $(COMMON) $(PROBE_MAIN) | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LDLIBS)

$(BIN_DIR)/lps-ozaki-test: $(COMMON) $(OZTEST_MAIN) | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LDLIBS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu | $(OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) -dc $< -o $@

$(TOBJ_DIR)/%.o: $(TEST_DIR)/%.cu | $(TOBJ_DIR)
	$(NVCC) $(NVCCFLAGS) -dc $< -o $@

$(TOBJ_DIR)/%.o: $(TEST_DIR)/%.cpp | $(TOBJ_DIR)
	$(NVCC) $(NVCCFLAGS) -dc -x cu $< -o $@

$(OBJ_DIR) $(TOBJ_DIR) $(BIN_DIR):
	mkdir -p $@

clean:
	rm -rf build $(BIN_DIR)
