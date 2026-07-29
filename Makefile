# GPU backend. Common objects are listed once; each entry point gets its own
# *_MAIN variable and link rule, so a new binary is an entry here rather than
# a new build system.

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
OBJ_DIR   := build/gpu
BIN_DIR   := bin

# Shared objects, entry points excluded.
COMMON    := \
	$(OBJ_DIR)/definitions.o \
	$(OBJ_DIR)/error.o       \
	$(OBJ_DIR)/timing.o      \
	$(OBJ_DIR)/problem.o     \
	$(OBJ_DIR)/metrics.o     \
	$(OBJ_DIR)/solver.o      \
	$(OBJ_DIR)/solve_direct.o \
	$(OBJ_DIR)/factor_solve_vendor_irs.o \
	$(OBJ_DIR)/ozaki.o \
	$(OBJ_DIR)/solve_split_mpir.o

SWEEP_MAIN := $(OBJ_DIR)/main_sweep.o
OZTEST_MAIN := $(OBJ_DIR)/main_ozaki_test.o

.PHONY: all clean

all: $(BIN_DIR)/lps-sweep $(BIN_DIR)/lps-ozaki-test

$(BIN_DIR)/lps-sweep: $(COMMON) $(SWEEP_MAIN) | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LDLIBS)

$(BIN_DIR)/lps-ozaki-test: $(COMMON) $(OZTEST_MAIN) | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LDLIBS)


$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu | $(OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) -dc $< -o $@

$(OBJ_DIR) $(BIN_DIR):
	mkdir -p $@

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)
