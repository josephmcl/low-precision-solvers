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
# 'native' is resolved here rather than handed to nvcc, because nvcc's own
# -arch=native gives sm_90 on Hopper and the wgmma path then refuses to
# assemble ("Instruction 'wgmma.wait_group' not supported on .target
# 'sm_90'"). Make stops at that error, the previous binary is still on
# disk, and the run that follows silently measures the OLD build -- which
# is indistinguishable from a change having no effect. __nvcc_device_query
# prints the bare capability (e.g. 90); the 'a' suffix exists from 90 up,
# so append it there and leave older parts alone.
ifeq ($(CUDA_ARCH),native)
NATIVE_CC := $(shell __nvcc_device_query 2>/dev/null || \
                     /usr/local/cuda/bin/__nvcc_device_query 2>/dev/null)
NATIVE_CC := $(firstword $(NATIVE_CC))
ifeq ($(NATIVE_CC),)
ARCH_FLAGS := -arch=native
$(warning could not query the device capability; falling back to \
          -arch=native, which resolves to sm_90 on Hopper and drops wgmma)
else ifeq ($(shell test $(NATIVE_CC) -ge 90 && echo yes),yes)
ARCH_FLAGS := -gencode arch=compute_$(NATIVE_CC)a,code=sm_$(NATIVE_CC)a
else
ARCH_FLAGS := -gencode arch=compute_$(NATIVE_CC),code=sm_$(NATIVE_CC)
endif
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

QLU_OBJ     := build/gpu/qlu.o
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
	$(COM_OBJ)/matrix_market.o \
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
	$(QLU_OBJ) \
	$(OBJ_DIR)/oii_gemm.o \
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
OIIGUARD_MAIN:= $(TOBJ_DIR)/main_oii_guard.o
SAT_MAIN    := $(TOBJ_DIR)/main_saturation.o
CONS_MAIN   := $(TOBJ_DIR)/consistency/main_paths.o
ABS_MAIN    := $(TOBJ_DIR)/consistency/main_absorb.o
SR_MAIN     := $(TOBJ_DIR)/consistency/main_sr.o
MART_MAIN   := $(TOBJ_DIR)/consistency/main_martingale.o
CAMP_MAIN   := $(TOBJ_DIR)/campaign/main_accuracy.o
INSTR_DIR   := build/instrumented
# panel_persist.cu is a cooperative-launch TU and must stay relocatable
# (-dc); int8lu_factor references its launcher unconditionally, so it links
# even when PANELPERSIST is never set.

.PHONY: all clean gates

all: $(BIN_DIR)/lps-sweep $(BIN_DIR)/lps-probe $(BIN_DIR)/lps-ozaki-test \
     $(BIN_DIR)/lps-ablate $(BIN_DIR)/lps-kappa \
     $(BIN_DIR)/lps-rcheck $(BIN_DIR)/lps-profile \
     $(BIN_DIR)/lps-oom $(BIN_DIR)/lps-energy \
     $(BIN_DIR)/lps-kappacheck $(BIN_DIR)/lps-ddcheck \
     $(BIN_DIR)/lps-int8lu-probe $(BIN_DIR)/lps-exact-crt $(BIN_DIR)/lps-oii-crt \
     $(BIN_DIR)/lps-lift-dump $(BIN_DIR)/lps-oii-gemm \
     $(BIN_DIR)/lps-oii-guard $(BIN_DIR)/lps-saturation \
     $(BIN_DIR)/lps-consistency $(BIN_DIR)/lps-absorb $(BIN_DIR)/lps-sr \
     $(BIN_DIR)/lps-martingale \
     $(BIN_DIR)/lps-campaign

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
$(QLU_OBJ): $(SRC_DIR)/qlu.cu | $(OBJ_DIR)
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

# qlu.cu built a SECOND time, against the instrumented header, for the
# export harness only. Two TUs including int8lu.cuh in one binary is a
# duplicate symbol at device link, so lps-lift-dump links this object
# and NOT the ordinary qlu.o. Same -c rule and the same reason.
QLU_DUMP_OBJ := $(OBJ_DIR)/qlu_dump.o

$(QLU_DUMP_OBJ): $(SRC_DIR)/qlu.cu $(INSTR_DIR)/int8lu_instrumented.cuh \
                 | $(OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) -DLPS_DUMP -I$(INSTR_DIR) $(VENDOR_INC) -c $< -o $@

$(LIFTD_MAIN): $(TEST_DIR)/main_lift_dump.cu | $(TOBJ_DIR)
	$(NVCC) $(NVCCFLAGS) -dc $< -o $@

$(BIN_DIR)/lps-lift-dump: $(COM_OBJ)/definitions.o $(COM_OBJ)/error.o \
                          $(COM_OBJ)/timing.o $(COM_OBJ)/convert.o \
                          $(COM_OBJ)/problem.o $(COM_OBJ)/tuning.o \
                          $(COM_OBJ)/ozaki2.o $(OBJ_DIR)/oii_gemm.o \
                          $(QLU_DUMP_OBJ) \
                          $(VENDOR_OBJ) $(LIFTD_MAIN) \
                          | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LDLIBS)

$(BIN_DIR)/lps-oii-gemm: $(COM_OBJ)/ozaki2.o $(COM_OBJ)/error.o \
                         $(COM_OBJ)/definitions.o $(OBJ_DIR)/oii_gemm.o \
                         $(OIIGEMM_MAIN) | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LDLIBS)

# Cross-path equivalence tests live in their own directory: they check
# two code paths against each other rather than against a reference, so
# they are a different kind of thing from the gates.
$(TOBJ_DIR)/consistency/%.o: $(TEST_DIR)/consistency/%.cu | $(TOBJ_DIR)
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -dc $< -o $@

$(TOBJ_DIR)/campaign/%.o: $(TEST_DIR)/campaign/%.cu | $(TOBJ_DIR)
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCCFLAGS) -dc $< -o $@

$(BIN_DIR)/lps-campaign: $(COMMON) $(COM_OBJ)/reference.o $(CAMP_MAIN) | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LDLIBS)

$(BIN_DIR)/lps-martingale: $(COMMON) $(MART_MAIN) | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LDLIBS)

$(BIN_DIR)/lps-sr: $(COMMON) $(SR_MAIN) | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LDLIBS)

$(BIN_DIR)/lps-absorb: $(COMMON) $(ABS_MAIN) | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LDLIBS)

$(BIN_DIR)/lps-consistency: $(COMMON) $(CONS_MAIN) | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LDLIBS)

$(BIN_DIR)/lps-saturation: $(COMMON) $(SAT_MAIN) | $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) $^ -o $@ $(LDLIBS)

$(BIN_DIR)/lps-oii-guard: $(COM_OBJ)/ozaki2.o $(COM_OBJ)/error.o \
                          $(COM_OBJ)/definitions.o $(OBJ_DIR)/oii_gemm.o \
                          $(OIIGUARD_MAIN) | $(BIN_DIR)
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

# Every check that has a verdict, in one command, failing if any does.
# The set was previously a line I retyped, which is how two of them came
# to be quoted in notes without having been re-run after the code under
# them changed.
#
# The two Python checks need the simulation repo (LPS_SIM, default ~/fp)
# for its oracle and its dd machinery. They SKIP loudly when it is
# absent rather than failing: "not available here" and "wrong" are
# different outcomes, and a runner that conflates them teaches people to
# ignore it.
LPS_SIM ?= $(HOME)/fp

# The oracle needs Python >= 3.13: core_exp/ozaki2.py uses math.fma for
# Algorithm 2's line 11, and that is 3.13+. Override LPS_PY where the
# system python is older (uv python find 3.13 gives one).
LPS_PY  ?= python3

gates: all
	@fail=0; \
	for b in lps-ddcheck lps-exact-crt lps-oii-crt lps-oii-gemm \
	         lps-oii-guard lps-consistency lps-absorb lps-martingale \
	         lps-saturation; do \
	  printf '%-18s ' "$$b"; \
	  if ASYNC=1 FUSE=5 $(BIN_DIR)/$$b > /tmp/gate.$$b.log 2>&1; then \
	    echo "ok    $$(tail -1 /tmp/gate.$$b.log)"; \
	  else \
	    echo "FAIL  $$(tail -1 /tmp/gate.$$b.log)"; fail=1; \
	  fi; \
	done; \
	probe=$$($(LPS_PY) -c "import sys; sys.path.insert(0,'$(LPS_SIM)'); \
	         import core_exp.envelope_int, core_exp.ozaki2" 2>&1 | tail -1); \
	if [ -z "$$probe" ]; then \
	  printf '%-18s ' "lift_check"; \
	  $(BIN_DIR)/lps-lift-dump 256 64 4 --out /tmp/gate_lift.bin >/dev/null 2>&1; \
	  if LPS_SIM=$(LPS_SIM) $(LPS_PY) tools/lift_check.py /tmp/gate_lift.bin \
	       > /tmp/gate.lift.log 2>&1; then \
	    echo "ok    thm:lift lem:intloc thm:int lem:carrier lem:pivot lem:bridge thm:facenv cor:qhat thm:transfer cor:criteria thm:outer thm:sketch"; \
	  else echo "FAIL  see /tmp/gate.lift.log"; fail=1; fi; \
	  printf '%-18s ' "oii_envelope"; \
	  $(BIN_DIR)/lps-oii-gemm --emit /tmp/gate_prod.txt >/dev/null 2>&1; \
	  if LPS_SIM=$(LPS_SIM) $(LPS_PY) tools/oii_envelope.py /tmp/gate_prod.txt \
	       > /tmp/gate.env.log 2>&1; then \
	    echo "ok    $$(tail -1 /tmp/gate.env.log)"; \
	  else echo "FAIL  see /tmp/gate.env.log"; fail=1; fi; \
	  printf '%-18s ' "lift_check_oii"; \
	  $(BIN_DIR)/lps-lift-dump 256 64 8 --oii \
	      --out /tmp/gate_oii.bin >/dev/null 2>&1; \
	  if LPS_SIM=$(LPS_SIM) $(LPS_PY) tools/lift_check.py \
	       /tmp/gate_oii.bin > /tmp/gate.loii.log 2>&1; then \
	    echo "ok    thm:lift thm:oii lem:pivot on the oii arm"; \
	  else echo "FAIL  see /tmp/gate.loii.log"; fail=1; fi; \
	  printf '%-18s ' "sr_keyed"; \
	  $(BIN_DIR)/lps-sr --out /tmp/gate_sr.txt >/dev/null 2>&1; \
	  if LPS_SIM=$(LPS_SIM) $(LPS_PY) tools/sr_check.py /tmp/gate_sr.txt \
	       > /tmp/gate.sr.log 2>&1; then \
	    echo "ok  $$(tail -1 /tmp/gate.sr.log)"; \
	  else echo "FAIL  see /tmp/gate.sr.log"; fail=1; fi; \
	  printf '%-18s ' "oii_step_envelope"; \
	  if LPS_SIM=$(LPS_SIM) $(LPS_PY) tools/dump_to_oii_cases.py \
	       /tmp/gate_lift.bin /tmp/gate_steps.txt >/dev/null 2>&1 && \
	     $(BIN_DIR)/lps-oii-gemm /tmp/gate_steps.txt \
	       --emit /tmp/gate_steps_prod.txt >/dev/null 2>&1 && \
	     LPS_SIM=$(LPS_SIM) $(LPS_PY) tools/oii_envelope.py \
	       /tmp/gate_steps_prod.txt --ref /tmp/gate_steps.txt \
	       > /tmp/gate.step.log 2>&1; then \
	    echo "ok    lem:oiiloc $$(tail -1 /tmp/gate.step.log)"; \
	  else echo "FAIL  see /tmp/gate.step.log"; fail=1; fi; \
	else \
	  echo "lift_check         SKIP  $$probe"; \
	  echo "oii_envelope       SKIP  $$probe"; \
	  echo "lift_check_oii     SKIP  $$probe"; \
	  echo "oii_step_envelope  SKIP  $$probe"; \
	  if [ -z "$(LPS_ALLOW_SKIP)" ]; then fail=1; skipped=1; fi; \
	fi; \
	echo; \
	if [ -n "$$skipped" ]; then \
	  echo "GATES FAIL: four oracle-backed checks did not run."; \
	  echo "  A suite that reports PASS with a third of it skipped is the"; \
	  echo "  vacuous-gate failure one level up. Set LPS_PY to a 3.13"; \
	  echo "  interpreter and LPS_SIM to the oracle, or pass"; \
	  echo "  LPS_ALLOW_SKIP=1 to say out loud that you are skipping them."; \
	fi; \
	if [ $$fail -eq 0 ]; then echo "GATES PASS"; \
	else echo "GATES FAIL"; exit 1; fi

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

# The wildcard does NOT recurse, so a new object directory needs adding
# here or its headers stop being tracked -- test/consistency/ was built
# against a stale qlu.h and failed at link with an undefined
# reference to the old solve() signature.
-include $(wildcard $(COM_OBJ)/*.d $(OBJ_DIR)/*.d $(TOBJ_DIR)/*.d \
                    $(TOBJ_DIR)/consistency/*.d \
                    $(TOBJ_DIR)/campaign/*.d $(POBJ_DIR)/*.d)
