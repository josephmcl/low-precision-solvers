#!/usr/bin/env bash
#
# Generate the result CSVs, one file per experiment.
#
# Separate files rather than one wide table: each figure draws from exactly one
# of these, so a re-run of one experiment cannot invalidate a plot built from
# another, and a failed run leaves the rest intact. The cost is that joins
# happen downstream, which is the right place for them.
#
# Usage:  profile/run_all.sh [outdir]      (default: results/rtx5090)
#         SKIP_SLOW=1 profile/run_all.sh   (omit the >10 min runs)
#
# Every file is written atomically via a .part rename, so a killed run never
# leaves a half-written CSV that looks complete.

set -u
OUT="${1:-results/rtx5090}"
BIN="bin"
mkdir -p "$OUT"

# BASELINE FAIRNESS, applied to every run below.
#
# split-MPIR's shipped stopping rule chooses 3 passes. Two passes reach
# 3.97e-16 against R-IR's 8.83e-16 at n=8192 k=2048 — still MORE accurate than
# R-IR — for 300 ms against 437. Two passes is therefore split-MPIR's fastest
# configuration that is not also less accurate than the method it is being
# compared against, which is the only defensible place to compare speed.
#
# This is exported here rather than passed per-run because leaving it out is
# invisible: the CSVs were once generated without it while the write-up quoted
# it, so every figure drawn from them overstated R-IR's advantage by ~45% at
# k=2048 and put the k-crossover in the wrong place. The shipped-rule numbers
# are still worth having; generate them with MPIR_PASSES=0 and compare.
# OOM SWEEP SIZES MUST BE DERIVED FROM THE DEVICE, NOT HARDCODED.
#
# The experiment's whole content is that a 12n^2 method runs out of memory
# where an 8n^2 one does not, so the sizes have to BRACKET this card's 12n^2
# limit. They were once hardcoded for a 33 GB 5090 (20000..56000); run
# unchanged on a 96 GB card every one of those fits, the sweep reports no OOM
# at all, and that reads as "the resident method coped" rather than "the
# experiment was pointed at the wrong range".
#
# USE 16.6, NOT 12. This experiment's "resident" arm is R-IR holding A in fp64
# (8n^2) ON TOP OF its own working set (8n^2), and it measures 16.57n^2 --
# not the 12n^2 of split-MPIR. A first version divided by 12, which biases the
# whole sweep high: on the PRO 6000 it put three of four points above the
# boundary and captured the last success only in the lowest sample. On a card
# where that lowest point also landed above the boundary there would be NO
# "resident ok" row at all, and the sweep would show OOM everywhere -- which
# reads as a broken run, not as a mis-centred one.
#
# The bracket is deliberately wide on the low side for that reason: losing the
# last success costs the whole comparison, losing the first failure only costs
# a data point.
if [ -z "${OOM_SIZES:-}" ]; then
    mem_mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    OOM_SIZES=$(awk -v m="$mem_mib" 'BEGIN{
        nmax = sqrt(m * 1048576 / 16.6);
        printf "%d,%d,%d,%d,%d", nmax*0.70, nmax*0.90, nmax*1.05, nmax*1.20, nmax*1.35 }')
    echo "OOM sweep derived for ${mem_mib} MiB: $OOM_SIZES (resident limit ~16.6n^2)"
fi

MPIR_PASSES="${MPIR_PASSES:-2}"
if [ "$MPIR_PASSES" != "0" ]; then
    export LPS_MPIR_MAX_OUTER="$MPIR_PASSES"
    echo "split-MPIR capped at $MPIR_PASSES passes (fair baseline)"
else
    unset LPS_MPIR_MAX_OUTER
    echo "split-MPIR at its shipped stopping rule"
fi

run() {                       # run <name> <command...>
    local name="$1"; shift
    printf '  %-22s ' "$name"
    if "$@" > "$OUT/$name.csv.part" 2>"$OUT/$name.log"; then
        mv "$OUT/$name.csv.part" "$OUT/$name.csv"
        printf 'ok  (%s rows)\n' "$(($(wc -l < "$OUT/$name.csv") - 1))"
    else
        printf 'FAILED - see %s\n' "$OUT/$name.log"
        rm -f "$OUT/$name.csv.part"
    fi
}

echo "writing to $OUT/"

# ---- 1. size sweep: the headline speed/accuracy table --------------------
run size_sweep       "$BIN/lps-profile" \
    --sizes 4096:512,4096:2048,8192:512,8192:2048,12288:2048 \
    --families diag --repeats 2

# ---- 1b. the k crossover, MEASURED -----------------------------------------
#
# The one region where split-MPIR beats R-IR. It was previously reported as
# k ~ 760 from a two-point fit; ten points bracket it directly. R-IR pays a
# fixed Theta(n^3) build that split-MPIR has no counterpart for, then wins per
# right-hand side, so the two cross as k grows.
run k_crossover      "$BIN/lps-profile" \
    --sizes 8192:128,8192:256,8192:384,8192:512,8192:768,8192:1024,8192:1536,8192:2048,8192:3072,8192:4096 \
    --families diag --repeats 2

# ---- 1c. large n -----------------------------------------------------------
#
# R-IR's cost is dominated by a Theta(n^3) factor, split-MPIR's by a
# Theta(n^2 k) solve, so at FIXED k the two must converge and eventually cross
# as n grows. These points are where that shows.
run large_n          "$BIN/lps-profile" \
    --sizes 16384:2048,20480:2048,24576:2048 --families diag --repeats 2

# ---- 2. conditioning, on the shift family --------------------------------
run conditioning     "$BIN/lps-profile" \
    --sizes 4096:1024 --families shift --repeats 2

# ---- 3. the five standard spectral families ------------------------------
#
# randsvd (log-uniform singular values), clustered and arithmetic spectra with
# positive and signed eigenvalues. These are the families the mixed-precision
# refinement literature reports on, swept over log10(kappa) = 1..12.
run spectral         "$BIN/lps-profile" \
    --sizes 4096:512 \
    --families randsvd,clust_pos,clust_sgn,arith_pos,arith_sgn --repeats 2

# ---- 4. the older synthetic families -------------------------------------
run families         "$BIN/lps-profile" \
    --sizes 4096:512 --families graded,spd,wilkinson --repeats 2

# ---- 5. per-iteration error curves ---------------------------------------
run per_iteration    "$BIN/lps-profile" \
    --sizes 8192:2048 --families diag --iters --max-cap 6 --repeats 1

# ---- 6. out-of-memory boundary -------------------------------------------
run oom              "$BIN/lps-oom" \
    --sweep "$OOM_SIZES" 64 --csv

# ---- 7. energy -----------------------------------------------------------
#
# 20 solves per method, applied by the loop and invisible to the methods — the
# uniformity whose absence retracted the earlier energy results. --rounds 3
# reports a median with min/max: clocks cannot be pinned in a container, so
# boost variance is bounded statistically rather than controlled. Each method is
# thermally soaked and commonly warmed before its measured region, which is what
# took run-to-run spread from 30x (on the idle baseline) to under 1%.
run energy           "$BIN/lps-energy" 8192 --k 2048 --repeats 20 --rounds 3 --csv

# ---- 8. ablation ---------------------------------------------------------
run ablation         "$BIN/lps-ablate" 8192 2048 2 --csv

# ---- 9. SuiteSparse ------------------------------------------------------
#
# Seven square real matrices chosen for DENSITY, because every method here does
# a dense LU: filling a 1%-dense matrix to n^2 measures the harness, not the
# matrix. These are the densest square real entries in the collection in the
# n = 2000-15000 range, spanning six application domains:
#
#   heart3       2339  12.4%  2D/3D          densest square real in range
#   psmigr_1     3140   5.5%  economic       unsymmetric, integer-valued
#   raefsky2     3242   2.8%  CFD            unsymmetric
#   nasa2910     2910   2.1%  structural     symmetric
#   exdata_1     6001   6.3%  optimization   symmetric indefinite
#   nd3k         9000   4.0%  2D/3D          symmetric
#   human_gene2 14340   8.8%  graph          symmetric, largest
#
# n comes from each file; --sizes carries only k. Fetch with
# profile/fetch_suitesparse.sh. Skipped rather than failed when absent, because
# an empty run here is a missing input, not a broken experiment.
if compgen -G "suitesparse/*.mtx" > /dev/null; then
    run suitesparse  "$BIN/lps-profile" \
        --sizes 0:512 --matrices "$(ls suitesparse/*.mtx | tr '\n' ',')" --repeats 2
else
    printf '  %-22s skipped (no suitesparse/*.mtx)\n' suitesparse
fi

echo "done"
