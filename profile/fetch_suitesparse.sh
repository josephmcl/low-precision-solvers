#!/usr/bin/env bash
#
# Fetch the SuiteSparse matrices used by the suitesparse experiment.
#
# Chosen for DENSITY. Every method in this harness does a dense LU, so a
# matrix is densified to n^2 before it is solved. Running a 1%-dense matrix
# through that measures the harness rather than the matrix, and its LU bears no
# resemblance to what a sparse solver would do with it. These seven are the
# densest square real entries in the collection in the n = 2000-15000 range,
# and they span six application domains:
#
#   heart3       2339  12.4%  2D/3D          densest square real in range
#   psmigr_1     3140   5.5%  economic       unsymmetric, integer-valued
#   raefsky2     3242   2.8%  CFD            unsymmetric
#   nasa2910     2910   2.1%  structural     symmetric
#   exdata_1     6001   6.3%  optimization   symmetric indefinite
#   nd3k         9000   4.0%  2D/3D          symmetric
#   human_gene2 14340   8.8%  graph          symmetric, largest
#
# The upper bound is the harness's, not the method's: n^2 fp64 plus 12n^2 for
# split-MPIR is what bounds n here, around 30000 on a 33 GB card.
#
# Usage:  profile/fetch_suitesparse.sh [dir]     (default: suitesparse/)

set -u
DIR="${1:-suitesparse}"
URL=https://suitesparse-collection-website.herokuapp.com/MM
mkdir -p "$DIR"

MATRICES="
Norris/heart3
HB/psmigr_1
Simon/raefsky2
Nasa/nasa2910
GHS_indef/exdata_1
ND/nd3k
Belcastro/human_gene2
"

for m in $MATRICES; do
    b=$(basename "$m")
    if [ -f "$DIR/$b.mtx" ]; then
        printf '  %-14s have\n' "$b"
        continue
    fi
    printf '  %-14s ' "$b"
    if curl -sSL -m 900 -o "$DIR/$b.tar.gz" "$URL/$m.tar.gz" \
       && tar xzf "$DIR/$b.tar.gz" -C "$DIR" \
       && mv "$DIR/$b/$b.mtx" "$DIR/"; then
        rm -rf "$DIR/$b" "$DIR/$b.tar.gz"
        printf 'ok (%s)\n' "$(du -h "$DIR/$b.mtx" | cut -f1)"
    else
        rm -rf "$DIR/$b" "$DIR/$b.tar.gz"
        printf 'FAILED\n'
    fi
done
