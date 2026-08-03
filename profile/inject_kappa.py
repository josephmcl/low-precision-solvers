#!/usr/bin/env python3
"""Join measured kappa onto an existing profile CSV.

The spectral sweep's x-axis is the REQUESTED log10(kappa). This adds the
measured value as extra columns, keyed on (matrix, param, n), so the sweep does
not have to be regenerated to gain them -- kappa is a property of the matrix,
not of the solve, and re-running 240 rows would also re-roll every timing in
them for no reason.

Rows with no matching measurement keep the file's shape and get empty cells
rather than being dropped: a spectral row that silently vanished because the
key did not match would look like a sweep that was never run.

Usage:
    profile/inject_kappa.py spectral.csv kappacheck.csv [-o out.csv]

With no -o the input is rewritten in place via a .part rename.
"""

import argparse
import csv
import os
import sys

NEW_COLS = ["kappa_requested", "kappa_measured", "kappa_ratio",
            "kappa_converged"]


def key(matrix, param, n):
    """Key on the numeric value of param, not its spelling.

    The two files are written by different programs, so the same parameter can
    appear as "1" and "1.0" and string equality would silently match nothing --
    producing a file of empty kappa columns that looks like a failed
    measurement rather than a failed join.
    """
    try:
        p = float(param)
    except (TypeError, ValueError):
        p = param
    try:
        nn = int(float(n))
    except (TypeError, ValueError):
        nn = n
    return (matrix.strip(), p, nn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("target", help="profile CSV to annotate")
    ap.add_argument("kappa", help="lps-kappacheck CSV")
    ap.add_argument("-o", "--out", default=None)
    args = ap.parse_args()

    with open(args.kappa, newline="") as fh:
        measured = {
            key(r["matrix"], r["param"], r["n"]): r
            for r in csv.DictReader(fh)
        }
    if not measured:
        sys.exit(f"{args.kappa}: no rows")

    with open(args.target, newline="") as fh:
        rows = list(csv.DictReader(fh))
        if not rows:
            sys.exit(f"{args.target}: no rows")
        fields = list(rows[0].keys())

    for c in NEW_COLS:
        if c not in fields:
            fields.append(c)

    hits = 0
    for r in rows:
        m = measured.get(key(r["matrix"], r["param"], r["n"]))
        if m is None:
            for c in NEW_COLS:
                r.setdefault(c, "")
            continue
        hits += 1
        r["kappa_requested"] = m["kappa_requested"]
        r["kappa_measured"] = m["kappa_measured"]
        r["kappa_ratio"] = m["ratio"]
        r["kappa_converged"] = m["converged"]

    out = args.out or args.target
    tmp = out + ".part"
    with open(tmp, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, out)

    print(f"{out}: {hits}/{len(rows)} rows annotated "
          f"from {len(measured)} measurements")
    if hits == 0:
        sys.exit("no rows matched -- check that the two files share "
                 "(matrix, param, n) keys")


if __name__ == "__main__":
    main()
