"""Second pass: does R stratify the downsample on (subset, outcome)?

The first diagnostic showed no 2-cell rule reproduces R. Rule B
(actual denominator) matches every fold-8/9/10 cell but not fold 1,
where R splits 499/15881 while independent flooring gives 500/15880 --
the same total, a different allocation. That cannot come from flooring
two strata independently, which points at the strata themselves.

Hypothesis: R stratifies on (subset, outcome), so the ALL train set has
four cells, not two. SAME and OTHER have constant subset, so their four
cells collapse to two and the rules coincide -- which is exactly why
only ALL ever disagrees.

This script scores candidate rules over both stratification schemes,
reports per-cell and per-iteration-total agreement separately, and
lists the failing cells for the leading candidates.

Run from the repository root::

    uv run python analyses/diagnose_downsample_strata.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import polars as pl

DEFAULT_FIXTURE = Path.home() / "Documents/NAU/Grad/Research/ADSI/soak_fixture"
N_FOLDS = 10
EXPECTED_ROWS = 46010


def keeps(sizes: list[int], target: int, denom: int, mode: str) -> list[int]:
    """Per-cell keep counts under a denominator and rounding mode."""
    if mode == "floor":
        return [s * target // denom for s in sizes]
    if mode == "round":
        return [int(np.floor(s * target / denom + 0.5)) for s in sizes]
    if mode == "largest-remainder":
        # Floor everything, then hand the leftover to the cells with the
        # largest fractional parts until the total reaches target.
        exact = [s * target / denom for s in sizes]
        out = [int(np.floor(e)) for e in exact]
        short = min(target, sum(sizes)) - sum(out)
        order = np.argsort([-(e - np.floor(e)) for e in exact])
        for i in range(max(0, short)):
            out[order[i % len(out)]] += 1
        return out
    msg = f"unknown mode {mode}"
    raise ValueError(msg)


def main() -> int:
    root = Path(os.environ.get("NSCH_SOAK_FIXTURE", DEFAULT_FIXTURE))
    print(f"fixture: {root}")
    if not root.is_dir():
        print("FAIL: fixture directory not found", file=sys.stderr)
        return 2

    folds = pl.read_csv(root / "nsch_autism_folds.csv")
    design = pl.read_csv(root / "data_Classif" / "NSCH_autism.csv", columns=["survey_year", "y"])
    iterations = pl.read_csv(root / "nsch_autism_iterations_long.csv")
    if folds.height != EXPECTED_ROWS or design.height != EXPECTED_ROWS:
        print("FAIL: unexpected fixture row count", file=sys.stderr)
        return 2

    subset = folds["test.subset"].cast(pl.Utf8).to_numpy()
    fold_id = folds["fold"].to_numpy()
    outcome = design["y"].to_numpy().astype(str)

    labels = np.unique(subset)
    full_count = {s: int(np.sum(subset == s)) for s in labels}
    same_nom = {s: full_count[s] * (N_FOLDS - 1) // N_FOLDS for s in labels}
    all_nom = sum(same_nom.values())
    nominal = {
        s: {"same": same_nom[s], "other": all_nom - same_nom[s], "all": all_nom} for s in labels
    }

    print("\n== overall outcome balance ==")
    for y in np.unique(outcome):
        print(f"  y={y}: {int(np.sum(outcome == y))}")

    grouped = (
        iterations.filter((pl.col("downsampled")) & (pl.col("role") == "train"))
        .group_by(["test.subset", "train.subsets", "test.fold"])
        .agg(pl.col("row_id"))
    )
    r_train = {
        (str(s), str(src), int(f)): np.asarray(ids, dtype=np.int64) - 1
        for s, src, f, ids in grouped.iter_rows()
    }

    schemes = {
        "outcome only": lambda idx: [outcome[idx]],
        "subset x outcome": lambda idx: [subset[idx], outcome[idx]],
    }
    modes = ["floor", "round", "largest-remainder"]
    denoms = ["nominal", "actual"]

    results: dict[tuple[str, str, str], dict[str, int]] = {}
    detail: dict[tuple[str, str, str], list[str]] = {}

    for scheme_name, keyer in schemes.items():
        for denom_name in denoms:
            for mode in modes:
                cell_hits = cell_total = 0
                iter_hits = 0
                misses: list[str] = []
                for (s, src, fold), r_idx in sorted(r_train.items()):
                    mask = {
                        "same": (subset == s) & (fold_id != fold),
                        "other": (subset != s) & (fold_id != fold),
                        "all": fold_id != fold,
                    }[src]
                    actual_idx = np.flatnonzero(mask)
                    actual = len(actual_idx)
                    own = nominal[s][src]
                    target = min(nominal[s].values())
                    denom = own if denom_name == "nominal" else actual

                    parts = keyer(actual_idx)
                    combo = np.array(
                        ["|".join(p) for p in zip(*[list(a) for a in parts], strict=True)]
                    )
                    r_parts = keyer(r_idx)
                    r_combo = np.array(
                        ["|".join(p) for p in zip(*[list(a) for a in r_parts], strict=True)]
                    )

                    cells = sorted(set(combo.tolist()))
                    sizes = [int(np.sum(combo == c)) for c in cells]
                    r_counts = [int(np.sum(r_combo == c)) for c in cells]
                    pred = keeps(sizes, target, denom, mode)

                    for c, p, r in zip(cells, pred, r_counts, strict=True):
                        cell_total += 1
                        if p == r:
                            cell_hits += 1
                        elif len(misses) < 8:
                            misses.append(
                                f"    {s} {src:5s} fold {fold:2d} [{c}] "
                                f"size={int(np.sum(combo == c))} pred={p} R={r}"
                            )
                    if sum(pred) == sum(r_counts):
                        iter_hits += 1

                key = (scheme_name, denom_name, mode)
                results[key] = {
                    "cells": cell_hits,
                    "cell_total": cell_total,
                    "iters": iter_hits,
                }
                detail[key] = misses

    print("\n== candidate rules ==")
    print(f"{'stratification':18s} {'denom':9s} {'rounding':18s} {'cells':>12s} {'totals':>9s}")
    ranked = sorted(results.items(), key=lambda kv: (-kv[1]["cells"], -kv[1]["iters"]))
    for (scheme, denom, mode), sc in ranked:
        exact = sc["cells"] == sc["cell_total"]
        flag = "  <-- EXACT" if exact else ""
        print(
            f"{scheme:18s} {denom:9s} {mode:18s} "
            f"{sc['cells']:5d}/{sc['cell_total']:<6d} {sc['iters']:4d}/40{flag}"
        )

    print("\n== failing cells for the top three candidates ==")
    for key, _ in ranked[:3]:
        scheme, denom, mode = key
        sc = results[key]
        print(f"\n  {scheme} / {denom} / {mode}  ({sc['cells']}/{sc['cell_total']} cells)")
        for line in detail[key] or ["    none"]:
            print(line)

    print("\n== note ==")
    print("An EXACT rule is the one to port into iter_soak_splits. Row")
    print("membership still will not match R (different RNG); only the")
    print("per-stratum counts are portable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
