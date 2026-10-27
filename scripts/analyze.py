#!/usr/bin/env python3
"""Aggregate JSONL runs into per-cell summary CSV with bootstrap CIs."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def bootstrap_ci(x, n_boot=10000, alpha=0.05, seed=0):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inputs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    rows = []
    for path in args.inputs:
        for line in Path(path).read_text().splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("no rows to aggregate")

    rows_out = []
    for key, grp in df.groupby(["family", "n", "k", "solver"], dropna=False):
        objs = grp["objective"].astype(float).values
        obj_mean = float(np.nanmean(objs)) if np.any(np.isfinite(objs)) else float("nan")
        obj_std = float(np.nanstd(objs)) if np.any(np.isfinite(objs)) else float("nan")
        lo, hi = bootstrap_ci(objs)
        rows_out.append({
            "family": key[0], "n": int(key[1]), "k": int(key[2]), "solver": key[3],
            "n_seeds": int(grp["seed"].nunique()),
            "obj_mean": obj_mean, "obj_std": obj_std,
            "obj_ci_lo": lo, "obj_ci_hi": hi,
            "wall_mean": float(grp["wall_time_s"].mean()),
            "wall_std": float(grp["wall_time_s"].std(ddof=0)),
            "peak_mb_mean": float(grp["peak_mb"].mean()),
            "feasible_frac": float(grp["feasible"].mean()),
        })
    out = pd.DataFrame(rows_out).sort_values(["family", "n", "k", "solver"])
    out.to_csv(args.out, index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
