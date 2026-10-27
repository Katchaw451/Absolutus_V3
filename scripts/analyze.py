#!/usr/bin/env python3
"""Aggregate JSONL runs into per-cell summary CSV with bootstrap CIs.

Grouping key: (family, n, k, solver, engine, p, chi).
  - classical:  solver in {random,greedy,sa,tabu,exact}, engine='statevector', p=1, chi=None
  - QAOA-SV:    solver='qaoa', engine='statevector', p in {1,2,3}, chi=None
  - QAOA-MPS:   solver='qaoa', engine='mps', p in {1,2}, chi in {2,4,8,...}

`chi` may be present top-level (new run_cell.py) or inside `extra` (older).
Missing chi is stored as Python None (object dtype), sorted last.
"""
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


def _is_missing(v):
    if v is None:
        return True
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


def _extract_chi(row):
    """Return int(chi) or None (object dtype-safe; never pd.NA, never crash)."""
    v = row["chi"] if "chi" in row.index else None
    if not _is_missing(v):
        return int(v)
    e = row["extra"] if "extra" in row.index else None
    if isinstance(e, dict) and not _is_missing(e.get("chi")):
        return int(e["chi"])
    return None


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

    df["chi"] = df.apply(_extract_chi, axis=1)  # object dtype: int or None

    grp_cols = ["family", "n", "k", "solver", "engine", "p", "chi"]

    rows_out = []
    for key, grp in df.groupby(grp_cols, dropna=False):
        objs = (grp["objective"].astype(float).values
                if "objective" in grp.columns else np.array([np.nan]))
        obj_mean = float(np.nanmean(objs)) if np.isfinite(objs).any() else float("nan")
        obj_std = float(np.nanstd(objs)) if np.isfinite(objs).any() else float("nan")
        lo, hi = bootstrap_ci(objs)

        feas_frac_runs = (float(grp["feasible"].mean())
                          if "feasible" in grp.columns else float("nan"))
        feas_frac_shots = (float(grp["feasible_fraction"].astype(float).mean())
                           if "feasible_fraction" in grp.columns else float("nan"))
        planted_rec = (float(grp["planted_recovery"].astype(float).mean())
                       if "planted_recovery" in grp.columns
                       and grp["planted_recovery"].notna().any() else float("nan"))
        decoy_rec = (float(grp["decoy_recovery"].astype(float).mean())
                     if "decoy_recovery" in grp.columns
                     and grp["decoy_recovery"].notna().any() else float("nan"))
        success_frac = (float(grp["success"].mean())
                        if "success" in grp.columns else float("nan"))

        chi_val = key[6]
        chi_out = None if _is_missing(chi_val) else int(chi_val)

        rows_out.append({
            "family": key[0], "n": int(key[1]), "k": int(key[2]),
            "solver": key[3], "engine": key[4],
            "p": (int(key[5]) if key[5] is not None and not _is_missing(key[5]) else None),
            "chi": chi_out,
            "n_seeds": int(grp["seed"].nunique()) if "seed" in grp.columns else 0,
            "obj_mean": obj_mean, "obj_std": obj_std,
            "obj_ci_lo": lo, "obj_ci_hi": hi,
            "wall_mean": (float(grp["wall_time_s"].mean())
                          if "wall_time_s" in grp.columns else float("nan")),
            "wall_std": (float(grp["wall_time_s"].std(ddof=0))
                         if "wall_time_s" in grp.columns else float("nan")),
            "peak_mb_mean": (float(grp["peak_mb"].mean())
                             if "peak_mb" in grp.columns else float("nan")),
            "feasible_frac": feas_frac_runs,
            "feasible_frac_shots": feas_frac_shots,
            "planted_recovery_mean": planted_rec,
            "decoy_recovery_mean": decoy_rec,
            "success_frac": success_frac,
        })

    out = pd.DataFrame(rows_out)
    # Sort: chi=None (classical/SV) goes last within (family,n,k,solver,engine,p)
    out["_chi_sort"] = out["chi"].apply(lambda v: 10**9 if v is None else int(v))
    out = (out.sort_values(["family", "n", "k", "solver", "engine", "p", "_chi_sort"])
              .drop(columns="_chi_sort"))
    out.to_csv(args.out, index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
