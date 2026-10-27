#!/usr/bin/env python3
"""Generate the three Phase-5 figures from summary CSVs (real data only)."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

FAMILIES = ["random", "planted", "decoy"]


def _load(summary_paths):
    dfs = []
    for p in summary_paths:
        p = Path(p)
        if p.exists():
            dfs.append(pd.read_csv(p))
    if not dfs:
        raise SystemExit("no summary CSVs found")
    return pd.concat(dfs, ignore_index=True)


def fig_quality_vs_chi(df, outdir):
    q = df[(df["solver"] == "qaoa")].copy()
    # chi column comes from the tag; we re-derive from raw jsonl in reproduce.sh
    # Simpler: summary CSV has 'engine' via grouping; we need 'chi' if present.
    if "chi" not in q.columns:
        print("  [skip] quality-vs-chi: summary CSV missing 'chi' column "
              "(will populate from JSONL in a later pass)")
        return
    fig, axes = plt.subplots(1, len(FAMILIES), figsize=(4 * len(FAMILIES), 3.5),
                             sharey=True, squeeze=False)
    for j, fam in enumerate(FAMILIES):
        ax = axes[0, j]
        d = q[q["family"] == fam]
        for p, sub in d.groupby("p"):
            for n, s2 in sub.groupby("n"):
                s2 = s2.dropna(subset=["chi"]).sort_values("chi")
                if s2.empty:
                    continue
                ax.errorbar(s2["chi"], s2["obj_mean"],
                            yerr=[s2["obj_mean"] - s2["obj_ci_lo"],
                                  s2["obj_ci_hi"] - s2["obj_mean"]],
                            marker="o", label=f"n={n} p={p}", capsize=2)
        ax.set_title(f"{fam}")
        ax.set_xlabel("bond dimension chi (MPS)")
        ax.set_xscale("log", base=2)
        ax.grid(True, alpha=0.3)
    axes[0, 0].set_ylabel("objective mean (95% CI)")
    axes[0, -1].legend(fontsize=7)
    fig.tight_layout()
    out = Path(outdir) / "fig_quality_vs_chi.png"
    fig.savefig(out, dpi=150)
    print(f"  wrote {out}")


def fig_quality_vs_compute(df, outdir):
    fig, axes = plt.subplots(1, len(FAMILIES), figsize=(4 * len(FAMILIES), 3.5),
                             sharey=True, squeeze=False)
    for j, fam in enumerate(FAMILIES):
        ax = axes[0, j]
        d = df[df["family"] == fam]
        for solver, sub in d.groupby("solver"):
            for n, s2 in sub.groupby("n"):
                s2 = s2.dropna(subset=["wall_mean", "obj_mean"])
                if s2.empty:
                    continue
                ax.scatter(s2["wall_mean"], s2["obj_mean"],
                           label=f"{solver}/n={n}", s=18, alpha=0.7)
        ax.set_title(f"{fam}")
        ax.set_xlabel("wall mean (s)")
        ax.set_xscale("log")
        ax.grid(True, alpha=0.3)
    axes[0, 0].set_ylabel("objective mean")
    axes[0, -1].legend(fontsize=6, ncol=2)
    fig.tight_layout()
    out = Path(outdir) / "fig_quality_vs_compute.png"
    fig.savefig(out, dpi=150)
    print(f"  wrote {out}")


def fig_recovery(df, outdir):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), squeeze=False)
    # Left: planted recovery
    ax = axes[0, 0]
    for fam in ["planted", "decoy"]:
        d = df[(df["family"] == fam) & df["planted_recovery_mean"].notna()]
        for solver, sub in d.groupby("solver"):
            sub = sub.sort_values("n")
            ax.plot(sub["n"], sub["planted_recovery_mean"],
                    marker="o", label=f"{fam}/{solver}")
    ax.set_xlabel("n")
    ax.set_ylabel("planted recovery |S∩S*|/k")
    ax.set_title("Planted module recovery")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=6, ncol=2)
    # Right: decoy recovery
    ax = axes[0, 1]
    d = df[(df["family"] == "decoy") & df["decoy_recovery_mean"].notna()]
    for solver, sub in d.groupby("solver"):
        sub = sub.sort_values("n")
        ax.plot(sub["n"], sub["decoy_recovery_mean"], marker="o", label=solver)
    ax.set_xlabel("n")
    ax.set_ylabel("decoy recovery |S∩D|/|D|")
    ax.set_title("Decoy-hub capture (lower is better)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=6)
    fig.tight_layout()
    out = Path(outdir) / "fig_recovery.png"
    fig.savefig(out, dpi=150)
    print(f"  wrote {out}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--summaries", nargs="+", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args(argv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    df = _load(args.summaries)
    fig_quality_vs_compute(df, outdir)
    fig_recovery(df, outdir)
    fig_quality_vs_chi(df, outdir)


if __name__ == "__main__":
    main()
