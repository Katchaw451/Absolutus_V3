#!/usr/bin/env python3
"""Tune QAOA (gamma, beta) on dev seeds (0-4) only, per (family, n, p).

Search:
  p=1: 20x20 grid over [0, 2pi) x [0, pi)
  p>=2: 200 random restarts, keep best by mean <H_C>_analytic over dev seeds.

Saves results/tuning/<family>_n<n>_p<p>.json with best gammas/betas and score.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from modbench.generators import random_sparse, planted_scale_free, planted_decoy_hubs
from modbench.engines.statevector import StatevectorEngine

GEN = {"random": random_sparse, "planted": planted_scale_free, "decoy": planted_decoy_hubs}


def _eval(inst, p, gammas, betas, seed=0):
    r = StatevectorEngine().run_qaoa(
        inst, lambda_coh=1.0, lambda_red=0.1, p=p,
        gammas=np.asarray(gammas), betas=np.asarray(betas),
        n_shots=1, seed=seed)
    return r.energy_analytic


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=list(GEN))
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--p", type=int, required=True)
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--dev-seeds", type=int, nargs="*", default=[0, 1, 2, 3, 4])
    ap.add_argument("--grid", type=int, default=20)
    ap.add_argument("--restarts", type=int, default=200)
    ap.add_argument("--out-dir", default=str(ROOT / "results" / "tuning"))
    args = ap.parse_args(argv)

    n = args.n
    k = args.k or max(3, round(n / 5))
    t0 = time.perf_counter()

    insts = [GEN[args.family](n=n, k=k, seed=s) for s in args.dev_seeds]
    print(f"tuning {args.family} n={n} k={k} p={args.p} on {len(insts)} dev seeds",
          flush=True)

    def mean_H(p, gammas, betas):
        vals = [_eval(inst, p, gammas, betas, seed=0) for inst in insts]
        return float(np.mean(vals))

    if args.p == 1:
        gammas = np.linspace(0, 2 * np.pi, args.grid, endpoint=False)
        betas = np.linspace(0, np.pi, args.grid, endpoint=False)
        best = (np.inf, None, None)
        for g in gammas:
            for b in betas:
                v = mean_H(1, [g], [b])
                if v < best[0]:
                    best = (v, [float(g)], [float(b)])
        score, g_best, b_best = best
    else:
        rng = np.random.default_rng(42)
        best = (np.inf, None, None)
        for _ in range(args.restarts):
            gs = list(rng.uniform(0, 2 * np.pi, size=args.p))
            bs = list(rng.uniform(0, np.pi, size=args.p))
            v = mean_H(args.p, gs, bs)
            if v < best[0]:
                best = (v, gs, bs)
        score, g_best, b_best = best

    out = Path(args.out_dir) / f"{args.family}_n{n}_p{args.p}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "family": args.family, "n": n, "k": k, "p": args.p,
        "gammas": g_best, "betas": b_best,
        "score_energy_analytic": score,
        "dev_seeds": args.dev_seeds,
        "wall_time_s": time.perf_counter() - t0,
    }, indent=2))
    print(f"  -> best <H_C>={score:.6f} gammas={g_best} betas={b_best} "
          f"({time.perf_counter()-t0:.1f}s)", flush=True)
    print(f"  saved {out}")


if __name__ == "__main__":
    main()
