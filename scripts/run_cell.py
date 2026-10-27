#!/usr/bin/env python3
"""Run one (family, n, k, seed, solver) cell and print/write a JSONL record."""
from __future__ import annotations
import argparse
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from modbench.generators import random_sparse, planted_scale_free, planted_decoy_hubs
from modbench.solvers import SOLVERS
from modbench.engines import ENGINES
from modbench.solvers.qaoa import QAOASolver

GEN = {"random": random_sparse, "planted": planted_scale_free, "decoy": planted_decoy_hubs}


def peak_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(Path(__file__).resolve().parents[1]),
            text=True).strip()
    except Exception:
        return "unknown"


def _jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def run_one(args) -> dict:
    inst = GEN[args.family](n=args.n, k=args.k, seed=args.seed)

    if args.solver == "qaoa":
        eng_kw = json.loads(args.engine_kwargs)
        eng = ENGINES[args.engine](**eng_kw)
        gammas = np.asarray(args.gammas, dtype=float) if args.gammas else None
        betas = np.asarray(args.betas, dtype=float) if args.betas else None
        solver = QAOASolver(engine=eng, p=args.p, n_shots=args.n_shots,
                            gammas=gammas, betas=betas)
    else:
        solver = SOLVERS[args.solver](**json.loads(args.solver_kwargs))

    t0 = time.perf_counter()
    try:
        result = solver.solve(inst, lambda_coh=args.lambda_coh,
                              lambda_red=args.lambda_red, seed=args.seed,
                              time_limit=args.time_limit,
                              n_evals=args.n_evals_budget)
        success = True
        error = None
    except Exception as e:  # noqa: BLE001 - we record and continue
        success = False
        error = f"{type(e).__name__}: {e}"
        result = None
    wall = time.perf_counter() - t0
    mem = peak_mb()

    rec = {
        "family": args.family, "n": args.n, "k": args.k, "seed": args.seed,
        "solver": args.solver, "engine": args.engine, "p": args.p,
        "lambda_coh": args.lambda_coh, "lambda_red": args.lambda_red,
        "time_limit": args.time_limit, "n_evals_budget": args.n_evals_budget,
        "success": success, "error": error,
        "wall_time_s": wall, "peak_mb": mem,
        "env": {"git_head": git_head(),
                "numpy": np.__version__,
                "python": sys.version.split()[0]},
    }
    if result is not None:
        rec.update({
            "energy": float(result.energy),
            "objective": float(result.objective) if np.isfinite(result.objective) else None,
            "feasible": bool(result.feasible),
            "feasible_fraction": float(result.feasible_fraction),
            "n_evals": int(result.n_evals),
            "extra": {k: _jsonable(v) for k, v in result.extra.items()},
        })
    return rec


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--family", required=True, choices=list(GEN))
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--k", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--solver", required=True)
    p.add_argument("--engine", default="statevector")
    p.add_argument("--engine-kwargs", default="{}", dest="engine_kwargs")
    p.add_argument("--solver-kwargs", default="{}", dest="solver_kwargs")
    p.add_argument("--p", type=int, default=1)
    p.add_argument("--n-shots", type=int, default=1024, dest="n_shots")
    p.add_argument("--gammas", type=float, nargs="*", default=None)
    p.add_argument("--betas", type=float, nargs="*", default=None)
    p.add_argument("--lambda-coh", type=float, default=1.0, dest="lambda_coh")
    p.add_argument("--lambda-red", type=float, default=0.1, dest="lambda_red")
    p.add_argument("--time-limit", type=float, default=30.0, dest="time_limit")
    p.add_argument("--n-evals-budget", type=int, default=10000, dest="n_evals_budget")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    rec = run_one(args)
    line = json.dumps(rec)
    print(line)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "a") as f:
            f.write(line + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
