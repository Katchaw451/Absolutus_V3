#!/usr/bin/env python3
"""Matrix runner: iterate (family, n, k, seed, solver, engine, chi/p) cells.

Resumable: skips cells whose JSONL already contains a success line for the
exact tuple (solver, engine, p, chi, seed). Failures are recorded but not
silently skipped — they show up in `analyze.py` as success_frac < 1.

Usage:
  run_matrix.py --kind classical --seeds dev
  run_matrix.py --kind qaoa-sv   --seeds test --ps 1 2 3
  run_matrix.py --kind qaoa-mps  --seeds test --ps 1 2 --chi 2 4 8 16 32
  run_matrix.py --kind exact     --seeds test --max-n 30
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TUNING_DIR = ROOT / "results" / "tuning"


SEEDS = {"dev": list(range(0, 5)), "test": list(range(100, 120))}


def k_of(n: int) -> int:
    return max(3, round(n / 5))


def _out_path(phase: str, family: str, n: int, k: int, tag: str) -> Path:
    return ROOT / "results" / phase / family / f"n{n:02d}_k{k:02d}" / f"{tag}.jsonl"


def _existing_cells(path: Path) -> set:
    """Set of (solver, engine, p, chi, seed) that already have a success line."""
    if not path.exists():
        return set()
    seen = set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not r.get("success", False):
            continue
        chi = r.get("extra", {}).get("chi") if r.get("engine") == "mps" else None
        seen.add((r.get("solver"), r.get("engine"), r.get("p"), chi, r.get("seed")))
    return seen


def _run_one(family, n, k, seed, solver, engine, p, engine_kwargs,
             phase, tag, lambda_coh, lambda_red, time_limit, n_evals_budget):
    out = _out_path(phase, family, n, k, tag)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(ROOT / "scripts" / "run_cell.py"),
        "--family", family, "--n", str(n), "--k", str(k), "--seed", str(seed),
        "--solver", solver, "--engine", engine, "--p", str(p),
        "--engine-kwargs", json.dumps(engine_kwargs),
        "--lambda-coh", str(lambda_coh), "--lambda-red", str(lambda_red),
        "--time-limit", str(time_limit), "--n-evals-budget", str(n_evals_budget),
        "--out", str(out),
    ]
    t0 = time.perf_counter()
    r = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0
    status = "ok" if r.returncode == 0 else f"FAIL({r.returncode})"
    print(f"  [{status}] {family} n={n} seed={seed} {solver}/{engine}"
          f"{'/chi='+str(engine_kwargs.get('chi')) if 'chi' in engine_kwargs else ''}"
          f"/p={p} ({elapsed:.1f}s)", flush=True)
    if r.returncode != 0:
        print(f"  stderr: {r.stderr.strip()[:500]}", file=sys.stderr, flush=True)


def _load_tuning(family: str, n: int, p: int):
    """Return (gammas, betas) tuned on dev for (family, n, p) or None."""
    f = TUNING_DIR / f"{family}_n{n}_p{p}.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text())
    return d.get("gammas"), d.get("betas")


def run_classical(args, seeds):
    sizes = args.sizes or [12, 20, 30, 40, 60]
    families = args.families or ["random", "planted", "decoy"]
    solvers = args.solvers or ["random", "greedy", "sa", "tabu"]
    for family in families:
        for n in sizes:
            k = k_of(n)
            for solver in solvers:
                out = _out_path(args.phase, family, n, k, solver)
                done = _existing_cells(out)
                for seed in seeds:
                    if (solver, "statevector", 1, None, seed) in done:
                        continue
                    _run_one(family, n, k, seed, solver, "statevector", 1, {},
                             args.phase, solver, args.lambda_coh, args.lambda_red,
                             args.time_limit, args.n_evals_budget)


def run_exact(args, seeds):
    sizes = [n for n in (args.sizes or [12, 20, 30, 40, 60]) if n <= args.max_n]
    families = args.families or ["random", "planted", "decoy"]
    for family in families:
        for n in sizes:
            k = k_of(n)
            out = _out_path(args.phase, family, n, k, "exact")
            done = _existing_cells(out)
            for seed in seeds:
                if ("exact", "statevector", 1, None, seed) in done:
                    continue
                _run_one(family, n, k, seed, "exact", "statevector", 1, {},
                         args.phase, "exact", args.lambda_coh, args.lambda_red,
                         args.time_limit, args.n_evals_budget)


def run_qaoa_sv(args, seeds):
    sizes = [n for n in (args.sizes or [12, 20]) if n <= 20]
    families = args.families or ["random", "planted", "decoy"]
    ps = args.ps or [1, 2, 3]
    for family in families:
        for n in sizes:
            k = k_of(n)
            for p in ps:
                tuned = _load_tuning(family, n, p)
                if tuned is None:
                    print(f"  [skip] no dev tuning for {family} n={n} p={p};"
                          f" run scripts/tune_qaoa_dev.py first", flush=True)
                    continue
                gammas, betas = tuned
                out = _out_path(args.phase, family, n, k, f"qaoa_sv_p{p}")
                done = _existing_cells(out)
                for seed in seeds:
                    if ("qaoa", "statevector", p, None, seed) in done:
                        continue
                    _run_one(family, n, k, seed, "qaoa", "statevector", p, {},
                             args.phase, f"qaoa_sv_p{p}",
                             args.lambda_coh, args.lambda_red,
                             args.time_limit, args.n_evals_budget) if False else None
                    # Direct call with tuned gammas/betas:
                    out.parent.mkdir(parents=True, exist_ok=True)
                    cmd = [
                        sys.executable, str(ROOT / "scripts" / "run_cell.py"),
                        "--family", family, "--n", str(n), "--k", str(k),
                        "--seed", str(seed), "--solver", "qaoa",
                        "--engine", "statevector", "--p", str(p),
                        "--gammas", *map(str, gammas),
                        "--betas", *map(str, betas),
                        "--lambda-coh", str(args.lambda_coh),
                        "--lambda-red", str(args.lambda_red),
                        "--time-limit", str(args.time_limit),
                        "--n-evals-budget", str(args.n_evals_budget),
                        "--out", str(out),
                    ]
                    t0 = time.perf_counter()
                    r = subprocess.run(cmd, capture_output=True, text=True)
                    print(f"  [{'ok' if r.returncode==0 else 'FAIL'}]"
                          f" {family} n={n} seed={seed} qaoa/sv/p={p}"
                          f" ({time.perf_counter()-t0:.1f}s)", flush=True)
                    if r.returncode != 0:
                        print(f"  stderr: {r.stderr.strip()[:500]}",
                              file=sys.stderr, flush=True)


def run_qaoa_mps(args, seeds):
    sizes = args.sizes or [12, 20, 30, 40]
    families = args.families or ["random", "planted", "decoy"]
    ps = args.ps or [1, 2]
    chis = args.chi or [2, 4, 8, 16, 32]
    for family in families:
        for n in sizes:
            k = k_of(n)
            for p in ps:
                tuned = _load_tuning(family, n, p)
                if tuned is None:
                    print(f"  [skip] no dev tuning for {family} n={n} p={p}",
                          flush=True)
                    continue
                gammas, betas = tuned
                for chi in chis:
                    tag = f"qaoa_mps_p{p}_chi{chi}"
                    out = _out_path(args.phase, family, n, k, tag)
                    done = _existing_cells(out)
                    for seed in seeds:
                        if ("qaoa", "mps", p, chi, seed) in done:
                            continue
                        out.parent.mkdir(parents=True, exist_ok=True)
                        cmd = [
                            sys.executable, str(ROOT / "scripts" / "run_cell.py"),
                            "--family", family, "--n", str(n), "--k", str(k),
                            "--seed", str(seed), "--solver", "qaoa",
                            "--engine", "mps", "--p", str(p),
                            "--engine-kwargs", json.dumps({"chi": chi}),
                            "--gammas", *map(str, gammas),
                            "--betas", *map(str, betas),
                            "--lambda-coh", str(args.lambda_coh),
                            "--lambda-red", str(args.lambda_red),
                            "--time-limit", str(args.time_limit),
                            "--n-evals-budget", str(args.n_evals_budget),
                            "--out", str(out),
                        ]
                        t0 = time.perf_counter()
                        r = subprocess.run(cmd, capture_output=True, text=True)
                        print(f"  [{'ok' if r.returncode==0 else 'FAIL'}]"
                              f" {family} n={n} seed={seed} qaoa/mps/p={p}/chi={chi}"
                              f" ({time.perf_counter()-t0:.1f}s)", flush=True)
                        if r.returncode != 0:
                            print(f"  stderr: {r.stderr.strip()[:500]}",
                                  file=sys.stderr, flush=True)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--kind", required=True,
                   choices=["classical", "qaoa-sv", "qaoa-mps", "exact"])
    p.add_argument("--seeds", choices=["dev", "test"], default="test")
    p.add_argument("--phase", default=None,
                   help="results/<phase>/... (default: 'dev' or 'test' from --seeds)")
    p.add_argument("--families", nargs="*", default=None)
    p.add_argument("--sizes", nargs="*", type=int, default=None)
    p.add_argument("--solvers", nargs="*", default=None)
    p.add_argument("--ps", nargs="*", type=int, default=None)
    p.add_argument("--chi", nargs="*", type=int, default=None)
    p.add_argument("--max-n", type=int, default=30)
    p.add_argument("--lambda-coh", type=float, default=1.0)
    p.add_argument("--lambda-red", type=float, default=0.1)
    p.add_argument("--time-limit", type=float, default=30.0)
    p.add_argument("--n-evals-budget", type=int, default=10000)
    args = p.parse_args(argv)
    if args.phase is None:
        args.phase = "dev" if args.seeds == "dev" else "test"
    seeds = SEEDS[args.seeds]

    t0 = time.perf_counter()
    if args.kind == "classical":
        run_classical(args, seeds)
    elif args.kind == "exact":
        run_exact(args, seeds)
    elif args.kind == "qaoa-sv":
        run_qaoa_sv(args, seeds)
    elif args.kind == "qaoa-mps":
        run_qaoa_mps(args, seeds)
    print(f"done in {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()
