"""Command-line interface: modbench generate|run|report."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

from .generators import random_sparse, planted_scale_free, planted_decoy_hubs
from .instances import save
from .solvers.exact import solve_exact_milp


def _cmd_generate(args):
    gens = {"random_sparse": random_sparse,
            "planted_scale_free": planted_scale_free,
            "planted_decoy_hubs": planted_decoy_hubs}
    fn = gens[args.generator]
    inst = fn(n=args.n, k=args.k, seed=args.seed)
    out = Path(args.out)
    if args.exact:
        res = solve_exact_milp(inst, lambda_coh=args.lambda_coh,
                               lambda_red=args.lambda_red,
                               time_limit=args.time_limit)
        save(inst, out, best_known_value=res["objective"],
             optimality_status=res["optimality"])
        print(f"saved {out}.npz + .json | best_known={res['objective']:.6f} "
              f"| optimality={res['optimality']}")
    else:
        save(inst, out)
        print(f"saved {out}.npz + .json (no exact run)")


def _not_implemented(name):
    def _f(_args):
        raise SystemExit(f"`modbench {name}` lands in Phase 5")
    return _f


def main(argv=None):
    p = argparse.ArgumentParser(prog="modbench")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="generate an instance")
    g.add_argument("--generator", required=True,
                   choices=["random_sparse", "planted_scale_free", "planted_decoy_hubs"])
    g.add_argument("--n", type=int, required=True)
    g.add_argument("--k", type=int, required=True)
    g.add_argument("--seed", type=int, required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--exact", action="store_true")
    g.add_argument("--lambda-coh", type=float, default=1.0, dest="lambda_coh")
    g.add_argument("--lambda-red", type=float, default=0.1, dest="lambda_red")
    g.add_argument("--time-limit", type=float, default=60.0, dest="time_limit")
    g.set_defaults(func=_cmd_generate)

    for other in ("run", "report"):
        s = sub.add_parser(other, help=f"{other} (Phase 5)")
        s.set_defaults(func=_not_implemented(other))

    ns = p.parse_args(argv)
    return ns.func(ns)
