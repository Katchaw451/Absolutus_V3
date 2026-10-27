"""Exact MILP solver wrapped in the Solver interface."""
from __future__ import annotations
import time
import numpy as np
from .base import Solver, SolverResult
from .exact import solve_exact_milp


class ExactSolver(Solver):
    name = "exact"
    def __init__(self, milp_time_limit: float = 60.0):
        self.milp_time_limit = float(milp_time_limit)

    def solve(self, inst, *, lambda_coh, lambda_red, seed,
              time_limit=60.0, n_evals=None):
        t0 = time.perf_counter()
        res = solve_exact_milp(inst, lambda_coh, lambda_red,
                               time_limit=min(time_limit, self.milp_time_limit))
        x = res["x"] if res["x"] is not None else np.zeros(inst.n, dtype=np.int8)
        feasible = res["x"] is not None and int(x.sum()) == inst.k
        qubo = inst.to_qubo(lambda_coh, lambda_red)
        return SolverResult(
            x=x, energy=qubo.energy(x),
            objective=res["objective"] if res["objective"] is not None else float("nan"),
            feasible=bool(feasible),
            wall_time_s=time.perf_counter() - t0,
            n_evals=-1,  # MILP is not eval-counted; report -1 and note in extra
            extra={"lambda_coh": lambda_coh, "lambda_red": lambda_red,
                   "milp_status": res["optimality"], "milp_message": res["status"]},
        )
