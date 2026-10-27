"""QAOA solver: wraps an Engine."""
from __future__ import annotations
import time
import numpy as np
from .base import Solver, SolverResult
from ..engines.statevector import StatevectorEngine


class QAOASolver(Solver):
    name = "qaoa"
    def __init__(self, engine=None, p: int = 1, n_shots: int = 1024,
                 gammas=None, betas=None):
        self.engine = engine if engine is not None else StatevectorEngine()
        self.p = int(p)
        self.n_shots = int(n_shots)
        self.gammas = gammas
        self.betas = betas

    def solve(self, inst, *, lambda_coh, lambda_red, seed,
              time_limit=60.0, n_evals=None):
        rng = np.random.default_rng(seed)
        gammas = (np.asarray(self.gammas) if self.gammas is not None
                  else rng.uniform(0.0, 2 * np.pi, size=self.p))
        betas = (np.asarray(self.betas) if self.betas is not None
                 else rng.uniform(0.0, np.pi, size=self.p))
        t0 = time.perf_counter()
        r = self.engine.run_qaoa(inst, lambda_coh=lambda_coh, lambda_red=lambda_red,
                                 p=self.p, gammas=gammas, betas=betas,
                                 n_shots=self.n_shots, seed=seed,
                                 time_limit=time_limit)

        # Propagate all JSON-serializable engine metadata (chi, cutoff for MPS),
        # excluding the dense_state ndarray which is not needed downstream.
        engine_meta = {k: v for k, v in r.metadata.items()
                       if k not in ("dense_state", "gammas", "betas")}

        extra = {
            "engine": self.engine.name,
            "p": self.p,
            "lambda_coh": lambda_coh,
            "lambda_red": lambda_red,
            "gammas": list(map(float, gammas)),
            "betas": list(map(float, betas)),
            "energy_analytic": r.energy_analytic,
            "discarded_weight": r.discarded_weight,
            **engine_meta,
        }
        return SolverResult(
            x=r.x_best, energy=r.energy_best, objective=r.objective_best,
            feasible=bool(int(r.x_best.sum()) == inst.k),
            wall_time_s=time.perf_counter() - t0,
            n_evals=r.n_evals,
            feasible_fraction=r.feasible_fraction,
            extra=extra,
        )
