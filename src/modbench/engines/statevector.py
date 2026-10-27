"""Exact statevector QAOA (n <= 20, validation only)."""
from __future__ import annotations
import time
import numpy as np
from .base import Engine, EngineResult
from ._qaoa_common import all_energies, apply_rx


class StatevectorEngine(Engine):
    name = "statevector"
    def __init__(self, max_n: int = 20):
        self.max_n = int(max_n)

    def run_qaoa(self, inst, *, lambda_coh, lambda_red, p, gammas, betas,
                 n_shots=1024, seed=0, time_limit=60.0):
        n = inst.n
        if n > self.max_n:
            raise ValueError(f"statevector requires n<={self.max_n}")
        qubo = inst.to_qubo(lambda_coh, lambda_red)
        t0 = time.perf_counter()

        energies = all_energies(qubo)
        psi = np.ones(2 ** n, dtype=np.complex128) / np.sqrt(2 ** n)

        for l in range(p):
            psi = psi * np.exp(-1j * gammas[l] * energies)
            for q in range(n):
                psi = apply_rx(psi, q, 2 * betas[l], n)

        probs = np.abs(psi) ** 2
        probs /= probs.sum()
        expectation = float(np.sum(probs * energies))

        rng = np.random.default_rng(seed)
        idx = rng.choice(2 ** n, size=n_shots, p=probs)
        states = np.arange(2 ** n, dtype=np.int64)[idx]
        bits = ((states[:, None] >> np.arange(n)) & 1).astype(np.int8)

        feasible = bits.sum(axis=1) == inst.k
        obj = np.full(n_shots, np.nan)
        for i in np.where(feasible)[0]:
            obj[i] = inst.objective(bits[i], lambda_coh, lambda_red)

        energies_of_samples = energies[idx]
        best = int(np.argmin(energies_of_samples))
        return EngineResult(
            x_best=bits[best].copy(),
            energy_best=float(energies_of_samples[best]),
            energy_mean=float(energies_of_samples.mean()),
            energy_analytic=expectation,
            objective_best=float(obj[best]) if feasible[best] else float("nan"),
            feasible_fraction=float(feasible.mean()),
            discarded_weight=0.0,
            wall_time_s=time.perf_counter() - t0,
            n_evals=int(n_shots),
            metadata={"dense_state": psi.copy(),
                      "p": p, "gammas": list(map(float, gammas)),
                      "betas": list(map(float, betas)),
                      "lambda_coh": lambda_coh, "lambda_red": lambda_red},
        )
