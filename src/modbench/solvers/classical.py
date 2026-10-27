"""Classical solvers: random, greedy, SA, tabu."""
from __future__ import annotations
import time
import numpy as np
import dimod
import neal
import tabu
from .base import Solver, SolverResult


def _qubo_to_bqm(qubo, tight_penalty: bool = True):
    """Convert QUBO to a dimod BQM.

    tight_penalty=True uses the minimal sufficient cardinality penalty:

        P_min = max|h| + max_i Σ_j |J_ij| + 1

    Proof sketch: flipping one bit changes F by at most max|h| +
    max_i Σ_j |J_ij|; the penalty P·(Σx−k)² jumps by at least P for any
    single-bit change that changes cardinality.  Hence P_min makes any
    off-cardinality solution strictly worse than any feasible one.  This
    is mathematically equivalent to the conservative P in QUBO for the
    purpose of argmin over feasible sets.

    Why it matters here: neal and tabu auto-derive beta/temperature
    scales from |coefficients|.  With the conservative P (which can be
    ~7× larger), the auto schedule is too cold to freeze or too hot to
    search.  Using P_min keeps the landscape tractable without affecting
    feasibility: our acceptance still uses qubo.energy(x) with the full
    penalty.
    """
    n, k = qubo.n, qubo.k
    if tight_penalty:
        lin_max = float(np.max(np.abs(qubo.h))) if n > 0 else 0.0
        if qubo.J.nnz > 0:
            absJ = np.abs(qubo.J)
            row_sum = np.asarray(absJ.sum(axis=1)).ravel()
            row_max = float(row_sum.max())
        else:
            row_max = 0.0
        P = lin_max + row_max + 1.0
    else:
        P = qubo.penalty_coeff

    h = qubo.h + P * (1 - 2 * k)
    Jd = qubo.J.toarray()
    Q = {(i, i): float(h[i]) for i in range(n)}
    iu, ju = np.triu_indices(n, k=1)
    for i, j in zip(iu, ju):
        Q[(int(i), int(j))] = float(Jd[i, j] + 2.0 * P)
    return dimod.BinaryQuadraticModel.from_qubo(Q, offset=float(P * k * k))


def _sample_to_x(sample, n: int) -> np.ndarray:
    if hasattr(sample, "get"):
        return np.array([int(sample.get(i, 0)) for i in range(n)], dtype=np.int8)
    return np.asarray(sample, dtype=np.int8)


def _best_from_samples(samples, qubo, n: int):
    best_e, best_x = np.inf, None
    for sample in samples.samples():
        x = _sample_to_x(sample, n)
        e = qubo.energy(x)
        if e < best_e:
            best_e, best_x = e, x
    return best_e, best_x


def _wrap(inst, x, wall_time_s, n_evals, extra=None):
    lam = extra.get("lambda_coh", 1.0) if extra else 1.0
    lam_red = extra.get("lambda_red", 0.1) if extra else 0.1
    feasible = int(x.sum()) == inst.k
    obj = inst.objective(x, lam, lam_red) if feasible else float("nan")
    qubo = inst.to_qubo(lam, lam_red)
    return SolverResult(x=x, energy=qubo.energy(x), objective=obj,
                        feasible=feasible, wall_time_s=wall_time_s,
                        n_evals=n_evals, extra=extra or {})


class RandomSolver(Solver):
    name = "random"
    def __init__(self, n_samples: int = 10000):
        self.n_samples = int(n_samples)

    def solve(self, inst, *, lambda_coh, lambda_red, seed,
              time_limit=10.0, n_evals=None):
        rng = np.random.default_rng(seed)
        qubo = inst.to_qubo(lambda_coh, lambda_red)
        N = self.n_samples if n_evals is None else min(n_evals, self.n_samples)
        t0 = time.perf_counter()
        best_e, best_x = np.inf, None
        for _ in range(N):
            S = rng.choice(inst.n, size=inst.k, replace=False)
            x = np.zeros(inst.n, dtype=np.int8); x[S] = 1
            e = qubo.energy(x)
            if e < best_e:
                best_e, best_x = e, x
        return _wrap(inst, best_x, time.perf_counter() - t0, N,
                     {"lambda_coh": lambda_coh, "lambda_red": lambda_red})


class GreedySolver(Solver):
    name = "greedy"
    def __init__(self, restarts: int = 20):
        self.restarts = int(restarts)

    def solve(self, inst, *, lambda_coh, lambda_red, seed,
              time_limit=10.0, n_evals=None):
        rng = np.random.default_rng(seed)
        qubo = inst.to_qubo(lambda_coh, lambda_red)
        t0 = time.perf_counter()
        evals = 0
        best_e, best_x = np.inf, None
        n = inst.n

        for r in range(self.restarts):
            x = np.zeros(n, dtype=np.int8)
            for _ in range(inst.k):
                cand = np.where(x == 0)[0]
                scores = np.empty(len(cand))
                for ci, i in enumerate(cand):
                    x_try = x.copy(); x_try[i] = 1
                    scores[ci] = qubo.energy(x_try)
                if r > 0 and rng.random() < 0.5:
                    scores = scores + rng.normal(0, 1e-6, size=len(cand))
                evals += len(cand)
                x[cand[np.argmin(scores)]] = 1
            e = qubo.energy(x)
            if e < best_e:
                best_e, best_x = e, x

        return _wrap(inst, best_x, time.perf_counter() - t0, evals,
                     {"lambda_coh": lambda_coh, "lambda_red": lambda_red,
                      "restarts": self.restarts})


class SASolver(Solver):
    name = "sa"
    def __init__(self, num_reads: int = 200, num_sweeps: int = 3000):
        self.num_reads = int(num_reads)
        self.num_sweeps = int(num_sweeps)

    def solve(self, inst, *, lambda_coh, lambda_red, seed,
              time_limit=10.0, n_evals=None):
        qubo = inst.to_qubo(lambda_coh, lambda_red)
        bqm = _qubo_to_bqm(qubo, tight_penalty=True)
        reads = self.num_reads if n_evals is None else min(n_evals, self.num_reads)
        t0 = time.perf_counter()
        sampler = neal.SimulatedAnnealingSampler()
        samples = sampler.sample(bqm, num_reads=reads,
                                 num_sweeps=self.num_sweeps, seed=int(seed))
        _, x_best = _best_from_samples(samples, qubo, inst.n)
        return _wrap(inst, x_best, time.perf_counter() - t0, reads,
                     {"lambda_coh": lambda_coh, "lambda_red": lambda_red,
                      "num_reads": reads, "num_sweeps": self.num_sweeps,
                      "tight_penalty": True})


class TabuSolver(Solver):
    name = "tabu"
    def __init__(self, num_reads: int = 100):
        self.num_reads = int(num_reads)

    def solve(self, inst, *, lambda_coh, lambda_red, seed,
              time_limit=10.0, n_evals=None):
        qubo = inst.to_qubo(lambda_coh, lambda_red)
        bqm = _qubo_to_bqm(qubo, tight_penalty=True)
        reads = self.num_reads if n_evals is None else min(n_evals, self.num_reads)
        t0 = time.perf_counter()
        sampler = tabu.TabuSampler()
        samples = sampler.sample(bqm, num_reads=reads, seed=int(seed))
        _, x_best = _best_from_samples(samples, qubo, inst.n)
        return _wrap(inst, x_best, time.perf_counter() - t0, reads,
                     {"lambda_coh": lambda_coh, "lambda_red": lambda_red,
                      "num_reads": reads, "tight_penalty": True})
