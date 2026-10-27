"""Solver smoke tests and planted-recovery checks."""
import numpy as np
import pytest

from modbench.generators import planted_scale_free
from modbench.solvers.classical import RandomSolver, GreedySolver, SASolver, TabuSolver
from modbench.solvers.qaoa import QAOASolver


def test_random_solver_feasible():
    inst = planted_scale_free(n=20, k=5, seed=1)
    r = RandomSolver(n_samples=100).solve(inst, lambda_coh=1.0, lambda_red=0.1, seed=0)
    assert r.feasible and int(r.x.sum()) == inst.k


@pytest.mark.parametrize("solver_factory", [
    lambda: GreedySolver(restarts=20),
    lambda: SASolver(num_reads=200),
    lambda: TabuSolver(num_reads=100),
])
def test_classical_recovers_planted_easy(solver_factory):
    """On an easy planted instance, the solver should match the planted objective."""
    inst = planted_scale_free(n=20, k=4, seed=1, weight_signal=5.0, weight_noise=0.1)
    planted = np.zeros(inst.n, dtype=np.int8)
    planted[inst.planted] = 1
    planted_obj = inst.objective(planted, lambda_coh=0.5, lambda_red=0.0)

    r = solver_factory().solve(inst, lambda_coh=0.5, lambda_red=0.0, seed=0)
    assert r.feasible
    assert r.objective == pytest.approx(planted_obj, abs=1e-6), \
        f"{r.extra} got {r.objective}, planted {planted_obj}"


def test_qaoa_smoke():
    inst = planted_scale_free(n=6, k=3, seed=1)
    r = QAOASolver(p=1, n_shots=200, gammas=[0.3], betas=[0.4]).solve(
        inst, lambda_coh=1.0, lambda_red=0.1, seed=0)
    assert r.x.shape == (inst.n,)
    assert 0.0 <= r.feasible_fraction <= 1.0
