import numpy as np
import pytest

from modbench.generators import random_sparse, planted_scale_free


@pytest.mark.parametrize("gen,kwargs", [
    (random_sparse, dict(n=15, k=4)),
    (planted_scale_free, dict(n=20, k=5)),
])
@pytest.mark.parametrize("lam_coh,lam_red", [(1.0, 0.0), (2.0, 0.5)])
def test_qubo_energy_equals_negative_objective_on_feasible(gen, kwargs, lam_coh, lam_red):
    inst = gen(seed=3, **kwargs)
    rng = np.random.default_rng(0)
    for _ in range(20):
        S = rng.choice(inst.n, size=inst.k, replace=False)
        x = np.zeros(inst.n, dtype=np.int8); x[S] = 1
        q = inst.to_qubo(lam_coh, lam_red)
        assert q.energy(x) == pytest.approx(-inst.objective(x, lam_coh, lam_red), abs=1e-9)


def test_qubo_penalty_dominates_offcardinality():
    inst = random_sparse(n=12, k=3, seed=11)
    lam_coh, lam_red = 1.0, 0.1
    q = inst.to_qubo(lam_coh, lam_red)
    rng = np.random.default_rng(0)
    infeasible = []
    for _ in range(200):
        m = int(rng.integers(0, 6))
        if m == inst.k:
            continue
        S = rng.choice(inst.n, size=m, replace=False)
        x = np.zeros(inst.n, dtype=np.int8); x[S] = 1
        infeasible.append(q.energy(x))
    rng2 = np.random.default_rng(1)
    feasible = []
    for _ in range(50):
        S = rng2.choice(inst.n, size=inst.k, replace=False)
        x = np.zeros(inst.n, dtype=np.int8); x[S] = 1
        feasible.append(q.energy(x))
    assert min(infeasible) > max(feasible)
