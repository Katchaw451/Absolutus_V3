import itertools
import numpy as np
import pytest

from modbench.generators import random_sparse, planted_scale_free
from modbench.solvers.exact import solve_exact_milp


def _bruteforce(inst, lam_coh, lam_red):
    best = -np.inf
    for S in itertools.combinations(range(inst.n), inst.k):
        x = np.zeros(inst.n, dtype=np.int8); x[list(S)] = 1
        v = inst.objective(x, lam_coh, lam_red)
        if v > best:
            best = v
    return best


@pytest.mark.parametrize("gen,kwargs", [
    (random_sparse, dict(n=12, k=3)),
    (random_sparse, dict(n=14, k=4)),
    (planted_scale_free, dict(n=15, k=4)),
])
def test_milp_matches_bruteforce(gen, kwargs):
    inst = gen(seed=5, **kwargs)
    lam_coh, lam_red = 1.0, 0.1
    res = solve_exact_milp(inst, lam_coh, lam_red, time_limit=30.0)
    bf = _bruteforce(inst, lam_coh, lam_red)
    assert res["optimality"] == "optimal", res
    assert res["objective"] == pytest.approx(bf, abs=1e-9)
