import numpy as np
import pytest

from modbench.generators import random_sparse, planted_scale_free, planted_decoy_hubs


@pytest.mark.parametrize("gen,kwargs", [
    (random_sparse, dict(n=30, k=6)),
    (planted_scale_free, dict(n=40, k=8)),
    (planted_decoy_hubs, dict(n=50, k=8, n_decoys=8)),
])
def test_regeneration_bit_identical(gen, kwargs):
    a = gen(seed=7, **kwargs)
    b = gen(seed=7, **kwargs)
    assert a.n == b.n and a.k == b.k
    assert np.array_equal(a.weights, b.weights)
    assert (a.adj != b.adj).nnz == 0
    if a.planted is not None:
        assert np.array_equal(np.sort(a.planted), np.sort(b.planted))


def test_planted_solution_is_feasible_and_finite():
    inst = planted_scale_free(n=40, k=8, seed=1)
    assert inst.planted is not None and len(inst.planted) == inst.k
    x = np.zeros(inst.n, dtype=np.int8); x[inst.planted] = 1
    val = inst.objective(x, lambda_coh=1.0, lambda_red=0.1)
    assert isinstance(val, float) and np.isfinite(val)


def test_decoy_hubs_have_higher_degree_than_planted():
    inst = planted_decoy_hubs(n=60, k=8, seed=2, n_decoys=8, decoy_edges_per_node=4)
    d = inst.degrees()
    decoys = np.array(inst.meta["decoys"])
    assert d[decoys].mean() > d[inst.planted].mean()
