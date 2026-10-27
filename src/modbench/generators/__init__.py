"""Seeded instance generators.

All generators are deterministic for a given (seed, params) and return a
modbench.core.Instance.
"""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp
import networkx as nx

from ..core import Instance


def random_sparse(n: int, k: int, seed: int, density: float = 0.05,
                  weight_std: float = 1.0) -> Instance:
    """Erdős–Rényi sparse graph, iid Gaussian weights (no planted structure)."""
    if not 0 < density < 1:
        raise ValueError("density must be in (0, 1)")
    rng = np.random.default_rng(seed)
    m_edges = max(1, int(density * n * (n - 1) / 2))
    iu, ju = np.triu_indices(n, k=1)
    idx = rng.choice(len(iu), size=m_edges, replace=False)
    rows = np.concatenate([iu[idx], ju[idx]])
    cols = np.concatenate([ju[idx], iu[idx]])
    data = np.ones(rows.size, dtype=np.float64)
    A = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    w = rng.normal(0.0, weight_std, n)
    return Instance(n=n, k=k, weights=w, adj=A, planted=None,
                    meta={"generator": "random_sparse", "seed": seed,
                          "density": density, "m_edges": m_edges})


def planted_scale_free(n: int, k: int, seed: int, m: int = 2,
                       weight_signal: float = 2.0, weight_noise: float = 1.0) -> Instance:
    """Planted k-module on a Barabási–Albert graph with noisy scores."""
    G = nx.barabasi_albert_graph(n, m, seed=seed)
    A = nx.to_scipy_sparse_array(G, nodelist=list(range(n)), format="csr", dtype=np.float64)
    rng = np.random.default_rng(seed + 1)
    planted = rng.choice(n, size=k, replace=False)
    w = rng.normal(0.0, weight_noise, n)
    w[planted] += weight_signal
    return Instance(n=n, k=k, weights=w, adj=sp.csr_matrix(A), planted=planted,
                    meta={"generator": "planted_scale_free", "seed": seed,
                          "m_ba": m, "weight_signal": weight_signal,
                          "weight_noise": weight_noise})


def planted_decoy_hubs(n: int, k: int, seed: int, m: int = 2,
                       n_decoys: int | None = None, decoy_edges_per_node: int = 3,
                       weight_signal: float = 2.0, weight_noise: float = 1.0) -> Instance:
    """Planted k-module + decoy hubs (high degree, non-boosted weight).

    Exposes degree/study-bias artefacts: methods that only reward cohesion
    (λ_red = 0) tend to pick the decoy hubs; methods with λ_red > 0 recover
    the planted module.

    If n_decoys is None, it is auto-scaled to min(10, max(2, n//4)) so the
    generator is valid for small n. Pass an explicit value for full control.
    """
    if n_decoys is None:
        n_decoys = min(10, max(2, n // 4))
    if n_decoys + k > n:
        raise ValueError(f"n_decoys ({n_decoys}) + k ({k}) must be <= n ({n})")
    G = nx.barabasi_albert_graph(n, m, seed=seed)
    deg = np.array([d for _, d in G.degree()], dtype=np.int64)
    sorted_nodes = np.argsort(-deg)
    decoys = sorted_nodes[:n_decoys].tolist()
    remaining = np.array(sorted(set(range(n)) - set(decoys)), dtype=np.int64)
    rng = np.random.default_rng(seed + 1)
    planted = rng.choice(remaining, size=k, replace=False)
    extra_rng = np.random.default_rng(seed + 2)
    for u in decoys:
        for _ in range(decoy_edges_per_node):
            v = int(extra_rng.integers(0, n))
            if v != u and not G.has_edge(u, v):
                G.add_edge(u, v)
    A = nx.to_scipy_sparse_array(G, nodelist=list(range(n)), format="csr", dtype=np.float64)
    w = rng.normal(0.0, weight_noise, n)
    w[planted] += weight_signal
    return Instance(n=n, k=k, weights=w, adj=sp.csr_matrix(A), planted=planted,
                    meta={"generator": "planted_decoy_hubs", "seed": seed,
                          "m_ba": m, "n_decoys": n_decoys,
                          "decoy_edges_per_node": decoy_edges_per_node,
                          "decoys": decoys,
                          "weight_signal": weight_signal,
                          "weight_noise": weight_noise})
