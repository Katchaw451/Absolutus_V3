"""Core data structures: Instance and QUBO."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import scipy.sparse as sp


@dataclass
class Instance:
    """Cardinality-constrained module-selection instance.

    Direct objective (MAXIMISED), for selection mask x with |x| = k:

        F(x) = Σ_i w_i x_i
             + λ_coh · Σ_{i<j} A_ij x_i x_j
             - λ_red · Σ_i d_i x_i
    """
    n: int
    k: int
    weights: np.ndarray
    adj: sp.csr_matrix
    planted: Optional[np.ndarray] = None
    meta: dict = field(default_factory=dict)

    def degrees(self) -> np.ndarray:
        return np.asarray(self.adj.sum(axis=1)).ravel()

    def objective(self, x, lambda_coh: float, lambda_red: float) -> float:
        x = np.asarray(x, dtype=np.int8)
        if int(x.sum()) != self.k:
            raise ValueError(f"objective defined only for |x|=k (got {int(x.sum())})")
        evidence = float(self.weights @ x)
        cohesion = 0.5 * float(x @ (self.adj @ x))
        redundancy = float(self.degrees() @ x)
        return evidence + lambda_coh * cohesion - lambda_red * redundancy

    def to_qubo(self, lambda_coh: float, lambda_red: float,
                penalty: Optional[float] = None) -> "QUBO":
        """Exact-cardinality-penalty QUBO.

        E(x) = -F(x) + P · (Σ_i x_i − k)²

        We keep the objective (h, J) in sparse form and store P separately as
        `penalty_coeff`, so memory is O(nnz(A)) and energy evaluation is exact
        for *any* binary x (feasible or not).
        """
        n, k = self.n, self.k
        d = self.degrees()
        if penalty is None:
            w_max = float(np.max(np.abs(self.weights)) + 1e-12)
            e_sum = float(self.adj.sum()) + 1e-12
            d_max = float(np.max(d)) + 1e-12
            penalty = 2.0 * (k * w_max + lambda_coh * e_sum + k * lambda_red * d_max + 1.0)

        # Objective only: h_i = -(w_i - λ_red d_i)
        h = -(self.weights - lambda_red * d)

        # Objective only: J_ij = -λ_coh A_ij for i<j (upper triangle)
        A_up = sp.triu(self.adj, k=1).tocoo()
        J = sp.csr_matrix((-lambda_coh * A_up.data, (A_up.row, A_up.col)), shape=(n, n))

        return QUBO(n=n, k=k, h=h, J=J, offset=0.0,
                    penalty_coeff=float(penalty),
                    meta={"lambda_coh": float(lambda_coh),
                          "lambda_red": float(lambda_red),
                          "penalty": float(penalty),
                          "instance_meta": self.meta})


@dataclass
class QUBO:
    """QUBO with exact cardinality penalty tracked separately.

    Energy of a binary vector x:

        E(x) = offset + Σ_i h_i x_i
                     + Σ_{i<j} J_ij x_i x_j
                     + penalty_coeff · (Σ_i x_i - k)²

    `h` and `J` encode the objective only (sparse). `penalty_coeff` is applied
    via the direct squared-cardinality term (equivalent to the all-pairs
    expansion but O(n) per eval instead of O(n²) memory).
    """
    n: int
    k: int
    h: np.ndarray
    J: sp.csr_matrix
    offset: float
    penalty_coeff: float = 0.0
    meta: dict = field(default_factory=dict)

    def energy(self, x) -> float:
        x = np.asarray(x, dtype=np.int8)
        lin = float(self.h @ x)
        if self.J.nnz > 0:
            rows, cols = self.J.nonzero()
            quad = float(self.J.data @ (x[rows] * x[cols]))
        else:
            quad = 0.0
        card_pen = self.penalty_coeff * (float(x.sum()) - self.k) ** 2
        return self.offset + lin + quad + card_pen
