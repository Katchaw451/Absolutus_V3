"""Exact MILP solver using HiGHS via scipy.optimize.milp.

Hard-constraint formulation; see docs/FORMULATION.md.
"""
from __future__ import annotations
from typing import Optional
import numpy as np
import scipy.sparse as sp
from scipy.optimize import milp, LinearConstraint, Bounds

from ..core import Instance


def solve_exact_milp(inst: Instance, lambda_coh: float, lambda_red: float,
                     time_limit: Optional[float] = 60.0) -> dict:
    n, k = inst.n, inst.k
    d = inst.degrees()

    # Enumerate upper-triangle edges.
    A_up = sp.triu(inst.adj, k=1).tocoo()
    rows_e = A_up.row.astype(np.int64)
    cols_e = A_up.col.astype(np.int64)
    a_e = A_up.data.astype(np.float64)
    m = len(a_e)
    nv = n + m               # variables: x_0..x_{n-1}, y_0..y_{m-1}
    e_idx = np.arange(m, dtype=np.int64)

    # Objective (minimise -F).
    c = np.zeros(nv)
    c[:n] = -(inst.weights - lambda_red * d)
    c[n:] = -(lambda_coh * a_e)

    # Equality: Σ x_i = k.
    A_eq = sp.csr_matrix((np.ones(n),
                          (np.zeros(n, dtype=np.int64), np.arange(n))),
                         shape=(1, nv))
    b_eq = np.array([float(k)])

    # Block 1: y_e - x_i <= 0     (2 entries per edge)
    r1 = np.concatenate([e_idx, e_idx])
    c1 = np.concatenate([n + e_idx, rows_e])
    v1 = np.concatenate([np.ones(m), -np.ones(m)])

    # Block 2: y_e - x_j <= 0     (2 entries per edge)
    r2 = np.concatenate([m + e_idx, m + e_idx])
    c2 = np.concatenate([n + e_idx, cols_e])
    v2 = np.concatenate([np.ones(m), -np.ones(m)])

    # Block 3: x_i + x_j - y_e <= 1   (3 entries per edge)
    r3 = np.concatenate([2 * m + e_idx, 2 * m + e_idx, 2 * m + e_idx])
    c3 = np.concatenate([rows_e, cols_e, n + e_idx])
    v3 = np.concatenate([np.ones(m), np.ones(m), -np.ones(m)])

    r_idx = np.concatenate([r1, r2, r3])
    c_idx = np.concatenate([c1, c2, c3])
    v_all = np.concatenate([v1, v2, v3])
    nrows = 3 * m
    A_ub = sp.csr_matrix((v_all, (r_idx, c_idx)), shape=(nrows, nv))
    lb_ub = np.full(nrows, -np.inf)
    ub_ub = np.concatenate([np.zeros(m), np.zeros(m), np.ones(m)])

    constraints = [
        LinearConstraint(A_eq, lb=b_eq, ub=b_eq),
        LinearConstraint(A_ub, lb=lb_ub, ub=ub_ub),
    ]
    integrality = np.ones(nv, dtype=np.int64)
    bounds = Bounds(np.zeros(nv), np.ones(nv))
    options = {"disp": False}
    if time_limit is not None:
        options["time_limit"] = float(time_limit)

    res = milp(c=c, constraints=constraints, integrality=integrality,
               bounds=bounds, options=options)

    if res.x is None:
        return {"x": None, "objective": None,
                "status": str(res.message), "optimality": "infeasible",
                "milp_status": int(res.status)}

    x = np.round(res.x[:n]).astype(np.int8)
    obj = float(inst.objective(x, lambda_coh=lambda_coh, lambda_red=lambda_red))
    optimality = "optimal" if int(res.status) == 0 else "feasible_or_unknown"
    return {"x": x, "objective": obj, "status": str(res.message),
            "optimality": optimality, "milp_status": int(res.status),
            "milp_objective": float(-res.fun)}
