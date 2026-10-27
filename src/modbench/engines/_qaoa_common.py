"""Shared helpers: QUBO -> Z-basis decomposition, dense energies, RX application."""
from __future__ import annotations
import numpy as np


def qubo_z_coefficients(qubo):
    """Return (const, alpha, beta_upper) s.t.
         H_C(x) = const + Σ_i alpha_i Z_i + Σ_{i<j} beta_upper[i,j] Z_i Z_j
       for H_C(x) = qubo.energy(x) with x_i = (1 - Z_i)/2.
    """
    n, k = qubo.n, qubo.k
    P = qubo.penalty_coeff
    a = qubo.h + P * (1 - 2 * k)
    Jd = qubo.J.toarray()
    Jsym = Jd + Jd.T
    row_sum = Jsym.sum(axis=1)
    sum_b_i = row_sum + 2 * P * (n - 1)
    alpha = -a / 2 - sum_b_i / 4
    beta_upper = (Jd + 2 * P * np.triu(np.ones((n, n)), k=1)) / 4
    const = qubo.offset + P * k * k - alpha.sum() - beta_upper.sum()
    return float(const), alpha, beta_upper


def all_energies(qubo):
    """Energy of every computational basis state, shape (2^n,).

    Convention: qubit q = bit q of the state index (LSB-first).
    """
    n = qubo.n
    if n > 22:
        raise ValueError(f"dense energies require n<=22 (got {n})")
    N = 2 ** n
    states = np.arange(N, dtype=np.int64)
    bits = ((states[:, None] >> np.arange(n)) & 1).astype(np.float64)
    E = qubo.offset + bits @ qubo.h
    if qubo.J.nnz > 0:
        rows, cols = qubo.J.nonzero()
        E += (bits[:, rows] * bits[:, cols]) @ qubo.J.data
    E += qubo.penalty_coeff * (bits.sum(axis=1) - qubo.k) ** 2
    return E


def apply_rx(psi, q, theta, n):
    """Apply RX(theta) on qubit q (LSB-first: bit q of state index)."""
    arr = psi.reshape([2] * n)
    arr = np.moveaxis(arr, n - 1 - q, 0)
    c = np.cos(theta / 2)
    s = -1j * np.sin(theta / 2)
    a0 = arr[0].copy(); a1 = arr[1].copy()
    arr[0] = c * a0 + s * a1
    arr[1] = s * a0 + c * a1
    arr = np.moveaxis(arr, 0, n - 1 - q)
    return arr.reshape(-1)
