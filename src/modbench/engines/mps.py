"""MPS QAOA with bond-dimension truncation (chi sweep).

Note on bit order: quimb represents qubit q as axis q of a rank-n tensor.
With C-order flattening, qubit q is bit (n-1-q) of the flat state index.
Our QUBO helpers (all_energies, statevector engine) use the LSB-first
convention (qubit q = bit q). We therefore reverse axes when extracting
the dense state, so that probs_rev[k] pairs with E_all[k].
"""
from __future__ import annotations
import time
import numpy as np
import quimb.tensor as qtn
from .base import Engine, EngineResult
from ._qaoa_common import qubo_z_coefficients, all_energies


def _RZ(theta: float) -> np.ndarray:
    return np.array([[np.exp(-0.5j * theta), 0.0],
                     [0.0, np.exp(0.5j * theta)]], dtype=np.complex128)


def _RX(theta: float) -> np.ndarray:
    c = np.cos(theta / 2)
    s = -1j * np.sin(theta / 2)
    return np.array([[c, s], [s, c]], dtype=np.complex128)


def _dense_lsb_first(psi_mps, n: int) -> np.ndarray:
    """Extract dense statevector, converting quimb qubit-0=axis-0 to LSB-first."""
    dense = np.asarray(psi_mps.to_dense()).reshape(-1)
    # quimb axes: qubit q at axis q. Reshape to [2]*n and reverse axes.
    arr = dense.reshape([2] * n)
    arr = arr.transpose(tuple(range(n - 1, -1, -1)))
    return np.ascontiguousarray(arr).reshape(-1)


class MPSEngine(Engine):
    name = "mps"
    def __init__(self, chi: int = 32, cutoff: float = 1e-12, dense_threshold: int = 12):
        self.chi = int(chi)
        self.cutoff = float(cutoff)
        self.dense_threshold = int(dense_threshold)

    def run_qaoa(self, inst, *, lambda_coh, lambda_red, p, gammas, betas,
                 n_shots=1024, seed=0, time_limit=60.0):
        n = inst.n
        qubo = inst.to_qubo(lambda_coh, lambda_red)
        _, alpha, beta_mat = qubo_z_coefficients(qubo)
        t0 = time.perf_counter()

        circ = qtn.CircuitMPS(n, max_bond=self.chi, cutoff=self.cutoff)
        for q in range(n):
            circ.apply_gate("H", q)

        for l in range(p):
            gamma = float(gammas[l])
            beta = float(betas[l])
            for i in range(n):
                if alpha[i] != 0.0:
                    circ.apply_gate(_RZ(2.0 * gamma * float(alpha[i])), i)
            for i in range(n):
                for j in range(i + 1, n):
                    bij = float(beta_mat[i, j])
                    if bij != 0.0:
                        circ.apply_gate("CX", i, j)
                        circ.apply_gate(_RZ(2.0 * gamma * bij), j)
                        circ.apply_gate("CX", i, j)
            for i in range(n):
                circ.apply_gate(_RX(2.0 * beta), i)

        # ---- Sampling + analytic energy ----
        N = 2 ** n
        E_all = all_energies(qubo)
        rng = np.random.default_rng(seed)
        energy_analytic = float("nan")
        dense_rev = None

        if n <= self.dense_threshold:
            # Exact path: extract dense, compute analytic, sample from probs
            dense_rev = _dense_lsb_first(circ.psi, n)
            probs = np.abs(dense_rev) ** 2
            probs /= probs.sum()
            energy_analytic = float(np.sum(probs * E_all))

            idx = rng.choice(N, size=n_shots, p=probs)
            states = np.arange(N, dtype=np.int64)[idx]
            bits = ((states[:, None] >> np.arange(n)) & 1).astype(np.int8)
        else:
            # Large-n path: use quimb sampling.  quimb returns bitstrings
            # with qubit 0 as the leftmost character; our convention wants
            # bits[q] = x_q = qubit q, which is exactly s[q].
            try:
                samples = circ.sample(n_shots, seed=seed)
            except TypeError:
                samples = circ.sample(n_shots)
            if isinstance(samples, dict):
                bitstrings = []
                for bs, cnt in samples.items():
                    bitstrings.extend([bs] * int(cnt))
            else:
                bitstrings = list(samples)
            bits = np.zeros((len(bitstrings), n), dtype=np.int8)
            for idx, bs in enumerate(bitstrings):
                s = str(bs).replace(" ", "")
                if len(s) != n:
                    s = s.zfill(n)
                bits[idx] = np.array([int(c) for c in s], dtype=np.int8)

        energies = np.array([qubo.energy(x) for x in bits]) if len(bits) else np.array([])
        feasible = bits.sum(axis=1) == inst.k if len(bits) else np.array([], dtype=bool)
        obj = np.full(len(bits), np.nan)
        for i in np.where(feasible)[0]:
            obj[i] = inst.objective(bits[i], lambda_coh, lambda_red)

        best = int(np.argmin(energies)) if len(energies) else -1
        # Absent a dense-state extraction (n > dense_threshold), truncation is
        # not measurable in this release; report NaN (not 0.0) so downstream
        # analysis cannot mistake "unmeasured" for "zero truncation".
        discarded = (abs(energy_analytic - float(energies.mean()))
                     if np.isfinite(energy_analytic) and len(energies) else float("nan"))

        return EngineResult(
            x_best=bits[best].copy() if best >= 0 else np.zeros(n, dtype=np.int8),
            energy_best=float(energies[best]) if best >= 0 else float("nan"),
            energy_mean=float(energies.mean()) if len(energies) else float("nan"),
            energy_analytic=energy_analytic,
            objective_best=float(obj[best]) if best >= 0 and feasible[best] else float("nan"),
            feasible_fraction=float(feasible.mean()) if len(bits) else 0.0,
            discarded_weight=float(discarded),
            wall_time_s=time.perf_counter() - t0,
            n_evals=int(n_shots),
            metadata={"dense_state": dense_rev,
                      "p": p, "chi": self.chi, "cutoff": self.cutoff,
                      "gammas": list(map(float, gammas)),
                      "betas": list(map(float, betas)),
                      "lambda_coh": lambda_coh, "lambda_red": lambda_red},
        )
