"""Engine tests: QAOA p=1 vs independent dense calc; MPS vs statevector."""
import numpy as np
import pytest
from scipy.linalg import expm

from modbench.generators import planted_scale_free, random_sparse
from modbench.engines.statevector import StatevectorEngine
from modbench.engines.mps import MPSEngine
from modbench.engines._qaoa_common import qubo_z_coefficients


def _make_X_operator(q, n):
    """2^n x 2^n matrix for X on qubit q, LSB-first convention."""
    X = np.array([[0, 1], [1, 0]])
    I = np.eye(2)
    chain = [I] * n
    chain[n - 1 - q] = X
    M = chain[0]
    for o in chain[1:]:
        M = np.kron(M, o)
    return M


def test_qubo_z_coefficients():
    """H_C = const + Σ α_i Z_i + Σ β_ij Z_i Z_j reproduces QUBO energies."""
    inst = planted_scale_free(n=5, k=2, seed=1)
    qubo = inst.to_qubo(1.0, 0.1)
    const, alpha, beta_mat = qubo_z_coefficients(qubo)
    for s in range(2 ** 5):
        x = ((np.array(s) >> np.arange(5)) & 1).astype(np.int8)   # shape (5,)
        z = 1.0 - 2.0 * x.astype(float)                            # shape (5,)
        E_z = const + alpha @ z + z @ beta_mat @ z
        assert abs(E_z - qubo.energy(x)) < 1e-10, f"s={s}: {E_z} vs {qubo.energy(x)}"


@pytest.mark.parametrize("inst", [
    planted_scale_free(n=4, k=2, seed=1),
    random_sparse(n=5, k=2, seed=7),
])
def test_qaoa_p1_matches_independent_dense_calc(inst):
    """QAOA p=1 <H_C> from statevector == independent expm-based calculation."""
    n = inst.n
    N = 2 ** n
    qubo = inst.to_qubo(1.0, 0.1)
    gamma, beta = 0.3, 0.4

    all_x = ((np.arange(N)[:, None] >> np.arange(n)) & 1).astype(np.int8)
    energies = np.array([qubo.energy(all_x[s]) for s in range(N)])

    U_cost = np.diag(np.exp(-1j * gamma * energies))
    H_mix = sum(_make_X_operator(q, n) for q in range(n))
    U_mix = expm(-1j * beta * H_mix)

    psi0 = np.ones(N, dtype=complex) / np.sqrt(N)
    psi = U_mix @ (U_cost @ psi0)
    E_expected = float(np.real(psi.conj() @ (energies * psi)))

    r = StatevectorEngine().run_qaoa(
        inst, lambda_coh=1.0, lambda_red=0.1, p=1,
        gammas=np.array([gamma]), betas=np.array([beta]),
        n_shots=100, seed=0)
    assert abs(r.energy_analytic - E_expected) < 1e-8


def test_mps_equals_statevector_no_truncation():
    """With chi large enough (no truncation), MPS matches statevector."""
    inst = planted_scale_free(n=6, k=3, seed=1)
    gammas = np.array([0.3])
    betas = np.array([0.5])

    r_sv = StatevectorEngine().run_qaoa(
        inst, lambda_coh=1.0, lambda_red=0.1, p=1,
        gammas=gammas, betas=betas, n_shots=5000, seed=42)
    r_mps = MPSEngine(chi=64).run_qaoa(
        inst, lambda_coh=1.0, lambda_red=0.1, p=1,
        gammas=gammas, betas=betas, n_shots=5000, seed=42)

    if np.isfinite(r_mps.energy_analytic) and np.isfinite(r_sv.energy_analytic):
        assert abs(r_mps.energy_analytic - r_sv.energy_analytic) < 1e-6, \
            f"<H_C>: sv={r_sv.energy_analytic}, mps={r_mps.energy_analytic}"

    assert abs(r_mps.energy_best - r_sv.energy_best) < 1e-6
    assert abs(r_mps.feasible_fraction - r_sv.feasible_fraction) < 0.05
