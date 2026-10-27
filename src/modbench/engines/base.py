"""Engine interface for QAOA backends."""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy as np


@dataclass
class EngineResult:
    x_best: np.ndarray
    energy_best: float
    energy_mean: float
    energy_analytic: float         # <H_C> from wavefunction (exact, no sampling noise)
    objective_best: float
    feasible_fraction: float
    discarded_weight: float        # MPS truncation; 0 for statevector
    wall_time_s: float
    n_evals: int
    metadata: dict = field(default_factory=dict)


class Engine(ABC):
    """A QAOA simulation backend.

    Adding a new engine: subclass Engine, set `name`, implement `run_qaoa`,
    register in src/modbench/engines/__init__.py. ~20 lines.
    """
    name: str = "unnamed"

    @abstractmethod
    def run_qaoa(self, inst, *, lambda_coh: float, lambda_red: float,
                 p: int, gammas: np.ndarray, betas: np.ndarray,
                 n_shots: int = 1024, seed: int = 0,
                 time_limit: float = 60.0) -> EngineResult:
        ...
