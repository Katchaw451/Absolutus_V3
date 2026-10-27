"""Solver interface and result type."""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy as np


@dataclass
class SolverResult:
    x: np.ndarray
    energy: float
    objective: float
    feasible: bool
    wall_time_s: float
    n_evals: int
    feasible_fraction: float = 1.0
    extra: dict = field(default_factory=dict)


class Solver(ABC):
    """A solver maps an Instance -> SolverResult.

    Adding a new solver: subclass Solver, set `name`, implement `solve`,
    register in src/modbench/solvers/__init__.py. ~20 lines.
    """
    name: str = "unnamed"

    @abstractmethod
    def solve(self, inst, *, lambda_coh: float, lambda_red: float,
              seed: int, time_limit: float = 10.0,
              n_evals: int | None = None) -> SolverResult:
        ...
