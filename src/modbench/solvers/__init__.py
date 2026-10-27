from .base import Solver, SolverResult
from .classical import RandomSolver, GreedySolver, SASolver, TabuSolver
from .exact_solver import ExactSolver
from .qaoa import QAOASolver

SOLVERS = {
    "random": RandomSolver,
    "greedy": GreedySolver,
    "sa": SASolver,
    "tabu": TabuSolver,
    "exact": ExactSolver,
    "qaoa": QAOASolver,
}
