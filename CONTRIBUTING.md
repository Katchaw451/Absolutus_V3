# Contributing

Two rules above all: **no fabricated numbers**, and **no claim without a
baseline at equal compute**.

## Adding a solver (~20 lines)

1. `src/modbench/solvers/my_solver.py`:

       from .base import Solver, SolverResult
       class MySolver(Solver):
           name = "my_solver"
           def solve(self, inst, *, lambda_coh, lambda_red, seed,
                     time_limit=10.0, n_evals=None) -> SolverResult: ...

2. Register in `src/modbench/solvers/__init__.py`:
   `SOLVERS = { ..., "my_solver": MySolver }`

3. Add a test in `tests/test_solvers.py`: feasible on an easy planted
   instance, recovers the planted objective.

`scripts/run_cell.py`, `scripts/analyze.py`, and the CLI pick it up
automatically.

## Adding an engine (~20 lines)

1. `src/modbench/engines/my_engine.py`:

       from .base import Engine, EngineResult
       class MyEngine(Engine):
           name = "my_engine"
           def run_qaoa(self, inst, *, lambda_coh, lambda_red, p, gammas,
                        betas, n_shots=1024, seed=0, time_limit=60.0): ...

2. Register in `src/modbench/engines/__init__.py`:
   `ENGINES = { ..., "my_engine": MyEngine }`

3. For exact engines, follow the MPS test: with no truncation
   (chi large) must match `StatevectorEngine` to 1e-6.

## Benchmark rules

- **Seeds:** test seeds 100-119; dev seeds 0-4. Tune on dev only.
- **Equal compute:** T=30 s and E=10,000 QUBO evals per cell.
- **Statistics:** >=20 test seeds per cell; mean + 95% bootstrap CI;
  paired Wilcoxon for QAOA vs best classical.
- **Negative results** must be reported, including regimes where the new
  solver loses.
- **`env.git_head` recorded in every JSONL row.** Do not rewrite history
  on a results commit without flagging affected rows.

## Tests

    pytest -q
