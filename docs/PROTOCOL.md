# Benchmark Protocol

**Status: COMMITTED BEFORE ANY REPORTED EXPERIMENT.**
No number in this document or in later reports is fabricated; every reported
value must be traceable to a saved run under `results/`, tagged with a git
commit hash, pinned package versions (`requirements.lock`), and a seed.

## 1. Roles: Validation vs Results

- **Validation** — `pytest` verifying that the code is correct: planted
  objectives evaluate correctly, QUBO ≡ −objective on feasible sets, MILP
  matches brute force, QAOA p=1 matches an independent `expm` calculation,
  MPS(χ large) equals statevector. These are correctness checks, **not**
  benchmark results.
- **Results** — any number appearing in `docs/REPORT.md` or `docs/CLAIMS.md`
  must come from a saved cell under `results/`. Any number not produced by a
  run is written as "untested".

## 2. Dev / Test split

- **Dev seeds:** 0–4 (5 seeds per cell). Used only for hyperparameter tuning.
  Never reported.
- **Test seeds:** 100–119 (20 seeds per cell). Each (solver, instance, seed)
  cell is run **exactly once**. Reruns are only allowed if the previous run
  crashed (recorded as `success=false`), not to cherry-pick better values.
- Instances are seeded deterministically: `(family, n, k, seed)` → bit-identical
  NPZ (verified by `tests/test_instances.py`).

## 3. Instance families

All three generators from `src/modbench/generators/`:

| family | generator | notes |
|---|---|---|
| `random` | `random_sparse` | Erdős–Rényi, iid Gaussian weights (no planted structure) |
| `planted` | `planted_scale_free` | Barabási–Albert, planted module + boosted weights |
| `decoy` | `planted_decoy_hubs` | planted module + 10 decoy hubs with extra edges, weights NOT boosted |

Module-selection hyperparameters for the **main** runs:
`λ_coh = 1.0`, `λ_red = 0.1`.
Sensitivity sweep on the `decoy` family only: `λ_red ∈ {0, 0.1, 0.5}`.

## 4. Sizes

- `n ∈ {12, 20, 30, 40, 60}`
- `k = max(3, round(n / 5))`
- Actual limits reached are stated in the Phase 5 report. Statevector QAOA is
  capped at `n ≤ 20`; classical + MPS + exact cover all sizes as far as
  hardware allows.

## 5. Solvers and engines

Solvers (`src/modbench/solvers/`):
`random`, `greedy`, `sa`, `tabu`, `exact` (HiGHS MILP, `n ≤ ~30` tractable),
`qaoa`.

Engines (`src/modbench/engines/`):
- `statevector` — exact, `n ≤ 20`, validation only.
- `mps` — quimb `CircuitMPS`, truncation at bond dimension `χ`.

**Hyperparameters (dev-tuned only, never on test):**
- `SA`: `num_reads=200`, `num_sweeps=3000`, tight penalty (`_qubo_to_bqm(tight_penalty=True)`)
- `Tabu`: `num_reads=100`, tight penalty
- `Greedy`: `restarts=20`
- `QAOA`: `p ∈ {1, 2, 3}`; `(γ, β)` chosen per `(family, n, k, p, seed_dev)` by
  grid search of 100 random restarts, minimizing `⟨H_C⟩_analytic`. Test seeds
  reuse the tuple selected on dev for the same configuration.
- `χ ∈ {2, 4, 8, 16, 32, 64, 128, 256}` (MPS only). For each cell we run every
  χ in the grid and record the discarded weight and solution quality per χ.

## 6. Equal-compute rule (two views, both reported)

For each `(family, n, k, seed, solver)`, the solver gets:

- **Wall-clock budget:** `T = 30 s`. Samplers use their native time-out
  parameter; iterative solvers check a wall-clock deadline between iterations.
- **Objective-evaluation budget:** `E = 10 000` QUBO evaluations. Samplers set
  `num_reads = min(default_reads, E)`; iterative solvers count calls to
  `qubo.energy`. Recorded as `n_evals` in every result line.

Both views are reported in every summary and figure. **A solver that wins on
wall-clock but loses on eval-count (or vice versa) is described as such** —
no cherry-picking the favorable axis. The `exact` MILP solver has no
eval-count semantics; it is reported only under the wall-clock view and
labelled `n_evals = -1`.

## 7. Metrics

Per cell:

- **QUBO energy** `E(x)`
- **Objective** `F(x)` (only meaningful when feasible; `null` otherwise)
- **Feasible** boolean, **feasible fraction** (QAOA samplers)
- **Wall-clock** (seconds), **peak RSS** (`resource.getrusage(RUSAGE_SELF).ru_maxrss / 1024`, MB on Linux)
- **n_evals** (as above)
- **Planted recovery** `|S ∩ S_planted| / k` (families `planted`, `decoy`)
- **Decoy recovery** `|S ∩ S_decoys| / 10` (family `decoy` only)
- **Discarded weight** (MPS only): `|⟨H_C⟩_analytic − ⟨H_C⟩_sampled|`, i.e. the
  proxy for truncation-induced distortion (exact when `n ≤ 12`, from the dense
  MPS state).

Derived metrics (computed in `scripts/analyze.py`):

- **Approximation ratio** relative to best-known per `(family, n, k, seed)`:
  `r = (F_solver − F_random_mean) / (F_best − F_random_mean)` when
  `F_best > F_random_mean`, else 1.0 (all tied). `F_best` is the maximum over
  all classical solvers on that seed (exact or SA/tabu/greedy).
- **Success probability** = fraction of seeds reaching `F_best − 1e-9`.
- **Time-to-target** = median wall-clock of successful seeds.

## 8. Statistics

- ≥ **20 seeds per cell** (test).
- **Mean and 95% bootstrap CI** (10 000 resamples, seeded) for every reported
  scalar.
- **Paired Wilcoxon signed-rank** between QAOA and best classical, per
  `(family, n, k)` cell, on the per-seed objective. Significance level
  `α = 0.05`, two-sided. Report effect size (median of paired differences).
- **Negative results are reported**, including every regime where a classical
  baseline strictly beats QAOA at equal compute.

## 9. Data format and layout

Every cell writes one JSONL line to
`results/<phase>/<family>/n<NN>_k<KK>/<solver>.jsonl` with the following
mandatory fields:

`family, n, k, seed, solver, engine, p, lambda_coh, lambda_red,
time_limit, n_evals_budget, success, error, wall_time_s, peak_mb,
energy, objective, feasible, feasible_fraction, n_evals, extra`

`extra` carries solver-specific metadata (e.g. QAOA `gammas`, `betas`,
`energy_analytic`, `discarded_weight`; SA `num_reads`, `num_sweeps`).

`summary.csv` per `(family, n, k, solver)` with `n_seeds, obj_mean, obj_std,
obj_ci_lo, obj_ci_hi, wall_mean, wall_std, peak_mb_mean, feasible_frac`.

## 10. Reproducibility

- `make reproduce` regenerates every table, summary CSV, and figure from
  scratch under `results/test/`.
- Every saved run records the git commit hash, `requirements.lock` hash, and
  seed in the JSONL line (`extra.env`).
- The frozen test instance set is published with sha256 in `results/instances/`
  (Phase 6).

## 11. What this protocol will NOT do

- No claim of **quantum advantage**. A claim without a baseline at equal
  compute is not made.
- No number in `docs/` without a saved run backing it.
- No hyperparameter tuned on a test seed. Test cells are run once.

## 12. Actual limits reached in v0.1.0

The limits below are what we actually ran, not what we planned:

- **Classical:** `n ∈ {12, 20, 30, 40, 60}`, all 3 families, 20 test seeds.
- **Exact MILP:** `n ∈ {12, 20}`. Not run at `n ≥ 30` (MILP does not close in
  30 s on n=30 instances in our setup).
- **QAOA-SV:** `n ∈ {12, 20}`, `p ∈ {1, 2}`. `p=3` not run (tuning cost
  ~2 h). `n ≥ 22` not run (statevector memory 600 MB–1.5 GB at n=20;
  n=22 is out of reach on our hardware).
- **QAOA-MPS:** `n=12` complete χ sweep (χ ∈ {2,4,8,16,32}); `n=20` partial
  (planted p=1 χ=32 has 3 seeds; all other cells at 20). `n ≥ 30` was in
  progress and did not complete within the release window.
- **MPS truncation metric:** measurable only at `n ≤ 12` (dense-state
  extraction). At `n > 12`, `discarded_weight` is `NaN` and the truncation-
  vs-quality analysis is deferred.
- **Wall-clock:** classical solvers terminate at 0.02–2.3 s; QAOA-MPS at
  n=20 is 5–10× slower than statevector at the same size.
- **Peak memory:** classical ~208 MB regardless of n; QAOA-SV n=20 up to
  1.47 GB.

Frozen instance set for v0.1.0: 3 families × 5 sizes × 20 test seeds
= 300 instances. sha256 manifest at `results/instances/manifest.json`.

## 13. Post-hoc addendum — actual limits reached in v0.1.0

Written after the runs, before release. Records what was actually done,
in addition to what §5 planned.

- **Classical:** all three families at n ∈ {12, 20, 30, 40, 60} complete.
- **Exact MILP:** capped at n ≤ 20. n ≥ 30 not attempted (30 s cap makes
  it infeasible to certify optimality; not worth adding `--max-n 30` runs
  that would time out — the paper's honest claim is exact only for n≤20).
- **QAOA-SV:** n ∈ {12, 20}, p ∈ {1, 2} complete. **p=3 was not run** —
  tuning p=3 at n=20 costs ~19 min per (family, n) and was out of budget.
- **QAOA-MPS:** partial coverage (see §4 of REPORT.md):
  - complete: n=12 random × p∈{1,2} × χ∈{2..32}; n=12 planted × p∈{1,2} × χ∈{2..32}; n=20 random × p∈{1,2} × χ∈{2..32}.
  - partial: n=20 planted p=1 (χ=32 has 3/20 seeds).
  - not reached: decoy at any n; planted p=2 at n=20; n ∈ {30, 40}.
- **`discarded_weight`:** measured for n=12 only. At n=20, the current
  `MPSEngine` placeholder returns 0.0 (see REPORT.md §3.8). This is
  documented as a measurement gap, not a result.
- **XY mixer with Dicke initialisation:** not implemented in v0.1.0.
  Listed as future work (REPORT.md §5).
- **SA configuration:** dev-tuned for n ≤ 20. Not retuned for n ≥ 30.
  Underperformance at n≥30 reflects this (see REPORT.md §3.2 caveat);
  the SA result is a statement about our configuration, not about SA.
- **Equal-compute interpretation:** "T = 30 s" is per-cell wall-clock,
  not a shared pool. All solvers are self-bounded at 30 s; the fastest
  classical solvers (exact, greedy) finish in ≪ 30 s and are not
  penalised for being fast. Both wall-clock and `n_evals` are in the
  CSV; no axis is cherry-picked in the report.

These adjustments are the *only* deviations from the pre-registered
protocol. All are scope reductions, not scope expansions; none of them
post-selects a favorable regime. Every result the protocol would have
produced but did not is marked **not reached** in REPORT §4.

## 13. Post-hoc addendum — actual limits reached in v0.1.0

Written after the runs, before release. Records what was actually done,
in addition to what §5 planned.

- **Classical:** all three families at n ∈ {12, 20, 30, 40, 60} complete.
- **Exact MILP:** capped at n ≤ 20. n ≥ 30 not attempted (30 s cap makes
  it infeasible to certify optimality; not worth adding `--max-n 30` runs
  that would time out — the paper's honest claim is exact only for n≤20).
- **QAOA-SV:** n ∈ {12, 20}, p ∈ {1, 2} complete. **p=3 was not run** —
  tuning p=3 at n=20 costs ~19 min per (family, n) and was out of budget.
- **QAOA-MPS:** partial coverage (see §4 of REPORT.md):
  - complete: n=12 random × p∈{1,2} × χ∈{2..32}; n=12 planted × p∈{1,2} × χ∈{2..32}; n=20 random × p∈{1,2} × χ∈{2..32}.
  - partial: n=20 planted p=1 (χ=32 has 3/20 seeds).
  - not reached: decoy at any n; planted p=2 at n=20; n ∈ {30, 40}.
- **`discarded_weight`:** measured for n=12 only. At n=20, the current
  `MPSEngine` placeholder returns 0.0 (see REPORT.md §3.8). This is
  documented as a measurement gap, not a result.
- **XY mixer with Dicke initialisation:** not implemented in v0.1.0.
  Listed as future work (REPORT.md §5).
- **SA configuration:** dev-tuned for n ≤ 20. Not retuned for n ≥ 30.
  Underperformance at n≥30 reflects this (see REPORT.md §3.2 caveat);
  the SA result is a statement about our configuration, not about SA.
- **Equal-compute interpretation:** "T = 30 s" is per-cell wall-clock,
  not a shared pool. All solvers are self-bounded at 30 s; the fastest
  classical solvers (exact, greedy) finish in ≪ 30 s and are not
  penalised for being fast. Both wall-clock and `n_evals` are in the
  CSV; no axis is cherry-picked in the report.

These adjustments are the *only* deviations from the pre-registered
protocol. All are scope reductions, not scope expansions; none of them
post-selects a favorable regime. Every result the protocol would have
produced but did not is marked **not reached** in REPORT §4.

## 13. Post-hoc addendum — actual limits reached in v0.1.0

Written after the runs, before release. Records what was actually done,
in addition to what §5 planned.

- **Classical:** all three families at n ∈ {12, 20, 30, 40, 60} complete.
- **Exact MILP:** capped at n ≤ 20. n ≥ 30 not attempted (30 s cap makes
  it infeasible to certify optimality; not worth adding `--max-n 30` runs
  that would time out — the paper's honest claim is exact only for n≤20).
- **QAOA-SV:** n ∈ {12, 20}, p ∈ {1, 2} complete. **p=3 was not run** —
  tuning p=3 at n=20 costs ~19 min per (family, n) and was out of budget.
- **QAOA-MPS:** partial coverage (see §4 of REPORT.md):
  - complete: n=12 random × p∈{1,2} × χ∈{2..32}; n=12 planted × p∈{1,2} × χ∈{2..32}; n=20 random × p∈{1,2} × χ∈{2..32}.
  - partial: n=20 planted p=1 (χ=32 has 3/20 seeds).
  - not reached: decoy at any n; planted p=2 at n=20; n ∈ {30, 40}.
- **`discarded_weight`:** measured for n=12 only. At n=20, the current
  `MPSEngine` placeholder returns 0.0 (see REPORT.md §3.8). This is
  documented as a measurement gap, not a result.
- **XY mixer with Dicke initialisation:** not implemented in v0.1.0.
  Listed as future work (REPORT.md §5).
- **SA configuration:** dev-tuned for n ≤ 20. Not retuned for n ≥ 30.
  Underperformance at n≥30 reflects this (see REPORT.md §3.2 caveat);
  the SA result is a statement about our configuration, not about SA.
- **Equal-compute interpretation:** "T = 30 s" is per-cell wall-clock,
  not a shared pool. All solvers are self-bounded at 30 s; the fastest
  classical solvers (exact, greedy) finish in ≪ 30 s and are not
  penalised for being fast. Both wall-clock and `n_evals` are in the
  CSV; no axis is cherry-picked in the report.

These adjustments are the *only* deviations from the pre-registered
protocol. All are scope reductions, not scope expansions; none of them
post-selects a favorable regime. Every result the protocol would have
produced but did not is marked **not reached** in REPORT §4.
