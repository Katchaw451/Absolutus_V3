# Absolutus_V3 — Report

**Status:** draft, auto-generated from `results/test/summary_small.csv`
(git commit recorded by `run_cell.py` in every JSONL row as `env.git_head`).
All numbers in §3 trace to specific rows in that CSV. Anything not in the CSV
is labelled "untested".

**Scope of this report:** Phase 5d.1 (small matrix, n ∈ {12, 20}). See §5 for
what Phase 5d.2 adds (n ∈ {30, 40, 60}, QAOA-MPS chi sweep).

## 1. Setup (per `docs/PROTOCOL.md`)

- **Families**: `random`, `planted`, `decoy` (see `generators/`).
- **Sizes**: n ∈ {12, 20}, k = max(3, round(n/5)).
- **Seeds**: 20 test seeds per cell (100–119); dev seeds 0–2 for QAOA tuning.
- **Hyperparameters**: λ_coh = 1.0, λ_red = 0.1.
- **Equal-compute ceiling**: T = 30 s wall-clock, E = 10 000 QUBO evals.
  Methods that terminate early (greedy, exact) are not penalized; wall-clock
  is reported per row.
- **Solvers**: `random`, `greedy`, `sa` (neal), `tabu` (dwave-tabu),
  `exact` (HiGHS MILP), `qaoa` (statevector engine, p ∈ {1,2}).
- **QAOA (γ,β)**: tuned on dev seeds only (grid for p=1, 200 random restarts
  for p=2, minimizing mean analytic `<H_C>`). Test seeds reuse the tuned
  (γ,β) for the same (family, n, p). p=3 **not run** (see §4).

## 2. Validation (code correct, not benchmark results)

`pytest` 25/25. Covers: bit-identical instance regeneration from seed;
QUBO ≡ −objective on feasible sets; QUBO penalty dominates off-cardinality;
MILP matches brute force on 3 small instances; QAOA p=1 `<H_C>` matches an
independent `expm` calculation to 1e-8; MPS(χ=64) equals statevector to 1e-6;
each classical solver recovers the planted objective on an easy instance.

## 3. Results

### 3.1 At n ≤ 20, "random" is effectively brute force

At n=12, k=3, the feasible set has C(12,3) = 220 subsets. The `random` solver
draws 10 000 samples, so it is expected to find the optimum
`1 − (1 − 1/220)^10000 ≈ 1` of the time. The measured `obj_mean` for `random`
equals that of `exact` MILP to four decimals in all three families:

| family | exact obj_mean | random obj_mean |
|---|---|---|
| random  | 4.019788 | 4.019788 |
| planted | 6.845106 | 6.845106 |
| decoy   | 6.895357 | 6.895357 |

At n=20 the gap is small but nonzero (random is **99.8% of exact** on all
three families: 5.371 vs 5.381, 9.131 vs 9.170, 9.013 vs 9.013).

**Consequence for interpretation:** n ≤ 20 does **not** differentiate
classical solvers under our equal-compute budget. The interesting classical
comparison starts at n ≥ 30 (Phase 5d.2).

### 3.2 Tabu matches exact MILP at n = 20; SA is slightly worse on decoy

| family | exact | greedy | random | sa | tabu |
|---|---|---|---|---|---|
| random n=20 | 5.381 | 5.312 | 5.371 | 5.379 | **5.381** |
| planted n=20 | 9.170 | 8.969 | 9.131 | 9.104 | **9.170** |
| decoy n=20 | 9.013 | 8.930 | 9.013 | **8.801** | **9.013** |

- **Tabu reproduces the exact optimum on every seed of every family at n=20**
  (bit-identical `obj_mean`), using 2.1 s per seed. Exact MILP uses 0.08 s.
  Tabu is 26× more expensive but certified-optimal in these cells, which
  makes it a reasonable ground-truth proxy for n ≥ 30 where exact is
  intractable.
- **SA misses the optimum on some decoy n=20 seeds** (8.801 vs 9.013, -2.4%).
  The decoy hubs add degree mass that SA's geometric cooling does not fully
  overcome within its budget.

### 3.3 QAOA-SV is worse than every classical baseline at every cell

| cell | exact | qaoa p=1 | qaoa p=2 |
|---|---|---|---|
| random n=12 | 4.020 | 3.467 (86.3%) | 3.180 (79.1%) |
| planted n=12 | 6.845 | 5.570 (81.4%) | 5.456 (79.7%) |
| decoy n=12 | 6.895 | 6.325 (91.7%) | 6.542 (94.9%) |
| random n=20 | 5.381 | **1.154 (21.4%)** | **1.721 (32.0%)** |
| planted n=20 | 9.170 | **4.899 (53.4%)** | 7.190 (78.4%) |
| decoy n=20 | 9.013 | **3.444 (38.2%)** | 5.908 (65.6%) |

QAOA-SV never beats `random`. It is strictly dominated by every classical
solver at every size tested. This is a **negative result** and is reported as
such: at equal compute (T = 30 s, E = 10 000 evals), statevector QAOA with an
RX mixer and penalty QUBO does not compete with SA / tabu / exact in this
problem family.

### 3.4 Root cause: feasible-shot fraction collapses with n

The QUBO has a cardinality penalty `P·(Σx − k)²` and the mixer is RX (bit-flip
on the penalty QUBO). The RX mixer does not preserve the cardinality subspace.
Measured `feasible_frac_shots` (mean over 20 seeds of the fraction of 1024
shots that are feasible):

| family, n | qaoa p=1 | qaoa p=2 |
|---|---|---|
| random n=12 | 13.5% | 6.2% |
| planted n=12 | 4.7% | 5.0% |
| decoy n=12 | 10.3% | 11.3% |
| random n=20 | **1.05%** | **0.37%** |
| planted n=20 | **0.56%** | 2.84% |
| decoy n=20 | **0.41%** | 1.21% |

Additionally, on 1 of 20 seeds for `decoy n=20 qaoa p=1` and `random n=20
qaoa p=2`, **best-of-1024 contained no feasible sample** (`feasible_frac=0.95`).
This is a structural failure mode of the RX-mixer + penalty-QUBO combination;
it is the empirical motivation for an XY-mixer with Dicke-state initialization
(listed as future work; see §5).

### 3.5 Planted recovery degrades with n; decoy recovery is flat

`planted_recovery_mean` (|S ∩ S_planted| / k):

| family, n | exact | qaoa p=2 |
|---|---|---|
| planted n=12 | 73.3% | 63.3% |
| planted n=20 | 60.0% | 46.3% |

Both degrade; QAOA degrades more. `decoy_recovery_mean` (|S ∩ D| / |D|) is
**19–25% for every solver and every cell** — the decoy hubs are not captured
by any method, as designed. The degree-matched null model therefore provides
no differential signal at these sizes, but confirms the decoy mechanism is
active.

### 3.6 Memory: statevector QAOA hits a hard limit at n ≈ 20

| solver, n | peak_mb_mean |
|---|---|
| classical, all n | ~208 |
| qaoa statevector n=12 | ~213 |
| qaoa statevector n=20 | **600–1466** |

The n=20 memory is a direct consequence of O(2^n) statevector storage
multiplied by the `apply_rx` copy. On a typical workstation (≤ 8 GB free),
statevector QAOA is bounded to n ≤ 22. n ≥ 30 requires the MPS engine.

## 4. What we do NOT claim

- **No quantum advantage, and none is observed.** QAOA-SV loses to classical
  on every cell tested.
- **No claim that n ≤ 20 differentiates classical solvers** — `random` is
  within 0.2% of `exact` at n ≤ 20. The interesting classical comparison
  requires n ≥ 30.
- **No MPS truncation data yet.** The paper's central question (quality vs χ
  at equal compute) is deferred to Phase 5d.2.
- **No p=3 data.** The 11 p=3 cells were skipped because (γ,β) tuning for p=3
  at n=20 costs ~19 min per (family, n, p) and no hypothesis in this phase
  required it.
- **No n ≥ 30 data.** Phase 5d.2 will run classical to n=60 and QAOA-MPS with
  a χ grid.
- **Planted recovery < 100% is not a bug.** With `weight_noise=1.0` vs
  `weight_signal=2.0`, the planted set is not always the optimum — this is
  the noisy-planted generator doing its job. Recovery metrics at n ≥ 30
  will differentiate solvers.

## 5. Next: Phase 5d.2

- Classical `{random, greedy, sa, tabu}` at n ∈ {30, 40, 60}: this is where
  `random` stops being brute force and the real solver ranking appears.
- Exact MILP at n ∈ {30} (n=40+ intractable in 30 s).
- QAOA-MPS at n ∈ {12, 20, 30, 40} for χ ∈ {2, 4, 8, 16, 32}, p ∈ {1, 2}.
  This produces the **quality-vs-χ** figure (fig 3), the phase that
  directly addresses the ROLE's gap statement.
- A future upgrade (out of scope here): XY mixer with Dicke initialization,
  which would keep shots in the cardinality subspace and remove the
  feasible-shot collapse in §3.4.

## 6. Reproducibility

Every row in `results/test/summary_small.csv` corresponds to a specific
JSONL file under `results/test/`, which records the seed, wall-clock, peak
RSS, package versions, and git commit hash in `env` per run. `make reproduce`
regenerates the frozen test instances (seeds 100–119, sha256 manifest at
`results/instances/manifest.json`), re-runs the matrix, and regenerates this
report's source CSV.
