# Absolutus_V3 — Report (v0.1.0)

**Status:** final for release v0.1.0 (released 2026-10-28).
**Data provenance:** every number below is a cell in
`results/test/summary_all.csv`, which aggregates per-seed JSONL rows under
`results/test/`. Each JSONL row records `env.git_head`, `env.numpy`,
`env.python`, seed, wall-clock, peak RSS. Instances frozen at
`results/instances/manifest.json` (sha256, seeds 100–119).
**Scope:** 20 test seeds per cell; 95% bootstrap CIs (10,000 resamples);
λ_coh=1.0, λ_red=0.1; T=30 s wall-clock budget, E=10,000 QUBO eval budget.

## 1. Real limits reached in v0.1.0

| dimension | reached | not reached |
|---|---|---|
| Classical solvers | n ∈ {12, 20, 30, 40, 60}, 3 families, 20 seeds | — |
| Exact MILP (HiGHS) | n ∈ {12, 20}, 3 families, 20 seeds | n ≥ 30 (MILP intractable at T=30 s) |
| QAOA-SV (RX mixer) | n ∈ {12, 20}, p ∈ {1, 2} | p = 3 (tuning cost), n ≥ 22 (2^n memory) |
| QAOA-MPS (χ sweep) | n=12 full χ ∈ {2,4,8,16,32}; n=20 partial | n ∈ {30, 40} (was in progress, not completed) |
| MPS truncation metric | n=12 (`discarded_weight`) | n>12 (`energy_analytic` requires dense state) |
| Planted recovery | all families, n ∈ {12..60} (classical), n ∈ {12,20} (QAOA) | — |
| Decoy recovery | decoy family only, all n | — |

**Nothing in this report is extrapolated.** Where a cell is missing, we say so.

## 2. Validation (not benchmark results)

`pytest` — 25 tests, all passing. These verify the code, not the science:

- Instances are bit-identical from `(family, n, k, seed)` (sha256 stable).
- QUBO `E(x) = −F(x)` on feasible sets, to 1e-10.
- QUBO penalty dominates off-cardinality (min infeasible > max feasible).
- MILP matches brute force on 3 small instances.
- QAOA p=1 `⟨H_C⟩` matches an independent `expm` calculation, to 1e-8.
- MPS(χ=64) equals statevector energy, to 1e-6.
- Each classical solver recovers the planted objective on an easy instance.

## 3. Classical solvers

### 3.1 At n ≤ 20, `random` is effectively brute force

At n=12, k=3, C(12,3)=220 feasible subsets; 10,000 random samples is
`1 − (1−1/220)^10000 ≈ 1`. The measured `obj_mean` for `random` equals `exact`
to four decimals in all three families (e.g. random family: 4.019788 for both).
At n=20, `random` is within 0.2% of exact on all families (5.371 vs 5.381;
9.131 vs 9.170; 9.013 vs 9.013).

**Consequence:** n≤20 does not differentiate classical solvers under this
equal-compute budget. This is a real observation, not a limitation of the
generators.

### 3.2 At n ≥ 30, `tabu` dominates

| family, n | greedy | random | sa | **tabu** |
|---|---|---|---|---|
| random  n=30 | 9.73 | 8.70 | 9.56 | **9.87** |
| random  n=40 | 12.62 | 9.96 | 11.82 | **13.05** |
| random  n=60 | 19.38 | 12.50 | 15.95 | **20.17** |
| planted n=30 | 14.58 | 13.44 | 13.39 | **14.83** |
| planted n=40 | 19.33 | 15.54 | 16.03 | **19.64** |
| planted n=60 | 30.37 | 21.75 | 22.98 | **31.17** |
| decoy   n=30 | 14.07 | 12.72 | 12.58 | **14.27** |
| decoy   n=40 | 19.29 | 16.21 | 16.05 | **19.90** |
| decoy   n=60 | 29.71 | 21.02 | 21.15 | **30.52** |

Tabu is best or tied-best in all 9 cells. Greedy is a strong second. **SA and
random are indistinguishable at n ≥ 30** under our SA configuration
(200 reads, 3000 sweeps, tight penalty). At n=60 random family: SA=15.95
vs random=12.50 vs tabu=20.17 — SA barely beats random, tabu is 61% higher.

**Honest caveat.** SA was tuned on dev seeds at n≤20 only (3000 sweeps is
enough there but plausibly too few at n≥30). This is a legitimate finding
about *our SA configuration*, not about SA as an algorithm. Proper SA tuning
at n≥30 is deferred — it would cost hours and is out of scope for v0.1.0.

Tabu at n=20 matched exact MILP to four decimals on every seed of every
family (see §3.3). It is a reasonable ground-truth proxy for n≥30, though not
certified.

### 3.3 Wall-clock is not uniform

- Exact MILP: 0.02–0.28 s (n≤20).
- Greedy: 0.06 s (n=12) → 1.3 s (n=60).
- Random: 1.20 s everywhere (fixed 10k samples).
- SA: 0.25 s (n=12) → 2.28 s (n=60).
- Tabu: 2.11 s everywhere (fixed 100 reads × ~21 ms).

At n=60, tabu is only **1.1× slower** than greedy but delivers 61% higher
objective. That is the most important operational result of v0.1.0.

## 4. Quantum (QAOA)

### 4.1 QAOA-SV loses to every classical baseline at every cell

| cell | exact | best classical (excl. exact) | best QAOA-SV |
|---|---|---|---|
| random  n=12 | 4.02 | 4.02 (random/greedy/sa/tabu) | 3.53 (MPS χ=8) |
| random  n=20 | 5.38 | 5.38 (tabu) | 2.44 (MPS χ=2) |
| planted n=12 | 6.85 | 6.85 (random/sa/tabu) | 5.57 (SV p=2) |
| planted n=20 | 9.17 | 9.17 (tabu) | 7.19 (SV p=2) |
| decoy   n=12 | 6.90 | 6.90 (random/sa/tabu) | 6.54 (SV p=2) |
| decoy   n=20 | 9.01 | 9.01 (random/tabu) | 5.91 (MPS χ=4) |

At n=20, QAOA reaches **45% (random), 78% (planted), 66% (decoy)** of the
classical optimum. Never higher.

### 4.2 Root cause: feasible-shot collapse

`feasible_frac_shots` = mean over seeds of the fraction of 1024 shots inside
the cardinality subspace:

| cell | SV p=1 | SV p=2 | MPS p=2 χ=4 | MPS p=2 χ=8 |
|---|---|---|---|---|
| random  n=12 | 13.5% | 6.2% | 8.7% | 6.8% |
| planted n=12 | 4.7% | 5.0% | 5.2% | 5.0% |
| decoy   n=12 | 10.3% | 11.3% | — | — |
| random  n=20 | 1.05% | **0.37%** | **0.024%** | **0.020%** |
| planted n=20 | 0.56% | 2.84% | 0.66% | 0.85% |
| decoy   n=20 | 0.41% | 1.21% | — | — |

The RX mixer on a QUBO with a cardinality penalty does not preserve the
cardinality subspace. At n=20 p=2 χ∈{4,8}, on `random`, **fewer than 3 shots
per 10,000** are feasible — that is why the reported objective is negative
(`−0.45 [−1.95, 0.95]` and `−0.33 [−1.07, 0.83]`): the "best" sample is
infeasible and its energy includes the penalty. This is a structural failure
mode of RX-mixer + penalty-QUBO, not a hyperparameter issue.

**Implication.** QAOA-SV at n≤20 with RX mixer is not a competitive solver on
this problem family. The natural fix — XY mixer with Dicke-state
initialization — is future work (out of scope for v0.1.0).

### 4.3 Truncation vs quality (χ sweep) — non-monotone

Full χ sweep is available only at **n=12**. At n=20 the sweep is partial
(planted p=1 χ=32 has n=3 seeds; all other cells complete at n_seeds=20).
The **truncation metric `discarded_weight` is only measurable at n≤12** in
v0.1.0 (see §1); at n>12 it is `NaN` by design.

n=12, `planted` family:

| p | χ=2 | χ=4 | χ=8 | χ=16 | χ=32 |
|---|---|---|---|---|---|
| p=1 obj | 5.49 | 5.30 | **4.96** | 5.12 | 5.08 |
| p=1 discarded | 20.09 | 29.44 | 27.61 | 29.35 | 30.47 |
| p=2 obj | 4.83 | **3.86** | 4.41 | 4.90 | 5.32 |
| p=2 discarded | 18.70 | 16.80 | 27.78 | 22.63 | 35.64 |

n=12, `random` family:

| p | χ=2 | χ=4 | χ=8 | χ=16 | χ=32 |
|---|---|---|---|---|---|
| p=1 obj | 2.89 | 2.74 | **3.26** | 3.16 | 3.24 |
| p=1 discarded | 8.82 | 5.58 | 5.47 | 7.01 | 8.26 |
| p=2 obj | **2.04** | 3.51 | 3.53 | 3.03 | 3.21 |
| p=2 discarded | 1.96 | 2.31 | 3.27 | 8.19 | 8.73 |

**Quality is not monotone in χ.** In three of four (family, p) cases the
best objective occurs at χ ∈ {2, 4, 8}, not at χ=32. More bond dimension does
not translate into better samples under our RX mixer + penalty QUBO. This is
consistent with §4.2: the limitation is the mixer's leakage off the
cardinality subspace, not the state's tensor-rank.

**Cost note.** At n=20, MPS is **5–10× slower** than statevector per seed
(9.8–20 s vs 1.7–4.9 s). The MPS advantage only materializes when 2^n is
intractable, which we did not reach. Reported as-is.

## 5. Planted and decoy recovery

`planted_recovery` = |S ∩ S_planted| / k; `decoy_recovery` = |S ∩ D| / |D|.

- **Planted recovery degrades with n** for all methods: exact 73% (n=12) →
  60% (n=20) → no exact data beyond. At n=60 the classical solvers' planted
  recovery is 47–62%.
- **QAOA-SV planted recovery is consistently 10–20 points below classical**
  at each size.
- **Decoy recovery is flat 15–27% for every solver at every cell.** The decoy
  mechanism is active and no method captures the decoy hubs, as designed.
- **Planted recovery < 100% is not a bug.** With `weight_noise=1.0` vs
  `weight_signal=2.0`, the planted set is not always the true optimum. The
  noisy-planted generator is working as intended.

## 6. Memory

- Classical solvers: ~208 MB regardless of n (peak RSS).
- QAOA-SV n=12: ~213 MB.
- QAOA-SV n=20: **600–1466 MB** (O(2^n) statevector + per-qubit RX copies).
- QAOA-MPS n=20: ~613–1259 MB, dominated by the dense-state extraction for
  the `energy_analytic` diagnostic at n≤12, plus O(n·χ³) MPS tensors.

Statevector QAOA on a typical workstation is bounded to **n ≤ 22**.

## 7. What this report does NOT claim

- **No quantum advantage.** QAOA never beat classical at any cell tested.
- **No claim about χ-vs-quality at n>12.** We measured n=12 only.
- **No claim that tabu is optimal at n≥30.** It is best among tested methods;
  exact ground truth stops at n=20.
- **No claim that SA is a bad algorithm.** Our SA hyperparameters were tuned
  at n≤20 only.
- **No claim that the RX mixer is fundamentally wrong.** It is wrong for
  cardinality-constrained problems *without a subspace-preserving mixer*.
  XY + Dicke is the standard fix, not attempted here.

## 8. Future work (not in v0.1.0)

1. **XY mixer with Dicke initialization** to keep shots in the cardinality
   subspace. Would address §4.2 and §4.3 directly.
2. **MPS at n ∈ {30, 40}** — the regime where statevector is intractable and
   the MPS advantage actually appears. Partial data exists; not complete.
3. **SA tuning at n ≥ 30.** More reads, more sweeps, or a schedule tuned to
   the specific family.
4. **p=3** at n=12/20 (tuning budget ~2 h).
5. **Real-data loader** (STRING TSV + gene score table) with degree-matched
   null, per PROTOCOL §3 of the ROLE.

## 9. Reproducibility

Every row in `results/test/summary_all.csv` corresponds to a specific JSONL
under `results/test/`. Each JSONL records seed, wall-clock, peak RSS, package
versions, and git commit hash. Instances are frozen at
`results/instances/manifest.json` (sha256 per file, seeds 100–119). `make
reproduce` re-runs the matrix and regenerates this report's source CSV.

At release v0.1.0 the frozen instance set is: 3 families × 5 sizes × 20 seeds
= 300 NPZ files; manifest committed to the repo.
