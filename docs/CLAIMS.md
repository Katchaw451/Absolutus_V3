# Claims Register — v0.1.0

Every claim in `docs/REPORT.md`, with evidence file, experiment, status.
Status is one of: **supported**, **partial**, **not supported**, **untested**.

## C1. At n ≤ 20, random sampling is effectively brute force for the
cardinality-constrained QUBO used here.

- **Status:** supported.
- **Evidence:** `results/test/summary_all.csv`, rows where `solver=random` and
  `n ∈ {12, 20}`. At n=12 the `obj_mean` equals `exact` to 4 decimals
  (`random` family: 4.019788 for both; `planted`: 6.845106; `decoy`: 6.895357).
  At n=20 `random` is within 0.2% of `exact` on every family.
- **Experiment:** 20 test seeds per cell, 10,000 random draws per seed.
- **Caveat:** this is a property of the *instance size*, not the solver.
  n ≤ 20 does not discriminate classical solvers at any useful resolution.

## C2. At n ≥ 30, tabu dominates greedy, SA and random under equal compute.

- **Status:** supported.
- **Evidence:** `summary_all.csv`, rows with `n ∈ {30, 40, 60}`. Tabu is best
  in all 9 (family, n) cells; e.g. random family n=60: tabu 20.17
  [19.45, 20.92] vs greedy 19.38 [18.71, 20.09] vs sa 15.95 [15.18, 16.76]
  vs random 12.50 [11.84, 13.21].
- **Caveat:** "best among tested methods", not certified-optimal. Exact MILP
  is unavailable at n≥30 within the T=30 s budget.

## C3. Under our SA configuration, SA at n ≥ 30 is statistically
indistinguishable from random.

- **Status:** supported for our SA configuration; **not supported as a
  general claim about SA**.
- **Evidence:** `summary_all.csv`; random family n=60: SA 15.95 [15.18, 16.76]
  vs random 12.50 [11.84, 13.21] — SA is 28% above random, but the gap to
  tabu (20.17) is much larger.
- **Caveat:** SA was tuned at n≤20 only (200 reads, 3000 sweeps, tight
  penalty). Tuning for n≥30 is out of scope.

## C4. Tabu matches exact MILP at n=20, seed by seed.

- **Status:** supported.
- **Evidence:** every `(family, n=20)` cell in `summary_all.csv` has
  `exact` and `tabu` columns equal to 4 decimals (random: 5.381131 for both;
  planted: 9.169561; decoy: 9.013324).
- **Caveat:** exact-matching is observed at n=20, 3 families, 20 seeds.
  Extrapolation to n>20 is not certified.

## C5. QAOA-SV (RX mixer, p ∈ {1,2}) loses to every classical solver at
every cell tested.

- **Status:** supported.
- **Evidence:** `summary_all.csv`; §4.1 of `REPORT.md`. Worst gap: random
  n=20, best-QAOA 2.44 (MPS χ=2) vs classical 5.38 = 45% of optimum.
  Best gap: decoy n=12, best-QAOA 6.54 vs classical 6.90 = 95%.
- **Caveat:** this is a negative result about this mixer + this penalty QUBO.
  A subspace-preserving mixer (XY + Dicke) is the natural follow-up.

## C6. The RX mixer's leak off the cardinality subspace is the dominant
failure mode at n=20.

- **Status:** supported.
- **Evidence:** `feasible_frac_shots` in `summary_all.csv`. At n=20 p=2
  χ=4/8 on the `random` family, fewer than 0.03% of 1024 shots per seed are
  feasible. Several seeds have **zero feasible samples in best-of-1024**, so
  the "best" reported sample is infeasible and its energy includes the
  cardinality penalty (reported objective is negative for two cells:
  -0.45 and -0.33).
- **Caveat:** this is a property of *RX + penalty*; it does not extend to
  XY + Dicke, untested here.

## C7. QAOA-MPS quality is not monotone in bond dimension χ.

- **Status:** partial (only n=12 has a complete χ sweep with seeds=20).
- **Evidence:** `summary_all.csv` n=12 rows. `planted p=1`: best objective is
  χ=8 (4.96), not χ=32 (5.08). `random p=2`: best is χ=2 (2.04), not χ=32
  (3.21). In 3 of 4 (family, p) cells the best objective is at χ ∈ {2,4,8}.
- **Caveat:** at n=12 the state is already representable at low χ for this
  problem; the result likely reflects that the mixer, not the tensor rank, is
  the bottleneck. Extrapolation to larger n is untested.

## C8. `discarded_weight` is measurable only at n ≤ 12 in v0.1.0.

- **Status:** supported (a limitation, not a result).
- **Evidence:** `mps.py`'s `energy_analytic` requires dense state extraction
  (`n ≤ 12`); at n>12 the JSONL `extra.discarded_weight` field is `NaN` by
  design (see code comment in `engines/mps.py`).
- **Implication:** the truncation-quality analysis at n≥20 is deferred.
  The values of `discarded` reported at n=12 range 1.96–35.64 across χ; a
  larger value indicates the MPS energy at that χ diverges from the sampled
  energies more.

## C9. Statevector QAOA memory hits a hard limit at n ≈ 22.

- **Status:** supported.
- **Evidence:** `summary_all.csv` `peak_mb_mean`: classical and QAOA-SV at
  n=12 are ~208–213 MB; QAOA-SV at n=20 is 600–1466 MB. Doubling n from 12
  to 20 multiplies memory by 3–7×.
- **Caveat:** hard limit is a *workstation* limit; HPC or distributed SV
  could go further.

## C10. The decoy-hub mechanism does not capture any solver at n≤20 tested.

- **Status:** supported (design confirmation, not a discovery).
- **Evidence:** `decoy_recovery_mean` in `summary_all.csv`: 15–27% for every
  solver, every cell, every n tested. No method selects the decoy hubs.
- **Caveat:** this is by design — the decoy hubs have high degree but no
  weight boost; solvers with λ_red=0.1 correctly down-weight them. A
  complementary experiment with λ_red=0 (degree-blind) was in PROTOCOL §3
  but is deferred to a follow-up.

## C11. MPS at n=20 is 5–10× slower than statevector.

- **Status:** supported.
- **Evidence:** `summary_all.csv` `wall_mean` at n=20: `qaoa/statevector`
  1.7–4.9 s; `qaoa/mps` 9.8–20.0 s per seed per χ.
- **Caveat:** the MPS advantage only appears at n ≳ 22 where 2^n is
  intractable for statevector. We did not reach that regime.

## Untested claims (not made)

- Any claim of quantum advantage.
- Any claim about MPS truncation at n≥30.
- Any claim about SA or tabu performance under different hyperparameters.
- Any claim about p ≥ 3.
- Any claim about real biological data (STRING loader deferred).
