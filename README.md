# Absolutus_V3 (modbench)

An open, reproducible benchmark and toolkit for **cardinality-constrained
module selection** (QUBO / densest-k-subgraph style) on **biologically
structured networks**. Compares classical solvers (SA, tabu, greedy, exact
MILP) and **QAOA with statevector / MPS (tensor-network) simulation** at equal
compute.

**Release v0.1.0** — see `docs/REPORT.md` and `docs/CLAIMS.md` for measured
findings. Honest state of the benchmark as of v0.1.0, including every regime
where classical baselines win.

## Quickstart (5 commands)

    git clone https://github.com/Katchaw451/Absolutus_V3.git
    cd Absolutus_V3
    python3 -m venv .venv && source .venv/bin/activate
    pip install -e ".[dev]"
    pytest -q

Generate an instance and solve it exactly:

    modbench generate --generator planted_decoy_hubs --n 20 --k 4 --seed 42 \
        --out results/demo --exact --lambda-coh 1.0 --lambda-red 0.1

Reproduce the full benchmark matrix (2-4 h):

    make reproduce

## CLI

    modbench generate --generator {random_sparse,planted_scale_free,planted_decoy_hubs}
                      --n N --k K --seed S --out PATH [--exact]
    modbench run       # benchmark runner (Phase 5)
    modbench report    # regenerate figures + summary from results/

## Key findings in v0.1.0 (all in docs/REPORT.md)

- **n <= 20 does not differentiate classical solvers** under our equal-compute
  budget (`random` is effectively brute force at these sizes).
- **At n >= 30, tabu dominates** every tested classical solver under equal
  compute; SA under our config collapses to near-random.
- **QAOA-SV (RX mixer, p in {1,2}) loses to every classical solver** at every
  cell. The RX mixer on a cardinality-penalty QUBO leaves >99% of shots
  outside the feasible subspace at n=20.
- **MPS quality is not monotone in chi** at n=12; chi in {2,4,8} often beats
  chi=32. The mixer, not the tensor rank, is the bottleneck.
- **Decoy-hub null works**: no solver selects decoy hubs; planted recovery
  degrades with n as designed.

## When to use this and when NOT to

**Use modbench if:**

- You want a QUBO formulation of densest-k-subgraph with a degree-matched
  null on biological networks, with planted ground truth and decoy hubs to
  test for degree bias.
- You want an end-to-end reproducible benchmark from instance generation to
  figures, with seeds, versions, git hashes, and equal-compute rules.
- You are studying how tensor-network truncation (MPS bond dimension)
  affects QAOA solution quality at fixed compute.

**Do NOT use modbench if:**

- You want to benchmark real biological module discovery on real networks.
  modbench is a benchmark, not a discovery tool. Use `networkx`, `igraph`,
  or `scikit-network`.
- You need state-of-the-art QAOA at scale. For QAOA on hardware or larger
  simulators use `qiskit-aer`, `itensor`, `tenpy`, or D-Wave tools.
  modbench's MPS is honest but bounded: complete chi sweep at n=12, partial
  at n=20, not reached n>=30 in v0.1.0.
- You want to make a quantum advantage claim. Explicitly out of scope; the
  benchmark reports the opposite where the data says so.
- You want a large-scale MILP benchmark suite. See QUBOLib or QOBLIB for
  general-purpose QUBO; modbench is targeted at module selection.

## Adding a solver or engine (~20 lines)

See `CONTRIBUTING.md`. New solvers subclass `modbench.solvers.base.Solver`
and register in `src/modbench/solvers/__init__.py`. New engines subclass
`modbench.engines.base.Engine` and register in
`src/modbench/engines/__init__.py`.

## Reproducibility

- Every result row records seed, wall-clock, peak RSS, Python/NumPy
  versions, git commit hash.
- Frozen instance set for v0.1.0: 3 families x 5 sizes x 20 test seeds
  = 300 instances, sha256 manifest at `results/instances/manifest.json`.
- `make reproduce` re-runs everything and regenerates every figure.

## License

MIT — see `LICENSE`.

- **n <= 20 does not differentiate classical solvers** under our
  equal-compute budget (`random` is effectively brute force at these sizes).
- **At n >= 30, tabu dominates** every tested classical solver; SA under
  our config collapses to near-random.
- **QAOA-SV (RX mixer, p in {1,2}) loses to every classical solver** at
  every cell. The RX mixer leaves >99% of shots outside the feasible
  subspace at n=20.
- **MPS quality is not monotone in chi** at n=12; chi in {2,4,8} often
  beats chi=32. The mixer, not the tensor rank, is the bottleneck.
- **Decoy-hub null works**: no solver selects decoy hubs; planted recovery
  degrades with n as designed.

## When to use this and when NOT to

**Use modbench if:**

- You want a QUBO formulation of densest-k-subgraph with a degree-matched
  null on biological networks, with planted ground truth and decoy hubs.
- You want an end-to-end reproducible benchmark with seeds, versions, and
  equal-compute rules baked in.
- You are studying how MPS truncation affects QAOA solution quality at
  fixed compute.

**Do NOT use modbench if:**

- You want to benchmark real biological module discovery. Use `networkx`,
  `igraph`, or `scikit-network`.
- You need state-of-the-art QAOA at scale. For hardware or larger
  simulators use `qiskit-aer`, `itensor`, `tenpy`, or D-Wave tools.
- You want to make a quantum advantage claim. Out of scope; the benchmark
  reports the opposite where the data says so.
- You want a general QUBO benchmark suite. See QUBOLib or QOBLIB.

## Adding a solver or engine

See CONTRIBUTING.md — 20-line recipe for both.

## License

MIT — see LICENSE.
