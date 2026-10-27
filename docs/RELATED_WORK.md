# Related Work and Positioning

## 1. Existing Benchmark Libraries for QUBO and Quantum Optimisation

### QUBOLib
QUBOLib is a Julia-based benchmark library that stores QUBO instances in a
canonical form and provides a reproducibility layer for solver comparisons. It
focuses on physics-inspired and heuristic optimisers and records provenance,
solution bitstrings, and best-known incumbents. However, QUBOLib is a
*general-purpose* corpus: it does not provide biologically structured,
cardinality-constrained instances with planted ground truth, nor does it analyse
the trade-off between tensor-network truncation and solution quality at equal
compute.

### QOBLIB
QOBLIB (Quantum Optimization Benchmarking Library) offers ten challenging
problem classes with MIP and QUBO formulations, covering sizes from 100 to
10,000 decision variables. Its goal is fair, comparable benchmarking for quantum
optimisation methods. While it includes practically relevant problems, it lacks
a dedicated class for biological module selection with a degree-matched null
model and decoy hubs, and it does not provide a truncation-vs-quality study for
MPS-based QAOA simulation.

### Planted QUBO Benchmarks
There is a body of work on "planted" QUBO instances for benchmarking quantum
annealers, where a known solution is embedded into the QUBO. These works focus
on annealer benchmarking and do not target biologically structured networks,
nor do they include decoy hubs or a degree-bias null model.

### QAOA Simulation with MPS
Tensor-network methods, including MPS with truncation, have been used to
simulate QAOA for larger circuits. Several studies report that truncation can
degrade solution quality, but they do not systematically analyse this
degradation against classical baselines at equal compute, nor do they focus on
cardinality-constrained biological module selection.

## 2. Biological Network Module Selection and QUBO
Densest k-subgraph and related module-selection problems have been formulated as
QUBOs for biological applications. However, existing work either uses small, toy
instances or lacks a controlled benchmark with planted ground truth, decoy hubs,
and a degree-matched null model. There is no open, reproducible benchmark that
separates the effect of degree bias from true modular signal in a
cardinality-constrained QUBO setting.

## 3. Gap Statement

**Existing work does not provide:**

1. A benchmark of **biologically structured, cardinality-constrained QUBO
   instances** with **planted ground truth** and **decoy hubs** to expose
   degree/study-bias artefacts.
2. A **degree-matched null model** for validating module-selection results.
3. A systematic **truncation-vs-solution-quality analysis** for MPS-based QAOA
   at **equal compute**, compared against strong classical baselines (SA, tabu,
   exact MILP).
4. An **open, pip-installable Python toolkit** that integrates instance
   generation, multiple solvers, MPS engines with bond-dimension sweeps, and
   automatic figure generation from real runs.

Absolutus_V3 fills this gap. It is designed for quantum-optimisation researchers
who need hard, structured instances beyond toy sizes; for bioinformaticians who
want a principled QUBO formulation with a null model; and for students who need
a clean, documented reference implementation.

**If any of these claims are later found to be covered by existing work, this
phase will be revised and the project will stop or pivot.**
