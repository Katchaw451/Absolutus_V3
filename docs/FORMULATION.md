# Problem Formulation: Cardinality-Constrained Module Selection

## Notation

- Graph G = (V, E) with |V| = n. Adjacency matrix A (symmetric, non-negative,
  A_ii = 0). For biological networks, A_ij typically counts interactions or is
  thresholded to 0/1.
- Evidence vector w ∈ ℝ^n: per-node score (e.g. differential expression,
  effect size, prior knowledge).
- Node degrees d_i = Σ_j A_ij.
- Target module size k ∈ ℕ, k < n.
- Hyperparameters λ_coh ≥ 0 (cohesion weight), λ_red ≥ 0 (redundancy/degree
  penalty).

## Direct objective (to be maximised)

For a selection mask x ∈ {0,1}^n with Σ_i x_i = k:

    F(x) =  Σ_i w_i x_i                          (evidence)
          + λ_coh · Σ_{i<j} A_ij x_i x_j         (network cohesion)
          - λ_red · Σ_i d_i x_i                  (redundancy / degree bias)

The cohesion term rewards internal edges of the selected set. The redundancy
term is the *degree-matched null*: it penalises selecting highly-connected
nodes, which in biological networks often reflects study bias (well-studied
genes accumulate more reported interactions) rather than true modular signal.
With λ_red = 0 the objective reduces to the standard densest-k-subgraph
problem; with λ_red > 0 it explicitly compensates for hub bias.

## QUBO form (exact cardinality penalty)

We minimise E(x) = -F(x) + P · (Σ_i x_i − k)^2 over x ∈ {0,1}^n, with P > 0
chosen large enough that any off-cardinality solution has strictly higher
energy than any feasible one. Expanding the penalty (using x_i^2 = x_i for
binary variables):

    (Σ_i x_i − k)^2 = Σ_i x_i + 2 Σ_{i<j} x_i x_j − 2k Σ_i x_i + k^2
                    = Σ_i (1 − 2k) x_i + 2 Σ_{i<j} x_i x_j + k^2.

Collecting coefficients:

    E(x) = offset + Σ_i h_i x_i + Σ_{i<j} J_ij x_i x_j,

    h_i  = −(w_i − λ_red · d_i) + P · (1 − 2k)
    J_ij = −λ_coh · A_ij + 2P    (for i < j, and only for edges with A_ij ≠ 0)
    offset = P · k^2

For feasible x (|x| = k), the penalty vanishes and E(x) = −F(x). The default
penalty in code is

    P = 2 · ( k·w_max + λ_coh·Σ_ij A_ij + k·λ_red·d_max + 1 ),

which guarantees P > 2·max_x |F(x)| and hence that minimising E is equivalent
to maximising F on the feasible set.

## Hard-constraint MILP variant

For exact optimisation (HiGHS via `scipy.optimize.milp`) we drop the penalty
and solve the constrained problem directly:

    max   Σ_i (w_i − λ_red d_i) x_i + λ_coh · Σ_{e=(i,j), i<j} A_ij y_e
    s.t.  Σ_i x_i = k,
          y_e ≤ x_i,  y_e ≤ x_j,  y_e ≥ x_i + x_j − 1   for each edge e,
          x_i ∈ {0,1},  y_e ∈ {0,1}.

The linearisation y_e = x_i x_j is exact for binary x (McCormick constraints
reduce to equality). Its LP relaxation is tighter than the penalty QUBO, which
is why we use it for exact/bound work.

## Degree-matched null model

Given a selected set S with |S| = k and observed objective F(S), the null model
redraws the evidence vector w from a degree-matched permutation (degree-
preserving rewiring of G, then re-sampling w without replacement) and recomputes
F on the same selection rule. A module is considered statistically interesting
only if F(S) exceeds the 95th percentile of the null. (Implemented in Phase 2
loader; used in Phase 5 analyses.)
