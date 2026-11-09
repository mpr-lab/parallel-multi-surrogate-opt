# Surrogate pairs

The framework always runs exactly two surrogates per experiment. Which two
is a configuration choice (`--pair <name>`), not a code change — adding a
pair means adding one entry to `SURROGATE_PAIR_REGISTRY` in
`experiments/run_experiment.py`. The Orchestrator, CandidateMerge,
WorkerPool, DataStore, and metrics/plotting code never branch on which
pair is active; they only ever call `.name`, `.fit()`, and `.propose()` on
whatever `BaseSurrogate` instances they're given.

## `gp_rf` (Gaussian Process + Random Forest)

Both surrogates fit on the same shared `(X, y)` history each round, sample
a pool of candidates uniformly from the search space, score the pool with
their own predicted mean, and return the top candidates ranked by that
score. This is a "sample-then-rank" baseline for both surrogates — the
only difference between them is which regression model does the scoring.

## `gp_tpe` (Gaussian Process + Tree-structured Parzen Estimator)

The GP side is unchanged from `gp_rf`. The second surrogate is now a TPE
(`src/surrogates/tpe.py`) instead of a Random Forest, and it proposes
candidates through a genuinely different mechanism — not sample-a-uniform-
pool-then-rank, but sample-from-a-fitted-density.

### What TPE does, in plain language

1. **Fit.** Given the shared history, sort observations by `y` (higher is
   better, by convention). Split them into a **good** group (the top
   `gamma` fraction, default `gamma=0.25`, i.e. the best 25%) and a
   **bad** group (everyone else). Fit a separate density model over each
   group, one independent 1-D Gaussian kernel density estimate (KDE) per
   search-space dimension, with a bandwidth chosen by Silverman's rule.

2. **Propose.** To generate a candidate, pick a random point from the
   *good* group and jitter it with Gaussian noise sized by that
   dimension's bandwidth — so new candidates cluster near points that
   were already good, rather than being sampled uniformly at random
   across the whole space. A pool of these is generated this way, then
   every candidate is scored by how much more likely it is under the good
   density than the bad density (`log good_density − log bad_density`,
   in log-space for numerical stability). The pool is ranked by that
   score and the top-N candidates are returned.

This is the standard TPE idea (good/bad density split, sample from the
good side, rank by the density ratio) implemented at the level of detail
this framework needs. It is not a from-scratch research contribution and
makes no novelty claims — it exists so the framework has a second
proposal strategy that is structurally different from "rank a uniform
pool," for comparison against `gp_rf`.

### Determinism

`TPESurrogate.fit()` has no randomness at all. `TPESurrogate.propose()`
takes its randomness only from the `rng` argument the Orchestrator already
derives deterministically per-round per-surrogate
(`np.random.SeedSequence([experiment_seed, role, round_id])`) — the same
mechanism GP and RF already use. There is no internal `random_state` on
`TPESurrogate` because there is nothing for it to seed.

### Search-space handling

- **Continuous spaces:** candidates are jittered in raw `[low, high]`
  units and clipped to `search_space.low` / `search_space.high`.
- **Discrete / `ParameterSpace` workloads:** TPE only ever sees the
  normalized `[0, 1]` encoding, exactly like GP and RF. Decoding,
  deduplication against prior evaluations, and finite-space-exhaustion
  handling all happen downstream in `CandidateMerge`, unmodified.

### Round allocation at `batch_size=2`

With `n_proposals_per_surrogate=1` (the batch_size=2, 50/50 split), each
round asks GP for 1 ranked proposal and TPE for 1 ranked proposal, and
`CandidateMerge` takes exactly one candidate from each — so every non-
warm-start round evaluates exactly one GP-sourced and one TPE-sourced
candidate. Warm-start candidates are still `surrogate_source="random"`,
same as every other pair.
