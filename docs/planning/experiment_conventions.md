# Experiment Conventions

These conventions make surrogate-pair comparisons fair and reproducible.
They apply to every experiment run with the framework described in
[`docs/architecture/interfaces.md`](../architecture/interfaces.md). Nothing
here is surrogate-family-specific — these rules must hold no matter which two
surrogates (RF, GP, TPE, or others later) are being compared.

---

### A. Objective direction

Everything presented to the optimizer uses **higher `y` = better**, with no
exceptions. If a benchmark is naturally minimized, the transformation happens
inside the objective function itself (see `interfaces.md` section 2) — no
other module ever needs to know a benchmark was originally a minimization
problem.

---

### B. Shared initial design

For the same `(benchmark, seed)` pair, every surrogate pair being compared
must receive **the exact same warm-start points and the exact same observed
objective values**. The warm-start sample is drawn once per `(benchmark,
seed)` combination and reused across every surrogate pair evaluated under
that combination — it is not redrawn per surrogate pair.

**Why:** if warm-start points differ between pairs, any later performance gap
could be explained by differing starting data rather than by the surrogates
themselves.

---

### C. Fixed evaluation budget

Every surrogate pair in a comparison gets the same total number of real
objective evaluations (warm-start evaluations + all batch evaluations across
all rounds). The number of candidates any individual surrogate proposes must
never change the total number of real evaluations a pair receives — only
which candidates fill the fixed number of batch slots.

---

### D. Fixed batch size

The same batch size is used across all pairs in a given comparison, unless
batch size is itself the explicit independent variable of the experiment.
For initial pair screening, the two surrogates split the batch **50/50**
whenever the batch size is even. (Odd batch sizes and uneven allocation are
not addressed yet — see Open Decisions.)

---

### E. Same worker configuration

Comparable experiments use the same number of evaluation workers. Worker
count changes wall-clock time, not evaluation count, so it must be held fixed
within a comparison to keep wall-clock measurements comparable across pairs.

---

### F. Same benchmark / search space

For a given comparison, every pair sees identical bounds, preprocessing,
datasets, train/validation splits, and evaluation procedure. None of these
may vary between surrogate pairs within the same comparison.

---

### G. Reproducibility

A single experiment seed deterministically controls everything random in a
run:

- warm-start sampling
- surrogate-internal randomness (e.g. a model's own stochastic fitting)
- candidate sampling inside `propose()`
- objective randomness, where the benchmark accepts a seed

**Principle:** sub-seeds are derived deterministically from the one
experiment seed and a role label (e.g. `"warm_start"`, `"surrogate_a"`,
`"surrogate_b"`, `"round_3_candidate_sampling"`), rather than by mutating a
single global RNG that every component draws from in sequence. Drawing from
one shared, mutated global generator makes outcomes depend on *call order*,
not just on the seed — which breaks reproducibility the moment an unrelated
call is added or removed upstream. Deriving each sub-seed from
`(experiment_seed, role_label)` keeps every component's randomness
independent of what else happened to run before it.

The exact mechanism for deriving these sub-seeds (e.g. NumPy
`SeedSequence.spawn()`, hashing `(experiment_seed, role_label)`, or a fixed
per-role offset table) is **not decided yet** — see Open Decisions.

---

### H. Candidate duplication

A real configuration must not be evaluated twice within the same batch merely
because two surrogates independently proposed it. `CandidateMerge` is
responsible for deduplication and refilling from the next-ranked unused
candidate (see `interfaces.md` section 5). This is enforced at the merge
layer, not by the surrogates or the WorkerPool.

---

### I. Per-surrogate attribution

Every evaluated candidate preserves which surrogate proposed it. Minimum
fields, present on every stored record:

- `surrogate_source`
- `candidate_rank_within_source`

These two fields are the basis for metrics to be derived later, including:

- number of a surrogate's proposed candidates that were actually evaluated
  (after dedup/refill)
- number of global-best improvements attributable to each surrogate
- improvement rate per surrogate
- objective-value distribution per surrogate
- duplicate rate between the two surrogates' proposal lists

---

### J. Timing

Every run collects separate timing for at least:

- surrogate fitting (per surrogate)
- candidate proposal (per surrogate)
- objective evaluation (per batch)
- full optimization round (fit + propose + merge + evaluate)
- total run wall-clock time

See `interfaces.md` section 8 for the exact field shapes.

---

### K. Stopping condition

The primary stopping condition for scientific comparisons is a **fixed real-
evaluation budget**, not wall-clock time and not a fixed round count (round
count can vary if batch size varies, but evaluation budget must not).
Wall-clock time is always measured and reported as an *outcome* of a run —
it must never be used to decide when a run stops, since that would let faster
surrogates silently receive fewer evaluations than slower ones.

---

### L. Initial benchmark set

The currently planned first benchmark set is:

- Branin
- Hartmann6
- Rastrigin
- Logistic Regression

This set is intentionally not expanded further at this stage. Adding
benchmarks is a later decision, not part of Sprint 0.

---

### M. Initial surrogate screening

GP, RF, and TPE are documented as the **likely** initial surrogate families
to implement and compare — this is not a commitment to any final choice, and
actual availability depends on what gets implemented. The architecture
(`interfaces.md` section 4, `BaseSurrogate`) is required to keep all
surrogates interchangeable regardless of which families end up implemented
first.

---

## Open Decisions Before Implementation

- Exact method for deriving deterministic sub-seeds from
  `(experiment_seed, role_label)` (see section G).
- How to handle odd batch sizes or more than two surrogates when the
  allocation convention is "50/50" (see section D).
- Number of warm-start observations per run.
- Exact initial batch size.
- Number of seeds to use for pilot experiments (i.e. how many independent
  repeats per benchmark/pair combination before treating a result as
  meaningful).
- Exact evaluation budget (total real evaluations) for initial pilot runs.
- Whether worker count for initial pilots is 1 (fully synchronous, simplest
  to debug) or >1 (exercises the parallel path early).
