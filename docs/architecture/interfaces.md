# Module Interfaces

This document defines the minimal public contract for every module in the
synchronous parallel multi-surrogate optimization framework. It is written
before any implementation code so that every design decision can be reviewed
first.

Scope constraints that apply to every section below:

- Continuous, box-bounded search spaces only. No categorical/conditional
  spaces yet.
- The framework always maximizes. Any benchmark that is naturally minimized
  is transformed at the objective boundary.
- Synchronous rounds only: a batch must finish evaluating before the next
  round starts. No adaptive allocation, no async scheduling, no
  multi-fidelity logic.
- No module may assume a specific surrogate family (no GP-only or RF-only
  fields, no assumption that every surrogate exposes a standard deviation).
- No module may assume one surrogate is "fast" and the other "slow."
- The Orchestrator must never branch on which concrete surrogate class it is
  holding.

---

## 1. SearchSpace

### Responsibility
Represents a valid hyperparameter space and generates reproducible candidate
samples from it.

### Initial scope
Continuous box-bounded spaces only.

### Interface

```python
class SearchSpace:
    def __init__(self, bounds: list[tuple[float, float]]):
        """bounds[i] = (low_i, high_i) for dimension i."""
        ...

    @property
    def dim(self) -> int:
        ...

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Return an array of shape (n, dim). Every row is a valid
        configuration. Determinism is entirely controlled by the caller-
        supplied rng — SearchSpace holds no internal random state."""
        ...
```

### Notes
- `rng` is always passed in explicitly, never created or seeded inside the
  class. This keeps all randomness traceable to the single experiment seed
  (see experiment conventions, section G).
- No validation logic beyond box bounds is in scope yet (no conditional
  parameters, no integer/categorical dimensions).

---

## 2. Objective function

### Responsibility
Evaluate one configuration and return a scalar score.

### Interface

```python
def objective(x: np.ndarray, seed: int | None = None) -> float:
    """Return a scalar score for configuration x. HIGHER = BETTER, always."""
    ...
```

### Convention
If the underlying benchmark is naturally minimized (e.g. Rastrigin, log-loss),
the transformation (typically negation) happens **inside** the objective
function, at the boundary between the benchmark and the framework. Nothing
downstream of this function ever needs to know whether the original benchmark
was a minimization problem.

### Notes
- No base class. A bare callable satisfying this signature is the entire
  contract. Benchmarks are plain functions (or simple closures/partials), not
  objects, unless a concrete need for shared state (e.g. a cached dataset for
  the logistic regression benchmark) arises — and even then, the shared state
  can be captured via closure or `functools.partial` rather than a class
  hierarchy.
- `seed` is optional because not every benchmark is stochastic (Branin,
  Hartmann6, Rastrigin are deterministic; a benchmark that trains a model on
  resampled data would use `seed`).

---

## 3. DataStore

### Responsibility
The single shared source of truth for completed evaluations. Pure storage —
it makes no optimization decisions and contains no model-fitting or
candidate-selection logic.

### Record shape
Each stored record contains at minimum:

| field | type | meaning |
|---|---|---|
| `x` | `np.ndarray` shape `(dim,)` | evaluated configuration |
| `y` | `float` | observed objective value (already oriented higher-is-better) |
| `round_id` | `int` | which optimization round produced this evaluation |
| `candidate_id` | `str` or `int` | unique id for this evaluation |
| `surrogate_source` | `str` | which surrogate proposed this candidate (or `"warm_start"`) |
| `candidate_rank_within_source` | `int` | rank position within that surrogate's proposal list |
| `eval_start_time` | `float` | wall-clock start (e.g. `time.perf_counter()`) |
| `eval_end_time` | `float` | wall-clock end |
| `eval_duration` | `float` | `eval_end_time - eval_start_time` |
| `seed` | `int \| None` | seed passed to the objective, if any |

### Interface

```python
class DataStore:
    def append(self, record: dict) -> None: ...

    def append_batch(self, records: list[dict]) -> None: ...

    def get_X_y(self) -> tuple[np.ndarray, np.ndarray]:
        """X shape (n, dim), y shape (n,), in insertion order."""
        ...

    def get_all(self) -> list[dict]:
        """Full records, including metadata, for metrics/analysis."""
        ...

    def __len__(self) -> int: ...
```

### Notes
- `get_X_y()` exists specifically so surrogates never touch raw records —
  they only ever see `(X, y)`.
- `get_all()` is for metrics and analysis code, not for surrogate fitting.
- No update/delete methods. The store is append-only by design.

---

## 4. BaseSurrogate

### Responsibility
Fit a model to the shared observation history and propose new candidates to
evaluate. Every concrete surrogate (RF, GP, TPE, future neural surrogate)
must be interchangeable behind this contract.

### Interface

```python
class BaseSurrogate(ABC):
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit on the full shared history. Stateful: a later predict/propose
        call uses the most recent fit."""
        ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Point predictions, shape (n,). Does NOT need to return a standard
        deviation or any other uncertainty measure — not every model family
        has one."""
        ...

    def propose(
        self,
        search_space: SearchSpace,
        n_candidates: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Return up to n_candidates configurations, shape (n_candidates, dim),
        ordered best-first. Rank is implied by row position, not a separate
        returned value."""
        ...
```

### Design decision: `propose()` lives on the surrogate itself (for v0)

This was the central question for this interface. Two designs were
considered:

**Option 1 (chosen for v0): `propose()` is a method on `BaseSurrogate`.**
Each surrogate implementation owns its full pipeline — fit, internal
uncertainty/acquisition representation (if any), and candidate ranking — behind
one method.

**Option 2 (rejected for v0): separate the acquisition/candidate-generation
logic into its own module, independent of the surrogate.**
A generic `Acquisition` class would call `surrogate.predict()` (and
optionally some uncertainty hook) and rank candidates itself.

**Why Option 1 for the first version:** the project explicitly must not
assume every surrogate exposes a standard deviation or any other common
uncertainty channel. GP posterior variance, RF prediction-interval bootstrap,
and TPE's density-ratio scoring are structurally different — there is no
honest shared numeric interface to factor out yet without either (a) forcing
every surrogate to fake a variance it doesn't have, or (b) building a
premature abstraction with only one or two real implementations to generalize
from. Keeping `propose()` inside the surrogate avoids inventing that
abstraction before we have evidence of what it should look like.

**Tradeoff accepted:** this couples "model" and "acquisition strategy" inside
one class per surrogate. If later work wants to mix-and-match (e.g. GP model +
a different acquisition function), that will require refactoring. This is
listed as an explicit open decision below and is the first thing to revisit
once a second or third concrete surrogate exists.

### Notes
- `predict()` is kept in the contract even though the v0 orchestrator may
  never call it directly — it is needed for later metrics (e.g. surrogate
  agreement/calibration analysis) and keeps the contract honest about what a
  "model" is, independent of acquisition behavior.
- Nothing in this interface requires `fit()` to be called before `propose()`,
  but in practice the Orchestrator always calls them in that order.

---

## 5. CandidateMerge

### Responsibility
Combine the two surrogates' ranked proposal lists into one evaluation batch,
preserving attribution, removing exact duplicates, and refilling from the
next-ranked unused candidate — deterministically, with no adaptive logic.

### Inputs
- `proposals: dict[str, np.ndarray]` — keyed by `surrogate_source` label,
  each value a ranked array `(n_i, dim)` as returned by that surrogate's
  `propose()`.
- `allocation: dict[str, int]` — how many slots each source is entitled to
  in the final batch (fixed per round; 50/50 for the two-surrogate case per
  experiment conventions).

### Output
A list of records, one per batch slot:

```python
{
    "candidate": np.ndarray,              # shape (dim,)
    "surrogate_source": str,
    "candidate_rank_within_source": int,  # 0-indexed rank in that source's original proposal list
}
```

### Behavior (v0, fixed and deterministic)
1. Walk each source's allocation in rank order, taking its top-ranked
   unused candidates first.
2. If a candidate is an exact duplicate of one already placed in the batch
   (by any source, including itself), skip it and take that source's next-
   ranked candidate instead — repeat until the slot is filled or that
   source's proposal list is exhausted.
3. No randomness is introduced inside merge. Any necessary tie-break is
   resolved purely by rank order, never by re-sampling.

### Notes
- This function is pure: same inputs always produce the same output.
- It does not evaluate anything and does not touch the DataStore or
  WorkerPool.
- The exact duplicate-equality rule and the tie-break rule for when two
  *different* sources propose the identical point are open decisions (see
  bottom of this document).

---

## 6. WorkerPool

### Responsibility
Evaluate one merged batch of candidates against the real objective function.
Evaluations within a batch may run in parallel; the pool does not return
until the entire batch is complete (the framework-level loop stays
synchronous even though in-batch execution may be parallel).

### Inputs
- `batch: list[dict]` — the CandidateMerge output.
- `objective_fn: Callable[[np.ndarray, int | None], float]`.
- `round_id: int`.

### Output
A list of result records, one per evaluated candidate, each containing
everything required to build a DataStore record:

```python
{
    "x": np.ndarray,
    "y": float,
    "candidate_id": str | int,
    "surrogate_source": str,
    "candidate_rank_within_source": int,
    "round_id": int,
    "eval_start_time": float,
    "eval_end_time": float,
    "eval_duration": float,
    "seed": int | None,
}
```

### Notes
- No distributed-computing frameworks (e.g. Ray) in v0. A local executor
  (thread or process pool, or even a plain loop) is sufficient — see open
  decisions.
- `candidate_id` generation is the WorkerPool's (or Orchestrator's)
  responsibility, not CandidateMerge's — CandidateMerge only knows about
  rank and source, not about evaluation identity.

---

## 7. Orchestrator

### Responsibility
Coordinate the optimization loop end to end. Contains no model-specific
behavior — it only knows about the `BaseSurrogate`, `DataStore`,
`CandidateMerge`, and `WorkerPool` contracts.

### Loop (conceptual)

```python
def run(self):
    warm_X = search_space.sample(n_warm_start, warm_rng)
    warm_results = worker_pool.evaluate(warm_X, objective_fn, round_id=0, source="warm_start")
    datastore.append_batch(warm_results)

    round_id = 1
    while not budget_exhausted():
        X, y = datastore.get_X_y()

        surrogate_a.fit(X, y)
        surrogate_b.fit(X, y)

        proposals_a = surrogate_a.propose(search_space, n_a, rng_a)
        proposals_b = surrogate_b.propose(search_space, n_b, rng_b)

        batch = candidate_merge(
            {"surrogate_a": proposals_a, "surrogate_b": proposals_b},
            allocation={"surrogate_a": n_a, "surrogate_b": n_b},
        )

        results = worker_pool.evaluate(batch, objective_fn, round_id)
        datastore.append_batch(results)

        metrics.record_round(round_id, ...)
        round_id += 1
```

### Notes
- The Orchestrator holds references to `BaseSurrogate` instances, never
  concrete classes — it cannot tell GP from RF from TPE, and must not need
  to.
- Budget check (`budget_exhausted()`) is based on total real evaluations by
  default (see experiment conventions, section K), not elapsed wall-clock
  time or round count.
- The Orchestrator owns sub-seed derivation for each round (warm-start seed,
  per-surrogate seed, candidate-sampling seed) — see experiment conventions,
  section G.

---

## 8. Metrics / logging

### Responsibility
Derive analysis-ready aggregates from the DataStore's records. Metrics code
is read-only with respect to optimization behavior — it never feeds back
into what the Orchestrator does.

### Fields collected, from the start

Per-round timing:
- `fit_time: dict[str, float]` — keyed by `surrogate_source` label (e.g.
  `{"surrogate_a": 0.42, "surrogate_b": 1.1}`), not hardcoded model names.
- `proposal_time: dict[str, float]` — same keying.
- `merge_time: float`
- `eval_time: float` — total batch evaluation wall-clock.
- `round_total_time: float`

Per-round optimization quality:
- `num_candidates_by_source: dict[str, int]`
- `num_duplicates_removed: int`
- `cumulative_best_y: float`
- `num_global_best_improvements_by_source: dict[str, int]`

Run-level:
- `total_wall_clock_time: float`
- `total_evaluations: int`

### Design decision: model-name-keyed dicts, not `surrogate_a_fit_time` / `surrogate_b_fit_time` literal fields

The task prompt offered both as acceptable. Dict-keyed-by-source is chosen
because it does not hardcode an assumption of exactly two surrogates, and
`surrogate_source` is already the attribution key used everywhere else
(DataStore records, CandidateMerge output, WorkerPool output) — reusing it
for metrics keeps one consistent vocabulary across the whole framework
instead of introducing a second naming scheme.

### Notes
- Metrics are computed from `DataStore.get_all()` plus timing captured by
  the Orchestrator/WorkerPool at each stage — `metrics.py` does not need its
  own storage; it can be a pure aggregation module operating on records.

---

## Open Decisions Before Implementation

- Whether `propose()` belongs permanently inside `BaseSurrogate`, or should
  be split into a separate acquisition abstraction once a second concrete
  surrogate exists to generalize from (see section 4).
- Whether `propose()` should also return per-candidate scores alongside the
  ranked array, or whether rank-by-position is sufficient for v0.
- The exact duplicate-equality rule in `CandidateMerge` (exact floating-point
  equality vs. a numeric tolerance for continuous spaces).
- The tie-break rule in `CandidateMerge` when two *different* surrogates
  propose the identical point — which source's slot refills and which keeps
  attribution.
- `candidate_id` format (e.g. incrementing integer vs. UUID) and which
  component assigns it (WorkerPool vs. Orchestrator).
- WorkerPool executor choice for v0: plain sequential loop,
  `ThreadPoolExecutor`, or `ProcessPoolExecutor` — depends on whether the
  initial benchmarks are CPU-bound (model training) or cheap/synthetic
  (Branin, Hartmann6, Rastrigin), where parallelism overhead may dominate.
- DataStore backing representation: plain list of dicts, pandas DataFrame,
  or append-only file/log — no decision needed until persistence across
  runs is required.
- Whether every concrete surrogate must implement `predict()` in v0 even if
  the Orchestrator never calls it, or whether it may raise
  `NotImplementedError` until a metrics use case needs it.
- Naming convention for `surrogate_source` values (literal class name,
  user-supplied experiment label, or a fixed `"surrogate_a"`/`"surrogate_b"`
  role label independent of the underlying model).
