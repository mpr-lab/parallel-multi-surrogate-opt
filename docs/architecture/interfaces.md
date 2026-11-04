# Module Interfaces

This document defines the minimal public contract for every module in the
synchronous parallel multi-surrogate optimization framework. It is written
before any implementation code so that every design decision can be reviewed
first.

Scope constraints that apply to every section below:

- `SearchSpace` itself models continuous, box-bounded spaces only.
  Integer/discrete/categorical parameters are handled by a separate,
  optional layer (`ParameterSpace`, §1a) that sits *in front of*
  `SearchSpace` — it never gains categorical/conditional logic itself.
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
- No validation logic beyond box bounds is in scope (no conditional
  parameters, no integer/categorical dimensions) — `SearchSpace` itself
  stays exactly this simple forever; mixed parameter types are handled by
  `ParameterSpace` (§1a) sitting in front of it, never by extending this
  class.

---

## 1a. ParameterSpace (optional numeric-vector <-> decoded-config layer)

### Responsibility
Provide the **one canonical encoding/decoding system** between the raw
numeric vector surrogates/`SearchSpace` operate on and the decoded,
meaningful configuration an objective actually receives (e.g.
`{"C": 1, "penalty": "l2"}` for Logistic Regression). Covers integers,
discrete numeric choices, and categorical choices with a single mechanism
— all three are "pick one of N options" underneath, so there is no need
for three separate dimension types.

This is entirely optional and additive: every synthetic benchmark
(Branin/Hartmann6/Rastrigin) has no `ParameterSpace` at all and is
completely unaffected by this layer existing.

### Interface

```python
@dataclass(frozen=True)
class ChoiceDimension:
    name: str
    choices: Sequence[Any]

    def decode(self, raw_value: float) -> Any:
        """Map a raw value in [0, 1) to one of `choices`."""
        ...

    def encode(self, value: Any) -> float:
        """Inverse: a representative raw value for a given choice."""
        ...


class ParameterSpace:
    def __init__(self, dimensions: list[ChoiceDimension]): ...

    @property
    def dim(self) -> int: ...

    @property
    def bounds(self) -> list[tuple[float, float]]:
        """Always [(0.0, 1.0)] * dim -- this is what gets handed to the
        unmodified SearchSpace; surrogates only ever see this unit-cube
        encoding, never the decoded values."""
        ...

    @property
    def total_configurations(self) -> int:
        """Product of each dimension's choice count -- the exact size of
        the entire finite decoded configuration space."""
        ...

    def decode(self, x: np.ndarray) -> dict[str, Any]: ...
    def encode(self, params: dict[str, Any]) -> np.ndarray: ...
```

### How it threads through the pipeline
- `Orchestrator` takes an optional `parameter_space: ParameterSpace | None`
  constructor argument (default `None` — byte-for-byte the pre-existing
  behavior). When set:
  - `SearchSpace` is built from `parameter_space.bounds` (always the unit
    cube), so surrogates fit/propose on a fixed numeric representation
    exactly as before — they are never aware decoding exists.
  - Every candidate (warm-start and round-proposed) carries an extra
    `decoded_params` dict alongside its raw numeric `candidate`/`x`.
    `DataStore`'s `x` column is always the raw numeric vector;
    `decoded_params` rides along as ordinary extra metadata.
  - The *objective function* receives `decoded_params`, not the raw
    vector, when one is present (see §2 below) — surrogates see numbers,
    objectives see meaning.
- `CandidateMerge` (§5) deduplicates on the decoded configuration, not the
  raw vector, when a `decode_fn` is given — see §5 for the full
  collision/refill/exhaustion semantics.

### Notes
- No `ContinuousDimension` type exists yet — every current ML/HF
  parameter is a discrete grid. Adding one (identity decode) would be
  straightforward if a genuinely continuous ML hyperparameter is ever
  needed, but is not built speculatively.
- `total_configurations` is what lets `CandidateMerge` distinguish "the
  entire finite space has genuinely been evaluated" from "this specific
  proposal list just didn't happen to cover something new" (§5).

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
- When a `ParameterSpace` is in play (§1a), the first argument is the
  **decoded parameter dict**, not the raw numeric vector — e.g.
  `def objective(params: dict, seed=None) -> float`. This is still the
  same single-callable contract, just with a dict instead of an ndarray
  as the "configuration" being scored. Classical-ML/HF objectives are
  factories (`make_logistic_regression_objective(...) -> Callable`) that
  do one-time dataset/model setup *before* returning this callable, so
  that setup cost is never counted inside any single candidate's
  objective-evaluation timing (WorkerPool's `eval_start_time`/
  `eval_end_time` wrap only the returned closure's own body).

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
- `decode_fn: Callable[[np.ndarray], dict] | None = None` — optional (see
  §1a). When `None` (every synthetic benchmark), everything below about
  decoding/history/exhaustion is inert and behavior is exactly the v0
  raw-numeric-dedup design.
- `evaluated_configs: set[tuple] | None = None` — canonical decoded-config
  keys already evaluated in prior rounds/warm start (supplied by
  Orchestrator from its running history). Only meaningful when `decode_fn`
  is given.
- `total_configurations: int | None = None` — size of the entire finite
  decoded space (`ParameterSpace.total_configurations`). Only meaningful
  when `decode_fn` is given; lets this function distinguish true
  exhaustion from an unlucky proposal list (see Behavior, below).

### Output
A list of records, one per batch slot:

```python
{
    "candidate": np.ndarray,              # shape (dim,)
    "surrogate_source": str,
    "candidate_rank_within_source": int,  # 0-indexed rank in that source's original proposal list
    "decoded_params": dict,               # only present when decode_fn was given
}
```

### Behavior (v0, fixed and deterministic)
1. Walk each source's allocation in rank order, taking its top-ranked
   unused candidates first.
2. **Duplicate test.** With no `decode_fn`: a candidate is a duplicate if
   its raw numeric vector exactly matches one already placed in the batch.
   With a `decode_fn`: a candidate is a duplicate if its **decoded
   configuration** matches one already placed in the batch, OR one already
   present in `evaluated_configs` (i.e. already evaluated in a prior round
   or warm start) — so a numeric vector that happens to decode to a
   configuration the framework already knows the answer to is skipped just
   like an in-batch duplicate, never silently re-evaluated as if new.
3. On a duplicate, skip it and take that source's next-ranked candidate
   instead — repeat until the slot is filled or that source's proposal
   list is exhausted.
4. If a source's list is exhausted before its allocation is filled:
   - If `decode_fn` and `total_configurations` are both given, and the
     number of distinct configurations already used (history + this
     batch) has reached `total_configurations`, every possible
     configuration has genuinely been evaluated — raise
     `SearchSpaceExhaustedError` (a `ValueError` subclass), naming the
     real, provable cause.
   - Otherwise, raise a plain `ValueError` — the proposal list just
     wasn't long/lucky enough, even though unevaluated configurations
     likely exist elsewhere. This is Orchestrator's problem to reduce
     (by requesting a longer ranked list from `propose()`), not something
     this function can fix by inventing candidates.
5. No randomness is introduced inside merge. Any necessary tie-break is
   resolved purely by rank order, never by re-sampling.

### Notes
- This function is pure: same inputs always produce the same output.
- It does not evaluate anything and does not touch the DataStore or
  WorkerPool.
- `propose()`'s "top-N ranked from a pool" mechanism does **not**
  guarantee covering every decoded bucket even when N equals
  `total_configurations` — the discarded remainder of the pool may have
  covered buckets the top-N missed. Orchestrator compensates by
  requesting a generously oversampled N (not just `total_configurations`)
  from `propose()`, which also grows the underlying pool for every
  current concrete surrogate — see §7.
- The exact duplicate-equality rule for the *raw-numeric* (no `decode_fn`)
  path, and the tie-break rule for when two different sources' raw
  vectors are exactly equal, remain open decisions (see bottom of this
  document) — unaffected by the decode-aware path above.

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

### Optional ParameterSpace wiring (§1a)
When constructed with `parameter_space: ParameterSpace` (default `None`):
- Warm-start batch entries and every round's `merge_candidates` call get
  `decode_fn=parameter_space.decode`.
- The Orchestrator maintains `self._evaluated_configs: set[tuple]`,
  updated every time a result is appended (warm start included) — this is
  what flows into `merge_candidates(evaluated_configs=...)` so a
  configuration evaluated in round 1 is never silently re-evaluated in
  round 3.
- Each round, `propose()` is asked for a deliberately oversampled number
  of candidates (capped, not `n_proposals_per_surrogate` directly) — see
  §5 Notes for why exactly `total_configurations` is not itself enough.
- When `parameter_space` is `None` (every synthetic benchmark), all of the
  above is simply absent — `propose()` is asked for exactly
  `n_proposals_per_surrogate` and `merge_candidates` gets no decode
  arguments, byte-for-byte the pre-existing behavior.

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
- `_MAX_PROPOSAL_OVERSAMPLE` (currently a fixed constant, 500, in
  `orchestrator.py`) is an empirically-chosen cap, not a derived or
  configurable value. Whether it should scale with `total_configurations`,
  be CLI-configurable, or stay a fixed constant is unresolved — it was
  sized to clear the current ML parameter spaces (6-9 configurations)
  reliably, not derived from a formal coverage guarantee.
- There is no formal guarantee that oversampling `propose()` actually
  covers every bucket of a finite `ParameterSpace` — it empirically works
  for the current small discrete spaces (verified across many seeds) but
  relies on each concrete surrogate's `propose()` growing its underlying
  pool when asked for more candidates, which is an implementation detail
  of today's two surrogates, not a `BaseSurrogate` contract guarantee. A
  future surrogate that doesn't grow its pool with `n_candidates` could
  reintroduce spurious `ValueError`s on coarse discrete spaces.
- No `ContinuousDimension` exists in `ParameterSpace` yet (see §1a) —
  only discrete/categorical `ChoiceDimension`. Adding one is deferred
  until a genuinely continuous ML hyperparameter is needed.
- The real (non-smoke-test) classical-ML experiment dataset is not yet
  chosen — `breast_cancer`/`synthetic_classification` in
  `objectives/classical_ml.py` are explicitly labeled smoke-test-only.
  That choice is pending a review of historical profiling artifacts.
