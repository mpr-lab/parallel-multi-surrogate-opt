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

**Discrete/categorical workloads (ParameterSpace, `interfaces.md` §1a):**
duplication must be judged on the **decoded configuration**, not the raw
numeric vector — two numerically different proposals that decode to the
same configuration are the same real experiment and must not both be
evaluated. This extends across the whole run, not just one batch: a
configuration already evaluated in an earlier round (or warm start) must
not be silently re-evaluated as if new in a later round. `CandidateMerge`
takes the Orchestrator's running set of already-evaluated decoded
configurations for exactly this reason.

If the finite decoded configuration space has genuinely been exhausted
(every possible configuration already evaluated), the framework must say
so explicitly (`SearchSpaceExhaustedError`) rather than raising a generic
collision error or looping indefinitely trying to find something new.

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

**Classical ML objectives explicitly use one internal CPU thread so that
experiment-level worker count controls parallelism and wall-clock
comparisons are not confounded by library-specific threading defaults.**
Concretely: `RandomForestClassifier(..., n_jobs=1)` and
`XGBClassifier(..., n_jobs=1)` in `objectives/classical_ml.py`. Without
this, sklearn's `n_jobs=None` (1 core) and XGBoost's `n_jobs=None` (all
available cores) would silently give the two objectives a different
amount of hardware parallelism regardless of `--workers`, making any
wall-clock timing comparison between them meaningless.

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
- Random Forest
- XGBoost
- DistilBERT fine-tuning on IMDB (`distilbert_imdb`)

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

### N. Classical-ML / Hugging Face dataset and preprocessing reproducibility

For the classical-ML benchmarks (Logistic Regression, Random Forest,
XGBoost):

- Dataset choice is **explicit** (`--dataset`), never silently assumed by
  an objective function.
- **CPU threading is explicitly single-threaded per objective** (`n_jobs=1`
  on both `RandomForestClassifier` and `XGBClassifier`) — see section J
  for why.
- **`higgs` is now the frozen real-experiment dataset** for Logistic
  Regression, Random Forest, and XGBoost, recovered from the historical
  `hpc-bayes-opt` profiling artifacts (BoTorch Logistic Regression,
  Optuna Logistic Regression, Optuna XGBoost notebooks — all three loaded
  it identically and produced real result CSVs). `breast_cancer` and
  `synthetic_classification` remain smoke-test-only (bundled/synthetic,
  no download) — never treat either as a stand-in for a real run.
- **HIGGS loading recipe** (`objectives/classical_ml.py::_load_higgs`):
  read `HIGGS.csv` (no header, column 0 = label, columns 1: = 28
  features) from `--data-dir`; take a stratified 50,000-row subsample
  (`HIGGS_SUBSAMPLE_SIZE`); split 80/20 train/validation
  (`HIGGS_VALIDATION_FRACTION = 0.2`, the historically-recovered ratio —
  every other dataset keeps this framework's 0.25 smoke-test default).
  One shared loader is used identically by all three classical-ML
  objectives — only Logistic Regression additionally applies a
  `MinMaxScaler` (fit on train, applied to train+val) on top of that
  shared data, matching the historical notebooks; Random Forest/XGBoost
  consume the same split unscaled.
- **Not auto-downloaded.** The historical notebooks used `kagglehub`
  (requires Kaggle API credentials); this framework instead requires
  `HIGGS.csv` to already exist under `--data-dir` and raises a clear,
  actionable error if it doesn't — never a network call during module
  import, test collection, or objective construction.
- **Historical detail recovered but not replicated:** Random Forest was
  never actually run against HIGGS in the historical repo (its BoTorch/
  Optuna/DeepHyper notebooks all used `make_classification` or
  `load_breast_cancer` instead). A DeepHyper template notebook's comments
  stated the intent that the HIGGS pipeline "should be fully reusable for
  xgboost and rf," but this was never executed. Using it for RF now is
  this convention's decision, not a rediscovered historical fact.
- **Scaler type ambiguity:** two different scalers appear across the
  historical artifacts — `MinMaxScaler` (BoTorch/Optuna Logistic
  Regression and Optuna XGBoost, all actually executed) vs
  `StandardScaler` (the incomplete DeepHyper template). `MinMaxScaler`
  was chosen since it's what the actually-executed, result-producing runs
  used.
- The same dataset, preprocessing, and train/validation split must be used
  for every surrogate pair compared on the same benchmark + seed. The
  split's own randomness is derived from a dedicated sub-seed
  (`SeedSequence([experiment_seed, ROLE_DATASET_SPLIT])`), independent of
  warm-start/propose sub-seeds, so dataset-split determinism never
  collides with search-randomness.
- Dataset/model loading happens once per experiment (inside the
  `make_*_objective` factory, called before `Orchestrator.run()`), not
  once per candidate — this is also what keeps setup cost out of
  WorkerPool's per-candidate timing (`interfaces.md` §2 Notes).

For the Hugging Face benchmark (`distilbert_imdb`):

- Model/dataset identity, subset sizes, max token length, and epoch count
  are fixed by the historical spec (`distilbert-base-uncased` / `imdb`,
  2000 train / 1000 eval examples, max length 128, 1 epoch) — not
  silently varied.
- `learning_rate`/`batch_size` use the **exact historical grid**
  (`{1e-5, 3e-5, 1e-4}` x `{8, 16}`), not an approximated or re-derived one.
- Every candidate evaluation starts from a fresh pretrained model (trials
  are independent) — only dataset/tokenizer preparation is reused across
  candidates, never the trained model weights.
- `y = -eval_loss`, per the framework's higher-is-better convention
  (`interfaces.md` §2) — the historical executable returned `eval_loss`
  directly (despite one comment incorrectly mentioning accuracy); this
  framework negates it at the objective boundary like every other
  naturally-minimized benchmark.
- Cache/output directories are explicit CLI/config options
  (`--cache-dir`, `--model-output-dir`), never a hard-coded path — so a
  run can point them at `/work/...` on an HPC cluster without any library
  code knowing that path exists.

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
- ~~The real (non-smoke-test) classical-ML dataset~~ — **resolved**: HIGGS,
  recovered from historical profiling artifacts (see section N).
- ~~Whether the real-evaluation budget for a `ParameterSpace`-backed
  benchmark should be validated against `total_configurations` up
  front~~ — **resolved**: `run_experiment.py` now rejects
  `n_initial + n_rounds * batch_size > parameter_space.total_configurations`
  before building or running the experiment, rather than only discovering
  it via `SearchSpaceExhaustedError` partway through a run.
- The propose() oversample factor (`_PROPOSAL_OVERSAMPLE_FACTOR` in
  `orchestrator.py`, now scaled by `total_configurations` rather than a
  flat constant) still has no formal decode-coverage guarantee for
  arbitrary space sizes — see `interfaces.md` Open Decisions.
