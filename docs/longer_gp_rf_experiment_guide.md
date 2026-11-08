# Longer GP + RF Experiment Sweep

## Goal

The goal of this sweep is to repeat the same GP + Random Forest experiments I already ran, but with a longer optimization trajectory.

The question we are trying to answer is:

> If GP and Random Forest are allowed to run for longer, does one surrogate consistently dominate, or do they contribute useful candidates at different points in the optimization?

This is separate from the three-surrogate/TPE work. I am working on that side while Nargiz runs the longer GP + RF experiments.

---

## What I already ran

I ran the following seven workloads with the same GP + Random Forest setup:

1. Branin
2. Hartmann6
3. Rastrigin
4. Logistic Regression on HIGGS
5. Random Forest on HIGGS
6. XGBoost on HIGGS
7. DistilBERT on IMDB

The short matched sweep used:

```text
seed = 42
n_initial = 5
n_rounds = 5
batch_size = 2
workers = 1
pair = gp_rf
```

That gives:

```text
5 warm-start evaluations
+ 5 rounds × 2 candidates per round
= 15 total evaluations
```

After the warm start, each round evaluates:

- 1 candidate proposed by the Gaussian Process
- 1 candidate proposed by the Random Forest surrogate

So in the 15-evaluation version:

- random warm start = 5 candidates
- GP = 5 candidates
- RF = 5 candidates

---

## Longer-run setup

For the longer experiment, keep **everything the same** except the number of rounds.

Use:

```text
seed = 42
n_initial = 5
n_rounds = 25
batch_size = 2
workers = 1
pair = gp_rf
```

That gives:

```text
5 warm-start evaluations
+ 25 rounds × 2 candidates per round
= 55 total evaluations
```

So GP and RF each get **25 post-warm-start candidates**.

I think 55 evaluations is a good first longer-run budget because it is much longer than the original 15-evaluation experiment, but still realistic enough to run across all seven workloads.

The main thing we want to inspect is whether the surrogate-contribution pattern changes over time.

---

## Important: do not change anything else

Please keep the following fixed:

- seed = 42
- 5 warm-start points
- batch size = 2
- one GP candidate + one RF candidate per round
- workers = 1
- the same benchmark/search spaces already in the repo
- the same datasets and train/eval splits
- the same objective functions
- the same single-thread controls for the classical ML objectives
- the same plotting/measurement code

The only intended experiment change is:

```text
n_rounds: 5 -> 25
```

This is important because we want to compare the 15-evaluation and 55-evaluation runs cleanly.

---

# Environment setup

From Unity:

```bash
cd ~/parallel-multi-surrogate-opt
git pull
```

Activate the normal project environment for the synthetic and classical ML experiments:

```bash
source .venv/bin/activate
```

For HIGGS:

```bash
HIGGS_DIR=/work/pi_mrobson_smith_edu/moumer_smith_edu/kagglehub-cache/datasets/erikbiswas/higgs-uci-dataset/versions/1
```

Use a separate output folder so the longer runs do not overwrite the original 15-evaluation results:

```bash
RESULTS=results/long_55eval_seed42
mkdir -p "$RESULTS"
```

---

# 1. Branin

```bash
python experiments/run_experiment.py \
  --benchmark branin \
  --pair gp_rf \
  --seed 42 \
  --n-initial 5 \
  --n-rounds 25 \
  --batch-size 2 \
  --workers 1 \
  --output-dir "$RESULTS"
```

---

# 2. Hartmann6

```bash
python experiments/run_experiment.py \
  --benchmark hartmann6 \
  --pair gp_rf \
  --seed 42 \
  --n-initial 5 \
  --n-rounds 25 \
  --batch-size 2 \
  --workers 1 \
  --output-dir "$RESULTS"
```

---

# 3. Rastrigin

```bash
python experiments/run_experiment.py \
  --benchmark rastrigin \
  --pair gp_rf \
  --seed 42 \
  --n-initial 5 \
  --n-rounds 25 \
  --batch-size 2 \
  --workers 1 \
  --output-dir "$RESULTS"
```

---

# 4. Logistic Regression

```bash
python experiments/run_experiment.py \
  --benchmark logistic_regression_expanded \
  --pair gp_rf \
  --seed 42 \
  --dataset higgs \
  --data-dir "$HIGGS_DIR" \
  --n-initial 5 \
  --n-rounds 25 \
  --batch-size 2 \
  --workers 1 \
  --output-dir "$RESULTS"
```

---

# 5. Random Forest

```bash
python experiments/run_experiment.py \
  --benchmark random_forest_expanded \
  --pair gp_rf \
  --seed 42 \
  --dataset higgs \
  --data-dir "$HIGGS_DIR" \
  --n-initial 5 \
  --n-rounds 25 \
  --batch-size 2 \
  --workers 1 \
  --output-dir "$RESULTS"
```

This one is much slower than the other classical ML workloads because some objective configurations train hundreds of trees.

That is expected.

---

# 6. XGBoost

```bash
python experiments/run_experiment.py \
  --benchmark xgboost_expanded \
  --pair gp_rf \
  --seed 42 \
  --dataset higgs \
  --data-dir "$HIGGS_DIR" \
  --n-initial 5 \
  --n-rounds 25 \
  --batch-size 2 \
  --workers 1 \
  --output-dir "$RESULTS"
```

---

# 7. DistilBERT + IMDB

DistilBERT uses a separate environment because the Hugging Face dependency stack conflicts with the environment used for the other experiments.

Deactivate the normal environment if necessary:

```bash
deactivate
```

Activate the Hugging Face environment:

```bash
source /work/pi_mrobson_smith_edu/moumer_smith_edu/venvs/hf-distilbert/bin/activate
```

Set the Hugging Face cache location:

```bash
export HF_HOME=/work/pi_mrobson_smith_edu/moumer_smith_edu/huggingface
export HF_DATASETS_CACHE=$HF_HOME/datasets
mkdir -p "$HF_HOME"
```

Then run:

```bash
python experiments/run_experiment.py \
  --benchmark distilbert_imdb_expanded \
  --pair gp_rf \
  --seed 42 \
  --n-initial 5 \
  --n-rounds 25 \
  --batch-size 2 \
  --workers 1 \
  --train-subset-size 2000 \
  --eval-subset-size 1000 \
  --output-dir results/long_55eval_seed42
```

DistilBERT is very expensive on CPU. The original 15-evaluation run took roughly two hours, so the 55-evaluation run may take many hours.

For this one, it is better to use a long-running Unity/Slurm job rather than relying on a short interactive session.

---

# Plotting

After each run completes, generate the same plots used for the original sweep.

For example:

```bash
python analysis/plot_run_no_eval.py \
  --run-dir results/long_55eval_seed42/<RUN_DIRECTORY>
```

The run directory names should follow the same naming convention as the original experiments.

You can find them with:

```bash
find results/long_55eval_seed42 -maxdepth 1 -type d | sort
```

The main plots we care about are:

- best objective vs. evaluations
- best objective vs. wall-clock time
- surrogate contribution
- surrogate fit/proposal timing
- surrogate overhead breakdown

---

# What to look for

The most important thing is **surrogate contribution over a longer trajectory**.

The original 15-evaluation experiments only gave GP and RF five proposed candidates each.

With 55 total evaluations, they now each get 25 chances.

We want to see whether:

- GP continues to dominate on workloads like Branin/Rastrigin
- RF continues to dominate or become useful on workloads like Hartmann
- both models continue contributing on the ML workloads
- one surrogate contributes mainly early and the other later
- contribution alternates between them
- one surrogate becomes consistently unhelpful over time

The current contribution plot only shows total counts. If useful, we may later add a contribution-by-round plot so we can see *when* each surrogate produces improvements instead of only the final total.

---

# Results to save

Please do not overwrite the original matched 15-evaluation results.

Original short sweep:

```text
results/matched_15eval_seed42/
```

Longer sweep:

```text
results/long_55eval_seed42/
```

For each completed workload, keep:

- candidates.csv
- round/timing metrics
- metadata
- generated plots

---

# Short summary of the experiment

The original experiment asked:

> Does GP or RF contribute more useful candidates across different workload types?

The longer experiment asks:

> If we give both surrogates substantially more opportunities, does that preference remain stable or change over time?

The only intended change from the original matched sweep is the number of optimization rounds:

```text
5 rounds -> 25 rounds
15 total evaluations -> 55 total evaluations
```
