## Architecture Diagram

<img width="1241" height="8191" alt="Parallel multi-surrogate BO architecture" src="https://github.com/user-attachments/assets/7676c0cf-9a1a-466a-80ac-293d0fdedad1" />

## Setting Up the Development Environment

Clone the repository:

```bash
git clone https://github.com/mpr-lab/parallel-multi-surrogate-opt.git
cd parallel-multi-surrogate-opt

Create and actvate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate

Install the project dependencies:
```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt

Run the full test suite:
```bash
python -m pytest -q


Reactivate the environment via:
source .venv/bin/activate


## Running an Experiment
python experiments/run_experiment.py \
    --benchmark branin --pair gp_rf --seed 42 \
    --n-initial 10 --n-rounds 5 --batch-size 6 --workers 1 \
    --output-dir results

This writes `candidates.csv`, `round_metrics.json`, and `metadata.json` to
`results/<run_name>/`.

## Generating Analysis Plots
Plotting is a separate post-processing step: it only reads the files a run
already wrote and never touches the optimization code.

python analysis/plot_run.py --run-dir results/gaussian_process_random_forest_branin_seed42

This writes five PNGs to `results/<run_name>/plots/`:
- `best_objective_vs_evaluations.png`
- `best_objective_vs_wall_time.png`
- `timing_breakdown_by_round.png`
- `surrogate_timing_by_round.png`
- `surrogate_contribution.png`
