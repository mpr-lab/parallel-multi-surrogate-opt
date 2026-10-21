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

## Running the BoTorch GP+EI Baselines

`baselines/run_botorch_baseline.py` is a standalone sequential GP + Expected
Improvement runner (BoTorch, q=1) for Branin and Hartmann6. It is separate
from the GP+RF multi-surrogate framework -- it does not use Orchestrator,
WorkerPool, DataStore, or any surrogate class. It reuses the exact same
5 warm-start points and y values as the existing `gp_rf` pilot runs for the
same seed (reconstructed via the same deterministic derivation, not
re-randomized), then runs 10 (Branin) or 16 (Hartmann6) further sequential
BO iterations to reach a fixed total of 15 / 21 real evaluations.

This needs BoTorch, which is **not** part of the core `requirements.txt` --
it pulls in `torch`/`gpytorch`, a much heavier dependency than anything the
core framework needs.

### Local setup

```bash
python -m pip install -r requirements-baseline.txt
```

> On platforms where PyPI only ships an older `torch` wheel built against
> NumPy 1.x (e.g. Intel macOS), `requirements-baseline.txt` pins `numpy<2`
> to avoid an ABI crash on import. If your platform has newer `torch`
> wheels available, this pin may not be necessary.

### Running

```bash
python baselines/run_botorch_baseline.py --benchmark branin --seed 42 --output-dir results
python baselines/run_botorch_baseline.py --benchmark hartmann6 --seed 42 --output-dir results
```

This writes `candidates.csv` and `metadata.json` to:
```
results/botorch_gp_ei_branin_seed42/
results/botorch_gp_ei_hartmann6_seed42/
```

### Running on Unity

Each run completes in well under a minute (measured locally: ~2-10s), so an
interactive session is sufficient -- no need for a long-running batch job.

```bash
# from a login node
srun --partition=cpu --time=00:10:00 --mem=4G --cpus-per-task=2 --pty bash

# inside the interactive session
module load python/3.11.7   # adjust to whatever Python module your account has available
cd /path/to/parallel-multi-surrogate-opt
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-baseline.txt

python baselines/run_botorch_baseline.py --benchmark branin --seed 42 --output-dir results
python baselines/run_botorch_baseline.py --benchmark hartmann6 --seed 42 --output-dir results
```

If you'd rather submit a batch job than hold an interactive shell, a minimal
`sbatch` script:

```bash
#!/bin/bash
#SBATCH --job-name=botorch-gp-ei-baseline
#SBATCH --partition=cpu
#SBATCH --time=00:10:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2
#SBATCH --output=botorch_baseline_%j.log

module load python/3.11.7   # adjust to your account's available module
source .venv/bin/activate   # venv created ahead of time, as above

python baselines/run_botorch_baseline.py --benchmark branin --seed 42 --output-dir results
python baselines/run_botorch_baseline.py --benchmark hartmann6 --seed 42 --output-dir results
```

Submit with `sbatch run_botorch_baseline.slurm`. The exact partition name,
module name/version, and resource limits above are a standard starting
point -- check `module avail python` and `sinfo` on your account, since
Unity's available partitions/modules can differ by group allocation.
