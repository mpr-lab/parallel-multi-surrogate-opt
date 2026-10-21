# Pilot Validation Experiment Protocol

## Purpose

The goal of this pilot is to validate the rebuilt parallel multi-surrogate
Bayesian optimization framework before running a larger experimental study.

These runs are not intended to establish the final research result or evaluate
adaptive allocation. They are intended to verify that the GP + RF framework:

- runs correctly on Unity,
- produces reproducible outputs,
- records the expected optimization and timing information,
- behaves sensibly on known synthetic benchmark functions,
- and can later be used under a standardized experimental methodology.

The pilot begins with Branin and then Hartmann6.

---

## Experimental Environment

Run the experiments on the Unity HPC platform using a dedicated and consistent
JupyterLab session.

### Unity resource request

Use the following resource configuration for all pilot runs:

- **Partition:** `ood-shared,cpu-preempt,cpu`
- **Maximum job duration:** 2 hours
- **CPU cores:** 4
- **Memory:** 16 GB
- **GPU:** none
- **JupyterLab:** use the same version across all pilot runs
- **Modules:** none unless explicitly required
- **Debug mode:** off
- **Short QoS:** optional for jobs under four hours

These values should remain fixed throughout the pilot so that runtime
measurements are collected under the same resource allocation.

The exact Unity configuration used for each run should also be recorded in the
experiment notes.

> Note: Unity CPU allocation and framework worker count are different settings.
> The Unity session requests 4 CPU cores, while the initial framework runs use
> `--workers 1` so that framework correctness can be validated before testing
> parallel scaling.

---

## Software Setup

Before running experiments:

```bash
git pull
source .venv/bin/activate
python -m pytest -q
