---
name: Initial Setup
description: Guide for setting up and initializing the meta-evolution environment for Fractal Networks.
---

# Initial Setup Guide

This skill details the steps required to set up the environment and initialize the meta-evolution process for Fractal Networks.

## 0. Scope Constraints

**CRITICAL**: All modifications must be strictly contained within the `meta_evolution` directory. Do not change files outside this folder under any circumstances.

## 1. Environment Preparation

Before running the evolution script, ensure the environment is correctly configured.

### Dependencies
Ensure all dependencies are installed. Use the `requirements.txt` file located in this `meta_evolution` directory.
```bash
pip install -r ab/gpt/brute/ga/meta_evolution/requirements.txt
```

### PYTHONPATH
The scripts require the project root to be in the python path to resolve imports like `ab.gpt...`.
```bash
export PYTHONPATH=$PYTHONPATH:.
```
*Note: `run_fractal_evolution.py` attempts to add the repo root to `sys.path`, but setting `PYTHONPATH` explicitly is safer.*

### Virtual Environment & `nndataset` (CRITICAL)
**Crucial Warning**: Ensure that the `nndataset` module is correctly installed or linked in the active virtual environment (`venv`). 
**Why this matters**: Evaluation jobs *require* `nndataset` to process data for the models. If this dependency is missing or misconfigured in the environment where evaluation scripts are executed, the pipeline will fail silently or obscurely. This issue is known to have caused weeks of lost debugging time in the past!
Always verify `nndataset` visibility when setting up a new pod or test environment.

## 2. Directory Structure

The `run_fractal_evolution.py` script automatically creates necessary output directories, with strict naming conventions:
- `ga_fractal_arch/`: Stores the generated Python code for each unique fractal model.
  - **Format**: `img-classification_cifar_FractalNet-<FULL_MD5_HASH>.py`
- `stats/`: Stores evaluation statistics for each model within its own subdirectory.
  - **Structure**: `stats/img-classification_cifar_FractalNet-<FULL_MD5_HASH>/`
- `out/`: (Optional) Used for training summaries if available.

## 3. Running the Evolution

### Local Execution
To run the evolution locally (e.g., for debugging):
```bash
python3 ab/gpt/brute/ga/meta_evolution/run_fractal_evolution.py --pop 10 --gens 3
```
- `--pop`: Population size (default: 10)
- `--gens`: Number of generations (default: 3)
- `--clean`: Use this flag to delete existing checkpoints (`fractal_ga_ckpt.pkl`) and start fresh.

### Kubernetes/Cluster Execution
Refer to `Fracta_Evo.json` for the Job configuration. Ensure volume mounts map correctly to your storage:
- `/a/mm`: Maps to the project root.
- `/a/mm/data`: Maps to the dataset directory.

## 4. Cleaning State
To completely reset the experiment:
1. Delete the checkpoint file: `rm fractal_ga_ckpt.pkl` (or use `--clean`).
2. (Optional) Clear output directories if you want to remove old model artifacts:
   ```bash
   rm -rf ga_fractal_arch/* stats/*
   ```

## 5. Configuration
- **Evolution Parameters**: defined in `run_fractal_evolution.py` (mutation rate, elitism).
- **Evaluation Parameters**: defined in `fitness_function` (learning rate, epochs for proxy task).

## 6. Troubleshooting

### Error: `ContainerCreating` / Pod won't start

**Symptom**: `kubectl logs` returns `Error from server (BadRequest): container "ml" is waiting to start: ContainerCreating` for more than 3 minutes.

**Causes & Fixes**:

1. **Wrong volume path** — Cluster nodes require `/shared/ssd/home/...`, not `/home/...`. In `meta_evol_tune_nngpt.json`:
   ```json
   { "name": "v0", "hostPath": { "path": "/shared/ssd/home/b-a-singh/Thesis/nn-gpt", "type": "Directory" } }
   ```
   ⚠️ **IMPORTANT**: Changing `v0` to `/home/b-a-singh/Thesis/nn-gpt` (the local workspace path) always brings back this `ContainerCreating` error — the cluster node `w10-mi` cannot mount that path. **Always keep it as `/shared/ssd/home/...`.**
   Debug with: `kubectl describe pod <pod-name>` → check the **Events** section for `MountVolume.SetUp failed`.

2. **requirements file missing or wrong path** — Use the absolute path inside the container. The correct working `args` command is:
   ```
   cd /a/mm && pip install -r /a/mm/ab/gpt/brute/ga/meta_evolution/meta_requirements.txt && python3 -u -m ab.gpt.brute.ga.meta_evolution.meta_evolver
   ```
   Note: `meta_requirements.txt` must exist in the `meta_evolution` folder (it is a superset of the root `requirements.txt`).

3. **Broken `.venv`** — Do NOT use `source .venv/bin/activate`. The venv may have incompatible compiled extensions (e.g. `flash_attn`). Rely on fresh `pip install` instead.

### Error: `sqlite3.OperationalError: no such table: loader` during Evaluation

**Symptom**: While evaluating the model, the code crashes tracing through `ab/nn/util/db/Read.py` with `sqlite3.OperationalError: no such table: loader`.

**Causes & Fixes**:
This is a classic Python environment mismatch problem tied to how the `ab` module resolves libraries and the active environment.
1. In the error traceback, it typically points to a path like `/home/ab/.local/lib/python3.12/...`. This means the evaluation process is incorrectly picking up a global/user-level Python environment (like Python 3.12) instead of the intended virtual environment (like Python 3.10 inside your `.venv`).
2. The `ab.nn.loader` is searching for its SQLite cache/database in this wrong environment, which hasn't been properly initialized, meaning the `loader` table is literally missing there.
3. **Fix**: Ensure your jobs strictly use the virtual environment's python and site-packages (`/shared/ssd/home/b-a-singh/Thesis/nn-gpt/.venv/lib64/python3.10/site-packages/`). Double-check that your script/container executes using the absolute path to the `.venv` python executable, or verify that your `PYTHONPATH` correctly prioritizes the `.venv` site-packages so it finds `/shared/ssd/home/b-a-singh/Thesis/nn-gpt/.venv/lib64/python3.10/site-packages/ab/nn/loader` instead of falling back.
4. **Alternative Fix**: Pin `nn-dataset` to the exact stable version matching your database (e.g., `pip install nn-dataset==2.2.7`). Installing from `git+https://github.com/ABrain-One/nn-dataset` can pull a newer version with a different DB schema.

---

### Known Trade-off: Where Models Are Saved

There are **two different paths** for the project on this machine:

| Path | Accessible from | Used by |
|------|----------------|---------|
| `/home/b-a-singh/Thesis/nn-gpt` | Local shell / IDE | You (editing code, viewing files) |
| `/shared/ssd/home/b-a-singh/Thesis/nn-gpt` | Cluster nodes (e.g. `w10-mi`) | Kubernetes jobs |

Because the Kubernetes job MUST use `/shared/ssd/home/...` as the volume path, **all generated models, stats, and checkpoints are saved there** — NOT in your local `/home/...` workspace.

**To view models saved by the cluster job, always use:**
```bash
ls /shared/ssd/home/b-a-singh/Thesis/nn-gpt/ab/gpt/brute/ga/meta_evolution/ga_fractal_arch/
```

There is no workaround for this without changing the cluster node configuration. Do not attempt to use `/home/...` as the volume path — it will cause `ContainerCreating`.
