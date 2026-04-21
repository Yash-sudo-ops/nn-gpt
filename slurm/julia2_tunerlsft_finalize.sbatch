#!/bin/bash
#SBATCH --job-name=tunerl-finalize
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --chdir=/home/s471802/nn-gpt
#SBATCH --open-mode=append
#SBATCH --output=/home/s471802/nn-gpt/slurm_logs/%x-%j.out
#SBATCH --error=/home/s471802/nn-gpt/slurm_logs/%x-%j.err

set -euo pipefail

: "${NNGPT_RUN_ROOT:?Set NNGPT_RUN_ROOT before sbatch.}"
: "${NNGPT_RUN_DOC_PATH:?Set NNGPT_RUN_DOC_PATH before sbatch.}"

REPO_ROOT="/home/s471802/nn-gpt"
RUN_ROOT="$(python3 -c 'from pathlib import Path; import os; print(Path(os.environ["NNGPT_RUN_ROOT"]).expanduser().resolve())')"
DOC_PATH="$(python3 -c 'from pathlib import Path; import os; print(Path(os.environ["NNGPT_RUN_DOC_PATH"]).expanduser().resolve())')"

source /home/s471802/.venv/bin/activate
cd "${REPO_ROOT}"

python3 "${REPO_ROOT}/scripts/julia2_run_archive.py" update \
  --run-root "${RUN_ROOT}" \
  --doc-path "${DOC_PATH}" \
  --mode finish \
  --job-id "${NNGPT_MAIN_JOB_ID:-}" \
  --end-time "$(date -Is)"
