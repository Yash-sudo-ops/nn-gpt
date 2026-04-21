#!/bin/bash
#SBATCH --job-name=tunerl-isolated-run
#SBATCH --partition=gpu_computervision_long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --gres=gpu:2
#SBATCH --signal=TERM@300
#SBATCH --chdir=/home/s471802/nn-gpt
#SBATCH --open-mode=append
#SBATCH --output=/home/s471802/nn-gpt/slurm_logs/%x-%j.out
#SBATCH --error=/home/s471802/nn-gpt/slurm_logs/%x-%j.err

set -euo pipefail

: "${NNGPT_RUN_ID:?Set NNGPT_RUN_ID before sbatch.}"
: "${NNGPT_RUN_ROOT:?Set NNGPT_RUN_ROOT before sbatch.}"
: "${NNGPT_RUN_DOC_PATH:?Set NNGPT_RUN_DOC_PATH before sbatch.}"
: "${NNGPT_SEED_STAGE2_CHECKPOINT:?Set NNGPT_SEED_STAGE2_CHECKPOINT before sbatch.}"

REPO_ROOT="/home/s471802/nn-gpt"
RUN_ROOT="$(python3 -c 'from pathlib import Path; import os; print(Path(os.environ["NNGPT_RUN_ROOT"]).expanduser().resolve())')"
DOC_PATH="$(python3 -c 'from pathlib import Path; import os; print(Path(os.environ["NNGPT_RUN_DOC_PATH"]).expanduser().resolve())')"
RUN_ENV_FILE="${RUN_ROOT}/run_env.sh"
WORK_BASE="/tmp/${USER}/${NNGPT_RUN_ID}"
WORK_DIR="${WORK_BASE}/nn-gpt"
SLURM_DIR="${RUN_ROOT}/slurm"
SOURCE_DIR="${REPO_ROOT}"

mkdir -p "${SLURM_DIR}" "${RUN_ROOT}/out"
source /home/s471802/.venv/bin/activate
cd "${REPO_ROOT}"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
if [ -z "${MASTER_PORT:-}" ]; then
  if [ -n "${SLURM_JOB_ID:-}" ]; then
    export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
  else
    export MASTER_PORT=$((20000 + $$ % 20000))
  fi
fi
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"

rm -rf "${WORK_DIR}"
mkdir -p "${WORK_BASE}"
rsync -a \
  --exclude '.git' \
  --exclude 'grpo_backbone_outputs' \
  --exclude 'rl_output' \
  --exclude 'slurm_logs' \
  --exclude 'parallel_runs' \
  --exclude 'out' \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  "${SOURCE_DIR}/" "${WORK_DIR}/"

rm -rf "${WORK_DIR}/out"
ln -s "${RUN_ROOT}/out" "${WORK_DIR}/out"

if [ -f "${RUN_ENV_FILE}" ]; then
  # shellcheck disable=SC1090
  source "${RUN_ENV_FILE}"
fi

: "${NNGPT_SFT_DATASET_LIMIT:=500}"
: "${NNGPT_SFT_NUM_EPOCHS:=5}"
: "${NNGPT_SFT_SAVE_STEPS:=50}"
: "${NNGPT_SFT_SAVE_TOTAL_LIMIT:=4}"
: "${NNGPT_RL_STAGE1_EVAL_LIMIT_SECONDS:=900}"

export NNGPT_SFT_LOG_DIR="${RUN_ROOT}/rl_output"
export NNGPT_SFT_MODEL_OUT="${RUN_ROOT}/grpo_backbone_outputs"
export NNGPT_SFT_TRAINER_OUT="${RUN_ROOT}/grpo_backbone_outputs_trainer"

mkdir -p "${NNGPT_SFT_LOG_DIR}" "${NNGPT_SFT_MODEL_OUT}" "${NNGPT_SFT_TRAINER_OUT}"

unset NNGPT_SFT_RESUME_TRAINER_CHECKPOINT
unset NNGPT_SFT_RESUME_STAGE_CHECKPOINT
unset NNGPT_RL_RESUME_STAGE

latest_trainer_checkpoint="$(
  find "${NNGPT_SFT_TRAINER_OUT}" -maxdepth 1 -mindepth 1 -type d -name 'checkpoint-*' 2>/dev/null | sort -V | tail -n 1
)"
resume_source=""
resume_mode=""

if [ -n "${latest_trainer_checkpoint}" ]; then
  export NNGPT_SFT_RESUME_TRAINER_CHECKPOINT="${latest_trainer_checkpoint}"
  resume_mode="trainer"
  resume_source="本 run trainer checkpoint: ${latest_trainer_checkpoint}"
elif [ -d "${NNGPT_SFT_MODEL_OUT}/checkpoints/stage3_formal_optimize" ]; then
  export NNGPT_SFT_RESUME_STAGE_CHECKPOINT="${NNGPT_SFT_MODEL_OUT}/checkpoints/stage3_formal_optimize"
  export NNGPT_RL_RESUME_STAGE="stage3_formal_optimize"
  resume_mode="stage"
  resume_source="本 run stage checkpoint: ${NNGPT_SFT_RESUME_STAGE_CHECKPOINT}"
elif [ -d "${NNGPT_SFT_MODEL_OUT}/checkpoints/stage2_formal_explore" ]; then
  export NNGPT_SFT_RESUME_STAGE_CHECKPOINT="${NNGPT_SFT_MODEL_OUT}/checkpoints/stage2_formal_explore"
  export NNGPT_RL_RESUME_STAGE="stage2_formal_explore"
  resume_mode="stage"
  resume_source="本 run stage checkpoint: ${NNGPT_SFT_RESUME_STAGE_CHECKPOINT}"
else
  if [ ! -d "${NNGPT_SEED_STAGE2_CHECKPOINT}" ]; then
    echo "Seed stage2 checkpoint not found: ${NNGPT_SEED_STAGE2_CHECKPOINT}" >&2
    exit 1
  fi
  export NNGPT_SFT_RESUME_STAGE_CHECKPOINT="${NNGPT_SEED_STAGE2_CHECKPOINT}"
  export NNGPT_RL_RESUME_STAGE="stage2_formal_explore"
  resume_mode="seed-stage2"
  resume_source="共享 seed stage2 checkpoint: ${NNGPT_SFT_RESUME_STAGE_CHECKPOINT}"
fi

python3 "${REPO_ROOT}/scripts/julia2_run_archive.py" update \
  --run-root "${RUN_ROOT}" \
  --doc-path "${DOC_PATH}" \
  --mode start \
  --status "运行中" \
  --job-id "${SLURM_JOB_ID:-}" \
  --node "${SLURMD_NODENAME:-${SLURM_JOB_NODELIST:-}}" \
  --start-time "$(date -Is)" \
  --stdout-path "${SLURM_DIR}/${SLURM_JOB_NAME:-job}-${SLURM_JOB_ID:-0}.out" \
  --stderr-path "${SLURM_DIR}/${SLURM_JOB_NAME:-job}-${SLURM_JOB_ID:-0}.err" \
  --work-dir "${WORK_DIR}" \
  --initial-resume-source "${resume_source}"

echo "Run ID: ${NNGPT_RUN_ID}"
echo "Run root: ${RUN_ROOT}"
echo "Work dir: ${WORK_DIR}"
echo "Log dir: ${NNGPT_SFT_LOG_DIR}"
echo "Model out: ${NNGPT_SFT_MODEL_OUT}"
echo "Trainer out: ${NNGPT_SFT_TRAINER_OUT}"
echo "Resume mode: ${resume_mode}"
echo "Resume source: ${resume_source}"
echo "Master addr: ${MASTER_ADDR}"
echo "Master port: ${MASTER_PORT}"

cd "${WORK_DIR}"
exec python -m ab.gpt.TuneRLSft
