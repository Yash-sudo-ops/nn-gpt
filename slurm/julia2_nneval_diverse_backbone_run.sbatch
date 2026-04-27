#!/bin/bash
#SBATCH --job-name=nneval-bb-diverse
#SBATCH --partition=gpu_computervision
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --chdir=/home/s471802/nn-gpt
#SBATCH --output=/home/s471802/nn-gpt/slurm_logs/%x-%j.out
#SBATCH --error=/home/s471802/nn-gpt/slurm_logs/%x-%j.err

set -euo pipefail

mkdir -p /home/s471802/nn-gpt/slurm_logs
source /home/s471802/.venv/bin/activate
cd /home/s471802/nn-gpt

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="/home/s471802/nn-dataset:/home/s471802/nn-gpt:${PYTHONPATH:-}"

runtime_nn="/home/s471802/nn-gpt/ab/nn"
runtime_backups=()
runtime_links=()
for component in loader transform metric; do
  src="/home/s471802/nn-dataset/ab/nn/${component}"
  dst="${runtime_nn}/${component}"
  backup="${dst}.nneval_backup_${SLURM_JOB_ID:-$$}"
  if [ -L "${dst}" ]; then
    rm "${dst}"
  elif [ -e "${dst}" ]; then
    mv "${dst}" "${backup}"
    runtime_backups+=("${dst}:${backup}")
  fi
  ln -s "${src}" "${dst}"
  runtime_links+=("${dst}")
done

cleanup_runtime_links() {
  for dst in "${runtime_links[@]}"; do
    if [ -L "${dst}" ]; then
      rm "${dst}"
    fi
  done
  for pair in "${runtime_backups[@]}"; do
    dst="${pair%%:*}"
    backup="${pair#*:}"
    if [ -e "${backup}" ] && [ ! -e "${dst}" ]; then
      mv "${backup}" "${dst}"
    fi
  done
}
trap cleanup_runtime_links EXIT

export NNGPT_NNEVAL_MAX_WORKERS_PER_GPU="${NNGPT_NNEVAL_MAX_WORKERS_PER_GPU:-2}"
export NNGPT_NNEVAL_WORKER_BUDGET_GIB="${NNGPT_NNEVAL_WORKER_BUDGET_GIB:-8}"
export NNGPT_NNEVAL_RESERVED_HEADROOM_GIB="${NNGPT_NNEVAL_RESERVED_HEADROOM_GIB:-3}"

variants_per_pattern="${NNGPT_DIVERSE_BB_VARIANTS_PER_PATTERN:-20}"
prefix="${NNGPT_DIVERSE_BB_PREFIX:-rl-bb-struct1}"

python -m ab.gpt.brute.fract.backbone.NNAlterBackboneDiverse \
  --epochs 1 \
  --variants-per-pattern "${variants_per_pattern}"

exec python -m ab.gpt.NNEval \
  --only_epoch 0 \
  --nn_train_epochs 1 \
  --nn_name_prefix "${prefix}" \
  --transform norm_128_flip \
  --save_to_db
