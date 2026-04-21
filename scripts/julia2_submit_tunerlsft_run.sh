#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DOC_PATH="${REPO_ROOT}/run_archive_index.md"
SEED_STAGE2_CHECKPOINT_DEFAULT="${REPO_ROOT}/grpo_backbone_outputs/sft/checkpoints/stage2_formal_explore"

RUN_LABEL="formal10"
RUN_ID=""
RUN_NOTE=""
COMMIT_HASH=""
COMMIT_SUBJECT=""
PARTITION="gpu_computervision_long"
QOS=""
FINALIZE_PARTITION=""
FINALIZE_QOS=""
TIME_LIMIT=""
GPU_COUNT=""
MEMORY=""
CPUS=""
JOB_NAME=""
SEED_STAGE2_CHECKPOINT="${SEED_STAGE2_CHECKPOINT_DEFAULT}"
ENV_OVERRIDES=()

usage() {
  cat <<'EOF'
Usage:
  scripts/julia2_submit_tunerlsft_run.sh --run-note TEXT [options]

Options:
  --run-note TEXT              必填，本次 job 的改动说明
  --run-label LABEL            可读短标签，默认 formal10
  --run-id RUN_ID              指定已有 run_id；不传则自动生成
  --commit-hash HASH           可选，归档文档写入指定 commit hash
  --commit-subject TEXT        可选，归档文档写入指定 commit subject
  --partition PARTITION        sbatch 分区，默认 gpu_computervision_long
  --qos QOS                    可选 QoS
  --finalize-partition P       finalize job 分区，默认跟主 job 一致
  --finalize-qos QOS           finalize job QoS，默认跟主 job 一致
  --time HH:MM:SS              可选 time limit
  --gpus N                     可选 GPU 数
  --mem SIZE                   可选内存，如 64G
  --cpus N                     可选 cpus-per-task
  --job-name NAME              可选 job name
  --seed-stage2-checkpoint P   可选共享 stage2 checkpoint 路径
  --env NAME=VALUE             追加训练环境变量，可重复
  --help                       显示帮助

Examples:
  scripts/julia2_submit_tunerlsft_run.sh \
    --run-note '验证 10 epoch 正式脚本' \
    --partition gpu_computervision \
    --qos normal \
    --gpus 1 \
    --time 02:00:00 \
    --env NNGPT_SFT_DATASET_LIMIT=1 \
    --env NNGPT_SFT_NUM_GENERATIONS=2
EOF
}

slugify() {
  local raw="$1"
  local lowered
  lowered="$(printf '%s' "${raw}" | tr '[:upper:]' '[:lower:]')"
  lowered="$(printf '%s' "${lowered}" | sed -E 's/[^a-z0-9]+/_/g; s/^_+//; s/_+$//; s/_+/_/g')"
  if [[ -z "${lowered}" ]]; then
    lowered="run"
  fi
  printf '%s\n' "${lowered}"
}

run_id_exists() {
  local candidate="$1"
  if [[ -e "${REPO_ROOT}/parallel_runs/${candidate}" ]]; then
    return 0
  fi
  if [[ -f "${DOC_PATH}" ]] && grep -Fq "<!-- NNGPT_RUN:${candidate}:START -->" "${DOC_PATH}"; then
    return 0
  fi
  return 1
}

generate_run_id() {
  local label_slug="$1"
  local ts base candidate suffix
  ts="$(date '+%Y%m%d_%H%M')"
  base="${ts}_${label_slug}"
  candidate="${base}"
  suffix=2
  while run_id_exists "${candidate}"; do
    candidate="${base}_v${suffix}"
    suffix=$((suffix + 1))
  done
  printf '%s\n' "${candidate}"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-note)
      RUN_NOTE="$2"
      shift 2
      ;;
    --run-label)
      RUN_LABEL="$2"
      shift 2
      ;;
    --run-id)
      RUN_ID="$2"
      shift 2
      ;;
    --commit-hash)
      COMMIT_HASH="$2"
      shift 2
      ;;
    --commit-subject)
      COMMIT_SUBJECT="$2"
      shift 2
      ;;
    --partition)
      PARTITION="$2"
      shift 2
      ;;
    --qos)
      QOS="$2"
      shift 2
      ;;
    --finalize-partition)
      FINALIZE_PARTITION="$2"
      shift 2
      ;;
    --finalize-qos)
      FINALIZE_QOS="$2"
      shift 2
      ;;
    --time)
      TIME_LIMIT="$2"
      shift 2
      ;;
    --gpus)
      GPU_COUNT="$2"
      shift 2
      ;;
    --mem)
      MEMORY="$2"
      shift 2
      ;;
    --cpus)
      CPUS="$2"
      shift 2
      ;;
    --job-name)
      JOB_NAME="$2"
      shift 2
      ;;
    --seed-stage2-checkpoint)
      SEED_STAGE2_CHECKPOINT="$2"
      shift 2
      ;;
    --env)
      ENV_OVERRIDES+=("$2")
      shift 2
      ;;
    --help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "${RUN_NOTE}" ]]; then
  echo "--run-note is required." >&2
  usage >&2
  exit 1
fi

RUN_LABEL_SLUG="$(slugify "${RUN_LABEL}")"
if [[ -z "${RUN_ID}" ]]; then
  RUN_ID="$(generate_run_id "${RUN_LABEL_SLUG}")"
fi
if [[ -z "${FINALIZE_PARTITION}" ]]; then
  FINALIZE_PARTITION="${PARTITION}"
fi
if [[ -z "${FINALIZE_QOS}" ]]; then
  FINALIZE_QOS="${QOS}"
fi

RUN_ROOT="${REPO_ROOT}/parallel_runs/${RUN_ID}"
SLURM_DIR="${RUN_ROOT}/slurm"
MAIN_SCRIPT="${REPO_ROOT}/slurm/julia2_tunerlsft_isolated_run.sbatch"
FINALIZE_SCRIPT="${REPO_ROOT}/slurm/julia2_tunerlsft_finalize.sbatch"
RUN_ENV_FILE="${RUN_ROOT}/run_env.sh"

mkdir -p "${RUN_ROOT}" "${SLURM_DIR}" "${RUN_ROOT}/out"

if [[ ! -d "${SEED_STAGE2_CHECKPOINT}" ]]; then
  echo "Seed stage2 checkpoint not found: ${SEED_STAGE2_CHECKPOINT}" >&2
  exit 1
fi

if [[ -z "${COMMIT_HASH}" ]]; then
  COMMIT_HASH="$(git -C "${REPO_ROOT}" rev-parse HEAD)"
fi
if [[ -z "${COMMIT_SUBJECT}" ]]; then
  COMMIT_SUBJECT="$(git -C "${REPO_ROOT}" log -1 --pretty=%s)"
fi
submit_time="$(date -Is)"

{
  echo "#!/bin/bash"
  echo "set -euo pipefail"
  for entry in "${ENV_OVERRIDES[@]}"; do
    if [[ "${entry}" != *=* ]]; then
      echo "Invalid --env entry: ${entry}" >&2
      exit 1
    fi
    name="${entry%%=*}"
    value="${entry#*=}"
    if [[ ! "${name}" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
      echo "Invalid environment variable name: ${name}" >&2
      exit 1
    fi
    printf 'export %s=%q\n' "${name}" "${value}"
  done
} > "${RUN_ENV_FILE}"
chmod 700 "${RUN_ENV_FILE}"

stdout_path="${SLURM_DIR}/${JOB_NAME:-tunerl-${RUN_LABEL_SLUG}}-%j.out"
stderr_path="${SLURM_DIR}/${JOB_NAME:-tunerl-${RUN_LABEL_SLUG}}-%j.err"

python3 "${REPO_ROOT}/scripts/julia2_run_archive.py" init \
  --run-root "${RUN_ROOT}" \
  --doc-path "${DOC_PATH}" \
  --run-id "${RUN_ID}" \
  --run-label "${RUN_LABEL}" \
  --run-note "${RUN_NOTE}" \
  --commit-hash "${COMMIT_HASH}" \
  --commit-subject "${COMMIT_SUBJECT}" \
  --submit-time "${submit_time}" \
  --status "已提交" \
  --partition "${PARTITION}" \
  --qos "${QOS:-}" \
  --stdout-path "${stdout_path/\%j/<jobid>}" \
  --stderr-path "${stderr_path/\%j/<jobid>}" \
  --seed-stage2-checkpoint "${SEED_STAGE2_CHECKPOINT}"

sbatch_args=(
  --export="ALL,NNGPT_RUN_ID=${RUN_ID},NNGPT_RUN_ROOT=${RUN_ROOT},NNGPT_RUN_DOC_PATH=${DOC_PATH},NNGPT_SEED_STAGE2_CHECKPOINT=${SEED_STAGE2_CHECKPOINT}"
  --open-mode append
  --output "${stdout_path}"
  --error "${stderr_path}"
)

if [[ -n "${PARTITION}" ]]; then
  sbatch_args+=(-p "${PARTITION}")
fi
if [[ -n "${QOS}" ]]; then
  sbatch_args+=(--qos "${QOS}")
fi
if [[ -n "${TIME_LIMIT}" ]]; then
  sbatch_args+=(--time "${TIME_LIMIT}")
fi
if [[ -n "${GPU_COUNT}" ]]; then
  sbatch_args+=(--gres "gpu:${GPU_COUNT}")
fi
if [[ -n "${MEMORY}" ]]; then
  sbatch_args+=(--mem "${MEMORY}")
fi
if [[ -n "${CPUS}" ]]; then
  sbatch_args+=(--cpus-per-task "${CPUS}")
fi
if [[ -n "${JOB_NAME}" ]]; then
  sbatch_args+=(--job-name "${JOB_NAME}")
else
  sbatch_args+=(--job-name "tunerl-${RUN_LABEL_SLUG}")
fi

main_submit_output="$(sbatch "${sbatch_args[@]}" "${MAIN_SCRIPT}")"
main_job_id="$(printf '%s\n' "${main_submit_output}" | awk '/Submitted batch job/ {print $4}' | tail -n 1)"
if [[ -z "${main_job_id}" ]]; then
  echo "Failed to parse main job id from sbatch output: ${main_submit_output}" >&2
  exit 1
fi

resolved_stdout_path="${stdout_path/\%j/${main_job_id}}"
resolved_stderr_path="${stderr_path/\%j/${main_job_id}}"

python3 "${REPO_ROOT}/scripts/julia2_run_archive.py" update \
  --run-root "${RUN_ROOT}" \
  --doc-path "${DOC_PATH}" \
  --mode manual \
  --job-id "${main_job_id}" \
  --status "已提交" \
  --partition "${PARTITION}" \
  --qos "${QOS:-}" \
  --stdout-path "${resolved_stdout_path}" \
  --stderr-path "${resolved_stderr_path}"

finalize_submit_output="$(
  {
    finalize_args=(
      --dependency "afterany:${main_job_id}"
      -p "${FINALIZE_PARTITION}"
      --open-mode append
      --job-name "tunerl-finalize-${RUN_LABEL_SLUG}"
      --output "${SLURM_DIR}/tunerl-finalize-${RUN_LABEL_SLUG}-${main_job_id}.out"
      --error "${SLURM_DIR}/tunerl-finalize-${RUN_LABEL_SLUG}-${main_job_id}.err"
      --export "ALL,NNGPT_RUN_ROOT=${RUN_ROOT},NNGPT_RUN_DOC_PATH=${DOC_PATH},NNGPT_MAIN_JOB_ID=${main_job_id}"
    )
    if [[ -n "${FINALIZE_QOS}" ]]; then
      finalize_args+=(--qos "${FINALIZE_QOS}")
    fi
    sbatch "${finalize_args[@]}" "${FINALIZE_SCRIPT}"
  }
)"
finalize_job_id="$(printf '%s\n' "${finalize_submit_output}" | awk '/Submitted batch job/ {print $4}' | tail -n 1)"

printf 'run_id=%s\n' "${RUN_ID}"
printf 'main_job_id=%s\n' "${main_job_id}"
printf 'finalize_job_id=%s\n' "${finalize_job_id:-unknown}"
printf 'run_root=%s\n' "${RUN_ROOT}"
printf 'doc_path=%s\n' "${DOC_PATH}"
