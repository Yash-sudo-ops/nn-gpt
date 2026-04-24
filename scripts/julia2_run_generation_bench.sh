#!/bin/bash
#SBATCH --job-name=genbench-driver
#SBATCH --partition=small_cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --open-mode=append
#SBATCH --output=/home/s471802/nn-gpt/slurm_logs/genbench-driver-%j.out
#SBATCH --error=/home/s471802/nn-gpt/slurm_logs/genbench-driver-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${NNGPT_REPO_ROOT:-}" ]]; then
  REPO_ROOT="$(cd "${NNGPT_REPO_ROOT}" && pwd)"
else
  REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
  if [[ ! -d "${REPO_ROOT}/.git" && -d /home/s471802/nn-gpt/.git ]]; then
    REPO_ROOT=/home/s471802/nn-gpt
  fi
fi
BENCH_TS="$(date '+%Y%m%d_%H%M%S')"
BENCH_ID="${NNGPT_GENBENCH_ID:-genbench_${BENCH_TS}}"
BENCH_ROOT="${REPO_ROOT}/parallel_runs/${BENCH_ID}"
SUMMARY_JSON="${BENCH_ROOT}/summary.json"
SUMMARY_MD="${BENCH_ROOT}/summary.md"
VLLM_VENV="${NNGPT_GENBENCH_VLLM_VENV:-/home/s471802/.venv-vllm-smoke}"
BASE_VENV="${NNGPT_GENBENCH_BASE_VENV:-/home/s471802/.venv}"
PARTITION="${NNGPT_GENBENCH_PARTITION:-h100}"
QOS="${NNGPT_GENBENCH_QOS:-normal}"
GPUS="${NNGPT_GENBENCH_GPUS:-2}"
CPUS="${NNGPT_GENBENCH_CPUS:-16}"
MEM="${NNGPT_GENBENCH_MEM:-80G}"
MAX_STEPS="${NNGPT_GENBENCH_MAX_STEPS:-20}"
DATASET_LIMIT="${NNGPT_GENBENCH_DATASET_LIMIT:-96}"
SEED_STAGE2_CHECKPOINT="${NNGPT_GENBENCH_SEED_STAGE2_CHECKPOINT:-${REPO_ROOT}/rl_backbone_model_sft/checkpoints/stage2_formal_explore}"
VLLM_READY=0
PREVIOUS_MAIN_JOB=""

mkdir -p "${BENCH_ROOT}" "${REPO_ROOT}/slurm_logs"

ensure_vllm_venv() {
  if [[ -x "${VLLM_VENV}/bin/python" ]] && "${VLLM_VENV}/bin/python" - <<'PYVLLM' >/dev/null 2>&1
import vllm
PYVLLM
  then
    return 0
  fi
  if [[ ! -d "${VLLM_VENV}" ]]; then
    "${BASE_VENV}/bin/python" -m venv "${VLLM_VENV}" --copies
  fi
  "${VLLM_VENV}/bin/python" -m pip install --upgrade pip
  "${VLLM_VENV}/bin/python" -m pip install -r <("${BASE_VENV}/bin/python" -m pip freeze)
  "${VLLM_VENV}/bin/python" -m pip install 'vllm==0.19.1'
  "${VLLM_VENV}/bin/python" - <<'PYVLLM'
import vllm
print('vllm_import_ok')
PYVLLM
}

submit_case() {
  local name="$1"
  local max_completion_length="$2"
  local use_vllm="$3"
  local venv_path="${BASE_VENV}"
  local run_id="${BENCH_ID}_${name}"
  local commit_hash commit_subject submit_output main_job finalize_job
  commit_hash="$(git -C "${REPO_ROOT}" rev-parse HEAD)"
  commit_subject="$(git -C "${REPO_ROOT}" log -1 --pretty=%s)"
  local args=(
    --run-id "${run_id}"
    --run-label "${name}"
    --run-note "generation bench ${name}"
    --commit-hash "${commit_hash}"
    --commit-subject "${commit_subject}"
    --partition "${PARTITION}"
    --qos "${QOS}"
    --finalize-partition small_cpu
    --gpus "${GPUS}"
    --mem "${MEM}"
    --cpus "${CPUS}"
    --seed-stage2-checkpoint "${SEED_STAGE2_CHECKPOINT}"
    --env "NNGPT_SFT_NUM_GENERATIONS=8"
    --env "NNGPT_SFT_MAX_STEPS=${MAX_STEPS}"
    --env "NNGPT_SFT_NUM_EPOCHS=1"
    --env "NNGPT_SFT_DATASET_LIMIT=${DATASET_LIMIT}"
    --env "NNGPT_RL_FORMAL_REWARD_EPOCHS=1"
    --env "NNGPT_SFT_SAVE_STEPS=1000"
    --env "NNGPT_SFT_SAVE_TOTAL_LIMIT=1"
    --env "NNGPT_BENCH_GPU_MONITOR=1"
    --env "NNGPT_BENCH_GPU_MONITOR_INTERVAL_SECONDS=15"
  )
  if [[ -n "${PREVIOUS_MAIN_JOB}" ]]; then
    args+=(--dependency "afterany:${PREVIOUS_MAIN_JOB}")
  fi
  if [[ "${max_completion_length}" != "1536" ]]; then
    args+=(--env "NNGPT_SFT_MAX_COMPLETION_LENGTH=${max_completion_length}")
  fi
  if [[ "${use_vllm}" == "1" ]]; then
    venv_path="${VLLM_VENV}"
    args+=(
      --venv "${venv_path}"
      --env "NNGPT_SFT_USE_VLLM=1"
      --env "NNGPT_SFT_VLLM_MODE=colocate"
      --env "NNGPT_SFT_VLLM_GPU_MEMORY_UTILIZATION=0.25"
      --env "NNGPT_SFT_VLLM_ENABLE_SLEEP_MODE=1"
      --env "NNGPT_SFT_VLLM_TENSOR_PARALLEL_SIZE=1"
    )
  fi
  submit_output="$("${REPO_ROOT}/scripts/julia2_submit_tunerlsft_run.sh" "${args[@]}")"
  printf '%s\n' "${submit_output}" | tee "${BENCH_ROOT}/${name}_submit.txt"
  main_job="$(printf '%s\n' "${submit_output}" | awk -F= '/^main_job_id=/{print $2}')"
  finalize_job="$(printf '%s\n' "${submit_output}" | awk -F= '/^finalize_job_id=/{print $2}')"
  if [[ -z "${main_job}" ]]; then
    echo "Failed to parse main job id for ${name}" >&2
    exit 1
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${name}" "${run_id}" "${main_job}" "${finalize_job}" "${venv_path}" "${max_completion_length}" "${use_vllm}" >> "${BENCH_ROOT}/cases.tsv"
  PREVIOUS_MAIN_JOB="${main_job}"
}

wait_cases() {
  local jobs
  if [[ ! -f "${BENCH_ROOT}/cases.tsv" ]]; then
    return 0
  fi
  while IFS=$'\t' read -r name run_id main_job finalize_job venv_path max_completion_length use_vllm; do
    [[ -n "${name}" ]] || continue
    jobs="${main_job}"
    if [[ "${finalize_job}" =~ ^[0-9]+$ ]]; then
      jobs="${jobs},${finalize_job}"
    fi
    while squeue -j "${jobs}" -h 2>/dev/null | grep -q .; do
      sleep 60
    done
  done < "${BENCH_ROOT}/cases.tsv"
}

write_summary() {
  "${BASE_VENV}/bin/python" - "${BENCH_ROOT}" "${SUMMARY_JSON}" "${SUMMARY_MD}" <<'PYCODE'
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path

bench_root = Path(sys.argv[1])
summary_json = Path(sys.argv[2])
summary_md = Path(sys.argv[3])
rows = []

def stat(values, fn):
    return None if not values else float(fn(values))

def p90(values):
    if not values:
        return None
    return sorted(values)[int(0.9 * (len(values) - 1))]

def parse_reward_events(text):
    starts = []
    ends = []
    for line in text.splitlines():
        if '[Reward Precompute Local]' not in line:
            continue
        kind = None
        if '[Reward Precompute Local] start ' in line:
            kind = 'start'
        elif '[Reward Precompute Local] end ' in line:
            kind = 'end'
        if kind is None:
            continue
        fields = dict(re.findall(r'(\w+)=([^\s]+)', line))
        try:
            wall_time = float(fields['wall_time'])
        except (KeyError, ValueError):
            continue
        event = {
            'kind': kind,
            'wall_time': wall_time,
            'reward_batch_index': fields.get('reward_batch_index'),
            'entries': fields.get('entries'),
        }
        if kind == 'end':
            try:
                event['elapsed_seconds'] = float(fields['elapsed_seconds'])
            except (KeyError, ValueError):
                event['elapsed_seconds'] = None
            ends.append(event)
        else:
            starts.append(event)
    return starts, ends

def parse_bench_gpu(text):
    used_mib = []
    total_mib = []
    util = []
    for line in text.splitlines():
        if not line.startswith('[Bench GPU]'):
            continue
        match = re.search(r'iso_time=\S+\s+(.+)$', line)
        if not match:
            continue
        parts = [part.strip() for part in match.group(1).split(',')]
        if len(parts) < 5:
            continue
        try:
            used_mib.append(int(parts[-3]))
            total_mib.append(int(parts[-2]))
            util.append(int(parts[-1]))
        except ValueError:
            continue
    return {
        'bench_gpu_samples': len(used_mib),
        'bench_gpu_max_used_mib': max(used_mib) if used_mib else None,
        'bench_gpu_max_total_mib': max(total_mib) if total_mib else None,
        'bench_gpu_max_util_percent': max(util) if util else None,
    }

cases_path = bench_root / 'cases.tsv'
if cases_path.exists():
    for line in cases_path.read_text().splitlines():
        name, run_id, main_job, finalize_job, venv_path, max_completion_length, use_vllm = line.split('\t')
        run_root = bench_root.parent / run_id
        out_files = sorted((run_root / 'slurm').glob(f'*{main_job}.out'))
        err_files = sorted((run_root / 'slurm').glob(f'*{main_job}.err'))
        out_text = out_files[0].read_text(errors='replace') if out_files else ''
        err_text = err_files[0].read_text(errors='replace') if err_files else ''
        reward_starts, reward_ends = parse_reward_events(out_text)
        elapsed = [event['elapsed_seconds'] for event in reward_ends if event.get('elapsed_seconds') is not None]
        generation_update_gaps = []
        for index, end_event in enumerate(reward_ends[:-1]):
            if index + 1 < len(reward_starts):
                gap = reward_starts[index + 1]['wall_time'] - end_event['wall_time']
                if gap >= 0:
                    generation_update_gaps.append(gap)
        mean_lengths = [float(x) for x in re.findall(r"'completions/mean_length': ([0-9.]+)", out_text)]
        max_lengths = [float(x) for x in re.findall(r"'completions/max_length': ([0-9.]+)", out_text)]
        clipped = [float(x) for x in re.findall(r"'completions/clipped_ratio': ([0-9.]+)", out_text)]
        reward_std = [float(x) for x in re.findall(r"'reward_std': ([0-9.eE+-]+)", out_text)]
        zero_std = [float(x) for x in re.findall(r"'frac_reward_zero_std': ([0-9.eE+-]+)", out_text)]
        stage_counts = {}
        sample_path = run_root / 'rl_output' / 'generation_samples.jsonl'
        if sample_path.exists():
            for raw in sample_path.read_text(errors='replace').splitlines():
                try:
                    obj = json.loads(raw)
                except Exception:
                    continue
                stage = ((obj.get('api_result') or {}).get('error_stage') or 'ok')
                stage_counts[stage] = stage_counts.get(stage, 0) + 1
        sacct = subprocess.run(
            ['sacct', '-j', main_job, '--format=State,ExitCode,Elapsed', '-n', '-P'],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        ).stdout.splitlines()
        state, exit_code, wall = ('unknown', 'unknown', 'unknown')
        if sacct:
            parts = sacct[0].split('|')
            if len(parts) >= 3:
                state, exit_code, wall = parts[:3]
        gpu_summary = parse_bench_gpu(out_text)
        rows.append({
            'name': name,
            'run_id': run_id,
            'main_job': main_job,
            'finalize_job': finalize_job,
            'max_completion_length': int(max_completion_length),
            'use_vllm': use_vllm == '1',
            'venv_path': venv_path,
            'state': state,
            'exit_code': exit_code,
            'wall': wall,
            'reward_batches': len(elapsed),
            'reward_elapsed_mean': stat(elapsed, statistics.mean),
            'reward_elapsed_median': stat(elapsed, statistics.median),
            'reward_elapsed_p90': stat(elapsed, p90),
            'generation_update_gap_count': len(generation_update_gaps),
            'generation_update_gap_mean': stat(generation_update_gaps, statistics.mean),
            'generation_update_gap_median': stat(generation_update_gaps, statistics.median),
            'generation_update_gap_p90': stat(generation_update_gaps, p90),
            'completion_mean_length_last': mean_lengths[-1] if mean_lengths else None,
            'completion_max_length_last': max_lengths[-1] if max_lengths else None,
            'clipped_ratio_last': clipped[-1] if clipped else None,
            'reward_std_last': reward_std[-1] if reward_std else None,
            'frac_reward_zero_std_last': zero_std[-1] if zero_std else None,
            'stage_counts': stage_counts,
            'stdout': str(out_files[0]) if out_files else None,
            'stderr': str(err_files[0]) if err_files else None,
            'error_tail': '\n'.join(err_text.splitlines()[-30:]),
            **gpu_summary,
        })

setup_error_path = bench_root / 'vllm_setup_failed.txt'
summary = {
    'bench_root': str(bench_root),
    'vllm_setup_failed': setup_error_path.read_text(errors='replace') if setup_error_path.exists() else None,
    'cases': rows,
}
summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True))
lines = [
    '# Generation Bench Summary',
    '',
    f'bench_root: `{bench_root}`',
]
if summary['vllm_setup_failed']:
    lines.extend(['', 'vLLM setup failed:', '', '```text', summary['vllm_setup_failed'][-4000:], '```'])
lines.extend([
    '',
    '| case | state | wall | reward batches | reward mean s | gen/update gap mean s | mean len | max len | clipped | gpu max MiB | gpu max util % | stages |',
    '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|',
])
for row in rows:
    lines.append(
        f"| {row['name']} | {row['state']} | {row['wall']} | {row['reward_batches']} | "
        f"{row['reward_elapsed_mean']} | {row['generation_update_gap_mean']} | "
        f"{row['completion_mean_length_last']} | {row['completion_max_length_last']} | "
        f"{row['clipped_ratio_last']} | {row['bench_gpu_max_used_mib']} | "
        f"{row['bench_gpu_max_util_percent']} | {row['stage_counts']} |"
    )
summary_md.write_text('\n'.join(lines) + '\n')
PYCODE
}

cd "${REPO_ROOT}"
if [[ ! -d "${SEED_STAGE2_CHECKPOINT}" ]]; then
  echo "Seed stage2 checkpoint not found: ${SEED_STAGE2_CHECKPOINT}" >&2
  exit 1
fi
rm -f "${BENCH_ROOT}/cases.tsv" "${BENCH_ROOT}/vllm_setup_failed.txt"

submit_case original 1536 0
submit_case len896 896 0
if ensure_vllm_venv > >(tee "${BENCH_ROOT}/vllm_venv.log") 2> >(tee "${BENCH_ROOT}/vllm_venv.err" >&2); then
  VLLM_READY=1
else
  VLLM_READY=0
  {
    echo "vLLM setup failed; vLLM benchmark cases were not submitted."
    echo "See ${BENCH_ROOT}/vllm_venv.log and ${BENCH_ROOT}/vllm_venv.err."
  } | tee "${BENCH_ROOT}/vllm_setup_failed.txt"
fi
if [[ "${VLLM_READY}" == "1" ]]; then
  submit_case vllm 1536 1
  submit_case vllm_len896 896 1
fi
wait_cases
write_summary
printf 'bench_root=%s\n' "${BENCH_ROOT}"
printf 'summary_json=%s\n' "${SUMMARY_JSON}"
printf 'summary_md=%s\n' "${SUMMARY_MD}"
