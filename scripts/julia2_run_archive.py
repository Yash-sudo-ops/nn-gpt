#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any


RUN_MARKER_PREFIX = "NNGPT_RUN"
STATE_FILENAME = "run_state.json"


def _normalize_text(value: str | None, *, default: str = "-") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def _normalize_multiline(value: str | None, *, default: str = "-") -> str:
    normalized = _normalize_text(value, default=default)
    if normalized == default:
        return default
    return "<br>".join(part.strip() for part in normalized.splitlines() if part.strip()) or default


def _format_inline_code(value: str | None, *, default: str = "-") -> str:
    normalized = _normalize_text(value, default=default)
    if normalized == default:
        return default
    return f"`{normalized}`"


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object in {path}, got {type(payload).__name__}")
    return dict(payload)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def _state_path(run_root: Path) -> Path:
    return run_root / STATE_FILENAME


def _load_state(run_root: Path) -> dict[str, Any]:
    path = _state_path(run_root)
    if not path.exists():
        return {}
    return _read_json(path)


def _save_state(run_root: Path, payload: dict[str, Any]) -> None:
    _write_json(_state_path(run_root), payload)


def _format_resume_source(value: str | None) -> str:
    normalized = _normalize_text(value, default="-")
    if normalized == "-":
        return normalized
    return normalized.replace(str(Path.home()), "~")


def _section_markers(run_id: str) -> tuple[str, str]:
    start = f"<!-- {RUN_MARKER_PREFIX}:{run_id}:START -->"
    end = f"<!-- {RUN_MARKER_PREFIX}:{run_id}:END -->"
    return start, end


def _render_section(state: dict[str, Any]) -> str:
    run_id = _normalize_text(state.get("run_id"))
    start_marker, end_marker = _section_markers(run_id)
    partition = _normalize_text(state.get("partition"))
    qos = _normalize_text(state.get("qos"))
    partition_qos = partition if qos == "-" else f"{partition} / {qos}"
    commit_hash = _normalize_text(state.get("commit_hash"))
    commit_subject = _normalize_text(state.get("commit_subject"))
    commit_value = commit_hash if commit_subject == "-" else f"{commit_hash} {commit_subject}"
    lines = [
        f"## {run_id}",
        "",
        start_marker,
        f"- 运行 ID：{_format_inline_code(run_id)}",
        f"- 标签：{_format_inline_code(state.get('run_label'))}",
        f"- 状态：{_normalize_text(state.get('status'))}",
        f"- 提交时间：{_format_inline_code(state.get('submit_time'))}",
        f"- 开始时间：{_format_inline_code(state.get('start_time'))}",
        f"- 结束时间：{_format_inline_code(state.get('end_time'))}",
        f"- Job ID：{_format_inline_code(state.get('job_id'))}",
        f"- 分区 / QoS：{_format_inline_code(partition_qos)}",
        f"- 节点：{_format_inline_code(state.get('node'))}",
        f"- 提交 commit：{_format_inline_code(commit_value)}",
        f"- 本次改动：{_normalize_multiline(state.get('run_note'))}",
        f"- 输出目录：{_format_inline_code(state.get('run_root'))}",
        f"- 标准输出：{_format_inline_code(state.get('stdout_path'))}",
        f"- 标准错误：{_format_inline_code(state.get('stderr_path'))}",
        f"- 工作目录：{_format_inline_code(state.get('work_dir'))}",
        f"- 初始恢复来源：{_normalize_multiline(_format_resume_source(state.get('initial_resume_source')))}",
        f"- 训练结果：{_normalize_multiline(state.get('training_result'))}",
        f"- 主要缺陷：{_normalize_multiline(state.get('defects'))}",
        f"- 保留目录：{_format_inline_code(state.get('keep_dir') or state.get('run_root'))}",
        end_marker,
        "",
    ]
    return "\n".join(lines)


def _upsert_section(doc_path: Path, run_id: str, section_text: str) -> None:
    doc_path.parent.mkdir(parents=True, exist_ok=True)
    start_marker, end_marker = _section_markers(run_id)
    existing = doc_path.read_text(encoding="utf-8") if doc_path.exists() else ""
    pattern = re.compile(
        rf"(?ms)^## {re.escape(run_id)}\n\n{re.escape(start_marker)}.*?{re.escape(end_marker)}\n?"
    )
    if pattern.search(existing):
        updated = pattern.sub(section_text.rstrip() + "\n", existing, count=1)
    else:
        separator = "" if not existing.strip() else "\n"
        updated = existing.rstrip() + separator + section_text.rstrip() + "\n"
    doc_path.write_text(updated, encoding="utf-8")


def _merge_updates(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    payload = dict(base)
    for key, value in updates.items():
        if value is None:
            continue
        payload[key] = value
    return payload


def _collect_sacct(job_id: str | None) -> dict[str, str]:
    if not job_id:
        return {}
    try:
        completed = subprocess.run(
            [
                "sacct",
                "-j",
                job_id,
                "--parsable2",
                "--noheader",
                "--format=JobIDRaw,State,ExitCode,NodeList,Start,End",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return {}
    if completed.returncode != 0 or not completed.stdout.strip():
        return {}
    for line in completed.stdout.splitlines():
        fields = line.split("|")
        if len(fields) < 6:
            continue
        if fields[0].strip() != str(job_id).strip():
            continue
        return {
            "state": fields[1].strip() or "-",
            "exit_code": fields[2].strip() or "-",
            "node": fields[3].strip() or "-",
            "start": fields[4].strip() or "-",
            "end": fields[5].strip() or "-",
        }
    return {}


def _collect_finish_updates(run_root: Path, state: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    job_id = _normalize_text(args.job_id or state.get("job_id"), default="")
    sacct_info = _collect_sacct(job_id)
    slurm_state = _normalize_text(sacct_info.get("state"), default="-")
    status = "已结束" if slurm_state in {"-", "COMPLETED"} else f"已结束({slurm_state})"
    return {
        "status": status,
        "end_time": args.end_time or sacct_info.get("end") or state.get("end_time") or "-",
        "node": state.get("node") if _normalize_text(state.get("node")) != "-" else sacct_info.get("node"),
        "keep_dir": str(run_root),
    }


def _run_init(args: argparse.Namespace) -> None:
    run_root = Path(args.run_root).expanduser().resolve()
    doc_path = Path(args.doc_path).expanduser().resolve()
    payload = {
        "run_id": args.run_id,
        "run_label": args.run_label,
        "run_note": args.run_note,
        "status": args.status,
        "submit_time": args.submit_time,
        "start_time": "-",
        "end_time": "-",
        "job_id": args.job_id or "-",
        "partition": args.partition or "-",
        "qos": args.qos or "-",
        "node": "-",
        "commit_hash": args.commit_hash,
        "commit_subject": args.commit_subject,
        "run_root": str(run_root),
        "stdout_path": args.stdout_path or "-",
        "stderr_path": args.stderr_path or "-",
        "work_dir": "-",
        "initial_resume_source": "-",
        "training_result": "待手写",
        "defects": "待手写",
        "keep_dir": str(run_root),
        "seed_stage2_checkpoint": args.seed_stage2_checkpoint or "-",
    }
    _save_state(run_root, payload)
    _upsert_section(doc_path, args.run_id, _render_section(payload))


def _run_update(args: argparse.Namespace) -> None:
    run_root = Path(args.run_root).expanduser().resolve()
    doc_path = Path(args.doc_path).expanduser().resolve()
    state = _load_state(run_root)
    if not state:
        raise FileNotFoundError(f"Missing run state under {run_root}")
    updates = {
        "status": args.status,
        "job_id": args.job_id,
        "partition": args.partition,
        "qos": args.qos,
        "node": args.node,
        "start_time": args.start_time,
        "end_time": args.end_time,
        "stdout_path": args.stdout_path,
        "stderr_path": args.stderr_path,
        "work_dir": args.work_dir,
        "initial_resume_source": args.initial_resume_source,
        "training_result": args.training_result,
        "defects": args.defects,
        "keep_dir": args.keep_dir,
    }
    payload = _merge_updates(state, updates)
    if args.mode == "finish":
        payload = _merge_updates(payload, _collect_finish_updates(run_root, payload, args))
    _save_state(run_root, payload)
    _upsert_section(doc_path, _normalize_text(payload.get("run_id")), _render_section(payload))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Maintain Julia2 run archive sections.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init")
    init_parser.add_argument("--run-root", required=True)
    init_parser.add_argument("--doc-path", required=True)
    init_parser.add_argument("--run-id", required=True)
    init_parser.add_argument("--run-label", required=True)
    init_parser.add_argument("--run-note", required=True)
    init_parser.add_argument("--commit-hash", required=True)
    init_parser.add_argument("--commit-subject", required=True)
    init_parser.add_argument("--submit-time", required=True)
    init_parser.add_argument("--status", default="已提交")
    init_parser.add_argument("--job-id")
    init_parser.add_argument("--partition")
    init_parser.add_argument("--qos")
    init_parser.add_argument("--stdout-path")
    init_parser.add_argument("--stderr-path")
    init_parser.add_argument("--seed-stage2-checkpoint")
    init_parser.set_defaults(func=_run_init)

    update_parser = subparsers.add_parser("update")
    update_parser.add_argument("--run-root", required=True)
    update_parser.add_argument("--doc-path", required=True)
    update_parser.add_argument("--mode", choices=("start", "finish", "manual"), required=True)
    update_parser.add_argument("--status")
    update_parser.add_argument("--job-id")
    update_parser.add_argument("--partition")
    update_parser.add_argument("--qos")
    update_parser.add_argument("--node")
    update_parser.add_argument("--start-time")
    update_parser.add_argument("--end-time")
    update_parser.add_argument("--stdout-path")
    update_parser.add_argument("--stderr-path")
    update_parser.add_argument("--work-dir")
    update_parser.add_argument("--initial-resume-source")
    update_parser.add_argument("--training-result")
    update_parser.add_argument("--defects")
    update_parser.add_argument("--keep-dir")
    update_parser.set_defaults(func=_run_update)

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
