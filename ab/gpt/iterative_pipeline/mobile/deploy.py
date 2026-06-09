"""
Mobile deploy track: copy successful eval models → export .pth → TFLite → optional bench.
Separate from core iterative pipeline; only runs when --mobile_deploy is set.
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from ab.gpt.iterative_pipeline.mobile.export_weights import (
    export_weights_pth,
    infer_input_size,
    load_model_for_export,
)
from ab.gpt.iterative_pipeline.mobile.torch2tflite import (
    bench_tflite_cpu,
    convert_fp32_int8,
    mobile_score,
)

logger = logging.getLogger(__name__)


def _read_accuracy(model_dir: Path) -> float:
    for name in ("1.json", "eval_info.json"):
        p = model_dir / name
        if not p.exists():
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, list) and data:
                acc = data[0].get("accuracy")
                if acc is not None:
                    return float(acc)
            if isinstance(data, dict):
                acc = data.get("accuracy")
                if acc is not None:
                    return float(acc)
        except Exception:
            pass
    return 0.0


def _read_server_duration(model_dir: Path) -> float:
    p = model_dir / "1.json"
    if not p.exists():
        return 0.0
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, list) and data:
            return float(data[0].get("duration", 0) or 0)
    except Exception:
        pass
    return 0.0


def deploy_one_model(
    src_dir: Path,
    dest_dir: Path,
    *,
    input_size: int,
    max_params: int,
    export_tflite: bool,
    bench_desktop: bool,
    bench_runs: int = 20,
) -> Dict[str, Any]:
    """Process a single evaluated model directory."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    record: Dict[str, Any] = {
        "source": str(src_dir),
        "dest": str(dest_dir),
        "success": False,
    }

    code_src = src_dir / "new_nn.py"
    if not code_src.exists():
        record["error"] = "new_nn.py missing in source"
        return record

    for fname in (
        "new_nn.py",
        "hp.json",
        "tr.py",
        "1.json",
        "eval_info.json",
        "dat.json",
        "weights.pth",
        "best_model.pth",
    ):
        sp = src_dir / fname
        if sp.exists():
            shutil.copy2(sp, dest_dir / fname)

    size = infer_input_size(src_dir, default=input_size)
    record["input_size"] = size

    weights_path, wmeta = export_weights_pth(dest_dir, input_size=size)
    record["weights"] = wmeta
    if not wmeta.get("success"):
        record["error"] = wmeta.get("error", "weights export failed")
        return record

    param_count = wmeta.get("param_count")
    if param_count is None and weights_path and weights_path.exists():
        try:
            import torch
            ckpt = torch.load(weights_path, map_location="cpu")
            sd = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            param_count = sum(v.numel() for v in sd.values())
            record["weights"]["param_count"] = param_count
        except Exception:
            pass

    if param_count is not None and param_count > max_params:
        record["skipped"] = True
        record["error"] = f"param_count {param_count} > max_params {max_params}"
        return record

    accuracy = _read_accuracy(src_dir)
    server_duration = _read_server_duration(src_dir)
    record["accuracy"] = accuracy
    record["server_duration"] = server_duration
    record["server_score"] = (accuracy / server_duration) if server_duration > 0 else 0.0

    tflite_meta: Dict[str, Any] = {}
    if export_tflite:
        model, load_meta = load_model_for_export(dest_dir, input_size=size)
        record["load_model"] = load_meta
        if model is None:
            record["error"] = load_meta.get("error", "model load failed")
            return record

        tflite_dir = dest_dir / "tflite"
        tflite_meta = convert_fp32_int8(model, input_size=size, out_dir=tflite_dir, stem="model")
        record["tflite"] = tflite_meta

        if bench_desktop:
            benches: Dict[str, Any] = {}
            for mode, key in (("fp32", "fp32_path"), ("int8", "int8_path")):
                path_str = tflite_meta.get(key)
                if path_str:
                    benches[mode] = bench_tflite_cpu(
                        Path(path_str), input_size=size, runs=bench_runs
                    )
            record["desktop_bench"] = benches
            best_ms = None
            for mode, b in benches.items():
                if b.get("status") == "ok":
                    ms = b["avg_ms"]
                    if best_ms is None or ms < best_ms:
                        best_ms = ms
                        record["best_bench_mode"] = mode
            if best_ms is not None:
                record["mobile_latency_ms"] = best_ms
                record["mobile_score"] = mobile_score(accuracy, best_ms)

    record["success"] = True
    summary_path = dest_dir / "mobile_deploy.json"
    summary_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def run_mobile_deploy_for_cycle(
    cycle: int,
    nneval_dir: Path,
    output_base: Path,
    *,
    input_size: int = 32,
    max_params: int = 500_000,
    export_tflite: bool = True,
    bench_desktop: bool = True,
    min_accuracy: float = 0.0,
    only_successful: bool = True,
) -> Dict[str, Any]:
    """
    Scan nneval_dir for gen_* folders with new_nn.py (+ optional 1.json success).
    Write to output_base/cycle_{N}/mobile_deploy/gen_XXXX/.
    """
    nneval_dir = Path(nneval_dir)
    mobile_root = Path(output_base) / f"cycle_{cycle}" / "mobile_deploy"
    mobile_root.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Any] = {
        "cycle": cycle,
        "nneval_dir": str(nneval_dir),
        "mobile_root": str(mobile_root),
        "models": [],
        "deployed": 0,
        "failed": 0,
        "skipped": 0,
    }

    if not nneval_dir.exists():
        summary["error"] = f"nneval_dir not found: {nneval_dir}"
        return summary

    candidates = sorted(
        [p for p in nneval_dir.iterdir() if p.is_dir() and p.name.startswith("gen_")],
        key=lambda p: p.name,
    )
    logger.info(f"[mobile] Cycle {cycle}: {len(candidates)} candidate dirs in {nneval_dir}")

    for src in candidates:
        if not (src / "new_nn.py").exists():
            continue

        acc = _read_accuracy(src)
        if only_successful and acc <= 0 and not (src / "1.json").exists():
            summary["skipped"] += 1
            summary["models"].append({"source": str(src), "skipped": True, "reason": "no successful eval"})
            continue
        if acc < min_accuracy:
            summary["skipped"] += 1
            summary["models"].append({"source": str(src), "skipped": True, "reason": f"accuracy {acc} < {min_accuracy}"})
            continue

        dest = mobile_root / src.name
        try:
            rec = deploy_one_model(
                src,
                dest,
                input_size=input_size,
                max_params=max_params,
                export_tflite=export_tflite,
                bench_desktop=bench_desktop,
            )
            summary["models"].append(rec)
            if rec.get("success"):
                summary["deployed"] += 1
            elif rec.get("skipped"):
                summary["skipped"] += 1
            else:
                summary["failed"] += 1
        except Exception as exc:
            summary["failed"] += 1
            summary["models"].append({"source": str(src), "success": False, "error": str(exc)})

    out_file = mobile_root / "mobile_deploy_summary.json"
    out_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info(
        f"[mobile] Cycle {cycle} done: deployed={summary['deployed']} "
        f"failed={summary['failed']} skipped={summary['skipped']}"
    )
    return summary
