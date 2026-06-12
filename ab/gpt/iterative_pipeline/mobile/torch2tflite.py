"""
FP32 + INT8 TFLite export (nn-lite style).
Requires: pip install -r requirements-mobile.txt
Uses ai-edge-torch-nightly + ai_edge_tensorflow (not PyPI tensorflow).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch


def _import_tf():
    """tf.lite API from ai_edge_tensorflow (nn-lite stack), not PyPI tensorflow."""
    import tensorflow as tf  # noqa: F401  # provided by ai_edge_tensorflow
    return tf


def _try_import_ai_edge():
    try:
        import ai_edge_torch  # noqa: F401
        tf = _import_tf()
        return True, ai_edge_torch, tf
    except ImportError:
        return False, None, None


def convert_fp32_int8(
    model: torch.nn.Module,
    *,
    input_size: int = 32,
    out_dir: Path,
    stem: str = "model",
) -> Dict[str, Any]:
    """
    Export model_fp32.tflite and model_int8.tflite under out_dir.
    INT8 failure is non-fatal (same as nn-lite torch2tflite.py).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result: Dict[str, Any] = {
        "fp32_path": None,
        "int8_path": None,
        "fp32_ok": False,
        "int8_ok": False,
        "converter": None,
        "errors": [],
    }

    ok, ai_edge_torch, tf = _try_import_ai_edge()
    if not ok:
        result["errors"].append(
            "Mobile TFLite stack not installed. "
            "Install with: pip install -r requirements-mobile.txt "
            "(ai-edge-torch-nightly + ai_edge_tensorflow; no PyPI tensorflow)"
        )
        return result

    result["converter"] = "ai_edge_torch"
    dummy_input = (torch.randn(1, 3, input_size, input_size),)
    fp32_path = out_dir / f"{stem}_fp32.tflite"
    int8_path = out_dir / f"{stem}_int8.tflite"

    try:
        ai_edge_torch.convert(model, dummy_input).export(str(fp32_path))
        result["fp32_path"] = str(fp32_path)
        result["fp32_ok"] = fp32_path.exists()
    except Exception as exc:
        result["errors"].append(f"fp32: {type(exc).__name__}: {exc}")
        return result

    try:
        def rep():
            for _ in range(50):
                yield [np.random.randn(1, 3, input_size, input_size).astype(np.float32)]

        ai_edge_torch.convert(
            model,
            dummy_input,
            _ai_edge_converter_flags={
                "optimizations": [tf.lite.Optimize.DEFAULT],
                "representative_dataset": rep,
                "target_spec": {"supported_ops": [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]},
                "inference_input_type": tf.int8,
                "inference_output_type": tf.int8,
            },
        ).export(str(int8_path))
        result["int8_path"] = str(int8_path)
        result["int8_ok"] = int8_path.exists()
    except Exception as exc:
        result["errors"].append(f"int8: {type(exc).__name__}: {exc}")

    return result


def bench_tflite_cpu(
    tflite_path: Path,
    *,
    input_size: int = 32,
    runs: int = 20,
    warmup: int = 3,
) -> Dict[str, Any]:
    """Desktop proxy latency via TFLite interpreter (no ADB)."""
    out: Dict[str, Any] = {"path": str(tflite_path), "status": "failed", "runs": runs}
    try:
        tf = _import_tf()
    except ImportError:
        out["error"] = "ai_edge_tensorflow not installed (pip install -r requirements-mobile.txt)"
        return out

    if not tflite_path.exists():
        out["error"] = "file missing"
        return out

    try:
        interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
        interpreter.allocate_tensors()
        inp = interpreter.get_input_details()[0]
        dtype = inp["dtype"]
        shape = inp["shape"]
        # Build input matching shape
        if dtype == np.int8:
            data = np.random.randint(-128, 127, size=shape, dtype=np.int8)
        else:
            data = np.random.randn(*shape).astype(np.float32)

        def _run_once():
            interpreter.set_tensor(inp["index"], data)
            interpreter.invoke()

        for _ in range(warmup):
            _run_once()

        times_ms: List[float] = []
        for _ in range(runs):
            t0 = time.perf_counter()
            _run_once()
            times_ms.append((time.perf_counter() - t0) * 1000.0)

        arr = np.array(times_ms)
        out.update(
            {
                "status": "ok",
                "avg_ms": float(arr.mean()),
                "min_ms": float(arr.min()),
                "max_ms": float(arr.max()),
                "std_ms": float(arr.std()),
            }
        )
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def mobile_score(accuracy: float, latency_ms: float) -> float:
    if latency_ms <= 0:
        return 0.0
    return float(accuracy) / float(latency_ms)
