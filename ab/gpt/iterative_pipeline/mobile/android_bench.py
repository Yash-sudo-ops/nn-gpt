"""Optional on-device TFLite benchmark via ADB. Skipped when no device attached."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any, Dict


def adb_available() -> bool:
    try:
        res = subprocess.run(["adb", "get-state"], capture_output=True, text=True, timeout=10)
        return res.returncode == 0 and "device" in (res.stdout or "")
    except Exception:
        return False


def run_bench(model_path: str, backend: str, runs: int) -> Dict[str, Any]:
    flag = {"cpu": "--use_xnnpack=false", "gpu": "--use_gpu", "npu": "--use_nnapi"}.get(backend, "")
    cmd = f"/data/local/tmp/benchmark_model --graph={model_path} --num_runs={runs} {flag}"
    res = subprocess.run(["adb", "shell", cmd], capture_output=True, text=True, timeout=120)
    out = (res.stdout or "") + (res.stderr or "")
    if "ERROR:" in out or "Failed to compute" in out or "avg=" not in out.replace(" ", ""):
        return {"avg": 0, "status": "failed", "error": "benchmark failed"}
    result: Dict[str, Any] = {"status": "ok", "avg": 0.0}
    for key in ("avg", "min", "max", "std"):
        match = re.search(rf"{key}=([\d\.]+)", out.replace(" ", ""))
        if match:
            result[key] = float(match.group(1)) * 1000.0
    return result


def benchmark_tflite_on_device(tflite_path: Path, *, runs: int = 20) -> Dict[str, Any]:
    if not adb_available():
        return {"status": "skipped", "reason": "no adb device"}
    dev_p = f"/data/local/tmp/{tflite_path.name}"
    push = subprocess.run(["adb", "push", str(tflite_path), dev_p], capture_output=True, text=True, timeout=120)
    if push.returncode != 0:
        return {"status": "failed", "error": push.stderr or "adb push failed"}
    out = {
        "cpu": run_bench(dev_p, "cpu", runs),
        "gpu": run_bench(dev_p, "gpu", runs),
        "npu": run_bench(dev_p, "npu", runs),
    }
    subprocess.run(["adb", "shell", f"rm {dev_p}"], capture_output=True)
    return out
