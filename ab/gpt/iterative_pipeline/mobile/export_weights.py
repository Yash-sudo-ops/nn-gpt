"""Load generated new_nn.py and export PyTorch state_dict to .pth."""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch


DEFAULT_PRM = {
    "lr": 0.01,
    "momentum": 0.9,
    "batch": 16,
    "dropout": 0.0,
    "transform": "echo_32",
    "epoch": 1,
}


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def infer_input_size(model_dir: Path, default: int = 32) -> int:
    """Guess spatial size from hp.json, eval_info, or transform code."""
    hp = _load_json(model_dir / "hp.json")
    if isinstance(hp, dict):
        transform = hp.get("transform", "")
        if isinstance(transform, str):
            m = re.search(r"(\d+)", transform)
            if m:
                return int(m.group(1))

    eval_info = _load_json(model_dir / "eval_info.json")
    if isinstance(eval_info, dict):
        cli = eval_info.get("cli_args") or {}
        transform = cli.get("transform", "")
        if isinstance(transform, str):
            m = re.search(r"(\d+)", transform)
            if m:
                return int(m.group(1))

    stat = _load_json(model_dir / "1.json")
    if isinstance(stat, list) and stat and isinstance(stat[0], dict):
        transform = stat[0].get("transform", "")
        if isinstance(transform, str):
            m = re.search(r"(\d+)", transform)
            if m:
                return int(m.group(1))

    tr_path = model_dir / "tr.py"
    if tr_path.exists():
        m = re.search(r"Resize\s*\(\s*\(?\s*(\d+)", tr_path.read_text(encoding="utf-8"))
        if m:
            return int(m.group(1))
    return default


def load_hyperparameters(model_dir: Path) -> Dict[str, Any]:
    hp = _load_json(model_dir / "hp.json")
    if isinstance(hp, dict):
        prm = dict(DEFAULT_PRM)
        prm.update(hp)
        return prm

    eval_info = _load_json(model_dir / "eval_info.json")
    if isinstance(eval_info, dict):
        cli = eval_info.get("cli_args") or {}
        prm = dict(DEFAULT_PRM)
        for key in ("lr", "batch", "dropout", "momentum", "transform"):
            if key in cli and cli[key] is not None:
                prm[key] = cli[key]
        return prm

    return dict(DEFAULT_PRM)


def load_net_module(code_path: Path):
    spec = importlib.util.spec_from_file_location(f"mobile_nn_{code_path.parent.name}", code_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {code_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "Net"):
        raise AttributeError(f"No Net class in {code_path}")
    return mod


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def export_weights_pth(
    model_dir: Path,
    *,
    input_size: int = 32,
    num_classes: int = 10,
    device: str = "cpu",
    weights_name: str = "weights.pth",
) -> Tuple[Optional[Path], Dict[str, Any]]:
    """
    Instantiate Net from new_nn.py and save state_dict.
    Uses existing weights.pth in model_dir if present (e.g. from a future trainer hook).
    """
    code_path = model_dir / "new_nn.py"
    meta: Dict[str, Any] = {"code_file": str(code_path), "success": False}
    if not code_path.exists():
        meta["error"] = "new_nn.py missing"
        return None, meta

    existing = model_dir / weights_name
    best = model_dir / "best_model.pth"
    if existing.exists():
        meta.update(
            {
                "success": True,
                "weights_file": str(existing),
                "weights_source": "existing",
                "skipped_instantiate": True,
                "trained": True,
            }
        )
        return existing, meta
    if best.exists():
        try:
            ckpt = torch.load(best, map_location="cpu")
            state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            torch.save(
                {"state_dict": state, "weights_source": str(best), "trained": True},
                existing,
            )
            meta.update(
                {
                    "success": True,
                    "weights_file": str(existing),
                    "weights_source": "best_model.pth",
                    "trained": True,
                }
            )
            return existing, meta
        except Exception as exc:
            meta["best_model_error"] = str(exc)

    prm = load_hyperparameters(model_dir)
    try:
        mod = load_net_module(code_path)
        in_shape = (3, input_size, input_size)
        out_shape = (num_classes,)
        model = mod.Net(in_shape, out_shape, prm, device)
        model.to(device)
        model.eval()
        param_count = count_parameters(model)
        out_path = model_dir / weights_name
        torch.save({"state_dict": model.state_dict(), "prm": prm, "in_shape": in_shape, "out_shape": out_shape}, out_path)
        meta.update(
            {
                "success": True,
                "weights_file": str(out_path),
                "weights_source": "instantiated_state_dict",
                "param_count": param_count,
                "trained": False,
                "note": "Exported module state after NNEval; for trained weights, add weights.pth from trainer.",
            }
        )
        return out_path, meta
    except Exception as exc:
        meta["error"] = f"{type(exc).__name__}: {exc}"
        return None, meta


def load_model_for_export(
    model_dir: Path,
    *,
    input_size: int = 32,
    num_classes: int = 10,
    device: str = "cpu",
) -> Tuple[Optional[torch.nn.Module], Dict[str, Any]]:
    """Load nn.Module for TFLite conversion (from weights.pth or fresh instantiate)."""
    code_path = model_dir / "new_nn.py"
    prm = load_hyperparameters(model_dir)
    in_shape = (3, input_size, input_size)
    out_shape = (num_classes,)
    weights_path = model_dir / "weights.pth"

    try:
        mod = load_net_module(code_path)
        model = mod.Net(in_shape, out_shape, prm, device)
        model.to(device)
        if weights_path.exists():
            ckpt = torch.load(weights_path, map_location=device)
            state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            model.load_state_dict(state, strict=False)
        model.eval()
        return model, {"success": True, "prm": prm, "in_shape": in_shape}
    except Exception as exc:
        return None, {"success": False, "error": f"{type(exc).__name__}: {exc}"}
