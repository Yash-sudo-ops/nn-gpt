import ast
import hashlib
import os
import shutil
import random
import inspect
import re
import sys
import tempfile
import textwrap
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple
from torch.utils.data import Dataset as TorchDataset
from ab.gpt.util.Const import conf_dir


# ── SFT runtime configuration ─────────────────────────────────────────────
SFT_BASE_MODEL_ID = "ABrain/NNGPT-Backbone-deepseek-coder-6.7b-instruct"
SFT_INIT_ADAPTER = ""
SFT_LOAD_INITIAL_ADAPTER = False
SFT_SAVE_RL_MODEL = False
SFT_MODEL_OUT = "rl_backbone_model_sft"
SFT_LOG_DIR = "rl_output/sft"
SFT_EPOCH_ROOT = "out/nngpt/llm/epoch_sft"
SFT_TRAINER_OUT = "grpo_backbone_outputs/sft"
SFT_TEMPERATURE = 1.1
SFT_NUM_GENERATIONS = 8
SFT_GRAD_ACCUM = 8
SFT_MAX_COMPLETION_LENGTH = 1536
SFT_DATASET_LIMIT = 500
SFT_FEEDBACK_CHAR_BUDGET = 1200
SFT_LR = 5e-5
SFT_NUM_EPOCHS = 5
SFT_KL_COEF = 5e-4
SFT_LORA_R = 16
SFT_LORA_ALPHA = 32
SFT_LORA_DROPOUT = 0.05
SFT_DEEPSPEED_DEFAULT_CONFIG = str(conf_dir / "DeepSpeedSftGrpo.json")
SFT_MODE_DEFAULT = "auto"

# CIFAR-10 reward evaluation via nn-dataset / NNEval-aligned formal acc.
SFT_EVAL_IMAGE_SIZE = 128
SFT_EVAL_BATCH_SIZE = 64
SFT_EVAL_TRAIN_SUBSET = 256
SFT_EVAL_VAL_SUBSET = 128
SFT_EVAL_TRAIN_EPOCHS = 1
SFT_EVAL_VAL_BATCHES = 2
SFT_EVAL_FULL_TEST_ACC = True
SFT_EVAL_RUN_UNFROZEN = False
SFT_EVAL_LIMIT_SECONDS = 900
SFT_EVAL_FORMAL_EPOCH_LIMIT_MINUTES = 30
SFT_EVAL_DATA_ROOT = "data_v2"
SFT_EVAL_DOWNLOAD = True
SFT_VAL_METRIC_BASELINE = 0.10

# Local desktop cache roots.
# Keep this block on the local machine if you want Hugging Face downloads/cache on
# the mounted disk. On the server, if the model is already placed under
# `out/llm/ABrain/NNGPT-Backbone-deepseek-coder-6.7b-instruct`, you can comment
# out this whole block and the script will still load from `out/llm` first.
# SFT_HF_HOME = "/media/xi/Data/hf-cache"
# SFT_HF_HUB_CACHE = "/media/xi/Data/hf-cache/hub"
# SFT_TRANSFORMERS_CACHE = "/media/xi/Data/hf-cache/transformers"

# os.environ["HF_HOME"] = SFT_HF_HOME
# os.environ["HF_HUB_CACHE"] = SFT_HF_HUB_CACHE
# os.environ["HUGGINGFACE_HUB_CACHE"] = SFT_HF_HUB_CACHE
# os.environ["TRANSFORMERS_CACHE"] = SFT_TRANSFORMERS_CACHE

import ab.gpt.TuneRL as TuneRL
import ab.gpt.util.Reward as RewardUtil
import ab.gpt.util.SFTUtil as SFTUtil
import ab.gpt.util.training_runtime as TrainingRuntime

SFT_EVAL_TRANSFORM = TuneRL.FORMAL_REWARD_TRANSFORM


_TRAIN_GPU_TOKENS_ENV = "NNGPT_TRAIN_GPU_TOKENS"
_AUX_GPU_TOKENS_ENV = "NNGPT_AUX_GPU_TOKENS"
_REWARD_GPU_TOKENS_ENV = "NNGPT_REWARD_GPU_TOKENS"

REQUIRED_BACKBONE_NAMES = ("backbone_a", "backbone_b")
BLOCK_SIGNATURE = "def drop_conv3x3_block(in_channels, out_channels, stride=1, padding=1, bias=False, dropout_prob=0.0):"
INIT_SIGNATURE = "def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:"
FORWARD_SIGNATURE = "def forward(self, x: torch.Tensor, is_probing: bool = False) -> torch.Tensor:"

_BLOCKED_ATTRS = {
    "device",
    "use_amp",
    "_input_spec",
    "pattern",
    "classifier",
    "infer_dimensions_dynamically",
    "train_setup",
    "learn",
    "criterion",
    "optimizer",
    "_scaler",
}

_EXTRACTION_META_CACHE: Dict[str, Dict[str, object]] = {}


def clear_extraction_meta_cache() -> None:
    _EXTRACTION_META_CACHE.clear()


class RawCodeLogger(TuneRL.SimpleCodeLogger):
    def __init__(self, output_dir: str = "rl_output/raw"):
        super().__init__(output_dir)
        self.samples_file = os.path.join(output_dir, "generation_samples.jsonl")

    def log_generation(self, prompt: str, completion: str, reward: float, api_result=None):
        super().log_generation(prompt, completion, reward, api_result)


def _strip_outer_code_fences(text: str) -> str:
    if not text:
        return ""
    text = text.strip()
    text = re.sub(r"^```(?:python)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _extract_xml_tag(text: str, tag: str) -> str:
    match = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", text, re.IGNORECASE | re.DOTALL)
    return TuneRL.clean_block(match.group(1)) if match else ""


def _extract_function_block(text: str, fn_name: str) -> str:
    lines = text.splitlines()
    for start_index, line in enumerate(lines):
        if not re.match(rf"^\s*def {re.escape(fn_name)}\s*\(", line):
            continue
        indent = len(line) - len(line.lstrip())
        end_index = len(lines)
        for scan_index in range(start_index + 1, len(lines)):
            stripped = lines[scan_index].lstrip()
            if not stripped:
                continue
            indent_scan = len(lines[scan_index]) - len(stripped)
            if (stripped.startswith("def ") or stripped.startswith("class ")) and indent_scan <= indent:
                end_index = scan_index
                break
        return textwrap.dedent("\n".join(lines[start_index:end_index])).strip()
    return ""


def _dedupe_keep_order(items: Sequence[str]) -> List[str]:
    deduped: List[str] = []
    seen = set()
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def _clean_source_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("```python", "").replace("```", "")
    text = text.replace("<s=", "=")
    text = text.replace("torch.concat(", "torch.cat(")
    text = text.replace("torch.concatenate(", "torch.cat(")
    text = text.replace("self.adaptive_pool_flatten(", "adaptive_pool_flatten(")
    return text.strip()


def _completion_cache_key(text: str) -> str:
    return hashlib.sha1((text or "").encode("utf-8")).hexdigest()


def _infer_attr_role(attr_name: str) -> str:
    lowered = attr_name.lower()
    if "fractal" in lowered:
        return "fractal"
    if lowered.startswith("backbone"):
        return "backbone"
    if "stem" in lowered:
        return "stem"
    if any(token in lowered for token in ("project", "bridge", "adapter", "align")):
        return "project"
    if any(token in lowered for token in ("fuse", "merge", "gate", "mixer")):
        return "fuse"
    return "generic"


def _has_structural_attr(attrs: Sequence[str]) -> bool:
    return any(
        _infer_attr_role(attr) in {"stem", "project", "fuse", "backbone", "fractal"}
        for attr in attrs
    )


def _scan_raw_attrs(*texts: str) -> List[str]:
    attrs: List[str] = []
    for text in texts:
        if not text:
            continue
        for attr in re.findall(r"self\.([A-Za-z_]\w*)\s*(?:\(|=)", text):
            if attr in _BLOCKED_ATTRS or attr.startswith("__"):
                continue
            attrs.append(attr)
    return _dedupe_keep_order(attrs)


def _prepare_completion_for_xml(completion: str) -> str:
    stripped = _strip_outer_code_fences(completion or "").lstrip()
    if "<block>" not in stripped and "</block>" in stripped and "<init>" in stripped:
        return stripped
    if "<block>" not in stripped and "</block>" not in stripped and "<init>" in stripped:
        init_pos = stripped.find("<init>")
        pre_init = stripped[:init_pos].strip()
        rest = stripped[init_pos:]
        if pre_init:
            return f"<block>\n{BLOCK_SIGNATURE}\n{pre_init}\n</block>\n{rest}"
        return (
            f"<block>\n{BLOCK_SIGNATURE}\n"
            "    return nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, stride, padding, bias=bias))\n"
            f"</block>\n{rest}"
        )
    return stripped


def _normalize_required_function(code: str, fn_name: str, signature: str) -> str:
    code = _strip_outer_code_fences(code)
    if not code:
        return ""
    code = textwrap.dedent(code).strip()
    if not code:
        return ""

    lines = code.splitlines()
    if lines and re.match(rf"^\s*def {re.escape(fn_name)}\s*\(", lines[0]):
        body_lines = lines[1:]
    else:
        body_lines = lines

    body_text = textwrap.dedent("\n".join(body_lines)).strip("\n")
    if not body_text.strip():
        return ""

    normalized_body = [f"    {line}" if line.strip() else "" for line in body_text.splitlines()]
    return f"{signature}\n" + "\n".join(normalized_body)


def _normalize_block_code(block_code: str) -> str:
    return _normalize_required_function(block_code, "drop_conv3x3_block", BLOCK_SIGNATURE)


def _find_last_body_line_index(lines: Sequence[str], prefixes: Sequence[str]) -> int:
    last_index = 0
    for index, raw_line in enumerate(lines):
        stripped = raw_line.strip()
        if any(stripped.startswith(prefix) for prefix in prefixes):
            last_index = index + 1
    return last_index


def _repair_init_abi(init_code: str) -> str:
    if not init_code:
        return ""
    lines = init_code.splitlines()
    if not lines:
        return init_code

    signature = lines[0]
    body_lines = list(lines[1:])
    if not body_lines:
        return init_code

    repaired_body: List[str] = []
    for raw_line in body_lines:
        if raw_line.strip().startswith("self.infer_dimensions_dynamically("):
            continue
        repaired_body.append(raw_line)

    if not any(line.strip().startswith("self.device =") for line in repaired_body):
        insert_at = _find_last_body_line_index(repaired_body, ("super().__init__()",))
        repaired_body.insert(insert_at, "    self.device = device")

    if not any(line.strip().startswith("self.use_amp =") for line in repaired_body):
        insert_at = _find_last_body_line_index(repaired_body, ("super().__init__()", "self.device ="))
        repaired_body.insert(insert_at, "    self.use_amp = torch.cuda.is_available()")

    if not any("self._input_spec" in line and "=" in line for line in repaired_body):
        insert_at = _find_last_body_line_index(
            repaired_body,
            ("super().__init__()", "self.device =", "self.use_amp =", "self.pattern ="),
        )
        repaired_body.insert(insert_at, "    self._input_spec = tuple(in_shape[1:])")

    while repaired_body and not repaired_body[-1].strip():
        repaired_body.pop()
    repaired_body.append("    self.infer_dimensions_dynamically(out_shape[0])")
    return "\n".join([signature, *repaired_body])


def _normalize_init_code(init_code: str) -> str:
    normalized = _normalize_required_function(init_code, "__init__", INIT_SIGNATURE)
    return _repair_init_abi(normalized)


def _normalize_forward_code(forward_code: str) -> str:
    return _normalize_required_function(forward_code, "forward", FORWARD_SIGNATURE)


def _extract_defined_backbones(init_code: str) -> List[str]:
    return _dedupe_keep_order(re.findall(r"self\.(backbone_[A-Za-z]\w*)\s*=", init_code or ""))


def _extract_used_backbones(forward_code: str) -> List[str]:
    return _dedupe_keep_order(re.findall(r"self\.(backbone_[A-Za-z]\w*)\b", forward_code or ""))


def _extract_backbone_model_names(init_code: str) -> List[str]:
    matches: Dict[str, str] = {}
    patterns = (
        r"self\.(backbone_[ab])\s*=\s*TorchVision\(\s*model\s*=\s*['\"]([^'\"]+)['\"]",
        r"self\.(backbone_[ab])\s*=\s*TorchVision\(\s*['\"]([^'\"]+)['\"]",
    )
    for pattern in patterns:
        for match in re.finditer(pattern, init_code or ""):
            matches.setdefault(match.group(1), match.group(2))
    return [matches[name] for name in REQUIRED_BACKBONE_NAMES if name in matches]


def _count_xml_tags(text: str, tag: str) -> Tuple[int, int]:
    return (
        len(re.findall(rf"<{tag}>", text, re.IGNORECASE)),
        len(re.findall(rf"</{tag}>", text, re.IGNORECASE)),
    )


def _build_extraction_meta(
    completion: str,
    candidate: str,
    block_code: str,
    init_code: str,
    forward_code: str,
) -> Dict[str, object]:
    xml_tag_count = sum(bool(code) for code in (block_code, init_code, forward_code))
    xml_counts = {tag: _count_xml_tags(candidate, tag) for tag in ("block", "init", "forward")}
    class_count = len(re.findall(r"^\s*class\s+\w+", candidate, re.MULTILINE))
    import_count = len(re.findall(r"^\s*(?:from|import)\s+\w+", candidate, re.MULTILINE))
    bad_signature_count = len(re.findall(r"\)\s*-\s*:", candidate))
    raw_attrs = _scan_raw_attrs(candidate, block_code, init_code, forward_code)
    structural_attr_detected = _has_structural_attr(raw_attrs)

    defined_backbones = _extract_defined_backbones(init_code)
    used_backbones = _extract_used_backbones(forward_code)
    backbone_model_names = _extract_backbone_model_names(init_code)
    required_backbone_set = set(REQUIRED_BACKBONE_NAMES)
    dual_backbone_init_ok = set(defined_backbones) == required_backbone_set and len(defined_backbones) == 2
    dual_backbone_forward_ok = required_backbone_set.issubset(set(used_backbones)) and len(set(used_backbones)) == 2
    dual_backbone_ok = dual_backbone_init_ok and dual_backbone_forward_ok

    exact_xml = all(start_count == 1 and end_count == 1 for start_count, end_count in xml_counts.values())
    exact_signatures = {
        "block": block_code.startswith(BLOCK_SIGNATURE),
        "init": init_code.startswith(INIT_SIGNATURE),
        "forward": forward_code.startswith(FORWARD_SIGNATURE),
    }

    quality_score = 0
    quality_score += 2 if exact_xml else 0
    quality_score += sum(1 for ok in exact_signatures.values() if ok)
    quality_score += 2 if dual_backbone_ok else 0
    quality_score += 1 if structural_attr_detected else 0
    quality_score -= min(class_count, 2)
    quality_score -= min(import_count, 2)
    quality_score -= min(bad_signature_count, 2)

    return {
        "xml_tag_count": xml_tag_count,
        "xml_tag_exact": exact_xml,
        "xml_counts": xml_counts,
        "class_count": class_count,
        "import_count": import_count,
        "bad_signature_count": bad_signature_count,
        "structural_attr_detected": structural_attr_detected,
        "quality_score": quality_score,
        "exact_block_signature": exact_signatures["block"],
        "exact_init_signature": exact_signatures["init"],
        "exact_forward_signature": exact_signatures["forward"],
        "defined_backbones": defined_backbones,
        "used_backbones": used_backbones,
        "backbone_model_names": backbone_model_names,
        "dual_backbone_init_ok": dual_backbone_init_ok,
        "dual_backbone_forward_ok": dual_backbone_forward_ok,
        "dual_backbone_ok": dual_backbone_ok,
        "candidate_line_count": len(candidate.splitlines()),
    }


def extract_completion_payload_tolerant(completion: str) -> Tuple[Tuple[str, str, str], Dict[str, object]]:
    cache_key = _completion_cache_key(completion or "")
    cached = _EXTRACTION_META_CACHE.get(cache_key)
    if cached:
        return (
            (cached["block_code"], cached["init_code"], cached["forward_code"]),
            dict(cached["meta"]),
        )

    candidate = _prepare_completion_for_xml(completion or "")
    block_code = _normalize_block_code(_extract_xml_tag(candidate, "block"))
    init_code = _normalize_init_code(_extract_xml_tag(candidate, "init"))
    forward_code = _normalize_forward_code(_extract_xml_tag(candidate, "forward"))
    meta = _build_extraction_meta(completion or "", candidate, block_code, init_code, forward_code)

    _EXTRACTION_META_CACHE[cache_key] = {
        "block_code": block_code,
        "init_code": init_code,
        "forward_code": forward_code,
        "meta": meta,
    }
    return ((block_code, init_code, forward_code), meta)


def extract_completion_blocks_tolerant(completion: str) -> Tuple[str, str, str]:
    blocks, _ = extract_completion_payload_tolerant(completion)
    return blocks


def extract_completion_meta(completion: str) -> Dict[str, object]:
    _, meta = extract_completion_payload_tolerant(completion)
    return meta


def raw_reward_fn(
    completion: str,
    *,
    seed_accuracy_baseline: float,
    precomputed_eval_result: Dict[str, Any] | None = None,
    graph_info=None,
    batch_graph_hashes: List[str] = None,
    batch_family_hashes: List[str] = None,
    batch_descriptor_keys: List[str] = None,
    batch_backbone_signatures: List[str] = None,
    batch_cnn_signatures: List[str] = None,
    prompt_goal_tags: List[str] = None,
    archive_snapshot_family_counts: Dict[str, int] = None,
    archive_snapshot_descriptor_counts: Dict[str, int] = None,
    archive_snapshot_backbone_signature_counts: Dict[str, int] = None,
    archive_snapshot_cnn_signature_counts: Dict[str, int] = None,
    archive_snapshot_backbone_cnn_pair_counts: Dict[str, int] = None,
    group_baseline_train_acc: float | None = None,
    group_baseline_reward_target_acc: float | None = None,
    reward_batch_index: int | None = None,
    reward_group_id: int | None = None,
    group_warmup: bool = False,
    completion_index: int | None = None,
    batch_last_item: bool = False,
):
    res = TuneRL.base_discovery_reward_fn(
        completion,
        seed_accuracy_baseline=seed_accuracy_baseline,
        precomputed_eval_result=precomputed_eval_result,
        graph_info=graph_info,
        batch_graph_hashes=batch_graph_hashes,
        batch_family_hashes=batch_family_hashes,
        batch_descriptor_keys=batch_descriptor_keys,
        batch_backbone_signatures=batch_backbone_signatures,
        batch_cnn_signatures=batch_cnn_signatures,
        prompt_goal_tags=prompt_goal_tags,
        archive_snapshot_family_counts=archive_snapshot_family_counts,
        archive_snapshot_descriptor_counts=archive_snapshot_descriptor_counts,
        archive_snapshot_backbone_signature_counts=archive_snapshot_backbone_signature_counts,
        archive_snapshot_cnn_signature_counts=archive_snapshot_cnn_signature_counts,
        archive_snapshot_backbone_cnn_pair_counts=archive_snapshot_backbone_cnn_pair_counts,
        group_baseline_train_acc=group_baseline_train_acc,
        group_baseline_reward_target_acc=group_baseline_reward_target_acc,
        reward_batch_index=reward_batch_index,
        reward_group_id=reward_group_id,
        group_warmup=group_warmup,
        completion_index=completion_index,
        batch_last_item=batch_last_item,
    )
    meta = extract_completion_meta(completion)
    raw_delta = 0.0

    raw_delta -= 0.35 * min(int(meta.get("class_count", 0)), 2)
    raw_delta -= 0.12 * min(int(meta.get("import_count", 0)), 2)
    raw_delta -= 0.35 * min(int(meta.get("bad_signature_count", 0)), 2)

    xml_tag_count = int(meta.get("xml_tag_count", 0))
    if xml_tag_count < 3:
        raw_delta -= 0.45 * (3 - xml_tag_count)
    if not meta.get("xml_tag_exact"):
        raw_delta -= 1.20
    if not meta.get("exact_block_signature"):
        raw_delta -= 0.50
    if not meta.get("exact_init_signature"):
        raw_delta -= 0.60
    if not meta.get("exact_forward_signature"):
        raw_delta -= 0.60

    if meta.get("dual_backbone_ok"):
        raw_delta += 0.45
    else:
        if not meta.get("dual_backbone_init_ok"):
            raw_delta -= 1.75
        if not meta.get("dual_backbone_forward_ok"):
            raw_delta -= 1.75

    res["reward"] = TuneRL._apply_trainability_clamp(
        res,
        float(res.get("reward", -2.0)) + raw_delta,
        graph_info,
    )

    core_format_violation = bool(
        not meta.get("xml_tag_exact")
        or not meta.get("exact_block_signature")
        or not meta.get("exact_init_signature")
        or not meta.get("exact_forward_signature")
    )
    class_count = int(meta.get("class_count", 0))
    import_count = int(meta.get("import_count", 0))
    minor_hygiene_violation = class_count > 0 or import_count > 0
    severe_hygiene_violation = class_count > 2 or import_count > 2

    if core_format_violation:
        res["reward"] = min(float(res["reward"]), -3.0)
    elif severe_hygiene_violation:
        res["reward"] = min(float(res["reward"]), -2.0)
    elif minor_hygiene_violation:
        res["reward"] = min(float(res["reward"]), -1.5)

    if not meta.get("dual_backbone_ok"):
        res["reward"] = min(float(res["reward"]), -3.5)
    elif group_warmup and TuneRL._is_trainable_candidate(res, graph_info):
        res["reward"] = float(res.get("warmup_dense_reward") or 0.0)

    res["raw_extraction"] = {
        **meta,
        "raw_delta": raw_delta,
    }
    res.setdefault("backbone_model_names", list(meta.get("backbone_model_names", [])))
    return res


def _sft_runtime_state_hooks() -> TrainingRuntime.RuntimeStateHooks:
    # SFT reuses the RL reward runtime state so checkpoint resume stays compatible.
    return TrainingRuntime.RuntimeStateHooks(
        capture=TuneRL.capture_reward_runtime_state,
        restore=TuneRL.restore_reward_runtime_state,
        reset=TuneRL.reset_reward_runtime_state,
    )


def _sft_runtime_state_hooks() -> TrainingRuntime.RuntimeStateHooks:
    # SFT reuses the RL reward runtime state so checkpoint resume stays compatible.
    return TrainingRuntime.RuntimeStateHooks(
        capture=TuneRL.capture_reward_runtime_state,
        restore=TuneRL.restore_reward_runtime_state,
        reset=TuneRL.reset_reward_runtime_state,
    )


def _repo_model_dir(model_id: str) -> Path:
    return Path("out/llm") / model_id


def _repo_tokenizer_dir(model_id: str) -> Path:
    return Path("out/tokenizer") / model_id


def _has_model_files(model_dir: Path) -> bool:
    if not model_dir.is_dir():
        return False
    if not (model_dir / "config.json").exists():
        return False
    return any(
        (model_dir / filename).exists()
        for filename in (
            "model.safetensors",
            "model.safetensors.index.json",
            "pytorch_model.bin",
            "pytorch_model.bin.index.json",
        )
    )


def _has_tokenizer_files(tokenizer_dir: Path) -> bool:
    if not tokenizer_dir.is_dir():
        return False
    return any(
        (tokenizer_dir / filename).exists()
        for filename in (
            "tokenizer_config.json",
            "tokenizer.json",
            "tokenizer.model",
            "vocab.json",
        )
    )


def resolve_sft_model_sources() -> tuple[str, str, str]:
    repo_model_dir = _repo_model_dir(SFT_BASE_MODEL_ID)
    repo_tokenizer_dir = _repo_tokenizer_dir(SFT_BASE_MODEL_ID)

    if _has_model_files(repo_model_dir):
        if _has_tokenizer_files(repo_model_dir):
            return str(repo_model_dir), str(repo_model_dir), "out/llm"
        if _has_tokenizer_files(repo_tokenizer_dir):
            return str(repo_model_dir), str(repo_tokenizer_dir), "out/llm+out/tokenizer"
        return str(repo_model_dir), SFT_BASE_MODEL_ID, "out/llm+model-id-tokenizer"

    if _has_tokenizer_files(repo_tokenizer_dir):
        return SFT_BASE_MODEL_ID, str(repo_tokenizer_dir), "model-id+out/tokenizer"

    return SFT_BASE_MODEL_ID, SFT_BASE_MODEL_ID, "model-id-download"


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return bool(default)
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_str(name: str, default: str) -> str:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return str(default)
    return str(raw).strip()


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw == "":
        return int(default)
    return int(raw)


def resolve_sft_init_adapter() -> str:
    return _env_str("NNGPT_SFT_INIT_ADAPTER", SFT_INIT_ADAPTER)


def resolve_sft_load_initial_adapter() -> bool:
    return _env_flag("NNGPT_SFT_LOAD_INITIAL_ADAPTER", SFT_LOAD_INITIAL_ADAPTER)


def resolve_sft_save_rl_model() -> bool:
    return _env_flag("NNGPT_SFT_SAVE_RL_MODEL", SFT_SAVE_RL_MODEL)


def resolve_sft_model_out() -> str:
    return _env_str("NNGPT_SFT_MODEL_OUT", SFT_MODEL_OUT)


def resolve_sft_log_dir() -> str:
    return _env_str("NNGPT_SFT_LOG_DIR", SFT_LOG_DIR)


def resolve_sft_epoch_root() -> str:
    return _env_str("NNGPT_SFT_EPOCH_ROOT", SFT_EPOCH_ROOT)


def resolve_sft_trainer_out() -> str:
    return _env_str("NNGPT_SFT_TRAINER_OUT", SFT_TRAINER_OUT)


def resolve_sft_num_epochs() -> int:
    return _env_int("NNGPT_SFT_NUM_EPOCHS", SFT_NUM_EPOCHS)


def resolve_sft_save_steps() -> int:
    return max(1, _env_int("NNGPT_SFT_SAVE_STEPS", 5))


def resolve_sft_save_total_limit() -> int:
    return max(1, _env_int("NNGPT_SFT_SAVE_TOTAL_LIMIT", 4))


def _resolve_sft_resume_spec() -> Dict[str, Any]:
    resume_spec = TrainingRuntime.resolve_resume_spec(
        trainer_env="NNGPT_SFT_RESUME_TRAINER_CHECKPOINT",
        stage_env="NNGPT_SFT_RESUME_STAGE_CHECKPOINT",
        initial_adapter_active=resolve_sft_load_initial_adapter(),
        initial_adapter_label="NNGPT_SFT_LOAD_INITIAL_ADAPTER",
        legacy_state_filenames=("reward_state.json",),
    )
    return {
        "mode": resume_spec.mode,
        "trainer_checkpoint": resume_spec.trainer_checkpoint,
        "stage_checkpoint_dir": resume_spec.stage_checkpoint_dir,
        "stage_adapter_dir": resume_spec.stage_adapter_dir,
        "active": resume_spec.active,
    }


def _sft_grpo_signature_parameters() -> Dict[str, inspect.Parameter]:
    return dict(inspect.signature(TuneRL.GRPOConfig.__init__).parameters)


def _sft_trainer_checkpoint_supported() -> bool:
    signature_parameters = _sft_grpo_signature_parameters()
    return "save_strategy" in signature_parameters and "save_steps" in signature_parameters


def _runtime_is_main_process(runtime: Dict[str, Any]) -> bool:
    return int(runtime.get("rank", 0)) == 0


def _resolved_visible_cuda_device_tokens(visible_cuda_devices: int) -> List[str]:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is not None:
        raw = raw.strip()
        if raw in {"", "-1"}:
            return []
        tokens = [token.strip() for token in raw.split(",") if token.strip()]
        if tokens:
            return tokens
    return [str(index) for index in range(max(0, int(visible_cuda_devices)))]


def _resolve_sft_mode(visible_gpu_tokens: List[str]) -> Dict[str, str]:
    role_plan = TrainingRuntime.resolve_role_plan(
        visible_gpu_tokens=visible_gpu_tokens,
        requested_mode_env="NNGPT_SFT_MODE",
        default_mode=SFT_MODE_DEFAULT,
    )
    return {
        "requested_mode": str(role_plan.requested_mode),
        "resolved_mode": str(role_plan.resolved_mode),
    }


def _configure_sft_gpu_role_env(visible_cuda_devices: int) -> Dict[str, List[str] | str]:
    visible_gpu_tokens = _resolved_visible_cuda_device_tokens(visible_cuda_devices)
    role_plan = TrainingRuntime.resolve_role_plan(
        visible_gpu_tokens=visible_gpu_tokens,
        requested_mode_env="NNGPT_SFT_MODE",
        default_mode=SFT_MODE_DEFAULT,
    )
    train_gpu_tokens = list(role_plan.train_gpu_tokens)
    reward_gpu_tokens = list(role_plan.aux_gpu_tokens)
    if train_gpu_tokens:
        os.environ[_TRAIN_GPU_TOKENS_ENV] = ",".join(train_gpu_tokens)
    else:
        os.environ.pop(_TRAIN_GPU_TOKENS_ENV, None)
    if reward_gpu_tokens:
        os.environ[_AUX_GPU_TOKENS_ENV] = ",".join(reward_gpu_tokens)
        os.environ[_REWARD_GPU_TOKENS_ENV] = ",".join(reward_gpu_tokens)
    else:
        os.environ.pop(_AUX_GPU_TOKENS_ENV, None)
        os.environ.pop(_REWARD_GPU_TOKENS_ENV, None)
    return {
        "requested_mode": str(role_plan.requested_mode),
        "resolved_mode": str(role_plan.resolved_mode),
        "visible_gpu_tokens": list(role_plan.visible_gpu_tokens),
        "train_gpu_tokens": list(train_gpu_tokens),
        "reward_gpu_tokens": list(reward_gpu_tokens),
    }


def _suggested_sft_worker_count(runtime: Dict[str, Any]) -> int:
    _ = runtime
    return 1


def _validate_sft_visible_worker_count(runtime: Dict[str, Any]) -> None:
    world_size = int(runtime.get("world_size", 1))
    if world_size == 1:
        return
    raise RuntimeError(
        "SFT RL runs with a single training rank so reward workers can use assigned GPUs: "
        f"world_size={world_size}, "
        f"visible_cuda_devices={int(runtime.get('visible_gpu_count', 0))}, "
        f"suggested_nproc_per_node={_suggested_sft_worker_count(runtime)}. "
        "Launch without torchrun or use --nproc_per_node=1."
    )


def _resolve_sft_local_rank(runtime: Dict[str, Any]) -> Dict[str, Any]:
    visible_cuda_devices = int(runtime.get("visible_gpu_count", 0))
    rank = int(runtime.get("rank", 0))
    world_size = int(runtime.get("world_size", 1))

    if visible_cuda_devices < 1:
        raise RuntimeError(
            f"SFT RL requires at least one visible CUDA device, got {visible_cuda_devices}"
        )
    local_rank = 0

    return {
        "rank": rank,
        "world_size": world_size,
        "raw_local_rank": int(runtime.get("raw_local_rank", 0)),
        "local_rank": local_rank,
        "visible_cuda_devices": visible_cuda_devices,
        "resolution": "single_process_training",
    }


def _maybe_relaunch_sft_with_visible_gpu_workers() -> None:
    if os.getenv("NNGPT_SFT_AUTO_TORCHRUN_DONE", "") == "1":
        return
    if os.getenv("WORLD_SIZE") not in (None, "", "1"):
        return
    if os.getenv("LOCAL_RANK") not in (None, ""):
        return

    import torch

    if not torch.cuda.is_available():
        return
    visible_cuda_devices = int(torch.cuda.device_count())
    _configure_sft_gpu_role_env(visible_cuda_devices)
    if visible_cuda_devices <= 1:
        return

    master_port = _resolve_sft_master_port()
    os.environ["NNGPT_SFT_AUTO_TORCHRUN_DONE"] = "1"
    print(
        "[SFT RL] Relaunching with single training worker: "
        f"nproc_per_node=1 master_addr={os.environ['MASTER_ADDR']} master_port={master_port}"
    )
    os.execvpe(
        sys.executable,
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--nproc_per_node=1",
            f"--master_addr={os.environ['MASTER_ADDR']}",
            f"--master_port={master_port}",
            "-m",
            "ab.gpt.TuneRLSft",
        ],
        os.environ,
    )


def _maybe_init_single_process_deepspeed_group(
    *,
    use_deepspeed: bool,
    world_size: int,
    visible_cuda_devices: int,
    local_rank: int,
) -> None:
    import torch

    if not use_deepspeed:
        return
    if int(world_size) > 1:
        return
    if not torch.distributed.is_available():
        return
    if torch.distributed.is_initialized():
        return

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    init_dir = Path(tempfile.mkdtemp(prefix="nngpt_sft_pg_"))
    init_file = (init_dir / "store").resolve()
    init_file.touch()
    init_method = init_file.as_uri()

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    torch.distributed.init_process_group(
        backend=backend,
        init_method=init_method,
        world_size=1,
        rank=0,
    )
    print(
        "[SFT RL] Initialized single-process distributed group for DeepSpeed: "
        f"backend={backend} local_rank={local_rank} visible_cuda_devices={visible_cuda_devices}"
    )


def resolve_sft_runtime_settings(runtime: Dict[str, Any]) -> Dict[str, int]:
    grad_accum = _env_int("NNGPT_SFT_GRAD_ACCUM", SFT_GRAD_ACCUM)
    generation_plan = TuneRL.resolve_generation_plan(
        runtime,
        env_name="NNGPT_SFT_NUM_GENERATIONS",
        default=SFT_NUM_GENERATIONS,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=grad_accum,
    )
    return {
        "dataset_limit": _env_int(
            "NNGPT_SFT_DATASET_LIMIT",
            SFT_DATASET_LIMIT,
        ),
        "grad_accum": grad_accum,
        "max_completion_length": _env_int(
            "NNGPT_SFT_MAX_COMPLETION_LENGTH",
            SFT_MAX_COMPLETION_LENGTH,
        ),
        "effective_train_batch_size": generation_plan["effective_train_batch_size"],
        "requested_global_num_generations": generation_plan["requested_global_num_generations"],
        "global_num_generations": generation_plan["global_num_generations"],
        "effective_global_num_generations": generation_plan["effective_global_num_generations"],
        "global_num_generations_adapted": generation_plan["global_num_generations_adapted"],
        "valid_generation_values": generation_plan["valid_generation_values"],
    }


def _resolve_sft_deepspeed_enabled(runtime: Dict[str, Any]) -> bool:
    raw = os.getenv("NNGPT_SFT_USE_DEEPSPEED")
    if raw is None or raw == "":
        return int(runtime.get("world_size", 1)) > 1
    return _env_flag("NNGPT_SFT_USE_DEEPSPEED", False)


def _resolve_sft_deepspeed_config_path() -> str:
    raw_path = os.getenv("NNGPT_SFT_DEEPSPEED_CONFIG", SFT_DEEPSPEED_DEFAULT_CONFIG)
    config_path = Path(raw_path).expanduser()
    if not config_path.exists():
        raise FileNotFoundError(f"SFT DeepSpeed config not found: {config_path}")
    return str(config_path)


def _maybe_init_hf_deepspeed_config(config_path: str) -> Any:
    last_error: Exception | None = None
    for module_name in ("transformers.integrations", "transformers.deepspeed"):
        try:
            module = __import__(module_name, fromlist=["HfDeepSpeedConfig"])
            config_cls = getattr(module, "HfDeepSpeedConfig", None)
            if config_cls is not None:
                return config_cls(config_path)
        except Exception as exc:
            last_error = exc
    raise RuntimeError(
        "DeepSpeed ZeRO-3 requested for SFT GRPO, but HfDeepSpeedConfig could not be imported"
    ) from last_error


def _bootstrap_run_token(runtime: Dict[str, Any] | None = None) -> str:
    runtime = runtime or {}
    for value in (
        os.getenv("TORCHELASTIC_RUN_ID"),
        os.getenv("MASTER_PORT"),
        os.getenv("SLURM_JOB_ID"),
    ):
        if value:
            return str(value)
    return f"rank0_world{int(runtime.get('world_size', 1))}"


def _resolve_sft_master_port() -> str:
    master_port = str(os.getenv("MASTER_PORT", "")).strip()
    if master_port:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        return master_port

    slurm_job_id = str(os.getenv("SLURM_JOB_ID", "")).strip()
    if slurm_job_id.isdigit():
        master_port = str(20000 + (int(slurm_job_id) % 20000))
    else:
        master_port = str(20000 + (os.getpid() % 20000))
    os.environ["MASTER_PORT"] = master_port
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    return master_port


def _bootstrap_sentinel_path(log_dir: str, runtime: Dict[str, Any] | None = None) -> Path:
    return Path(log_dir) / f".sft_bootstrap_complete.{_bootstrap_run_token(runtime)}"


def _wait_for_bootstrap_sentinel(
    sentinel_path: Path,
    *,
    timeout_seconds: float = 600.0,
) -> None:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if sentinel_path.exists():
            return
        time.sleep(1.0)
    raise TimeoutError(f"Timed out waiting for rank0 bootstrap sentinel: {sentinel_path}")


def _cifar_eval_error(error: Exception, *, seed_accuracy_baseline: float | None = None) -> Dict[str, Any]:
    error_type = type(error).__name__
    error_msg = f"{error_type}: {error}"
    return {
        "reward": 0.0,
        "components": {
            "reward": 0.0,
            "r_build": 0.0,
            "r_forward_shape": 0.0,
            "r_backward": 0.0,
            "r_loss_drop": 0.0,
            "r_forward": 0.0,
            "r_trainstep": 0.0,
            "r_metric": 0.0,
            "r_eff": 0.0,
            "r_critic": 0.0,
            "r_kl": 0.0,
        },
        "test_acc": None,
        "val_metric": None,
        "built_ok": False,
        "forward_ok": False,
        "forward_shape_ok": False,
        "trained_step_ok": False,
        "backward_ok": False,
        "loss_start": None,
        "loss_end": None,
        "loss_drop": None,
        "loss_drop_ok": False,
        "train_acc": None,
        "seed_accuracy_baseline": seed_accuracy_baseline,
        "seed_train_acc_gap": None,
        "seed_train_acc_improved": False,
        "accuracy_baseline": seed_accuracy_baseline,
        "train_acc_gain": None,
        "train_acc_improved": False,
        "group_baseline_train_acc": None,
        "group_train_acc_gain": None,
        "group_train_acc_improved": False,
        "reward_batch_index": None,
        "reward_group_id": None,
        "group_warmup": False,
        "latency_ms": None,
        "params_m": None,
        "timed_out": False,
        "estimated_total_seconds": None,
        "eval_limit_seconds": None,
        "warmup_dense_reward": None,
        "backbone_model_names": [],
        "frozen_train_acc": None,
        "frozen_test_acc": None,
        "unfrozen_train_acc": None,
        "unfrozen_test_acc": None,
        "frozen_eval": None,
        "unfrozen_eval": None,
        "reward_target_metric": "frozen_test_acc",
        "reward_target_value": None,
        "error": error_msg,
    }


def _torch_hub_checkpoints_dir() -> Path:
    import torch

    return Path(torch.hub.get_dir()) / "checkpoints"


def _print_runtime_cache_roots() -> None:
    print(f"[SFT RL] HF_HOME={os.environ.get('HF_HOME', '')!r}")
    print(f"[SFT RL] TORCH_HOME={os.environ.get('TORCH_HOME', '')!r}")
    print(f"[SFT RL] Torch hub checkpoints dir: {_torch_hub_checkpoints_dir()}")


def _configured_device_tokens(env_name: str) -> List[str]:
    raw = os.getenv(env_name, "")
    return [token.strip() for token in raw.split(",") if token.strip()]


def _validate_configured_gpu_role_tokens(runtime: Dict[str, Any]) -> Dict[str, Any]:
    visible_tokens = _resolved_visible_cuda_device_tokens(int(runtime.get("visible_gpu_count", 0)))
    mode_info = _resolve_sft_mode(visible_tokens)
    resolved_mode = str(mode_info["resolved_mode"])
    train_tokens = _configured_device_tokens(_TRAIN_GPU_TOKENS_ENV)
    reward_tokens = _configured_device_tokens(_REWARD_GPU_TOKENS_ENV)
    expected_train_tokens = visible_tokens[:1]
    expected_reward_tokens = list(visible_tokens) if resolved_mode == "split" else list(expected_train_tokens)
    if train_tokens != expected_train_tokens:
        raise RuntimeError(
            "Invalid configured training GPU tokens for SFT mode: "
            f"resolved_mode={resolved_mode} train_tokens={train_tokens} expected={expected_train_tokens}"
        )
    if reward_tokens != expected_reward_tokens:
        raise RuntimeError(
            "Invalid configured reward GPU tokens for SFT mode: "
            f"resolved_mode={resolved_mode} reward_tokens={reward_tokens} expected={expected_reward_tokens}"
        )
    return {
        "requested_mode": str(mode_info["requested_mode"]),
        "resolved_mode": resolved_mode,
        "visible_tokens": visible_tokens,
        "train_tokens": train_tokens,
        "reward_tokens": reward_tokens,
    }


def _validate_gpu_reward_worker_bindings(warmup_diagnostics: Dict[str, Any]) -> None:
    workers = list(warmup_diagnostics.get("workers", []) or [])
    failures: List[str] = []
    for worker in workers:
        if worker.get("assigned_gpu") is None:
            continue
        cuda_visible_devices = str(worker.get("cuda_visible_devices", "")).strip()
        if int(worker.get("cuda_device_count", 0) or 0) != 1:
            failures.append(
                f"slot={worker.get('slot')} expected cuda_device_count=1 got {worker.get('cuda_device_count')!r}"
            )
        if str(worker.get("worker_device", "")) != "cuda:0":
            failures.append(
                f"slot={worker.get('slot')} expected worker_device='cuda:0' got {worker.get('worker_device')!r}"
            )
        if not cuda_visible_devices:
            failures.append(
                f"slot={worker.get('slot')} expected non-empty CUDA_VISIBLE_DEVICES in worker handshake"
            )
        if not bool(worker.get("physical_binding_verified", False)):
            failures.append(
                f"slot={worker.get('slot')} physical_binding_verified=False"
            )
    if failures:
        raise RuntimeError(
            "SFT reward worker GPU binding hard validation failed: " + "; ".join(failures)
        )


def evaluate_code_and_reward_cifar(
    code: str,
    *,
    stage_name=None,
    in_shape=(1, 3, 256, 256),
    out_shape=(10,),
    prm=None,
    device: str = "cpu",
    val_metric_baseline=None,
    seed_accuracy_baseline=None,
    cfg=None,
    reward_batch_index: int | None = None,
    completion_index: int | None = None,
    batch_last_item: bool = False,
) -> Dict[str, Any]:
    try:
        import torch

        eval_device = "cuda" if torch.cuda.is_available() else "cpu"
        if prm is None:
            prm = {"lr": 1e-2, "momentum": 0.9, "dropout": 0.3}
        defaults = {
            "lr": 1e-2,
            "momentum": 0.9,
            "batch": SFT_EVAL_BATCH_SIZE,
            "epoch": 1,
            "transform": SFT_EVAL_TRANSFORM,
        }
        prm = {**defaults, **prm}
        cfg = build_sft_reward_eval_cfg(
            stage_name=stage_name,
            in_shape=in_shape,
            out_shape=out_shape,
            prm=prm,
            cfg=cfg,
            device=eval_device,
        )

        return RewardUtil.evaluate_code_and_reward(
            code,
            in_shape=in_shape,
            out_shape=out_shape,
            prm=prm,
            device=eval_device,
            val_metric_baseline=val_metric_baseline,
            seed_accuracy_baseline=seed_accuracy_baseline,
            cfg=cfg,
            reward_batch_index=reward_batch_index,
            completion_index=completion_index,
            batch_last_item=batch_last_item,
        )
    except Exception as exc:
        return _cifar_eval_error(exc, seed_accuracy_baseline=seed_accuracy_baseline)


def build_sft_reward_eval_cfg(
    *,
    stage_name=None,
    in_shape=(1, 3, 256, 256),
    out_shape=(10,),
    prm=None,
    cfg=None,
    device=None,
    **_unused_kwargs,
):
    import torch

    del _unused_kwargs

    eval_device = str(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if prm is None:
        prm = {"lr": 1e-2, "momentum": 0.9, "dropout": 0.3}
    defaults = {
        "lr": 1e-2,
        "momentum": 0.9,
        "batch": SFT_EVAL_BATCH_SIZE,
        "epoch": 1,
        "transform": SFT_EVAL_TRANSFORM,
    }
    prm = {**defaults, **prm}
    effective_stage_name = str(stage_name or TuneRL.current_stage_name)

    if cfg is None:
        cfg = TuneRL.build_stage_eval_cfg(
            stage_name=effective_stage_name,
            in_shape=tuple(in_shape),
            out_shape=tuple(out_shape),
            prm=prm,
            device=eval_device,
        )

    return RewardUtil.EvalConfig(
        device=eval_device,
        input_shape=tuple(getattr(cfg, "input_shape", in_shape)),
        n_classes=int(getattr(cfg, "n_classes", out_shape[0])),
        train_epochs=int(prm.get("epoch", getattr(cfg, "train_epochs", SFT_EVAL_TRAIN_EPOCHS)) or SFT_EVAL_TRAIN_EPOCHS),
        train_steps=getattr(cfg, "train_steps", None),
        max_val_batches=SFT_EVAL_VAL_BATCHES,
        default_batch_size=SFT_EVAL_BATCH_SIZE,
        train_subset_size=SFT_EVAL_TRAIN_SUBSET,
        val_subset_size=SFT_EVAL_VAL_SUBSET,
        data_root=SFT_EVAL_DATA_ROOT,
        download=SFT_EVAL_DOWNLOAD,
        measure_latency=getattr(cfg, "measure_latency", True),
        kl_div=getattr(cfg, "kl_div", None),
        critic_fn=getattr(cfg, "critic_fn", None),
        weights=getattr(cfg, "weights", None),
        eval_limit_seconds=getattr(cfg, "eval_limit_seconds", SFT_EVAL_LIMIT_SECONDS),
        budget_probe_batches=getattr(cfg, "budget_probe_batches", None),
        run_unfrozen_backbone_eval=False,
        full_test_acc=SFT_EVAL_FULL_TEST_ACC,
        reward_target_metric=getattr(cfg, "reward_target_metric", "frozen_test_acc"),
        formal_nn_eval=getattr(cfg, "formal_nn_eval", False),
        static_only=getattr(cfg, "static_only", False),
        formal_task=getattr(cfg, "formal_task", "img-classification"),
        formal_dataset=getattr(cfg, "formal_dataset", "cifar-10"),
        formal_metric=getattr(cfg, "formal_metric", "acc"),
        formal_epoch_limit_minutes=getattr(
            cfg,
            "formal_epoch_limit_minutes",
            SFT_EVAL_FORMAL_EPOCH_LIMIT_MINUTES,
        ),
    )


def _is_trainable_architecture(res: Dict[str, Any], graph_info) -> bool:
    discovery_meta = res.get("open_discovery", {})
    parse_ok = bool(getattr(graph_info, "parse_ok", False) or discovery_meta.get("parse_ok", False))
    return bool(
        parse_ok
        and res.get("built_ok")
        and res.get("forward_shape_ok")
        and res.get("backward_ok")
        and res.get("loss_drop_ok")
    )


def _reapply_trainability_clamp(res: Dict[str, Any], reward_value: float, graph_info) -> float:
    discovery_meta = res.get("open_discovery", {})
    parse_ok = bool(getattr(graph_info, "parse_ok", False) or discovery_meta.get("parse_ok", False))

    if not parse_ok:
        reward_value = min(reward_value, -0.25)
    if not res.get("built_ok"):
        build_partial = float(res.get("r_build_partial", 0.0))
        reward_value = min(reward_value, -0.8 + build_partial)
    elif not res.get("forward_shape_ok"):
        reward_value = min(reward_value, -0.50)
    elif not res.get("backward_ok"):
        reward_value = min(reward_value, -0.10)
    elif not res.get("loss_drop_ok"):
        reward_value = min(reward_value, 0.0)
    return reward_value


def sft_reward_fn(
    completion: str,
    *,
    seed_accuracy_baseline: float,
    precomputed_eval_result: Dict[str, Any] | None = None,
    graph_info=None,
    batch_graph_hashes: List[str] = None,
    batch_family_hashes: List[str] = None,
    batch_descriptor_keys: List[str] = None,
    batch_backbone_signatures: List[str] = None,
    batch_cnn_signatures: List[str] = None,
    prompt_goal_tags: List[str] = None,
    archive_snapshot_family_counts: Dict[str, int] = None,
    archive_snapshot_descriptor_counts: Dict[str, int] = None,
    archive_snapshot_backbone_signature_counts: Dict[str, int] = None,
    archive_snapshot_cnn_signature_counts: Dict[str, int] = None,
    archive_snapshot_backbone_cnn_pair_counts: Dict[str, int] = None,
    group_baseline_train_acc: float | None = None,
    group_baseline_reward_target_acc: float | None = None,
    reward_batch_index: int | None = None,
    reward_group_id: int | None = None,
    group_warmup: bool = False,
    completion_index: int | None = None,
    batch_last_item: bool = False,
):
    res = raw_reward_fn(
        completion,
        seed_accuracy_baseline=seed_accuracy_baseline,
        precomputed_eval_result=precomputed_eval_result,
        graph_info=graph_info,
        batch_graph_hashes=batch_graph_hashes,
        batch_family_hashes=batch_family_hashes,
        batch_descriptor_keys=batch_descriptor_keys,
        batch_backbone_signatures=batch_backbone_signatures,
        batch_cnn_signatures=batch_cnn_signatures,
        prompt_goal_tags=prompt_goal_tags,
        archive_snapshot_family_counts=archive_snapshot_family_counts,
        archive_snapshot_descriptor_counts=archive_snapshot_descriptor_counts,
        archive_snapshot_backbone_signature_counts=archive_snapshot_backbone_signature_counts,
        archive_snapshot_cnn_signature_counts=archive_snapshot_cnn_signature_counts,
        archive_snapshot_backbone_cnn_pair_counts=archive_snapshot_backbone_cnn_pair_counts,
        group_baseline_train_acc=group_baseline_train_acc,
        group_baseline_reward_target_acc=group_baseline_reward_target_acc,
        reward_batch_index=reward_batch_index,
        reward_group_id=reward_group_id,
        group_warmup=group_warmup,
        completion_index=completion_index,
        batch_last_item=batch_last_item,
    )
    res["reward"] = _reapply_trainability_clamp(res, float(res.get("reward", -2.0)), graph_info)
    res["anti_collapse"] = {
        "goal_key": TuneRL.primary_goal_key(prompt_goal_tags),
        "trainable_ok": _is_trainable_architecture(res, graph_info),
        "anti_collapse_delta": 0.0,
    }
    return res


SFT_DISCOVERY_PROMPT_TEMPLATE = SFTUtil.open_discovery_rl_prompt_template


class DynamicSFTPromptDataset(TorchDataset):
    column_names = ["prompt", "accuracy", "goal_name", "target_tags", "goal_profile_id"]

    def __init__(
        self,
        rows: List[Dict[str, Any]],
        tokenizer,
        *,
        block_signature: str,
        init_signature: str,
        forward_signature: str,
    ) -> None:
        self.rows = list(rows)
        self.tokenizer = tokenizer
        self.block_signature = block_signature
        self.init_signature = init_signature
        self.forward_signature = forward_signature

    def __len__(self) -> int:
        return len(self.rows)

    def select(self, indices) -> "DynamicSFTPromptDataset":
        if hasattr(indices, "tolist"):
            indices = indices.tolist()
        return DynamicSFTPromptDataset(
            [self.rows[int(index)] for index in indices],
            self.tokenizer,
            block_signature=self.block_signature,
            init_signature=self.init_signature,
            forward_signature=self.forward_signature,
        )

    def _render_prompt(self, row: Dict[str, Any]) -> str:
        profile = SFTUtil.open_discovery_goal_profiles[int(row["goal_profile_id"])]
        target_pattern = SFTUtil.goal_profile_target_pattern(profile)
        module_hints = (
            "self.backbone_a",
            "self.backbone_b",
            *profile["module_hints"],
        )
        user_prompt = SFT_DISCOVERY_PROMPT_TEMPLATE.format(
            accuracy=row["accuracy"],
            skeleton_code=SFTUtil.open_discovery_skeleton_code,
            available_backbones=", ".join(SFTUtil.available_backbones),
            legacy_patterns=", ".join(SFTUtil.legacy_patterns),
            goal_name=profile["name"],
            target_tags=", ".join(profile["tags"]),
            target_pattern=target_pattern,
            design_brief=profile["brief"],
            tag_realization=profile.get("realization", profile["brief"]),
            goal_tag_parser_cues=SFTUtil.goal_tag_parser_cues(profile["tags"]),
            module_hints=", ".join(module_hints),
            block_signature=self.block_signature,
            init_signature=self.init_signature,
            forward_signature=self.forward_signature,
        )
        feedback_text = TuneRL.render_prompt_feedback_text(
            feedback_char_budget=SFT_FEEDBACK_CHAR_BUDGET,
        )
        feedback_section = "\n\n### Current Optimization Feedback\n" + feedback_text.strip() + "\n"
        marker = "### Output Requirement (STRICT)"
        if marker in user_prompt:
            user_prompt = user_prompt.replace(marker, feedback_section + "\n" + marker, 1)
        else:
            user_prompt = user_prompt + feedback_section
        messages = [{"role": "user", "content": user_prompt}]
        return self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.rows[int(index)]
        return {
            "prompt": self._render_prompt(row),
            "accuracy": row["accuracy"],
            "goal_name": row["goal_name"],
            "target_tags": row["target_tags"],
            "goal_profile_id": row["goal_profile_id"],
        }


def load_rl_dataset_sft(tokenizer) -> TuneRL.Dataset:
    """Load SFT-aligned RL prompts while rendering feedback lazily at access time."""
    runtime_settings = resolve_sft_runtime_settings(RewardUtil.get_distributed_runtime_info())
    data = TuneRL.api.data(task="img-classification", nn_prefixes=("rl-bb-test1",))
    if data.empty:
        print("No 'rl-bb-test1' data found, falling back to all img-classification")
        data = TuneRL.api.data(only_best_accuracy=True, task="img-classification", dataset="cifar-10")

    print(f"Loaded {len(data)} examples for SFT RL")
    TuneRL.bootstrap_trainset_reference_library(data)

    rows: List[Dict[str, Any]] = []

    for _, row in data.iterrows():
        accuracy = TuneRL._coerce_accuracy_baseline(row.get("accuracy"), context="seed row accuracy")
        for profile_id, profile in enumerate(SFTUtil.open_discovery_goal_profiles):
            rows.append(
                {
                    "accuracy": accuracy,
                    "goal_name": profile["name"],
                    "target_tags": ", ".join(profile["tags"]),
                    "goal_profile_id": profile_id,
                }
            )

    random.Random(42).shuffle(rows)
    if len(rows) > runtime_settings["dataset_limit"]:
        rows = rows[:runtime_settings["dataset_limit"]]
    return DynamicSFTPromptDataset(
        rows,
        tokenizer,
        block_signature=BLOCK_SIGNATURE,
        init_signature=INIT_SIGNATURE,
        forward_signature=FORWARD_SIGNATURE,
    )


def sft_run_epoch_dir(*args) -> Path:
    epoch_dir = Path(resolve_sft_epoch_root())
    for value in args:
        epoch_dir = epoch_dir / f"A{value}"
    return epoch_dir


def _build_sft_grpo_config(
    *,
    precision: Dict[str, Any],
    use_deepspeed: bool,
    deepspeed_config_path: str | None,
    runtime_settings: Dict[str, int],
) -> Any:
    signature_parameters = _sft_grpo_signature_parameters()
    config_kwargs: Dict[str, Any] = {
        "temperature": SFT_TEMPERATURE,
        "learning_rate": SFT_LR,
        "max_completion_length": runtime_settings["max_completion_length"],
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": runtime_settings["grad_accum"],
        "lr_scheduler_type": "cosine",
        "num_train_epochs": resolve_sft_num_epochs(),
        "remove_unused_columns": False,
        "logging_steps": 1,
        "output_dir": resolve_sft_trainer_out(),
        "eval_strategy": "no",
        "bf16": precision["bf16"],
        "fp16": precision["fp16"],
        "gradient_checkpointing": True,
        "num_generations": runtime_settings["global_num_generations"],
    }
    if "gradient_checkpointing_kwargs" in signature_parameters:
        config_kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    if "save_strategy" in signature_parameters:
        config_kwargs["save_strategy"] = "steps"
    if "save_steps" in signature_parameters:
        config_kwargs["save_steps"] = resolve_sft_save_steps()
    if "save_total_limit" in signature_parameters:
        config_kwargs["save_total_limit"] = resolve_sft_save_total_limit()
    explicit_kl_coef = TuneRL.env_float("NNGPT_RL_KL_COEF", SFT_KL_COEF)
    if "beta" in signature_parameters:
        config_kwargs["beta"] = explicit_kl_coef
    elif "kl_coef" in signature_parameters:
        config_kwargs["kl_coef"] = explicit_kl_coef
    else:
        raise RuntimeError("Installed GRPOConfig does not expose `beta` or `kl_coef`; cannot set explicit KL control")
    if use_deepspeed:
        if "deepspeed" not in signature_parameters:
            raise RuntimeError("Installed GRPOConfig does not support the `deepspeed` argument")
        config_kwargs["deepspeed"] = deepspeed_config_path
        if "ds3_gather_for_generation" in signature_parameters:
            config_kwargs["ds3_gather_for_generation"] = False
    return TuneRL.GRPOConfig(**config_kwargs)


def run_sft_training():
    import torch
    from transformers import BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise RuntimeError("SFT RL requires CUDA for GRPO training, but no CUDA device is available")
    resume_spec = _resolve_sft_resume_spec()
    load_initial_adapter = resolve_sft_load_initial_adapter()
    init_adapter_path = resolve_sft_init_adapter()
    save_rl_model = resolve_sft_save_rl_model()
    model_out_path = resolve_sft_model_out()
    trainer_checkpoint_supported = _sft_trainer_checkpoint_supported()
    if resume_spec["mode"] == "trainer":
        train_signature = inspect.signature(TuneRL.GRPOTrainer.train)
        if "resume_from_checkpoint" not in train_signature.parameters:
            raise RuntimeError("Installed GRPOTrainer does not support trainer resume_from_checkpoint.")
        if not trainer_checkpoint_supported:
            raise RuntimeError(
                "Installed GRPOConfig does not support save_strategy/save_steps, so trainer checkpoint resume is unavailable."
            )
    runtime = RewardUtil.get_distributed_runtime_info()
    runtime_settings = resolve_sft_runtime_settings(runtime)
    _validate_sft_visible_worker_count(runtime)
    local_rank_resolution = _resolve_sft_local_rank(runtime)
    visible_cuda_devices = int(local_rank_resolution["visible_cuda_devices"])
    local_rank = int(local_rank_resolution["local_rank"])
    raw_local_rank = int(local_rank_resolution["raw_local_rank"])
    rank = int(local_rank_resolution["rank"])
    world_size = int(local_rank_resolution["world_size"])
    use_deepspeed = _resolve_sft_deepspeed_enabled(runtime)
    deepspeed_config_path = _resolve_sft_deepspeed_config_path() if use_deepspeed else None
    os.environ["NNGPT_SFT_USE_DEEPSPEED"] = "1" if use_deepspeed else "0"
    if deepspeed_config_path is not None:
        os.environ["NNGPT_SFT_DEEPSPEED_CONFIG"] = deepspeed_config_path
    torch.cuda.set_device(local_rank)
    train_device = f"cuda:{local_rank}"
    _maybe_init_single_process_deepspeed_group(
        use_deepspeed=use_deepspeed,
        world_size=world_size,
        visible_cuda_devices=visible_cuda_devices,
        local_rank=local_rank,
    )

    torch.cuda.empty_cache()
    trainer_checkpoint = resume_spec["trainer_checkpoint"]
    stage_checkpoint_dir = resume_spec["stage_checkpoint_dir"]
    stage_adapter_dir = resume_spec["stage_adapter_dir"]
    resume_state_dir = trainer_checkpoint if trainer_checkpoint is not None else stage_checkpoint_dir
    resume_stage_override = os.getenv("NNGPT_RL_RESUME_STAGE", "").strip()
    # Restore pipeline runtime bookkeeping the same way for trainer and stage checkpoints.
    restored_runtime_state_path = TrainingRuntime.restore_or_reset_runtime_state(
        resume_state_dir,
        _sft_runtime_state_hooks(),
        legacy_state_filenames=("reward_state.json",),
    )
    if resume_state_dir is not None and resume_stage_override:
        current_state_stage = str(TuneRL.current_stage_name)
        if current_state_stage != resume_stage_override:
            print(
                "[SFT RL] Resume stage override "
                f"checkpoint_stage={current_state_stage} requested_stage={resume_stage_override}"
            )
            TuneRL.current_stage_name = resume_stage_override
    precision = TuneRL.best_mixed_precision()
    grpo_config = _build_sft_grpo_config(
        precision=precision,
        use_deepspeed=use_deepspeed,
        deepspeed_config_path=deepspeed_config_path,
        runtime_settings=runtime_settings,
    )
    hf_deepspeed_config = _maybe_init_hf_deepspeed_config(deepspeed_config_path) if use_deepspeed else None
    if not trainer_checkpoint_supported:
        print(
            "[SFT RL] Warning: installed GRPOConfig does not support save_strategy/save_steps; "
            "trainer checkpoints will not be produced."
        )
    if restored_runtime_state_path is not None:
        print(f"[SFT RL] Restored runtime state from {restored_runtime_state_path}")

    print(f"Using RL base model: {TuneRL.base_model}")
    print(
        "[SFT RL] Distributed Runtime: "
        f"rank={rank} local_rank={local_rank} raw_local_rank={raw_local_rank} world_size={world_size}"
    )
    if use_deepspeed and world_size <= 1:
        print(
            "[SFT RL] DeepSpeed single-process fallback active: "
            f"visible_cuda_devices={visible_cuda_devices} local_rank={local_rank}"
        )
    print(f"[SFT RL] DeepSpeed Enabled: {use_deepspeed}")
    if deepspeed_config_path is not None:
        print(f"[SFT RL] DeepSpeed Config: {deepspeed_config_path}")
    print(f"[SFT RL] Fixed training device: {train_device}")
    print(f"[SFT RL] Visible CUDA devices: {visible_cuda_devices}")
    print(f"[SFT RL] Mixed precision: {precision['label']} (torch_dtype={precision['torch_dtype']})")
    print(f"[SFT RL] Current stage: {TuneRL.current_stage_name}")
    print(
        "[SFT RL] Runtime limits: "
        f"dataset_limit={runtime_settings['dataset_limit']} "
        f"max_completion_length={runtime_settings['max_completion_length']} "
        f"grad_accum={runtime_settings['grad_accum']} "
        f"effective_train_batch_size={runtime_settings['effective_train_batch_size']} "
        f"requested_global_num_generations={runtime_settings['requested_global_num_generations']} "
        f"global_num_generations={runtime_settings['global_num_generations']} "
        f"effective_global_num_generations={runtime_settings['effective_global_num_generations']}"
    )
    if runtime_settings["global_num_generations_adapted"]:
        print(
            "[SFT RL] Generation plan adapted "
            f"requested={runtime_settings['requested_global_num_generations']} "
            f"effective={runtime_settings['effective_global_num_generations']} "
            f"valid_generation_values={runtime_settings['valid_generation_values']} "
            f"world_size={world_size}"
        )
    reward_worker_plan = RewardUtil.get_reward_worker_plan()
    mode_validation = _validate_configured_gpu_role_tokens(runtime)
    print(
        "[SFT RL] Reward Worker Plan: "
        f"mode={reward_worker_plan['mode']} "
        f"reward_workers_per_gpu={reward_worker_plan.get('workers_per_gpu', 1)} "
        f"per_gpu_worker_counts={reward_worker_plan.get('per_gpu_worker_counts', [])} "
        f"rank={reward_worker_plan['rank']} "
        f"local_rank={reward_worker_plan['local_rank']} "
        f"world_size={reward_worker_plan['world_size']} "
        f"train_gpu={reward_worker_plan['train_gpu']} "
        f"reward_gpu_indices={reward_worker_plan['reward_gpu_indices']} "
        f"reward_gpu_tokens={reward_worker_plan.get('reward_gpu_tokens', [])} "
        f"reason={reward_worker_plan.get('reason', '')!r}"
    )
    print(
        "[SFT RL] GPU role validation: "
        f"requested_mode={mode_validation['requested_mode']} "
        f"resolved_mode={mode_validation['resolved_mode']} "
        f"train_tokens={mode_validation['train_tokens']} "
        f"reward_tokens={mode_validation['reward_tokens']}"
    )
    _print_runtime_cache_roots()
    tokenizer_source = getattr(TuneRL, "tokenizer_source", TuneRL.base_model)
    if tokenizer_source != TuneRL.base_model:
        print(f"Using RL tokenizer: {tokenizer_source}")
    tokenizer = TuneRL.AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rl_dataset = TuneRL.load_rl_dataset(tokenizer)
    if len(rl_dataset) > runtime_settings["dataset_limit"]:
        rl_dataset = rl_dataset.select(range(runtime_settings["dataset_limit"]))

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=precision["torch_dtype"],
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    model_load_kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
        "quantization_config": bnb_config,
        "torch_dtype": precision["torch_dtype"],
    }
    if not use_deepspeed:
        model_load_kwargs["device_map"] = {"": train_device}
    model = TuneRL.AutoModelForCausalLM.from_pretrained(
        TuneRL.base_model,
        **model_load_kwargs,
    )
    _ = hf_deepspeed_config

    if load_initial_adapter:
        if not init_adapter_path:
            raise ValueError("SFT_INIT_ADAPTER is empty, but SFT_LOAD_INITIAL_ADAPTER is True.")
        if not os.path.exists(init_adapter_path):
            raise FileNotFoundError(f"Initial adapter not found: {init_adapter_path}")
        print(f"Loading initial SFT adapter from {init_adapter_path}...")
        model = TuneRL.PeftModel.from_pretrained(model, init_adapter_path)
        model = model.merge_and_unload()

    model = TuneRL.prepare_model_for_kbit_training(model)
    TuneRL.align_generation_head_dtype(model, precision["torch_dtype"])

    peft_config = TuneRL.LoraConfig(
        r=SFT_LORA_R,
        lora_alpha=SFT_LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=SFT_LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    if stage_adapter_dir is not None:
        if not stage_adapter_dir.exists():
            raise FileNotFoundError(f"Missing adapter directory under SFT stage checkpoint: {stage_adapter_dir}")
        print(f"[SFT RL] Loading stage checkpoint adapter from {stage_adapter_dir}...")
        model = TuneRL.PeftModel.from_pretrained(model, str(stage_adapter_dir), is_trainable=True)
    else:
        model = TuneRL.get_peft_model(model, peft_config)
    TuneRL.align_generation_head_dtype(model, precision["torch_dtype"])

    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    try:
        model.enable_input_require_grads()
    except Exception:
        pass
    gc_patch_stats = TuneRL.enforce_non_reentrant_gradient_checkpointing(model)
    print(
        "[SFT RL] Gradient checkpointing enforcement: "
        f"roots={gc_patch_stats['roots']} modules={gc_patch_stats['modules']} use_reentrant=False"
    )
    model.print_trainable_parameters()
    TuneRL.active_rl_model = model
    TuneRL.active_rl_tokenizer = tokenizer

    trainer = TuneRL.GRPOTrainer(
        model=model,
        train_dataset=rl_dataset,
        reward_funcs=TuneRL.compute_reward,
        args=grpo_config,
    )
    trainer_gc_patch_stats = TuneRL.enforce_non_reentrant_gradient_checkpointing(trainer.model)
    print(
        "[SFT RL] Trainer gradient checkpointing enforcement: "
        f"roots={trainer_gc_patch_stats['roots']} modules={trainer_gc_patch_stats['modules']} use_reentrant=False"
    )
    if trainer_checkpoint_supported:
        # Keep legacy reward_state.json in trainer checkpoints so old resumes still work.
        runtime_callback = TrainingRuntime.build_trainer_checkpoint_callback(
            _sft_runtime_state_hooks(),
            state_aliases=("reward_state.json",),
        )
        if runtime_callback is not None:
            trainer.add_callback(runtime_callback)
    warmup_diagnostics = RewardUtil.prewarm_eval_workers(timeout_seconds=60.0, require_gpu=True)
    _validate_gpu_reward_worker_bindings(warmup_diagnostics)
    TuneRL.register_stage_checkpoint_signal_handlers()

    print("Starting GRPO training for Backbone Search...")
    try:
        if trainer_checkpoint is not None:
            print(f"[SFT RL] Resuming trainer state from {trainer_checkpoint}...")
            trainer.train(resume_from_checkpoint=str(trainer_checkpoint))
        else:
            trainer.train()
    except Exception as exc:
        if TuneRL.is_cuda_oom_error(exc):
            TuneRL.log_cuda_oom_diagnostics("sft/trainer.train", exc)
        raise
    finally:
        RewardUtil.shutdown_eval_worker()

    if save_rl_model:
        print(f"Saving model to {model_out_path}...")
        model.save_pretrained(model_out_path)
        print("Model saved successfully!")
    else:
        print("[SFT RL] Skipping RL adapter save. Next run will start from the same initial model.")
    if _runtime_is_main_process(runtime):
        TuneRL._save_stage_checkpoint(
            "completed",
            stage_name=TuneRL.current_stage_name,
            reason="sft_trainer_completed",
        )

    return model


def patch_sft_runtime() -> tuple[str, str, str]:
    """Patch TuneRL to use the SFT runtime and CIFAR-aware reward."""
    model_source, tokenizer_source, source_mode = resolve_sft_model_sources()
    load_initial_adapter = resolve_sft_load_initial_adapter()
    init_adapter_path = resolve_sft_init_adapter()
    TuneRL.base_model = model_source
    TuneRL.tokenizer_source = tokenizer_source
    TuneRL.LOAD_EXISTING_MODEL = load_initial_adapter
    TuneRL.SAVED_MODEL_PATH = init_adapter_path if load_initial_adapter else ""
    TuneRL.PROMPT_TEMPLATE = SFT_DISCOVERY_PROMPT_TEMPLATE
    TuneRL.extract_completion_blocks = extract_completion_blocks_tolerant
    TuneRL.clear_extraction_meta_cache = clear_extraction_meta_cache
    TuneRL.evaluate_code_and_reward = evaluate_code_and_reward_cifar
    setattr(TuneRL.evaluate_code_and_reward, "_nngpt_eval_cfg_builder", build_sft_reward_eval_cfg)
    TuneRL.reward_fn = sft_reward_fn
    TuneRL.load_rl_dataset = load_rl_dataset_sft
    TuneRL.run_log_dir = resolve_sft_log_dir
    TuneRL.run_model_out = resolve_sft_model_out
    TuneRL.run_epoch_dir = sft_run_epoch_dir
    return model_source, tokenizer_source, source_mode


def bootstrap_sft_runtime() -> None:
    """Initialize logging and reset extraction cache."""
    clear_extraction_meta_cache()
    RewardUtil.shutdown_eval_worker()
    runtime = RewardUtil.get_distributed_runtime_info()
    is_main = _runtime_is_main_process(runtime)
    resume_spec = _resolve_sft_resume_spec()
    if resume_spec["active"]:
        os.environ["NNGPT_SFT_APPEND_LOGS"] = "1"
    else:
        os.environ.pop("NNGPT_SFT_APPEND_LOGS", None)

    log_dir = TuneRL.run_log_dir()
    os.makedirs(log_dir, exist_ok=True)
    sentinel_path = _bootstrap_sentinel_path(log_dir, runtime)
    trainer_out_dir = Path(resolve_sft_trainer_out())
    stale_files = (
        "generation_samples.jsonl",
        "group_progress.jsonl",
        "group_feedback_samples.jsonl",
        "best_group_feedback.json",
    )
    if is_main:
        if sentinel_path.exists():
            sentinel_path.unlink()
        if resume_spec["active"]:
            print(
                f"[SFT RL] Resume mode active: preserving logs and trainer outputs "
                f"(mode={resume_spec['mode']}, log_dir={log_dir}, trainer_out={trainer_out_dir})"
            )
        else:
            for filename in stale_files:
                path = Path(log_dir) / filename
                if path.exists():
                    print(f"Removing stale runtime log: {path}")
                    path.unlink()
            print(f"Cleaning existing models in {TuneRL.run_epoch_dir()}...")
            shutil.rmtree(TuneRL.run_epoch_dir(), ignore_errors=True)
            print(f"Cleaning existing trainer outputs in {trainer_out_dir}...")
            shutil.rmtree(trainer_out_dir, ignore_errors=True)
        TuneRL.code_logger = RawCodeLogger(log_dir)
        sentinel_path.write_text(str(os.getpid()), encoding="utf-8")
        return

    TuneRL.code_logger = TuneRL.NullCodeLogger()
    _wait_for_bootstrap_sentinel(sentinel_path)


def main() -> None:
    import torch

    resume_spec = _resolve_sft_resume_spec()
    gpu_role_plan: Dict[str, List[str] | str] = {
        "requested_mode": "auto",
        "resolved_mode": "single",
        "visible_gpu_tokens": [],
        "train_gpu_tokens": [],
        "reward_gpu_tokens": [],
    }
    if torch.cuda.is_available():
        gpu_role_plan = _configure_sft_gpu_role_env(int(torch.cuda.device_count()))
        print(
            "[SFT RL] GPU mode plan: "
            f"requested_mode={gpu_role_plan['requested_mode']} "
            f"resolved_mode={gpu_role_plan['resolved_mode']} "
            f"visible_gpu_tokens={gpu_role_plan['visible_gpu_tokens']} "
            f"train_gpu_tokens={gpu_role_plan['train_gpu_tokens']} "
            f"reward_gpu_tokens={gpu_role_plan['reward_gpu_tokens']}"
        )
    _maybe_relaunch_sft_with_visible_gpu_workers()
    _validate_sft_visible_worker_count(RewardUtil.get_distributed_runtime_info())
    model_source, tokenizer_source, source_mode = patch_sft_runtime()
    bootstrap_sft_runtime()

    print(f"[SFT RL] Base model id: {SFT_BASE_MODEL_ID}")
    print(f"[SFT RL] Base model source ({source_mode}): {model_source}")
    if tokenizer_source != model_source:
        print(f"[SFT RL] Tokenizer source: {tokenizer_source}")
    print(f"[SFT RL] Resume mode: {resume_spec['mode']}")
    print(f"[SFT RL] Load init adapter: {resolve_sft_load_initial_adapter()}")
    if resolve_sft_load_initial_adapter():
        print(f"[SFT RL] Init adapter path: {resolve_sft_init_adapter()}")
    if resume_spec["trainer_checkpoint"] is not None:
        print(f"[SFT RL] Resume trainer checkpoint: {resume_spec['trainer_checkpoint']}")
    if resume_spec["stage_checkpoint_dir"] is not None:
        print(f"[SFT RL] Resume stage checkpoint: {resume_spec['stage_checkpoint_dir']}")
    print(f"[SFT RL] Log dir: {resolve_sft_log_dir()}")
    print(f"[SFT RL] Trainer out: {resolve_sft_trainer_out()}")
    print(f"[SFT RL] Model out: {resolve_sft_model_out()}")
    print(f"[SFT RL] Stage1 only: {TuneRL.env_flag('NNGPT_RL_STAGE1_ONLY', False)}")
    print(f"[SFT RL] Num epochs: {resolve_sft_num_epochs()}")
    print(f"[SFT RL] Temperature: {SFT_TEMPERATURE}")
    print(f"[SFT RL] KL coef: {TuneRL.env_float('NNGPT_RL_KL_COEF', SFT_KL_COEF):.6f}")
    print(
        f"[SFT RL] Eval plan: stage1=static_only(no-check_nn), stage2/3=nn-dataset-formal(cifar-10), "
        f"transform={SFT_EVAL_TRANSFORM}, resize={SFT_EVAL_IMAGE_SIZE}, batch={SFT_EVAL_BATCH_SIZE}, "
        f"train_set=full, test_set=full, train_epochs={SFT_EVAL_TRAIN_EPOCHS}, "
        f"freeze_only_backbone_eval=True, formal_epoch_limit_minutes={SFT_EVAL_FORMAL_EPOCH_LIMIT_MINUTES}, "
        f"worker_eval_limit_seconds={SFT_EVAL_LIMIT_SECONDS}, "
        f"baseline={SFT_VAL_METRIC_BASELINE:.2f}"
    )
    print(f"[SFT RL] Save RL adapter: {resolve_sft_save_rl_model()}")

    run_sft_training()


if __name__ == "__main__":
    main()
