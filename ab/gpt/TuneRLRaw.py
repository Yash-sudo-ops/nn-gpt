import os
import ast
import hashlib
import json
import re
import shutil
import textwrap
from typing import Any, Dict, List, Sequence, Tuple


# RAW_HF_HOME = "/media/xi/Data/hf-cache"
# RAW_HF_HUB_CACHE = "/media/xi/Data/hf-cache/hub"
# RAW_TRANSFORMERS_CACHE = "/media/xi/Data/hf-cache/transformers"
RAW_TRANSFORMERS_CACHE = "out/llm"
RAW_LOG_DIR = "rl_output/raw"
RAW_EPOCH_ROOT = "out/nngpt/llm/epoch_raw"
RAW_TRAINER_OUT = "grpo_backbone_outputs/raw"
RAW_MAX_COMPLETION_LENGTH = "1536"
RAW_TEMPERATURE = "0.9"
RAW_NUM_GENERATIONS = "8"
RAW_GRAD_ACCUM = "8"

# Force the raw runtime to use the mounted Windows cache and stable raw output dirs.
# os.environ["HF_HOME"] = RAW_HF_HOME
# os.environ["HF_HUB_CACHE"] = RAW_HF_HUB_CACHE
# os.environ["HUGGINGFACE_HUB_CACHE"] = RAW_HF_HUB_CACHE
os.environ["TRANSFORMERS_CACHE"] = RAW_TRANSFORMERS_CACHE
os.environ["NNGPT_RL_LOG_DIR"] = RAW_LOG_DIR
os.environ["NNGPT_RL_EPOCH_ROOT"] = RAW_EPOCH_ROOT
os.environ["NNGPT_RL_TRAINER_OUT"] = RAW_TRAINER_OUT
os.environ["NNGPT_RL_MAX_COMPLETION_LENGTH"] = RAW_MAX_COMPLETION_LENGTH
os.environ["NNGPT_RL_TEMPERATURE"] = RAW_TEMPERATURE
os.environ["NNGPT_RL_NUM_GENERATIONS"] = RAW_NUM_GENERATIONS
os.environ["NNGPT_RL_GRAD_ACCUM"] = RAW_GRAD_ACCUM

import ab.gpt.TuneRL as TuneRL
import ab.gpt.util.SFTUtil as SFTUtil


RAW_BASE_MODEL = "deepseek-ai/deepseek-coder-6.7b-instruct"
RAW_MODEL_OUT = "rl_backbone_model_raw"
REQUIRED_BACKBONE_NAMES = ("backbone_a", "backbone_b")
BLOCK_SIGNATURE = "def drop_conv3x3_block(in_channels, out_channels, stride=1, padding=1, bias=False, dropout_prob=0.0):"
INIT_SIGNATURE = "def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:"
FORWARD_SIGNATURE = "def forward(self, x: torch.Tensor, is_probing: bool = False) -> torch.Tensor:"

RAW_DISCOVERY_PROMPT_TEMPLATE = SFTUtil.open_discovery_rl_prompt_template

BLOCKED_ATTRS = {
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

PLACEHOLDER_PATTERN_NAMES = {
    "",
    "OpenMotif",
    "OpenMotifRaw",
    "NovelPattern",
    "NewPattern",
    "CustomPattern",
}

PROJECT_NAME_POOL = [
    "project_a",
    "project_b",
    "bridge",
    "project_c",
    "adapter",
]

RAW_ASSISTANT_PREFIX = ""

EXTRACTION_META_CACHE: Dict[str, Dict[str, object]] = {}


def clear_extraction_meta_cache() -> None:
    EXTRACTION_META_CACHE.clear()


def extraction_meta_cache_size() -> int:
    return len(EXTRACTION_META_CACHE)


class RawCodeLogger(TuneRL.SimpleCodeLogger):
    def __init__(self, output_dir: str = "rl_output/raw"):
        super().__init__(output_dir)
        self.samples_file = os.path.join(output_dir, "generation_samples.jsonl")

    def log_generation(self, prompt: str, completion: str, reward: float, api_result=None):
        super().log_generation(prompt, completion, reward, api_result)


def configure_raw_defaults() -> None:
    # Keep raw experiments isolated from the SFT trainer outputs with code-owned defaults.
    # os.environ["HF_HOME"] = RAW_HF_HOME
    # os.environ["HF_HUB_CACHE"] = RAW_HF_HUB_CACHE
    # os.environ["HUGGINGFACE_HUB_CACHE"] = RAW_HF_HUB_CACHE
    os.environ["TRANSFORMERS_CACHE"] = RAW_TRANSFORMERS_CACHE
    os.environ["NNGPT_RL_LOG_DIR"] = RAW_LOG_DIR
    os.environ["NNGPT_RL_EPOCH_ROOT"] = RAW_EPOCH_ROOT
    os.environ["NNGPT_RL_TRAINER_OUT"] = RAW_TRAINER_OUT
    os.environ["NNGPT_RL_MODEL_OUT"] = RAW_MODEL_OUT
    os.environ["NNGPT_RL_MAX_COMPLETION_LENGTH"] = RAW_MAX_COMPLETION_LENGTH
    os.environ["NNGPT_RL_TEMPERATURE"] = RAW_TEMPERATURE
    os.environ["NNGPT_RL_NUM_GENERATIONS"] = RAW_NUM_GENERATIONS
    os.environ["NNGPT_RL_GRAD_ACCUM"] = RAW_GRAD_ACCUM


def _extract_xml_tag(text: str, tag: str) -> str:
    match = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", text, re.IGNORECASE | re.DOTALL)
    return TuneRL.clean_block(match.group(1)) if match else ""


def _strip_outer_code_fences(text: str) -> str:
    if not text:
        return ""
    text = text.strip()
    text = re.sub(r"^```(?:python)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _extract_function_block(text: str, fn_name: str) -> str:
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if not re.match(rf"^\s*def {re.escape(fn_name)}\s*\(", line):
            continue
        indent = len(line) - len(line.lstrip())
        end = len(lines)
        for j in range(i + 1, len(lines)):
            stripped = lines[j].lstrip()
            if not stripped:
                continue
            indent_j = len(lines[j]) - len(stripped)
            if (stripped.startswith("def ") or stripped.startswith("class ")) and indent_j <= indent:
                end = j
                break
        return textwrap.dedent("\n".join(lines[i:end])).strip()
    return ""


def _dedupe_keep_order(items: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


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


def _hash_pick(seed_text: str, options: Sequence[str]) -> str:
    if not options:
        return ""
    digest = hashlib.sha1((seed_text or "raw").encode("utf-8")).hexdigest()
    return options[int(digest[:8], 16) % len(options)]


def _extract_pattern_name(*texts: str) -> str:
    for text in texts:
        if not text:
            continue
        match = re.search(r"self\.pattern\s*=\s*['\"]([^'\"]+)['\"]", text)
        if match:
            pattern_name = match.group(1).strip()
            if pattern_name not in PLACEHOLDER_PATTERN_NAMES:
                return pattern_name
    return ""


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


def _module_ctor(attr_name: str) -> str:
    return _module_ctor_with_input(attr_name, "48")


def _module_ctor_with_input(attr_name: str, in_channels_expr: str) -> str:
    role = _infer_attr_role(attr_name)
    if role == "stem":
        return (
            "nn.Sequential("
            f"drop_conv3x3_block({in_channels_expr}, 48, dropout_prob=dropout_prob), "
            "drop_conv3x3_block(48, 48, dropout_prob=dropout_prob)"
            ")"
        )
    if role == "fractal":
        return (
            "nn.Sequential("
            f"drop_conv3x3_block({in_channels_expr}, 48, dropout_prob=dropout_prob), "
            "FractalBlock(48, 48, num_columns=3, loc_drop_prob=0.0, dropout_prob=dropout_prob)"
            ")"
        )
    if role == "fuse" or "bridge" in attr_name.lower():
        return (
            "nn.Sequential("
            f"drop_conv3x3_block({in_channels_expr}, 48, dropout_prob=dropout_prob), "
            "drop_conv3x3_block(48, 48, dropout_prob=dropout_prob)"
            ")"
        )
    return f"drop_conv3x3_block({in_channels_expr}, 48, dropout_prob=dropout_prob)"


def _collect_completion_attrs(*texts: str) -> List[str]:
    attrs: List[str] = []
    for text in texts:
        if not text:
            continue
        for attr in re.findall(r"self\.([A-Za-z_]\w*)\s*(?:\(|=)", text):
            if attr in BLOCKED_ATTRS or attr.startswith("__"):
                continue
            attrs.append(attr)
    attrs = _dedupe_keep_order(attrs)
    if not attrs:
        return ["stem", "project_a", "project_b", "fuse"]
    if not _has_structural_attr(attrs):
        return ["stem", *attrs[:2], "fuse"]
    return attrs


def _scan_raw_attrs(*texts: str) -> List[str]:
    attrs: List[str] = []
    for text in texts:
        if not text:
            continue
        for attr in re.findall(r"self\.([A-Za-z_]\w*)\s*(?:\(|=)", text):
            if attr in BLOCKED_ATTRS or attr.startswith("__"):
                continue
            attrs.append(attr)
    return _dedupe_keep_order(attrs)


def _normalize_attr_list(attrs: Sequence[str]) -> List[str]:
    normalized = [
        attr
        for attr in _dedupe_keep_order(list(attrs))
        if attr and attr not in BLOCKED_ATTRS and not attr.startswith("__")
    ]
    if not normalized:
        return ["stem", "project_a", "project_b", "fuse"]
    if not _has_structural_attr(normalized):
        return _dedupe_keep_order(["stem", *normalized[:2], "fuse"])
    return normalized


def _candidate_body_lines(text: str, fn_name: str) -> List[str]:
    cleaned = _clean_source_text(text)
    fn_block = _extract_function_block(cleaned, fn_name)
    source = fn_block or cleaned
    lines = source.splitlines()
    if lines and lines[0].lstrip().startswith(f"def {fn_name}"):
        lines = lines[1:]

    body: List[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        stripped = stripped.split("#", 1)[0].strip()
        if not stripped:
            continue
        if stripped.startswith(("<", ">", "```", "class ", "def ", "from ", "import ")):
            continue
        if stripped.startswith(("if ", "elif ", "else:", "for ", "while ", "with ", "try:", "except", "finally:")):
            continue
        if "self.pattern" in stripped:
            continue
        if stripped in {"pass", "..."}:
            continue
        body.append(stripped)
    return body


def _register_attr(attr_name: str, used_attrs: List[str]) -> None:
    if attr_name in BLOCKED_ATTRS or attr_name.startswith("__"):
        return
    if attr_name not in used_attrs:
        used_attrs.append(attr_name)


def _make_unique_attr_name(base_name: str, used_names: set[str]) -> str:
    if base_name not in used_names:
        used_names.add(base_name)
        return base_name
    suffix = 2
    while f"{base_name}_{suffix}" in used_names:
        suffix += 1
    unique_name = f"{base_name}_{suffix}"
    used_names.add(unique_name)
    return unique_name


def _canonical_attr_mapping(attrs: Sequence[str]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    used_names: set[str] = set()
    project_idx = 0
    backbone_idx = 0

    for attr in _normalize_attr_list(attrs):
        lowered = attr.lower()
        role = _infer_attr_role(attr)
        if role == "stem":
            base_name = "stem"
        elif role == "fuse":
            base_name = "fuse"
        elif role == "fractal":
            base_name = "fractal"
        elif role == "project":
            if "bridge" in lowered:
                base_name = "bridge"
            elif project_idx < len(PROJECT_NAME_POOL):
                base_name = PROJECT_NAME_POOL[project_idx]
                project_idx += 1
            else:
                base_name = f"project_{project_idx - len(PROJECT_NAME_POOL) + 4}"
                project_idx += 1
        elif role == "backbone":
            base_name = f"backbone_{chr(ord('a') + backbone_idx)}"
            backbone_idx += 1
        else:
            if project_idx < len(PROJECT_NAME_POOL):
                base_name = PROJECT_NAME_POOL[project_idx]
                project_idx += 1
            else:
                base_name = f"project_{project_idx - len(PROJECT_NAME_POOL) + 4}"
                project_idx += 1

        mapping[attr] = _make_unique_attr_name(base_name, used_names)
    return mapping


def _rename_expr_tokens(expr: str, mapping: Dict[str, str]) -> str:
    renamed = expr
    for old_name in sorted(mapping, key=len, reverse=True):
        renamed = re.sub(rf"\b{re.escape(old_name)}\b", mapping[old_name], renamed)
    return renamed


def _canonicalize_assignments(
    assignments: Sequence[Tuple[str, str]],
    used_attrs: Sequence[str],
) -> Tuple[List[Tuple[str, str]], List[str]]:
    attr_mapping = _canonical_attr_mapping(used_attrs)
    renamed_assignments = [
        (lhs, _rename_expr_tokens(expr, {f"self.{k}": f"self.{v}" for k, v in attr_mapping.items()}))
        for lhs, expr in assignments
    ]

    compacted: List[Tuple[str, str]] = []
    var_mapping: Dict[str, str] = {"x": "x"}
    known_vars = {"x"}
    for lhs, expr in renamed_assignments:
        expr = _rename_expr_tokens(expr, var_mapping)
        if expr in known_vars:
            var_mapping[lhs] = expr
            continue
        new_lhs = f"x{len(compacted)}"
        compacted.append((new_lhs, expr))
        var_mapping[lhs] = new_lhs
        known_vars.add(new_lhs)

    canonical_attrs = _dedupe_keep_order([attr_mapping[attr] for attr in used_attrs if attr in attr_mapping])
    return compacted, canonical_attrs


def _channel_sum(parts: Sequence[str]) -> str:
    numeric_total = 0
    symbolic_parts: List[str] = []
    for part in parts:
        if not part:
            continue
        if re.fullmatch(r"\d+", part):
            numeric_total += int(part)
        else:
            symbolic_parts.append(part)
    if numeric_total:
        symbolic_parts.append(str(numeric_total))
    if not symbolic_parts:
        return "48"
    if len(symbolic_parts) == 1:
        return symbolic_parts[0]
    return " + ".join(symbolic_parts)


def _infer_expr_channels(
    node: ast.AST,
    var_channels: Dict[str, str],
    attr_inputs: Dict[str, str],
) -> str:
    if isinstance(node, ast.Name):
        return var_channels.get(node.id, "48")

    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _infer_expr_channels(node.left, var_channels, attr_inputs)

    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Attribute):
            base = node.func.value
            if isinstance(base, ast.Name) and base.id == "self":
                attr_name = node.func.attr
                child_channels = (
                    _infer_expr_channels(node.args[0], var_channels, attr_inputs)
                    if node.args else "48"
                )
                attr_inputs.setdefault(attr_name, child_channels)
                return "48"

        func_name = ""
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            func_name = f"{node.func.value.id}.{node.func.attr}"

        if func_name in {"torch.cat", "cat"} and node.args and isinstance(node.args[0], (ast.List, ast.Tuple)):
            child_channels = [
                _infer_expr_channels(item, var_channels, attr_inputs)
                for item in node.args[0].elts
            ]
            return _channel_sum(child_channels)

        if node.args:
            return _infer_expr_channels(node.args[0], var_channels, attr_inputs)

    return "48"


def _infer_attr_input_channels(assignments: Sequence[Tuple[str, str]]) -> Dict[str, str]:
    var_channels: Dict[str, str] = {"x": "in_shape[1]"}
    attr_inputs: Dict[str, str] = {}
    for lhs, expr in assignments:
        try:
            node = ast.parse(expr, mode="eval").body
            var_channels[lhs] = _infer_expr_channels(node, var_channels, attr_inputs)
        except Exception:
            var_channels[lhs] = "48"
    return attr_inputs


def _derive_pattern_name(
    attrs: Sequence[str],
    assignments: Sequence[Tuple[str, str]],
    completion_text: str,
) -> str:
    role_tokens: List[str] = []
    if any(_infer_attr_role(attr) == "stem" for attr in attrs):
        role_tokens.append("Stem")
    if any(_infer_attr_role(attr) == "project" for attr in attrs):
        role_tokens.append("Project")
    if any(_infer_attr_role(attr) == "fractal" for attr in attrs):
        role_tokens.append("Fractal")
    if any("torch.cat(" in expr or " + " in expr for _, expr in assignments):
        role_tokens.append("Merge")
    if any(_infer_attr_role(attr) == "fuse" for attr in attrs):
        role_tokens.append("Fuse")
    base_name = "".join(role_tokens[:4]) or "OpenMotif"
    digest_source = "||".join([*attrs, *[expr for _, expr in assignments], completion_text])
    digest = hashlib.sha1(digest_source.encode("utf-8")).hexdigest()[:6].upper()
    return f"{base_name}_{digest}"


def _render_expr(
    node: ast.AST,
    known_vars: Sequence[str],
    used_attrs: List[str],
    var_aliases: dict[str, str],
) -> str:
    if isinstance(node, ast.Name):
        if node.id in var_aliases:
            return var_aliases[node.id]
        return node.id if node.id in known_vars or node.id == "x" else "x"

    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return repr(node.value)
        return ""

    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self":
        _register_attr(node.attr, used_attrs)
        return f"self.{node.attr}"

    if isinstance(node, ast.IfExp):
        body_expr = _render_expr(node.body, known_vars, used_attrs, var_aliases)
        else_expr = _render_expr(node.orelse, known_vars, used_attrs, var_aliases)
        if body_expr and "self." in body_expr:
            return body_expr
        if else_expr:
            return else_expr
        return body_expr

    if isinstance(node, ast.List):
        items = [_render_expr(item, known_vars, used_attrs, var_aliases) for item in node.elts]
        items = [item for item in items if item]
        return ", ".join(items)

    if isinstance(node, ast.Tuple):
        items = [_render_expr(item, known_vars, used_attrs, var_aliases) for item in node.elts]
        items = [item for item in items if item]
        return ", ".join(items)

    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _render_expr(node.left, known_vars, used_attrs, var_aliases)
        right = _render_expr(node.right, known_vars, used_attrs, var_aliases)
        if left and right:
            return f"({left} + {right})"
        return left or right

    if isinstance(node, ast.Call):
        func_name = ""
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            base = node.func.value
            if isinstance(base, ast.Name):
                func_name = f"{base.id}.{node.func.attr}"
            elif isinstance(base, ast.Attribute) and isinstance(base.value, ast.Name) and base.value.id == "self":
                func_name = f"self.{base.attr}.{node.func.attr}"

        if func_name in {"adaptive_pool_flatten", "pool_flatten"}:
            child = _render_expr(node.args[0], known_vars, used_attrs, var_aliases) if node.args else "x"
            return child

        if func_name in {"torch.cat", "cat", "torch.stack", "stack"}:
            parts: List[str] = []
            if node.args and isinstance(node.args[0], (ast.List, ast.Tuple)):
                for item in node.args[0].elts:
                    rendered = _render_expr(item, known_vars, used_attrs, var_aliases)
                    if rendered:
                        parts.append(rendered)
            else:
                for item in node.args:
                    rendered = _render_expr(item, known_vars, used_attrs, var_aliases)
                    if rendered:
                        parts.append(rendered)
            parts = _dedupe_keep_order(parts)
            if not parts:
                return ""
            if len(parts) == 1:
                return parts[0]
            return f"torch.cat([{', '.join(parts)}], dim=1)"

        if isinstance(node.func, ast.Attribute):
            base = node.func.value
            if isinstance(base, ast.Name) and base.id == "self":
                attr_name = node.func.attr
                _register_attr(attr_name, used_attrs)
                child = _merge_call_args(node.args, known_vars, used_attrs, var_aliases)
                return f"self.{attr_name}({child})"

            if isinstance(base, ast.Attribute) and isinstance(base.value, ast.Name) and base.value.id == "self":
                attr_name = base.attr
                _register_attr(attr_name, used_attrs)
                child = _render_expr(node.args[0], known_vars, used_attrs, var_aliases) if node.args else "x"
                if node.func.attr in {"flatten", "view", "reshape", "contiguous", "mean"}:
                    return f"self.{attr_name}({child})"
                return f"self.{attr_name}({child})"

            child = _render_expr(base, known_vars, used_attrs, var_aliases)
            if node.func.attr in {"flatten", "view", "reshape", "contiguous", "mean"}:
                return child
            if node.func.attr in {"float", "to", "cpu", "cuda"}:
                return child

        if node.args:
            return _render_expr(node.args[0], known_vars, used_attrs, var_aliases)
        return ""

    return ""


def _merge_call_args(
    args: Sequence[ast.AST],
    known_vars: Sequence[str],
    used_attrs: List[str],
    var_aliases: dict[str, str],
) -> str:
    rendered_args = [
        _render_expr(arg, known_vars, used_attrs, var_aliases)
        for arg in args
    ]
    rendered_args = [arg for arg in rendered_args if arg]
    rendered_args = _dedupe_keep_order(rendered_args)
    if not rendered_args:
        return "x"
    if len(rendered_args) == 1:
        return rendered_args[0]
    return f"torch.cat([{', '.join(rendered_args)}], dim=1)"


def _sanitize_expr_text(
    expr_text: str,
    known_vars: Sequence[str],
    used_attrs: List[str],
    var_aliases: dict[str, str],
) -> str:
    expr_text = expr_text.strip().rstrip(",")
    if not expr_text:
        return ""
    expr_text = expr_text.replace("self.adaptive_pool_flatten(", "adaptive_pool_flatten(")
    expr_text = expr_text.replace("torch.concat(", "torch.cat(")
    expr_text = expr_text.replace("torch.concatenate(", "torch.cat(")
    try:
        node = ast.parse(expr_text, mode="eval").body
        rendered = _render_expr(node, known_vars, used_attrs, var_aliases)
        if rendered:
            return rendered
    except Exception:
        pass

    calls = re.findall(r"self\.([A-Za-z_]\w*)\s*\(\s*([A-Za-z_]\w*)", expr_text)
    parts = []
    for attr_name, var_name in calls:
        if attr_name in BLOCKED_ATTRS:
            continue
        _register_attr(attr_name, used_attrs)
        arg_name = var_aliases.get(var_name, var_name if var_name in known_vars or var_name == "x" else "x")
        parts.append(f"self.{attr_name}({arg_name})")

    if "torch.cat" in expr_text and parts:
        merged = _dedupe_keep_order(parts)
        if len(merged) == 1:
            return merged[0]
        return f"torch.cat([{', '.join(merged)}], dim=1)"

    if "+" in expr_text and len(parts) >= 2:
        return " + ".join(_dedupe_keep_order(parts[:3]))

    if parts:
        return parts[0]

    for name in reversed(list(known_vars)):
        if name in expr_text:
            return name
    return "x"


def _prepare_completion_for_xml(completion: str) -> str:
    text = _strip_outer_code_fences(completion or "")
    stripped = text.lstrip()
    # Case 1: Model output has </block> but no <block> (prefix was pre-filled)
    if "<block>" not in stripped and "</block>" in stripped and "<init>" in stripped:
        return f"{RAW_ASSISTANT_PREFIX}{stripped}"
    # Case 2: Model output has <init> and <forward> but no block tags at all
    # (model skipped the block section entirely)
    if "<block>" not in stripped and "</block>" not in stripped and "<init>" in stripped:
        # Wrap everything before <init> as a block
        init_pos = stripped.find("<init>")
        pre_init = stripped[:init_pos].strip()
        rest = stripped[init_pos:]
        if pre_init:
            return f"<block>\n{BLOCK_SIGNATURE}\n{pre_init}\n</block>\n{rest}"
        else:
            # No block code at all; use a minimal pass-through block
            return f"<block>\n{BLOCK_SIGNATURE}\n    return nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, stride, padding, bias=bias))\n</block>\n{rest}"
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

    normalized_body = []
    for line in body_text.splitlines():
        normalized_body.append(f"    {line}" if line.strip() else "")
    return f"{signature}\n" + "\n".join(normalized_body)


def _normalize_block_code(block_code: str) -> str:
    return _normalize_required_function(block_code, "drop_conv3x3_block", BLOCK_SIGNATURE)


def _normalize_init_code(init_code: str) -> str:
    return _normalize_required_function(init_code, "__init__", INIT_SIGNATURE)


def _normalize_forward_code(forward_code: str) -> str:
    return _normalize_required_function(forward_code, "forward", FORWARD_SIGNATURE)


def _extract_defined_backbones(init_code: str) -> List[str]:
    return _dedupe_keep_order(
        re.findall(r"self\.(backbone_[A-Za-z]\w*)\s*=", init_code or "")
    )


def _extract_used_backbones(forward_code: str) -> List[str]:
    return _dedupe_keep_order(
        re.findall(r"self\.(backbone_[A-Za-z]\w*)\b", forward_code or "")
    )


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
    text = completion or ""
    candidate = candidate or ""
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
    text = completion or ""
    cache_key = _completion_cache_key(text)
    cached = EXTRACTION_META_CACHE.get(cache_key)
    if cached:
        return (
            (cached["block_code"], cached["init_code"], cached["forward_code"]),
            dict(cached["meta"]),
        )

    candidate = _prepare_completion_for_xml(text)
    block_code = _normalize_block_code(_extract_xml_tag(candidate, "block"))
    init_code = _normalize_init_code(_extract_xml_tag(candidate, "init"))
    forward_code = _normalize_forward_code(_extract_xml_tag(candidate, "forward"))

    meta = _build_extraction_meta(
        text,
        candidate,
        block_code,
        init_code,
        forward_code,
    )
    EXTRACTION_META_CACHE[cache_key] = {
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
    prompt_goal_tags: List[str] = None,
    archive_snapshot_family_counts: Dict[str, int] = None,
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
        prompt_goal_tags=prompt_goal_tags,
        archive_snapshot_family_counts=archive_snapshot_family_counts,
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

    # --- Tiered format violation clamps ---
    # Core XML structure violations (strict)
    core_format_violation = bool(
        not meta.get("xml_tag_exact")
        or not meta.get("exact_block_signature")
        or not meta.get("exact_init_signature")
        or not meta.get("exact_forward_signature")
    )
    # Minor code hygiene violations (lenient)
    class_count = int(meta.get("class_count", 0))
    import_count = int(meta.get("import_count", 0))
    minor_hygiene_violation = class_count > 0 or import_count > 0
    severe_hygiene_violation = class_count > 2 or import_count > 2

    if core_format_violation:
        res["reward"] = min(float(res["reward"]), -3.0)
    elif severe_hygiene_violation:
        res["reward"] = min(float(res["reward"]), -2.0)
    elif minor_hygiene_violation:
        # Small class/import count: softer clamp so near-correct code isn't destroyed
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


def load_rl_dataset_raw(tokenizer):
    data = TuneRL.api.data(task="img-classification", nn_prefixes=("rl-bb-test1",))
    if data.empty:
        print("No 'rl-bb-test1' data found, falling back to all img-classification")
        data = TuneRL.api.data(only_best_accuracy=True, task="img-classification", dataset="cifar-10")

    print(f"Loaded {len(data)} examples for RL")
    TuneRL.bootstrap_trainset_reference_library(data)

    prompts = []
    legacy_patterns = ", ".join(SFTUtil.legacy_patterns)
    goal_profiles = SFTUtil.open_discovery_goal_profiles

    for _, row in data.iterrows():
        accuracy = TuneRL._coerce_accuracy_baseline(row.get("accuracy"), context="seed row accuracy")
        for profile in goal_profiles:
            module_hints = (
                "self.backbone_a",
                "self.backbone_b",
                *profile["module_hints"],
            )
            user_prompt = RAW_DISCOVERY_PROMPT_TEMPLATE.format(
                accuracy=accuracy,
                skeleton_code=SFTUtil.open_discovery_skeleton_code,
                available_backbones=", ".join(SFTUtil.available_backbones),
                legacy_patterns=legacy_patterns,
                goal_name=profile["name"],
                target_tags=", ".join(profile["tags"]),
                design_brief=profile["brief"],
                module_hints=", ".join(module_hints),
                block_signature=BLOCK_SIGNATURE,
                init_signature=INIT_SIGNATURE,
                forward_signature=FORWARD_SIGNATURE,
            )

            messages = [{"role": "user", "content": user_prompt}]
            prompt_str = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
            )

            prompts.append(
                {
                    "prompt": prompt_str,
                    "accuracy": accuracy,
                    "goal_name": profile["name"],
                    "target_tags": ", ".join(profile["tags"]),
                }
            )

    return TuneRL.Dataset.from_list(prompts).shuffle(seed=42)


def patch_raw_runtime() -> None:
    TuneRL.base_model = RAW_BASE_MODEL
    TuneRL.SAVED_MODEL_PATH = RAW_MODEL_OUT
    TuneRL.PROMPT_TEMPLATE = RAW_DISCOVERY_PROMPT_TEMPLATE
    TuneRL.extract_completion_blocks = extract_completion_blocks_tolerant
    TuneRL.reward_fn = raw_reward_fn
    TuneRL.load_rl_dataset = load_rl_dataset_raw
    global RAW_ASSISTANT_PREFIX
    RAW_ASSISTANT_PREFIX = ""


def bootstrap_raw_runtime() -> None:
    clear_extraction_meta_cache()
    log_dir = TuneRL.run_log_dir()
    os.makedirs(log_dir, exist_ok=True)
    TuneRL.code_logger = RawCodeLogger(log_dir)

    print(f"Cleaning existing models in {TuneRL.run_epoch_dir()}...")
    shutil.rmtree(TuneRL.run_epoch_dir(), ignore_errors=True)


def main() -> None:
    configure_raw_defaults()
    patch_raw_runtime()
    bootstrap_raw_runtime()
    TuneRL.main()


if __name__ == "__main__":
    main()
