#!/usr/bin/env python3
"""
Fine-tune Qwen3-8B (Unsloth 4-bit) for outcome prediction with early stopping.

Pipeline:
  1. prepare_llm_datasets — raw JSONL → ChatML train/val/test splits
  2. train_model          — QLoRA fine-tune with validation + early stopping

Usage:
    python TuneAccuracyPrediction.py
"""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path
from typing import Optional

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from unsloth import FastLanguageModel
    from trl import SFTTrainer
    from transformers import EarlyStoppingCallback, TrainingArguments
    from datasets import Dataset
    import torch
    from tqdm import tqdm
except ImportError as e:
    raise RuntimeError(
        "Missing required package. Install: pip install unsloth[colab-new] trl datasets transformers"
    ) from e

RAW_INPUT_PATH = SCRIPT_DIR / "data" / "llm_finetuning_data.jsonl"
PREP_OUTPUT_DIR = SCRIPT_DIR / "data"
DEFAULT_TRAIN_PATH = PREP_OUTPUT_DIR / "train_llm_dataset.jsonl"
DEFAULT_VAL_PATH = PREP_OUTPUT_DIR / "val_llm_dataset.jsonl"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "model2"

MODEL_NAME = "unsloth/Qwen3-8B-unsloth-bnb-4bit"
MODEL_FALLBACKS = ("unsloth/Qwen3-8B-bnb-4bit", "Qwen/Qwen3-8B")
DEFAULT_MAX_SEQ_LEN = 6144

PREDICTOR_HF_REPO = "ABrain/Accuracy-Prediction"
PREDICTOR_MAX_SEQ_LEN = DEFAULT_MAX_SEQ_LEN
PREDICTOR_MAX_NEW_TOKENS = 64
PREDICTOR_DEFAULT_MAX_EPOCHS = 50

_predictor_model = None
_predictor_tokenizer = None

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.013
WARMUP_RATIO = 0.02
MAX_GRAD_NORM = 0.5
LORA_R = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 8
NUM_EPOCHS = 20
MAX_STEPS = 0
SEED = 42
EARLY_STOPPING_PATIENCE = 3
SAVE_TOTAL_LIMIT = 2

TRAIN_ARCH_FAMILIES = frozenset({"cnn", "segmentation"})
VAL_ARCH_FAMILIES = frozenset({"detector", "transformer", "rnn"})
TEST_ARCH_FAMILIES = frozenset({"other"})

SYSTEM_PROMPT = """You are a strict JSON generator.
You must output exactly ONE JSON object and nothing else.

The JSON must contain exactly the keys:
best_accuracy
best_epoch

Rules:
best_accuracy must be a float in [0,100] rounded to 2 decimals.
best_epoch must be a positive integer representing the absolute epoch where peak validation accuracy occurs.

Do not explain.
Do not add text.
Stop immediately after the closing brace }.
"""


def _safe_float(val, default: float = 0.0) -> float:
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def _safe_int(val, default: int = 0) -> int:
    if val is None:
        return default
    try:
        return int(val)
    except (TypeError, ValueError):
        return default


def _resolve_call_name(node: ast.expr) -> Optional[str]:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _resolve_call_name(node.value)
        if base is None:
            return None
        return f"{base}.{node.attr}"
    return None


def _layer_counts(nn_code: str) -> dict[str, int]:
    """Count layer types in nn_code (AST parse with regex fallback)."""
    counts = {
        "conv2d": 0,
        "attention": 0,
        "lstm": 0,
        "gru": 0,
        "rnn": 0,
    }
    nn_code = nn_code or ""

    try:
        tree = ast.parse(nn_code)

        class LayerCounter(ast.NodeVisitor):
            def visit_Call(self, node: ast.Call) -> None:
                call_name = _resolve_call_name(node.func)
                if call_name is None:
                    self.generic_visit(node)
                    return

                if call_name in ("nn.Conv2d", "torch.nn.Conv2d", "Conv2d"):
                    counts["conv2d"] += 1
                if call_name in (
                    "nn.MultiheadAttention",
                    "torch.nn.MultiheadAttention",
                    "MultiheadAttention",
                ):
                    counts["attention"] += 1
                if call_name in ("nn.LSTM", "torch.nn.LSTM", "LSTM"):
                    counts["lstm"] += 1
                if call_name in ("nn.GRU", "torch.nn.GRU", "GRU"):
                    counts["gru"] += 1
                if call_name in ("nn.RNN", "torch.nn.RNN", "RNN"):
                    counts["rnn"] += 1

                self.generic_visit(node)

        LayerCounter().visit(tree)
    except (SyntaxError, ValueError):
        counts["conv2d"] = len(re.findall(r"\b(nn|torch\.nn)\.Conv2d\s*\(|\bConv2d\s*\(", nn_code))
        counts["attention"] = len(
            re.findall(
                r"\b(nn|torch\.nn)\.MultiheadAttention\s*\(|\bMultiheadAttention\s*\(",
                nn_code,
            )
        )
        counts["lstm"] = len(re.findall(r"\b(nn|torch\.nn)\.LSTM\s*\(|\bLSTM\s*\(", nn_code))
        counts["gru"] = len(re.findall(r"\b(nn|torch\.nn)\.GRU\s*\(|\bGRU\s*\(", nn_code))
        counts["rnn"] = len(re.findall(r"\b(nn|torch\.nn)\.RNN\s*\(|\bRNN\s*\(", nn_code))

    return counts


def infer_arch_family(nn_code: str, task: str) -> str:
    """
    Infer architecture family from nn_code and task.

    Mirrors the arch_family logic in extractor.extract_model_summary.
    Returns one of: detector, segmentation, transformer, rnn, cnn, other.
    """
    nn_code = nn_code or ""
    task = (task or "").lower()
    counts = _layer_counts(nn_code)

    if "detect" in task or re.search(r"\b(SSD|YOLO|RCNN|RetinaNet)\b", nn_code, re.IGNORECASE):
        return "detector"
    if "segmentation" in task or re.search(r"\b(FCN|DeepLab|UNet|LRASPP)\b", nn_code, re.IGNORECASE):
        return "segmentation"
    if counts["attention"] > 0 or re.search(r"\b(Transformer|MultiheadAttention|ViT|Swin)\b", nn_code):
        return "transformer"
    if counts["lstm"] > 0 or counts["gru"] > 0 or counts["rnn"] > 0:
        return "rnn"
    if counts["conv2d"] > 0:
        return "cnn"
    return "other"


def _arch_family(record: dict, nn_code: str) -> str:
    return infer_arch_family(nn_code, record.get("task", "") or "")


def _compute_targets(record: dict) -> dict:
    best_accuracy = max(0.0, min(_safe_float(record.get("best_accuracy"), 0.0), 100.0))
    max_epochs = _safe_int(record.get("total_epochs"), 50)
    best_epoch = max(1, min(_safe_int(record.get("epochs_to_best"), 1), max_epochs))
    return {"best_accuracy": best_accuracy, "best_epoch": best_epoch}


def _build_user_message(record: dict, nn_code: str) -> str:
    max_epochs = _safe_int(record.get("total_epochs"), 50)
    lines = [
        "INPUT",
        f"task: {record.get('task', '')}",
        f"dataset: {record.get('dataset', '')}",
        f"metric: {record.get('metric', '')}",
        "",
        "TRAINING_BUDGET",
        f"max_epochs: {max_epochs}",
        "",
        "EARLY_TRAINING_SIGNAL",
        f"epoch_1_accuracy: {round(_safe_float(record.get('accuracy_epoch_1'), 0.0), 6)}",
        f"epoch_2_accuracy: {round(_safe_float(record.get('accuracy_epoch_2'), 0.0), 6)}",
       # f"epoch_3_accuracy: {round(_safe_float(record.get('accuracy_epoch_3'), 0.0), 6)}",
        "",
        "NEURAL_NETWORK_CODE",
        "```python",
        (nn_code or "").strip(),
        "```",
        "",
        "Analyze the training dynamics, architecture complexity, and optimization hyperparameters to estimate the final training outcome.",
        "",
        "Important signals to consider:",
        "- Early learning progress (epoch accuracies)",
        "- Saturation of improvement across epochs",
        "- Architecture depth and complexity",
        "- Optimization scale (learning rate, batch size, effective_lr)",
        "",
        "Using these signals, estimate the final best validation accuracy and the epoch where it occurs.",
        "",
        "The value best_epoch represents the training epoch where the highest validation accuracy is reached.",
        "",
        "Constraints:",
        "- best_epoch must be an integer between 1 and max_epochs.",
        "",
        "OUTPUT (JSON ONLY):",
    ]
    return "\n".join(lines)


def _parse_prediction_json(text: str) -> dict | None:
    text = text.strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    if "```" in text:
        text = text.split("```", 1)[0].strip()
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def _get_predictor_model():
    global _predictor_model, _predictor_tokenizer
    if _predictor_model is None or _predictor_tokenizer is None:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=PREDICTOR_HF_REPO,
            max_seq_length=PREDICTOR_MAX_SEQ_LEN,
            dtype=None,
            load_in_4bit=True,
            trust_remote_code=True,
        )
        FastLanguageModel.for_inference(model)
        _predictor_model = model
        _predictor_tokenizer = tokenizer
    return _predictor_model, _predictor_tokenizer


def _apply_chat_template_for_inference(tokenizer, messages: list[dict]) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def predict_best_accuracy(
    task: str,
    dataset: str,
    metric: str,
    nn_code: str,
    epoch_1_accuracy: float,
    epoch_2_accuracy: float,
) -> tuple[float, int]:
    """Predict final best_accuracy and best_epoch using ABrain/Accuracy-Prediction."""
    record = {
        "task": task,
        "dataset": dataset,
        "metric": metric,
        "total_epochs": PREDICTOR_DEFAULT_MAX_EPOCHS,
        "accuracy_epoch_1": epoch_1_accuracy,
        "accuracy_epoch_2": epoch_2_accuracy,
        
    }
    user_content = _build_user_message(record, nn_code)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    model, tokenizer = _get_predictor_model()
    text_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    prompt_text = _apply_chat_template_for_inference(text_tokenizer, messages)
    input_ids = text_tokenizer.encode(prompt_text, add_special_tokens=False)
    if len(input_ids) > PREDICTOR_MAX_SEQ_LEN:
        input_ids = input_ids[:PREDICTOR_MAX_SEQ_LEN]

    input_ids_t = torch.tensor([input_ids], dtype=torch.long, device=model.device)
    attention_mask = torch.ones_like(input_ids_t)
    stop_token_ids = []
    for tok_str in ("}", "<|im_end|>", "<|endoftext|>"):
        ids = text_tokenizer.encode(tok_str, add_special_tokens=False)
        if ids:
            stop_token_ids.append(ids[-1])
    if text_tokenizer.eos_token_id is not None:
        stop_token_ids.append(text_tokenizer.eos_token_id)
    stop_token_ids = list(set(stop_token_ids))

    with torch.no_grad():
        outputs = model.generate(
            **{"input_ids": input_ids_t, "attention_mask": attention_mask},
            max_new_tokens=PREDICTOR_MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=text_tokenizer.pad_token_id or text_tokenizer.eos_token_id,
            eos_token_id=stop_token_ids or None,
            use_cache=False,
        )

    generated = text_tokenizer.decode(
        outputs[0][input_ids_t.shape[1] :],
        skip_special_tokens=True,
    )
    parsed = _parse_prediction_json(generated)

    fallback_acc = max(float(epoch_1_accuracy), float(epoch_2_accuracy))
    fallback_epoch = 2 if float(epoch_2_accuracy) >= float(epoch_1_accuracy) else 1

    if not parsed:
        return fallback_acc, fallback_epoch

    try:
        best_accuracy = float(parsed.get("best_accuracy", fallback_acc))
    except (TypeError, ValueError):
        best_accuracy = fallback_acc

    try:
        best_epoch = int(parsed.get("best_epoch", fallback_epoch))
        if best_epoch < 1:
            best_epoch = fallback_epoch
    except (TypeError, ValueError):
        best_epoch = fallback_epoch

    return best_accuracy, best_epoch


def _record_to_chatml(record: dict) -> dict | None:
    required = {"task", "dataset", "metric", "nn", "nn_code", "best_accuracy", "epochs_to_best"}
    if not all(k in record for k in required):
        return None

    nn_code = record.get("nn_code", "") or ""
    if not isinstance(nn_code, str):
        return None

    targets = _compute_targets(record)
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_message(record, nn_code)},
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "best_accuracy": round(targets["best_accuracy"], 2),
                        "best_epoch": int(targets["best_epoch"]),
                    },
                    separators=(",", ":"),
                ),
            },
        ]
    }


def _stream_jsonl(path: Path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _split_by_architecture_family(records: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    family_to_records: dict[str, list[dict]] = {}
    for rec in records:
        raw = rec["raw"]
        nn_code = raw.get("nn_code", "") or ""
        family = _arch_family(raw, nn_code) if isinstance(nn_code, str) else "other"
        family_to_records.setdefault(family, []).append(rec)

    train_records: list[dict] = []
    val_records: list[dict] = []
    test_records: list[dict] = []
    for family, recs in family_to_records.items():
        if family in TRAIN_ARCH_FAMILIES:
            train_records.extend(recs)
        elif family in VAL_ARCH_FAMILIES:
            val_records.extend(recs)
        elif family in TEST_ARCH_FAMILIES:
            test_records.extend(recs)
    return train_records, val_records, test_records


def _write_jsonl(path: Path, data: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for rec in data:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def prepare_llm_datasets(
    input_path: Path = RAW_INPUT_PATH,
    output_dir: Path = PREP_OUTPUT_DIR,
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    records: list[dict] = []
    skipped = 0
    for raw in tqdm(_stream_jsonl(input_path), desc="Processing"):
        chatml = _record_to_chatml(raw)
        nn = raw.get("nn")
        if chatml is None or not raw.get("task") or not raw.get("dataset") or not nn:
            skipped += 1
            continue
        records.append({"nn": str(nn), "chatml": chatml, "raw": raw})

    if not records:
        raise ValueError(f"No valid records loaded from {input_path} (skipped {skipped})")

    train_data, val_data, test_data = _split_by_architecture_family(records)
    train_path = output_dir / "train_llm_dataset.jsonl"
    val_path = output_dir / "val_llm_dataset.jsonl"
    test_path = output_dir / "test_llm_dataset.jsonl"

    def _with_arch_id(rec: dict) -> dict:
        out = dict(rec["chatml"])
        out["architecture_id"] = rec["nn"]
        return out

    _write_jsonl(train_path, [_with_arch_id(r) for r in train_data])
    _write_jsonl(val_path, [_with_arch_id(r) for r in val_data])
    _write_jsonl(test_path, [_with_arch_id(r) for r in test_data])
    return train_path, val_path, test_path


def _load_messages(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    examples: list[dict] = []
    for row in _stream_jsonl(path):
        if "messages" in row and row.get("architecture_id"):
            examples.append({"messages": row["messages"], "architecture_id": str(row["architecture_id"])})
    if not examples:
        raise ValueError(f"No examples loaded from {path}")
    return examples


def _load_model(model_name: str, max_seq_len: int):
    candidates = [model_name, *(m for m in MODEL_FALLBACKS if m != model_name)]
    last_error: Exception | None = None

    for model_id in candidates:
        try:
            return FastLanguageModel.from_pretrained(
                model_name=model_id,
                max_seq_length=max_seq_len,
                dtype=None,
                load_in_4bit=True,
                trust_remote_code=True,
            )
        except Exception as exc:
            last_error = exc

    raise RuntimeError(f"Could not load any Qwen3-8B checkpoint. Tried: {candidates}") from last_error


def _format_dataset(examples: dict, tokenizer) -> dict:
    texts = []
    for messages in examples["messages"]:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        texts.append(text + tokenizer.eos_token)
    return {"text": texts}


def train_model(
    train_path: Path,
    val_path: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    model_name: str = MODEL_NAME,
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
) -> None:
    train_examples = _load_messages(train_path)
    val_examples = _load_messages(val_path)

    train_ids = {ex["architecture_id"] for ex in train_examples}
    val_ids = {ex["architecture_id"] for ex in val_examples}
    overlap = train_ids & val_ids
    if overlap:
        raise ValueError(
            f"Architecture leakage: {len(overlap)} architecture_id(s) appear in both train and validation."
        )

    model, tokenizer = _load_model(model_name, max_seq_len)
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
        use_rslora=False,
        loftq_config=None,
    )
    FastLanguageModel.for_training(model)

    train_dataset = Dataset.from_list([{"messages": ex["messages"]} for ex in train_examples])
    val_dataset = Dataset.from_list([{"messages": ex["messages"]} for ex in val_examples])
    train_dataset = train_dataset.map(
        _format_dataset,
        fn_kwargs={"tokenizer": tokenizer},
        batched=True,
        remove_columns=train_dataset.column_names,
    )
    val_dataset = val_dataset.map(
        _format_dataset,
        fn_kwargs={"tokenizer": tokenizer},
        batched=True,
        remove_columns=val_dataset.column_names,
    )

    steps_per_epoch = max(
        1,
        (len(train_dataset) + BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS - 1)
        // (BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS),
    )
    total_steps = MAX_STEPS if MAX_STEPS > 0 else steps_per_epoch * NUM_EPOCHS
    warmup_steps = max(1, int(total_steps * WARMUP_RATIO))
    if MAX_STEPS > 0 and warmup_steps >= MAX_STEPS:
        warmup_steps = max(1, MAX_STEPS // 10)

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    ta_kwargs = {
        "per_device_train_batch_size": BATCH_SIZE,
        "per_device_eval_batch_size": BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "warmup_steps": warmup_steps,
        "num_train_epochs": NUM_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "fp16": not use_bf16,
        "bf16": use_bf16,
        "logging_steps": 20,
        "logging_first_step": True,
        "optim": "adamw_8bit",
        "weight_decay": WEIGHT_DECAY,
        "lr_scheduler_type": "linear",
        "seed": SEED,
        "output_dir": str(output_dir),
        "report_to": "none",
        "save_strategy": "epoch",
        "eval_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "max_grad_norm": MAX_GRAD_NORM,
        "gradient_checkpointing": True,
        "dataloader_num_workers": 2,
        "save_total_limit": SAVE_TOTAL_LIMIT,
    }
    if MAX_STEPS > 0:
        ta_kwargs["max_steps"] = MAX_STEPS

    try:
        training_args = TrainingArguments(**ta_kwargs)
    except TypeError:
        ta_kwargs["evaluation_strategy"] = ta_kwargs.pop("eval_strategy")
        training_args = TrainingArguments(**ta_kwargs)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_len,
        packing=False,
        args=training_args,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)],
    )
    trainer.train()

    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

