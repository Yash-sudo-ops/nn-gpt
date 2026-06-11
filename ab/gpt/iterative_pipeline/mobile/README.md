# Mobile deploy track

Optional pipeline branch for edge deployment: **mobile-oriented code generation** → **NNEval** → **`.pth` + TFLite (FP32 / INT8)**.

The core iterative pipeline is unchanged unless you pass `--mobile_deploy`. Conversion follows the [nn-lite `torch2tflite.py`](https://github.com/ABrain-One/nn-lite/blob/main/ab/lite/torch2tflite.py) stack (`ai-edge-torch-nightly` + `ai_edge_tensorflow`, not PyPI `tensorflow`).

## Install (once)

```bash
pip install -r requirements-mobile.txt
```

Verify (no PyPI `tensorflow` package — `ai_edge_tensorflow` registers the `tensorflow` import used by the converter):

```bash
python -c "import ai_edge_torch; import tensorflow as tf; print('ok', ai_edge_torch.__version__, tf.__version__)"
```

If that fails, only install `requirements-mobile.txt`; do not run `pip install tensorflow`.

## `.pth` weights (vs nn-lite HuggingFace checkpoints)

[nn-lite `torch2tflite.py`](https://github.com/ABrain-One/nn-lite/blob/main/ab/lite/torch2tflite.py) downloads fixed `.pth` files from HuggingFace per architecture name.

This pipeline instead:

1. **Generates** `new_nn.py` from your LLM (mobile prompt when `--mobile_deploy`).
2. **Trains** on CIFAR-10 via NNEval (`check_nn`) and records accuracy in `1.json` / `eval_info.json`.
3. **Saves trained weights** to `nneval/gen_XXXX/weights.pth` when `--mobile_deploy` (uses `save_pth_weights=True` during eval).
4. **Exports TFLite** from that trained module + `weights.pth` (FP32 + INT8), same converter stack as nn-lite.

No HuggingFace checkpoint is required for your generated models.

## Output layout

After a cycle with `--mobile_deploy`:

```
out/curation_output/cycle_N/
  generation/accepted_code/B*/new_nn.py   # LLM output
  nneval/gen_XXXX/                        # trained + evaluated + weights.pth
  mobile_deploy/gen_XXXX/
    new_nn.py
    weights.pth
    tflite/model_fp32.tflite
    tflite/model_int8.tflite   # if INT8 conversion succeeds
    mobile_deploy.json
  mobile_deploy/mobile_deploy_summary.json
```

## 1. Quick test (skip fine-tuning)

Use this to validate **mobile prompt → `new_nn.py` → TFLite** without waiting for LoRA (~hours per cycle).

```bash
# Optional: clear stale generation/eval from an earlier run
rm -rf out/curation_output/cycle_1/generation out/curation_output/cycle_1/nneval

python -m ab.gpt.TuneNNGen \
  --run_iterative_pipeline \
  --cycles 1 \
  --models_per_cycle 2 \
  --llm_conf ds_coder_7b_instruct.json \
  --skip_finetuning \
  --mobile_deploy \
  --force_regenerate \
  --skip_data_augment
```

| Flag | Purpose |
|------|---------|
| `--skip_finetuning` | No LoRA; generate from base model in `llm_conf` (default: `out/llm/<org>/<model>`) |
| `--mobile_deploy` | `mobile_rag_rules.json` prompt + TFLite export after eval |
| `--force_regenerate` | Remove existing `cycle_N/generation` and regenerate |
| `--skip_data_augment` | Skip filter → training-data augment (faster smoke test) |

Optional: pin the model path explicitly:

```bash
  --generation_model out/llm/deepseek-ai/deepseek-coder-7b-instruct-v1.5
```

**Standalone mobile export** (if `nneval/` already exists):

```bash
python -m ab.gpt.iterative_pipeline.run_mobile_deploy --cycle 1
```

## 2. Full pipeline (with fine-tuning)

Run when mobile flow is verified; includes LoRA per cycle and training-data augmentation.

```bash
python -m ab.gpt.TuneNNGen \
  --run_iterative_pipeline \
  --cycles 6 \
  --models_per_cycle 3 \
  --num_train_epochs 2 \
  --llm_conf ds_coder_7b_instruct.json \
  --mobile_deploy
```

Do **not** pass `--skip_finetuning` or `--skip_data_augment` for the production run.

**Shape convention:** NNEval instantiates `Net` with `in_shape` from `get_in_shape()` as `(N, C, H, W)` (e.g. `(1, 3, 256, 256)`). Use `in_shape[1]` for input channels (matches training data), not `in_shape[0]` (batch size).

Resume example (after cycle 2 checkpoint exists):

```bash
python -m ab.gpt.TuneNNGen \
  --run_iterative_pipeline \
  --cycles 6 \
  --models_per_cycle 3 \
  --num_train_epochs 2 \
  --llm_conf ds_coder_7b_instruct.json \
  --mobile_deploy \
  --resume_from_cycle 3
```

## Optional flags

| Flag | Default | Description |
|------|---------|-------------|
| `--mobile_input_size` | 32 | Spatial size for TFLite trace (CIFAR-style) |
| `--mobile_max_params` | 500000 | Skip export if model is larger |
| `--no_mobile_tflite` | — | Copy + `weights.pth` only |
| `--no_mobile_bench` | — | Skip desktop TFLite CPU timing |
| `--benchmark_android` | off | ADB `benchmark_model` when a device is attached |

## Android benchmark (later)

When a phone is connected via USB:

```bash
python -m ab.gpt.TuneNNGen \
  ... \
  --mobile_deploy \
  --benchmark_android
```

Requires `benchmark_model` on the device (see nn-lite setup).

## Modules

| File | Role |
|------|------|
| `deploy.py` | Orchestrates per-cycle mobile deploy from `nneval/` |
| `export_weights.py` | Load `new_nn.py` → `weights.pth` |
| `torch2tflite.py` | FP32 + INT8 TFLite via `ai_edge_torch` |
| `android_bench.py` | Optional on-device benchmark via ADB |
| `../run_mobile_deploy.py` | CLI without running the full pipeline |
