from ab.nn.util.Const import base_module, ab_root_path, out_dir
import json
from pathlib import Path
NN_TRAIN_EPOCHS = 1# How many epochs to train the altered NN for evaluation

new_nn_file = 'new_nn.py'
hp_file = 'hp.txt'
transformer_file = 'tr.py'
new_out_file = 'full_output.txt'

gpt = 'gpt'
gpt_dir = ab_root_path / base_module / gpt
conf_dir = gpt_dir / 'conf'
conf_prompt_dir = conf_dir / 'prompt'
conf_test_dir = conf_prompt_dir / 'test'
conf_train_dir = conf_prompt_dir / 'train'
conf_llm_dir = conf_dir / 'llm'

nngpt_dir = out_dir / 'nngpt'

# ── Branch isolation override ─────────────────────────────────────────────────
# Setting NNGPT_DIR_OVERRIDE env var to redirect all nngpt output to a custom path.
# Used by CurriculumGenerationPipeline.py to isolate per-dataset/level/k runs.
import os as _os
_nngpt_override = _os.environ.get("NNGPT_DIR_OVERRIDE")
if _nngpt_override:
    nngpt_dir = Path(_nngpt_override)
    print(f"[Const] nngpt_dir overridden → {nngpt_dir}")
# ─────────────────────────────────────────────────────────────────────────────
acgpt_dir = out_dir / 'acgpt'
nnrag_dir = out_dir / 'rag'

new_dataset_dir = nngpt_dir / 'new_lemur'
new_lemur_nn_dir = new_dataset_dir / 'nn'
new_lemur_stat_dir = new_dataset_dir / 'train'

brute_dir = gpt_dir / 'brute'
ast_dir = brute_dir / 'ast'
ea_dir = brute_dir / 'ea'
fract_dir = brute_dir / 'fract'
trans_dir = brute_dir / 'trans'

config_file = conf_llm_dir / 'nngpt_ds_coder_1.3b_instruct.json' # 'nngpt_dsr1_distill_qwen_7b_r.json'
with open(config_file) as f:
    base_llm = json.load(f)['base_model_name']

# Hugging Face cache directories .. onnx sepcific
huggingface_cache = out_dir / 'llm'
huggingface_tokenizer_cache = out_dir / 'tokenizer'
default_huggingface_cache = huggingface_cache
default_huggingface_tokenizer_cache = huggingface_tokenizer_cache


_MODEL_CONTEXT: dict[str, tuple[int | None, int | None]] = {
    "open-r1/OlympicCoder-7B":                     (8192, 8192),
    "deepseek-ai/deepseek-coder-7b-instruct-v1.5": (8192, None),
    "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B":       (4096, None),
    "deepseek-ai/deepseek-coder-6.7b-instruct":    (4096, 4096),
    "ABrain/HPGPT-DeepSeek-R1-Distill-Qwen-7B-R":  (4096, None),
    "ABrain/NNGPT-UniqueArch-Rag":                  (4096, None),
    "unsloth/gpt-oss-20b":                         (4600, 1600),
    "Qwen/Qwen2.5-Coder-7B-Instruct":              (32768, None),
    "mistralai/Mistral-7B-Instruct-v0.3":           (4096, None),
}

def get_model_context(base_model_name: str) -> tuple[int | None, int | None]:
    return _MODEL_CONTEXT.get(base_model_name, (None, None))


_MODEL_FLAGS: dict[str, dict] = {
    "unsloth/gpt-oss-20b":                      {"use_unsloth": True,  "load_in_4bit": True},
    "deepseek-ai/deepseek-coder-6.7b-instruct": {"use_unsloth": False, "load_in_4bit": False},
}

def get_model_flags(base_model_name: str) -> dict:
    return _MODEL_FLAGS.get(base_model_name, {})


def model_dir(base):
    return base / 'llm'

def synth_dir(base):
    return base / 'synth_nn'

def tokenizer_dir(base):
    return base / 'tokenizer'

nngpt_model = model_dir(out_dir)
nngpt_upload = out_dir / 'llm_to_upload'
llm_tokenizer_out = tokenizer_dir(out_dir)

def llm_dir(base, name):
    return model_dir(base) / name

def llm_tokenizer_dir(base, name):
    return tokenizer_dir(base) / name

def epoch_dir(*args):
    e_dir = llm_dir(nngpt_dir, 'epoch')
    for d in args:
        e_dir = e_dir / f'A{d}'
    return e_dir