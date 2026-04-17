import argparse
from typing import Literal
import torch
from ab.gpt.util.Const import nngpt_dir, new_out_file, NN_TRAIN_EPOCHS
from ab.nn.util.Const import out_dir
from pathlib import Path
import json
from ab.gpt.util.Const import conf_llm_dir

RUN_META = out_dir / 'nngpt' / 'run_config.json'


def persist_llm_conf(llm_conf, enable_merge=False):
    RUN_META.parent.mkdir(parents=True, exist_ok=True)
    base_model_name = None
    llm_conf_path = conf_llm_dir / llm_conf
    if llm_conf_path.exists():
        try:
            with open(llm_conf_path) as f:
                config = json.load(f)
            base_model_name = config.get("base_model_name")
        except Exception as e:
            print(f"Failed to read base_model_name from {llm_conf}: {e}")
    run_config = {"llm_conf": llm_conf, "enable_merge": enable_merge}
    if base_model_name:
        run_config["base_model_name"] = base_model_name
    with open(RUN_META, "w") as f:
        json.dump(run_config, f, indent=2)
    print(f"Run config saved: {RUN_META}")


START_LAYER = 0
END_LAYER = 24
TUNE_LAYERS = range(START_LAYER, END_LAYER)
R = 32
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'up_proj', 'down_proj', 'gate_proj')
TASK_TYPE = 'CAUSAL_LM'
BIAS: Literal['none', 'all', 'lora_only'] = 'none'
LEARNING_RATE = 1e-6
MAX_GRAD_NORM = 1.0
ENABLE_MERGE = False
PEFT = None
SKIP_EPOCHES = -1
NUM_TRAIN_EPOCHS = 3
LR_SCHEDULER = 'cosine'
PER_DEVICE_TRAIN_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
WARMUP_RATIO = 0.05
TEST_NN = 10
LOGGING_STEPS = 96
OPTIMIZER = 'paged_adamw_8bit'
LLM_TUNE_CONF = 'my_train_prun.json'
NN_GEN_CONF = 'my_test_prun.json'
NN_GEN_CONF_ID = 'pruning_improvement_train'
LLM_CONF = 'ds_coder_7b_olympic.json'
MAX_PROMPTS = 4 * 1024
MAX_NEW_TOKENS = 16 * 1024
SAVE_LLM_OUTPUT = True
USE_DEEPSPEED = False
NN_NAME_PREFIX = None
TEMPERATURE = 0.2
TOP_K = 20
TOP_P = 0.5
TEST_METRIC = None
ONNX_RUN = False
UNSLOTH_OPT = False
TRANS_MODE = False
PROMPT_BATCH = 2
USE_AGENTS = False
USE_PREDICTOR = False


def _best_dtype_args():
    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    return {"bf16": bf16_ok, "fp16": not bf16_ok}


def main(num_train_epochs=NUM_TRAIN_EPOCHS, lr_scheduler=LR_SCHEDULER, max_grad_norm=MAX_GRAD_NORM, test_metric=TEST_METRIC,
         tune_layers=TUNE_LAYERS, r=R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT, target_modules=TARGET_MODULES,
         task_type=TASK_TYPE, bias=BIAS, learning_rate=LEARNING_RATE, llm_tune_conf=LLM_TUNE_CONF, nn_gen_conf=NN_GEN_CONF,
         nn_gen_conf_id=NN_GEN_CONF_ID, llm_conf=LLM_CONF, test_nn=TEST_NN, peft=PEFT, skip_epoches=SKIP_EPOCHES,
         per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE, gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
         warmup_ratio=WARMUP_RATIO, logging_steps=LOGGING_STEPS, optimizer=OPTIMIZER, max_prompts=MAX_PROMPTS,
         save_llm_output=SAVE_LLM_OUTPUT, max_new_tokens=MAX_NEW_TOKENS, use_deepspeed=USE_DEEPSPEED,
         nn_name_prefix=NN_NAME_PREFIX, nn_train_epochs=NN_TRAIN_EPOCHS, temperature=TEMPERATURE, top_k=TOP_K,
         top_p=TOP_P, data_dir=None, base_data_dir=None, output_dir=None, evaluation_strategy=None, eval_steps=None,
         save_strategy=None, save_steps=None, save_total_limit=None, load_best_model_at_end=False,
         metric_for_best_model=None, warmup_steps=None, weight_decay=None, per_device_eval_batch_size=None,
         onnx_run=ONNX_RUN, unsloth_opt=UNSLOTH_OPT, trans_mode=TRANS_MODE, prompt_batch=PROMPT_BATCH,
         enable_merge=False, run_iterative_pipeline=False, cycles=5, models_per_cycle=150, samples_per_prompt=1,
         accuracy_threshold=0.40, min_selected_k=15, fallback_threshold=0.35, adaptive_threshold=False,
         novelty_check=True, resume_from_cycle=None, max_retries=3, use_optimized_training=True,
         use_agents=USE_AGENTS, use_predictor=USE_PREDICTOR, use_backbone=False):

    persist_llm_conf(llm_conf, enable_merge)

    if run_iterative_pipeline:
        print("--- Initiating Iterative Fine-Tuning Pipeline ---")
        from ab.gpt.iterative_finetune import IterativeFinetuner
        pipeline = IterativeFinetuner(llm_conf=llm_conf, cycles=cycles, models_per_cycle=models_per_cycle,
                                       samples_per_prompt=samples_per_prompt, accuracy_threshold=accuracy_threshold,
                                       min_selected_k=min_selected_k, fallback_threshold=fallback_threshold,
                                       adaptive_threshold=adaptive_threshold, novelty_check=novelty_check,
                                       resume_from_cycle=resume_from_cycle, max_retries=max_retries,
                                       use_optimized_training=use_optimized_training, num_train_epochs=num_train_epochs)
        pipeline.run()
        return

    from peft import LoraConfig
    from transformers import TrainingArguments
    from ab.gpt.util.Tune_prun import tune, ds_conf

    print(f'''All hyperparameters: num_train_epochs={num_train_epochs}, lr_scheduler={lr_scheduler},
          learning_rate={learning_rate}, llm_tune_conf={llm_tune_conf}, nn_gen_conf={nn_gen_conf},
          llm_conf={llm_conf}, test_nn={test_nn}, temperature={temperature}, top_k={top_k}, top_p={top_p}''')

    dtype_flags = _best_dtype_args()

    training_kwargs = {
        'num_train_epochs': num_train_epochs,
        'lr_scheduler_type': lr_scheduler,
        'max_grad_norm': max_grad_norm,
        'report_to': [],
        'per_device_train_batch_size': per_device_train_batch_size,
        'gradient_accumulation_steps': gradient_accumulation_steps,
        'warmup_ratio': warmup_ratio,
        'learning_rate': learning_rate,
        'bf16': True,
        'logging_steps': logging_steps,
        'output_dir': nngpt_dir / 'outputs',
        'optim': optimizer,
        'deepspeed': ds_conf if use_deepspeed else None,
        'gradient_checkpointing': True,
        **dtype_flags,
    }

    training_args = TrainingArguments(**training_kwargs)

    peft_config = LoraConfig(r=r, lora_alpha=lora_alpha, target_modules=target_modules,
                             layers_to_transform=list(tune_layers), lora_dropout=lora_dropout,
                             bias=bias, task_type=task_type)

    tune(test_nn, nn_train_epochs, skip_epoches, peft, llm_tune_conf, nn_gen_conf, nn_gen_conf_id, llm_conf,
         training_args, peft_config, max_prompts=max_prompts, save_llm_output=save_llm_output,
         max_new_tokens=max_new_tokens, nn_name_prefix=nn_name_prefix, temperature=temperature,
         top_k=top_k, top_p=top_p, onnx_run=onnx_run, trans_mode=trans_mode, prompt_batch=prompt_batch,
         use_agents=use_agents, use_predictor=use_predictor, enable_merge=enable_merge)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-ne', '--num_train_epochs', type=int, default=NUM_TRAIN_EPOCHS)
    parser.add_argument('--llm_tune_conf', type=str, default=LLM_TUNE_CONF)
    parser.add_argument('--nn_gen_conf', type=str, default=NN_GEN_CONF)
    parser.add_argument('--nn_gen_conf_id', type=str, default=NN_GEN_CONF_ID)
    parser.add_argument('--llm_conf', type=str, default=LLM_CONF)
    parser.add_argument('-n', '--test_nn', type=int, default=TEST_NN)
    parser.add_argument('-k', '--skip_epoches', type=int, default=SKIP_EPOCHES)
    parser.add_argument('--temperature', type=float, default=TEMPERATURE)
    parser.add_argument('--top_k', type=int, default=TOP_K)
    parser.add_argument('--top_p', type=float, default=TOP_P)
    args = parser.parse_args()
    main(**vars(args))