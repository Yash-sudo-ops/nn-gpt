import json
import os
import shutil
from pathlib import Path

import ab.nn.api as nn_dataset
import deepspeed
import pandas as pd
import torch
from peft import LoraConfig
from transformers import (
    BitsAndBytesConfig, 
    TrainingArguments,
    Trainer,
    AutoModelForCausalLM,
    AutoTokenizer
)

from ab.gpt.util.CVModelEvaluator import CVModelEvaluator
from ab.gpt.util.Chatbot import ChatBot
from ab.gpt.util.LoRATrainer import LoRATrainer, find_all_linear_names
from ab.gpt.util.preprocessors.DSCodePreprocessor import EnhancedCodePromptPreprocessor

import copy
import json
from typing import Dict, Sequence
import torch
from transformers import PreTrainedTokenizerBase
import ab.nn as lemur


# 从DeepSeek代码借鉴的常量
IGNORE_INDEX = -100
EOT_TOKEN = "<|EOT|>"  # 用于标记响应结束

def build_instruction_prompt(instruction: str):
    """从DeepSeek借鉴的提示模板构建函数，但调整为CV代码生成场景"""
    return '''
You are an AI assistant that can write computer vision code based on user requirements.
### Instruction:
{}
### Response:
'''.format(instruction.strip()).lstrip()

def _tokenize_fn(strings: Sequence[str], tokenizer: PreTrainedTokenizerBase) -> Dict:
    """从DeepSeek借鉴的标记化函数"""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]

    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]

    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )

def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: PreTrainedTokenizerBase
) -> Dict:
    """从DeepSeek借鉴的数据预处理函数"""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]

    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX  # 只训练响应部分
    return dict(input_ids=input_ids, labels=labels)

class DataCollatorForSupervisedDataset(object):
    """从DeepSeek借鉴的数据收集器"""
    def __init__(self, tokenizer: PreTrainedTokenizerBase):
        self.tokenizer = tokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = [torch.tensor(x) for x in input_ids]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = [torch.tensor(x) for x in labels]
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

def safe_save_model_for_hf_trainer(trainer: Trainer, output_dir: str):
    """从DeepSeek借鉴的安全保存模型函数"""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)

def main():
    """主函数"""
    # 加载配置
    with open("./conf/config.json") as config_file:
        config = json.load(config_file)
    
    token_from_file = True if config["token_from_file"] == "True" else False
    base_model_name = config["base_model_name"]
    num_epochs = int(config["num_epochs"])
    num_test_epochs = int(config["num_test_epochs"])
    use_deepspeed = True if config["use_deepspeed"] == "True" else False
    
    access_token = None
    if token_from_file:
        with open("../../token") as f:
            access_token = f.readline()
    
    # DeepSpeed配置
    ds_config = os.path.join("conf", "deepspeed_config.json")
    
    # 创建BitsAndBytes配置
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    # 创建训练参数
    training_args = TrainingArguments(
        report_to=None,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=2,
        num_train_epochs=1,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1,
        output_dir="outputs",
        optim="paged_adamw_8bit",
        deepspeed=ds_config if use_deepspeed else None,
        disable_tqdm=True
    )
    
    # 加载测试提示
    with open('./util/gen_prompts.json') as prompt_file:
        prompt_dict = json.load(prompt_file)
    
    # 处理提示以生成测试样本
    prompts = []
    for key in prompt_dict.keys():
        # Legacy test_prompts handling
        if prompt_dict[key]['single_row']:
            for pr in prompt_dict[key]['prompts']:
                prompts.append((pr, None))
        else:
            prompt = ""
            for pr in prompt_dict[key]['prompts']:
                prompt += pr + "\n"
            # 获取神经网络数据集代码
            if prompt_dict[key]['task'] == "all":
                data = lemur.data(only_best_accuracy=True).groupby(by="nn").sample(n=1)
            elif prompt_dict[key]['task'] == "":
                data = None
            else:
                data = lemur.data(only_best_accuracy=True, task=prompt_dict[key]['task']).groupby(by="nn").sample(n=1)
            # 获取附加神经网络数据集代码
            if prompt_dict[key]['addon_task'] == "all":
                addon_data = lemur.data(only_best_accuracy=True)
            elif prompt_dict[key]['addon_task'] == "":
                addon_data = None
            elif prompt_dict[key]['addon_task'] == prompt_dict[key]['task']:
                addon_data = data  # 当它们相同时，避免重复采样
            else:
                addon_data = nn_dataset.data(only_best_accuracy=True, task=prompt_dict[key]['addon_task'])
            
            if data is None:
                prompts.append((pr, None))
            else:
                for _, row in data.iterrows():
                    para_dict = dict()
                    for it in prompt_dict[key]["input_list"]:
                        para_dict[it['para']] = row[it['value']]
                    if not (addon_data is None):
                        # 避免采样相同的神经网络代码
                        addon_row = addon_data.loc[addon_data.nn != row['nn']].sample(n=1).iloc[0]
                        for it in prompt_dict[key]["addon_list"]:
                            para_dict[it['para']] = addon_row[it['value']]
                    prompts.append((prompt.format(**para_dict), row))
    
    # 加载模型和分词器, todo
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        token=access_token
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        model_max_length=3000,  # 根据需要调整
        padding_side="right",
        use_fast=True,
        token=access_token
    )
    
    # 确保分词器有必要的特殊token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 如果使用DeepSpeed，初始化它
    if use_deepspeed:
        model = deepspeed.initialize(model=model, config=ds_config)[0]
    
    # 使用增强的预处理器
    data_processor = EnhancedCodePromptPreprocessor(3000, tokenizer, prompts_path='./util/gen_prompts.json')# diff
    dataset = data_processor.get_dataset()
    print("Dataset length:", len(dataset))
    ds_updated = False
    
    # 创建ChatBot实例用于生成和评估
    chat_bot = ChatBot(model, tokenizer)
    
    # 创建LoRA配置
    peft_config = LoraConfig(
        r=32,  # 更新矩阵的维度
        lora_alpha=64,  # 缩放参数
        target_modules=find_all_linear_names(model),
        lora_dropout=0.1,  # 层的dropout概率
        bias="none",
        task_type="CAUSAL_LM",
    )
    print("prompt_len:", len(prompts))
    # 训练和评估循环
    for epoch in range(num_epochs):
        out_path = f"../../Models/epochs/A{epoch}/"
        
        # 使用ChatBot生成新的CV模型
        for idx, prompt in enumerate(prompts):
            prompt, origdf = prompt
            code_file = Path(out_path + f"synth_cv_models/B{idx}/code.py")
            code_file.parent.mkdir(exist_ok=True, parents=True)
            
            # 生成代码
            code = chat_bot.chat(
                prompt,
                engineer_prompt=True,
                code_only=True,
                max_words=10000
            )
            # print(code)
            print("idx:", idx)
            with open(code_file, 'w') as file:
                file.write(code)
            
            # 保存原始DataFrame信息
            df_file = Path(out_path + f"synth_cv_models/B{idx}/dataframe.df")
            if origdf is None:
                if os.path.isfile(df_file):  # 如果这次没有生成额外信息，则清理dataframe.df
                    os.remove(df_file)
            else:
                # 存储DataFrame信息，主要用于传递参数给评估器
                origdf.to_pickle(df_file)
        
        # 评估生成的CV模型
        for cv_model in os.listdir(out_path + "synth_cv_models"):
            cv_model = str(os.fsdecode(cv_model))
            if os.path.isdir(out_path + "synth_cv_models/" + cv_model):
                for tries in range(2):
                    try:
                        df_file = Path(out_path + f"synth_cv_models/{cv_model}/dataframe.df")
                        if os.path.isfile(df_file):
                            df = pd.read_pickle(df_file)
                            prm = df['prm']
                            # prm['epoch'] = df['epoch']
                            prm['epoch'] = 1
                            evaluator = CVModelEvaluator(
                                f"../../Models/epochs/A{epoch}/synth_cv_models/{cv_model}",
                                task=df['task'],
                                dataset=df['dataset'],
                                metric=df['metric'],
                                prm=prm
                            )
                        else:
                            evaluator = CVModelEvaluator(f"../../Models/epochs/A{epoch}/synth_cv_models/{cv_model}")
                        
                        # 评估准确率
                        accuracy = evaluator.evaluate()
                        accuracies = {
                            str(evaluator.get_args()): accuracy
                        }
                        
                        # 保存准确率信息
                        with open(out_path + f"synth_cv_models/{cv_model}/accuracies.json", "w+") as acc_file:
                            json.dump(accuracies, acc_file)
                        
                        # 复制到数据集文件夹
                        Path(f"./Dataset/A{epoch}{cv_model}").mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(
                            out_path + f"synth_cv_models/{cv_model}/code.py",
                            f"./Dataset/A{epoch}{cv_model}/code.py"
                        )
                        shutil.copyfile(
                            out_path + f"synth_cv_models/{cv_model}/accuracies.json",
                            f"./Dataset/A{epoch}{cv_model}/accuracies.json"
                        )
                        
                        ds_updated = True
                        break
                    except Exception as error:
                        print(f"failed to determine accuracy for {cv_model}")
                        
                        # 记录错误信息
                        with open(out_path + f"synth_cv_models/{cv_model}/error_{tries}.txt", "w+") as error_file:
                            error_file.write(str(error))
                        
                        # 尝试修复代码
                        with open(out_path + f"synth_cv_models/{cv_model}/code.py", "r") as code_file:
                            code_txt = code_file.read()
                        
                        # 使用模型修复代码
                        new_code = chat_bot.chat(
                            f'The error "{error}" was occurred in the following code. fix this problem. '
                            "Provide only the code. Don't provide any explanation. Remove any text from this reply. + \n " +
                            code_txt,
                            engineer_prompt=False,
                            max_words=5000
                        )
                        
                        # 保存修复后的代码
                        os.remove(out_path + f"synth_cv_models/{cv_model}/code.py")
                        with open(out_path + f"synth_cv_models/{cv_model}/code.py", 'w') as file:
                            file.write(new_code)
        
        # 当数据集更新时，重新加载
        if ds_updated:
            data_processor = EnhancedCodePromptPreprocessor(3000, tokenizer, prompts_path='./util/train_nn_improvement_prompts.json')
            dataset = data_processor.get_dataset()
            ds_updated = False
        # 使用Trainer进行LoRA微调
        from peft import prepare_model_for_kbit_training, get_peft_model
        
        # 准备量化模型进行训练
        model_copy = prepare_model_for_kbit_training(model)
        
        # 应用LoRA配置
        peft_model = get_peft_model(model_copy, peft_config)
        peft_model.print_trainable_parameters()
        
        # 创建数据收集器
        data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
        
        # 创建Trainer实例
        trainer = Trainer(
            model=peft_model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
            tokenizer=tokenizer
        )
        # trainer = LoRATrainer(
        #     model=model,
        #     tokenizer=tokenizer,
        #     training_args=training_args,
        #     peft_config=peft_config,
        #     access_token=access_token
        # )

        
        # 训练模型
        trainer.train()
        
        # 安全保存模型
        safe_save_model_for_hf_trainer(trainer, "../../Models/" + base_model_name + "_tuned")

if __name__ == "__main__":
    main()