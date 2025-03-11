import copy
import json
from typing import Dict, Sequence
from pandas import DataFrame
from overrides import override
from transformers import PreTrainedTokenizerBase
from datasets import Dataset
import ab.nn as lemur

from ab.gpt.util.preprocessors.PreprocessorBase import PreprocessorBase

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

class EnhancedCodePromptPreprocessor(PreprocessorBase):
    """结合DeepSeek和原始代码的增强预处理器"""
    def __init__(self, max_len: int, tokenizer: PreTrainedTokenizerBase, prompts_path=None):
        super().__init__(max_len, tokenizer)
        self.prompts_path = prompts_path
    
    @override
    def get_raw_dataset(self) -> DataFrame:
        """获取原始数据集，保持与原始代码相同的逻辑"""
        return self._get_from_prompts()
    
    def _get_from_prompts(self) -> DataFrame:
        """从提示文件获取数据"""
        dataframe = DataFrame(columns=["instruction", "context", "response", "category", "text"])
        
        with open(self.prompts_path) as prompt_file:
            prompt_dict = json.load(prompt_file)
        assert isinstance(prompt_dict, dict)
        
        for key in prompt_dict.keys():
            if prompt_dict[key]['single_row']:
                continue  # 跳过单行提示
            prompt = ""
            for pr in prompt_dict[key]['prompts']:
                prompt += pr + "\n"
            
            # 获取神经网络数据集代码
            if prompt_dict[key]['task'] == "all":
                data = lemur.data(only_best_accuracy=True)
            elif prompt_dict[key]['task'] == "":
                data = None
            else:
                data = lemur.data(only_best_accuracy=True, task=prompt_dict[key]['task'])
            
            # 获取附加神经网络数据集代码
            if prompt_dict[key]['addon_task'] == "all":
                addon_data = lemur.data(only_best_accuracy=True)
            elif prompt_dict[key]['addon_task'] == "":
                addon_data = None
            elif prompt_dict[key]['addon_task'] == prompt_dict[key]['task']:
                addon_data = data  # 当它们相同时，避免重复采样
            else:
                addon_data = lemur.data(only_best_accuracy=True, task=prompt_dict[key]['addon_task'])
            
            if data is None:
                assert ValueError("Task must be specified (or set to 'all')")
            else:
                for _, row in data.iterrows():
                    para_dict = dict()
                    for it in prompt_dict[key]["input_list"]:
                        para_dict[it['para']] = row[it['value']]
                    
                    if not (addon_data is None):
                        # 应用不重复过滤器
                        filter = "nn==nn"  # 默认过滤器应始终为True
                        if 'no_repeat' in prompt_dict[key]:
                            for filter_it in prompt_dict[key]['no_repeat']:
                                if isinstance(row[filter_it], str):
                                    filter += f"&{filter_it}!='{row[filter_it]}'"
                                else:
                                    filter += f"&{filter_it}!={row[filter_it]}"
                        if 'keep_same' in prompt_dict[key]:
                            for filter_it in prompt_dict[key]['keep_same']:
                                if isinstance(row[filter_it], str):
                                    filter += f"&{filter_it}=='{row[filter_it]}'"
                                else:
                                    filter += f"&{filter_it}=={row[filter_it]}"
                        
                        addon_row = addon_data.query(filter)
                        if len(addon_row) > 0:
                            addon_row = addon_row.sample(n=1).iloc[0]
                        else:
                            continue  # 没有匹配要求的结果时跳过
                        
                        for it in prompt_dict[key]["addon_list"]:
                            para_dict[it['para']] = addon_row[it['value']]
                    
                    inst = prompt.format(**para_dict)
                    response = "```\n" + data.query(f"task=='{row['task']}'&nn!='{row['nn']}'").sample(n=1).iloc[0]['nn_code'] + "\n```"
                    text = self.tokenizer.apply_chat_template(
                        [
                            {"role": "user", "content": inst},
                            {"role": "assistant", "content": response}
                        ], tokenize=False
                    )
                    
                    dataframe.loc[len(dataframe)] = [inst, "", response, "", text]
        
        return dataframe
    
    def train_tokenize_function(self, examples):
        """DeepSeek风格的训练数据标记化函数"""
        sources = [
            build_instruction_prompt(instruction)
            for instruction in examples['instruction']
        ]
        targets = [f"{output}\n{EOT_TOKEN}" for output in examples['response']]
        data_dict = preprocess(sources, targets, self.tokenizer)
        return data_dict
    
    def get_dataset(self, data_path=None):
        """获取处理后的数据集"""
        # 首先获取原始数据集
        raw_dataset = self.get_raw_dataset()
        
        # 创建数据集字典
        dataset_dict = {
            'instruction': raw_dataset['instruction'].tolist(),
            'response': raw_dataset['response'].tolist()
        }
        
        # 使用DeepSeek风格的处理方式
        sources = [
            build_instruction_prompt(instruction)
            for instruction in dataset_dict['instruction']
        ]
        targets = [f"{output}\n{EOT_TOKEN}" for output in dataset_dict['response']]
        
        # 预处理为模型可用格式
        processed_dataset = preprocess(sources, targets, self.tokenizer)
        
        # 转换为HuggingFace Dataset格式
        return Dataset.from_dict({
            'input_ids': processed_dataset['input_ids'],
            'labels': processed_dataset['labels']
        })