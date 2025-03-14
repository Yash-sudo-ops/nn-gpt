import json
import re
import ab.nn.api as nn_dataset
import pandas as pd

import argparse

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from tqdm import tqdm

import shutil
import traceback

from ab.gpt.util.CVModelEvaluator import CVModelEvaluator

import json
import os
from pathlib import Path
import pandas as pd

import torch

import ab.nn.api as nn_dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', type=int, default=8,
                        help="Maximum number of generation epochs.")
    args = parser.parse_args()
    limit_epoch = args.epochs

    # Load test prompts
    with open('./util/test_prompts_generation.json') as prompt_file:
        prompt_dict = json.load(prompt_file)
    assert isinstance(prompt_dict, dict)

    print("Loading Tokenizer and Model...")
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", trust_remote_code=True)
    print("Load Tokenizer Complete")
    model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()
    print("Load Model Complete, Start Loop...")

    for epoch in range(limit_epoch):
        out_path = "../../Models/epochs/A" + str(epoch) + "/"

        # Generate Prompts
        prompts = []
        for key in prompt_dict.keys():
            # Legency test_prompts handling
            if prompt_dict[key]['single_row']:
                for pr in prompt_dict[key]['prompts']:
                    prompts.append((pr,None))
            else:
                prompt = ""
                for pr in prompt_dict[key]['prompts']:
                    prompt+=pr+"\n"
                # Get nn-dataset codes
                if prompt_dict[key]['task']=="all":
                    data = nn_dataset.data(only_best_accuracy=True).groupby(by="nn").sample(n=1)
                elif prompt_dict[key]['task']=="":
                    data = None
                else:
                    data = nn_dataset.data(only_best_accuracy=True,task=prompt_dict[key]['task']).groupby(by="nn").sample(n=1)
                # Get addon nn-dataset codes
                if prompt_dict[key]['addon_task']=="all":
                    addon_data = nn_dataset.data(only_best_accuracy=True)
                elif prompt_dict[key]['addon_task']=="":
                    addon_data = None
                elif prompt_dict[key]['addon_task']==prompt_dict[key]['task']:
                    addon_data = data # When they are the same, avoid sampling twice
                else:
                    addon_data = nn_dataset.data(only_best_accuracy=True,task=prompt_dict[key]['addon_task'])
                if data is None:
                    prompts.append((pr,None))
                else:
                    for _, row in data.iterrows():
                        para_dict = dict()
                        for it in prompt_dict[key]["input_list"]:
                            para_dict[it['para']]=row[it['value']]
                        if not (addon_data is None):
                            ## Avoid sampling the same nn_code
                            addon_row = addon_data.loc[addon_data.nn!=row['nn']].sample(n=1).iloc[0]
                            for it in prompt_dict[key]["addon_list"]:
                                para_dict[it['para']]=addon_row[it['value']]
                        prompts.append((prompt.format(**para_dict),row))
        
        # produce new CV models
        B_index = 0
        for idx, prompt in tqdm(enumerate(prompts),desc="Generate Codes"):
            prompt, origdf = prompt
            code_file = Path(out_path + "synth_cv_models/B" + str(B_index) + "/code.py")
            inputs = tokenizer.apply_chat_template([{ 'role': 'user', 'content': prompt},], add_generation_prompt=True, return_tensors="pt").to(model.device)
            # tokenizer.eos_token_id is the id of <｜end▁of▁sentence｜>  token
            outputs = model.generate(inputs, max_new_tokens=10000, do_sample=False if epoch==0 else True, top_k=50, top_p=0.95, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id)
            out = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
            print("Response Available!")
            if out.count("```")<2:
                print(f"[INFO]Not exactly 2 \"```\", got {out.count("```")}. Skip to avoid infinite loop.")
                continue
            x = re.search("```((.|\s)*?)```", out)
            if x:
                print(f"[INFO]Saving code to: {code_file}")
                code_file.parent.mkdir(exist_ok=True, parents=True) # Move here to avoid empty folder
                out = x.group()
                out = out.replace("```python", "")
                out = out.replace("```", "")
                with open(code_file, 'w') as file:
                    file.write(out)
                df_file = Path(out_path + "synth_cv_models/B" + str(B_index) + "/dataframe.df")
                if origdf is None:
                    if os.path.isfile(df_file): # Clean up dataframe.df, if no additional information generated this time.
                        os.remove(df_file)
                else:
                    # Store DataFrame information, mainly for passing parameters to evaluator.
                    origdf.to_pickle(df_file)
                B_index += 1
            else:
                print("[INFO]Response Invalid!")
                continue

if __name__ == "__main__":
    main()
