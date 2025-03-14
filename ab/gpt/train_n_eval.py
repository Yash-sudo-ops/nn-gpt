import json
import os
import shutil
from pathlib import Path
import deepspeed
import pandas as pd

import argparse

import torch
import torchvision
from transformers import BitsAndBytesConfig, TrainingArguments
from torchvision.transforms import transforms
from peft import LoraConfig

from ab.gpt.util.CVModelEvaluator import CVModelEvaluator
from ab.gpt.util.LoRATrainer import LoRATrainer, find_all_linear_names
from ab.gpt.util.preprocessors.CodeChgPrmPromptPreprocessorSFT import CodeChgPrmPromptPreprocessor as CodePromptPreprocessor
from ab.gpt.util.Chatbot import ChatBot
from ab.gpt.util.ModelLoader import ModelLoader

import ab.nn.api as nn_dataset

import traceback

from peft import PeftModel

with open("./conf/config.json") as config_file:
    config = json.load(config_file)
assert isinstance(config, dict)

token_from_file = True if config["token_from_file"] == "True" else False
base_model_name = config["base_model_name"]
num_epochs = int(config["num_epochs"])
num_test_epochs = int(config["num_test_epochs"])
use_deepspeed = True if config["use_deepspeed"] == "True" else False

access_token = None
if token_from_file:
    with open("../../token") as f:
        access_token = f.readline()

# Deepspeed
ds_config = os.path.join("conf","deepspeed_config.json")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--skip', type=int, default=-1,
                        help="Number of epoches to skip the generation.")
    parser.add_argument('-p', '--peft', type=str, default=None,
                        help="Path to saved lora layers.")
    args = parser.parse_args()
    skip_epoch = args.skip
    peft_path = args.peft
    print(f"[DEBUG]Argument Information:\nSkip generation until Epoch: {skip_epoch}\nPath to saved LoRA Layers: {peft_path}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

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
    )

    # Load test prompts
    with open('./util/test_prompts.json') as prompt_file:
        prompt_dict = json.load(prompt_file)
    assert isinstance(prompt_dict, dict)

    # Load model and tokenizer
    model_loader = ModelLoader(
        base_model_name,
        bnb_config,
        access_token=access_token,
        use_deepspeed=use_deepspeed,
        base_path="../../",
    )
    model, tokenizer = model_loader.get_model(), model_loader.get_tokenizer()
    if not (peft_path is None):
        print(f"Load saved LoRA layer from path:{peft_path}")
        model = PeftModel.from_pretrained(model,peft_path)
        model = model.merge_and_unload()

    # initialize deepspeed before we do infer in ChatBot, since trainer is not initialized now.
    if use_deepspeed:
        deepspeed.initialize(model=model, config_params=ds_config)

    print("Using Max Length:", model_loader.get_max_length())
    data_processor = CodePromptPreprocessor(model_loader.get_max_length(), tokenizer)
    dataset = data_processor.get_dataset()
    print("Dataset length:", len(dataset))
    ds_updated = False

    peft_config = LoraConfig(
        r=64,  # dimension of the updated matrices
        lora_alpha=64,  # parameter for scaling
        target_modules=find_all_linear_names(model),
        lora_dropout=0.1,  # dropout probability for layers
        bias="none",
        task_type="CAUSAL_LM",
        use_dora=True,
        inference_mode=False
    )

    already_peft = False # Prevent multiple LoRA layers
    # loop train and eval cycles
    chat_bot = ChatBot(model, tokenizer)# Only initialize ONCE

    for epoch in range(num_epochs):
        print(f"[INFO]Start Epoch {epoch}")
        out_path = "../../Models/epochs/A" + str(epoch) + "/"
        # Move inside the loop to create new prompt with newly created models.
        print("Prepairing prompts for generation, this might take a while...")
        prompts = []
        for key in prompt_dict.keys():
            if epoch<skip_epoch:
                continue # Prompts are useless when generation is skipped
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
        
        from tqdm import tqdm
        # produce new CV models
        for idx, prompt in tqdm(enumerate(prompts)):
            if epoch<skip_epoch:
                print(f"Skipped Epoch {epoch}")
                continue  # Skipped
            prompt, origdf = prompt
            code_file = Path(out_path + "synth_cv_models/B" + str(idx) + "/code.py")
            code_file.parent.mkdir(exist_ok=True, parents=True)
            code = chat_bot.chat(
                prompt,
                engineer_prompt=True,
                code_only=True,
                max_words=5000 ## Reduce memory usage
            )
            with open(code_file, 'w') as file:
                file.write(code)
            df_file = Path(out_path + "synth_cv_models/B" + str(idx) + "/dataframe.df")
            if origdf is None:
                if os.path.isfile(df_file): # Clean up dataframe.df, if no additional information generated this time.
                    os.remove(df_file)
                    print(f"[DEBUG]Removed unmatched file: {df_file}")
            else:
                orig_code_file = Path(out_path + "synth_cv_models/B" + str(idx) + "/code_"+origdf['nn']+".py")
                with open(orig_code_file, 'w') as file:
                    file.write(origdf['nn_code'])
                # Store DataFrame information, mainly for passing parameters to evaluator.
                origdf.to_pickle(df_file)

        # evaluate produced CV models
        for cv_model in os.listdir(out_path + "synth_cv_models"):
            cv_model = str(os.fsdecode(cv_model))
            if os.path.isdir(out_path + "synth_cv_models/" + cv_model):
                for tries in range(2):
                    try:
                        df = None
                        df_file = Path(out_path + "synth_cv_models/" + cv_model + "/dataframe.df")
                        if os.path.isfile(df_file): # The code has additional information provided
                            df = pd.read_pickle(df_file)
                            prm = df['prm']
                            # prm['epoch'] = df['epoch']
                            prm['epoch'] = 5 # Force evaluation being 5
                            # Reduce Memory Usage
                            if prm['transform'].__contains__("512"):
                                prm['transform'].replace("512","128")
                            elif prm['transform'].__contains__("256"):
                                prm['transform'].replace("256","128")
                            elif prm['transform'].__contains__("299"):
                                prm['transform'].replace("299","128")
                            print(f"Determining {cv_model} with prm{prm}")
                            print(f"Model Path:{out_path + 'synth_cv_models/' + cv_model}")
                            evaluator = CVModelEvaluator(out_path + "synth_cv_models/" + cv_model,
                                                     task = df['task'],dataset=df['dataset'],metric=df['metric'],prm=prm,save_to_db=True,prefix=df['nn'].split('-')[0],save_path=out_path + "synth_cv_models/" + cv_model)
                        else:
                            evaluator = CVModelEvaluator(out_path + "synth_cv_models/" + cv_model)
                        accuracy = evaluator.evaluate()
                        accuracies = {
                            # str(evaluator.get_args()): (accuracy, num_test_epochs)
                            str(evaluator.get_args()): accuracy
                        }
                        name,_a,_b = accuracy
                        with open(out_path + "synth_cv_models/" + cv_model + "/accuracies.json", "w+") as acc_file:
                            json.dump(accuracies, acc_file)
                        Path("./Dataset/nn/nn").mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(out_path + "synth_cv_models/" + cv_model + "/code.py", "./Dataset/nn/nn/" + name + ".py")
                        if df is None:
                            Path("./Dataset/nn/stat/" + name).mkdir(parents=True, exist_ok=True)
                            for epo in range(prm['epoch']):
                                shutil.copyfile(out_path + "synth_cv_models/" + cv_model + "/"+str(epo+1)+".json", "./Dataset/nn/stat/" + name + "/"+str(epo+1)+".json")
                        else:
                            Path("./Dataset/nn/stat/" + f"{df['task']}_{df['dataset']}_{df['metric']}_{name}").mkdir(parents=True, exist_ok=True)
                            for epo in range(prm['epoch']):
                                shutil.copyfile(out_path + "synth_cv_models/" + cv_model + "/"+str(epo+1)+".json", "./Dataset/nn/stat/" + f"{df['task']}_{df['dataset']}_{df['metric']}_{name}" + "/"+str(epo+1)+".json")
                        ds_updated = True
                        break
                    except Exception as error:
                        print("failed to determine accuracy for", cv_model)
                        with open(out_path + "synth_cv_models/" + cv_model + "/error_" + str(tries) + ".txt", "w+") as error_file:
                            error_file.write(str(traceback.format_exc())) # Track traceback to have rich information on the exception.
                        with open(out_path + "synth_cv_models/" + cv_model + "/code.py", "r") as code_file:
                            code_txt = code_file.read()
                        new_code = chat_bot.chat(
                            'The error "' + str(error) +
                            '" was occurred in the following code. fix this problem. '
                            "Provide only the code. Don't provide any explanation. Remove any text from this reply. + \n " +
                            code_txt,
                            engineer_prompt=False,
                            max_words=5000
                        )
                        os.remove(out_path + "synth_cv_models/" + cv_model + "/code.py")
                        with open(out_path + "synth_cv_models/" + cv_model + "/code.py", 'w') as file:
                            file.write(new_code)


        # fine tune model for 1 epoch / Using training_args and save copy
        print(f"[DEBUG]Perform finetune at epoch {epoch}.")
        if ds_updated:
            print(f"Epoch{epoch}[DEBUG]Regenerate dataset...")
            data_processor = CodePromptPreprocessor(model_loader.get_max_length(), tokenizer)
            dataset = data_processor.get_dataset()
            ds_updated = False

        lora_tuner = LoRATrainer(
                model,
                tokenizer,
                training_args=training_args,
                access_token=access_token,
                peft_config=peft_config,
                already_peft=already_peft
            )
        already_peft = True
        model = lora_tuner.train(dataset, out_path + base_model_name + "_tuned")

        # del model
        # del tokenizer
        #
        # model_loader = ModelLoader(
        #     base_model_name,
        #     bnb_config,
        #     access_token=access_token,
        #     local_path=out_path + base_model_name + "_tuned"
        # )
        #
        # # use updated model for next round
        # model, tokenizer = model_loader.get_model(), model_loader.get_tokenizer()


if __name__ == "__main__":
    main()
