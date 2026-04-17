import json
import re
import math
import pandas as pd
from pandas import DataFrame
from transformers import PreTrainedTokenizerBase
from overrides import override
from tqdm import tqdm
from ab.gpt.util.prompt.Prompt import Prompt
import ab.nn.api as lemur


def evaluate_delimited_formulas(text: str, para_dict: dict) -> str:
    """
    Find patterns like <<accuracy / duration>> and replace with calculated values.
    """
    pattern = r'<<(.*?)>>'
    
    def replace_match(match):
        formula = match.group(1).strip()
        try:
            expr = formula
            # Replace variable names with their values - order by length to prevent partial matches
            for key in sorted(para_dict.keys(), key=len, reverse=True):
                val = para_dict[key]
                try:
                    val = float(val)
                except (ValueError, TypeError):
                    pass
                if isinstance(val, (int, float)):
                    # Use regex with word boundaries
                    expr = re.sub(rf'\b{re.escape(key)}\b', str(val), expr)
            
            # Safe evaluation
            safe_globals = {
                "__builtins__": {},
                "math": math,
                "abs": abs,
                "round": round,
                "min": min,
                "max": max,
            }
            result = eval(expr, safe_globals)
            
            # Format result nicely
            if isinstance(result, float):
                if abs(result) < 0.001:
                    return f"{result:.2e}"
                elif result > 100:
                    return f"{result:.1f}"
                else:
                    return f"{result:.4f}"
            return str(result)
        except Exception as e:
            print(f"[FORMULA ERROR] '{formula}' - {e}")
            return f"<<{formula}>>"
    
    return re.sub(pattern, replace_match, text)


class NNGenPrompt(Prompt):
    def __init__(self, max_len: int, tokenizer: PreTrainedTokenizerBase, prompts_path):
        super().__init__(max_len, tokenizer)
        self.prompts_path = prompts_path

    @override
    def get_raw_dataset(self, only_best_accuracy, n_training_prompts=None) -> DataFrame:
        prompt_lists = []

        with open(self.prompts_path) as prompt_file:
            prompt_dict = json.load(prompt_file)
        assert isinstance(prompt_dict, dict)

        print("\n" + "=" * 60)
        print("GENERAL FORMULA EXTRACTOR ENABLED")
        print("Pattern: << formula >> will be evaluated")
        print("=" * 60 + "\n")

        for key in prompt_dict.keys():
            dataframe = DataFrame(columns=['instruction', 'context', 'response', 'category', 'text'])
            prompt_lists.append(dataframe)

            prompt_template = '\n'.join(prompt_dict[key]['prompt'])
            print(f'Preparing Data for key: {key}...', flush=True)

            # Fetch prun data
            data = lemur.prun_data(max_rows=n_training_prompts)

            if 'status' in data.columns:
                data = data[data['status'] == 'success']

            if data.empty:
                print("ERROR: No successful pruning records found!")
                return DataFrame()

            print(f"Fetched {len(data)} records from PRUN table")

            # Get nn_code from stat table (if needed)
            stat_data = lemur.data(only_best_accuracy=True, max_rows=1000)
            nn_code_map = {}
            transform_code_map = {}
            for _, stat_row in stat_data.iterrows():
                nn_name = stat_row.get('nn', '')
                if nn_name:
                    nn_code_map[nn_name] = stat_row.get('nn_code', '')
                    transform_code_map[nn_name] = stat_row.get('transform_code', '')

            for idx, row in tqdm(data.iterrows(), total=len(data)):
                if n_training_prompts and len(dataframe) >= n_training_prompts:
                    break

                para_dict = {}
                model_name = row.get('model_name', row.get('nn', ''))

                for it in prompt_dict[key]['input_list']:
                    para_name = it['para']
                    db_column = it['value']

                    if db_column == 'nn_code':
                        para_dict[para_name] = nn_code_map.get(model_name, f"# Model: {model_name}")
                    elif db_column == 'transform_code':
                        para_dict[para_name] = transform_code_map.get(model_name, "")
                    else:
                        para_dict[para_name] = row.get(db_column, "")

                # First format the prompt with regular variables
                try:
                    inst = prompt_template.format(**para_dict)
                except KeyError as e:
                    print(f"[WARNING] Missing key {e}, skipping row {idx}")
                    continue
                
                # Debug: Print before formula evaluation
                has_formula = '<<' in inst
                if has_formula:
                    print(f"\n[DEBUG] Found formula in prompt for {model_name}")
                    print(f"[DEBUG] Before evaluation: {inst[:200]}...")
                
                # CRITICAL: Evaluate formulas inside << >>
                inst = evaluate_delimited_formulas(inst, para_dict)
                
                # Debug: Print after formula evaluation
                if has_formula and '<<' not in inst:
                    print(f"[DEBUG] After evaluation: {inst[:200]}...")
                    print(f"[DEBUG] ✅ Formula successfully evaluated!")
                
                # Print full prompt (first 2 examples only to avoid spam)
                if idx < 2:
                    print(f"\n{'='*60}")
                    print(f"FULL PROMPT (Example {idx+1}):")
                    print(inst)
                    print(f"{'='*60}\n")

                output_template = '\n'.join(prompt_dict[key]['output'])
                try:
                    response = output_template.format(**para_dict)
                except KeyError:
                    response = output_template
                    for k, v in para_dict.items():
                        response = response.replace(f'{{{k}}}', str(v))
                
                response = evaluate_delimited_formulas(response, para_dict)

                text = self.tokenizer.apply_chat_template(
                    [{'role': 'user', 'content': inst}, {'role': 'assistant', 'content': response}],
                    tokenize=False)

                dataframe.loc[len(dataframe)] = [inst, "", response, "", text]

        if prompt_lists:
            print(f'\n✅ Generated {sum(len(df) for df in prompt_lists)} training examples')
            return pd.concat(prompt_lists, ignore_index=True)
        return DataFrame()