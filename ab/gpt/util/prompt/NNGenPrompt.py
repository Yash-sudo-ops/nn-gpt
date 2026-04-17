import json
import re
import math
import pandas as pd
from pandas import DataFrame
from transformers import PreTrainedTokenizerBase
from overrides import override
from tqdm import tqdm
<<<<<<< HEAD
from ab.gpt.util.prompt.Prompt import Prompt
import ab.nn.api as lemur
=======

from ab.nn.util.db.Query import JoinConf
>>>>>>> 9720e9a479988a293869e5fe8627a684754f11d7


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
<<<<<<< HEAD

        print("\n" + "=" * 60)
        print("GENERAL FORMULA EXTRACTOR ENABLED")
        print("Pattern: << formula >> will be evaluated")
        print("=" * 60 + "\n")
=======
>>>>>>> 9720e9a479988a293869e5fe8627a684754f11d7

        for key in prompt_dict.keys():
            dataframe = DataFrame(columns=['instruction', 'context', 'response', 'category', 'text'])
            prompt_lists.append(dataframe)
<<<<<<< HEAD

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
=======
            prompt = '\n'.join(prompt_dict[key]['prompt'])
            print('Preparing Data...', flush=True)
            key_dict = prompt_dict[key]
            num_joint_nns = key_dict.get('num_joint_nns') or 1

            # For JOIN queries, do NOT pass max_rows — the LIMIT applies before
            # the JOIN and causes an O(n²) correlated scan. Slice the result after.
            use_join = num_joint_nns >= 2
            # For JOIN queries, cap rows to avoid O(n²) scan on the temp table.
            # Slice to n_training_prompts after the fact instead of relying on LIMIT inside the JOIN.
            join_cap = 1000
            data = lemur.data(
                only_best_accuracy=only_best_accuracy,
                task=key_dict.get('task'),
                nn_prefixes=tuple(key_dict.get('nn_prefixes')),
                max_rows=join_cap if use_join else n_training_prompts,
                sql=None if not use_join else JoinConf(
                    num_joint_nns=num_joint_nns,
                    same_columns=tuple(key_dict.get('keep_same', [])),
                    diff_columns=tuple(key_dict.get('no_repeat', [])),
                    enhance_nn=key_dict.get('improve', False)
                )
            )

            print('Data acquisition complete', flush=True)

            # Check if this is delta mode
            use_delta = key_dict.get('use_delta', False) or 'delta' in key.lower()

            for _, row in tqdm(data.iterrows(), total=n_training_prompts or len(data)):
                # print(f'Row keys: {list(row.index)}', flush=True)
                if n_training_prompts and len(dataframe) >= n_training_prompts:
                    break
                para_dict = dict()
                for it in prompt_dict[key]['input_list']:
                    para_dict[it['para']] = row[it['value']]

                nn_code_max_chars = key_dict.get('nn_code_max_chars')
                if nn_code_max_chars and 'nn_code' in para_dict and isinstance(para_dict['nn_code'], str):
                    para_dict['nn_code'] = para_dict['nn_code'][:nn_code_max_chars]

                # Inject columns referenced in the output template but absent from input_list
                # (e.g. better_dataset for classification tasks). Only applies when output_type
                # is 'classification' so other tasks are unaffected.
                if key_dict.get('output_type') == 'classification':
                    output_template = '\n'.join(key_dict['output'])
                    for col in row.index:
                        if f'{{{col}}}' in output_template and col not in para_dict:
                            para_dict[col] = row[col]

                inst = prompt.format(**para_dict)

                # Compute delta if delta mode is enabled
                if use_delta and 'addon_nn_code' in para_dict and 'nn_code' in para_dict:
                    try:
                        from ab.gpt.util.DeltaUtil import compute_delta
                        baseline_code = para_dict.get('nn_code', '')
                        improved_code = para_dict.get('addon_nn_code', '')

                        if baseline_code and improved_code:
                            computed_delta = compute_delta(baseline_code, improved_code)
                            if not computed_delta:
                                computed_delta = ""
                        else:
                            computed_delta = ""

                        output = '\n'.join(prompt_dict[key]['output'])
                        try:
                            response = output.format(**para_dict)
                        except KeyError:
                            response = output
                            for k, v in para_dict.items():
                                response = response.replace(f'{{{k}}}', str(v))
                        response = response.replace('{computed_delta}', computed_delta)
                    except Exception as e:
                        print(f'[WARNING] Failed to compute delta for key {key}: {e}. Using regular output.', flush=True)
                        output = '\n'.join(prompt_dict[key]['output'])
                        try:
                            response = output.format(**para_dict)
                        except KeyError:
                            response = output
                            for k, v in para_dict.items():
                                response = response.replace(f'{{{k}}}', str(v))
                        response = response.replace('{computed_delta}', '')
                else:
                    # Regular mode: use output template as-is
                    output = '\n'.join(prompt_dict[key]['output'])
                    response = output.format(**para_dict)

                text = self.tokenizer.apply_chat_template(
                    [
                        {'role': 'user', 'content': inst},
                        {'role': 'assistant', 'content': response}
                    ], tokenize=False)

                # print(f"Prompt: {inst}", flush=True)
                # print(f"Output: {response}", flush=True)
>>>>>>> 9720e9a479988a293869e5fe8627a684754f11d7

                dataframe.loc[len(dataframe)] = [inst, "", response, "", text]

        if prompt_lists:
            print(f'\n✅ Generated {sum(len(df) for df in prompt_lists)} training examples')
            return pd.concat(prompt_lists, ignore_index=True)
        return DataFrame()