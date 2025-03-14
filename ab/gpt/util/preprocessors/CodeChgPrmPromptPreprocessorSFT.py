import ab.nn as lemur

from overrides import override
from pandas import DataFrame
from transformers import PreTrainedTokenizerBase

from ab.gpt.util.preprocessors.PreprocessorBase import PreprocessorBase

import json

class CodeChgPrmPromptPreprocessor(PreprocessorBase):
    """
    Assumes the existance of accuracies.json and folder based dataset
    """
    def __init__(self, max_len: int, tokenizer: PreTrainedTokenizerBase, prompts_path="./ab/gpt/util/train_nn_change_prm_prompts.json"):
        super().__init__(max_len, tokenizer)
        self.prompts_path = prompts_path

    @override
    def get_raw_dataset(self) -> DataFrame:
        """
        :return:
            pandas.Dataframe object with columns described in nn_api.data()
        """
        dataframe = DataFrame(columns=["instruction", "context", "response", "category", "text"])

        with open(self.prompts_path) as prompt_file:
            prompt_dict = json.load(prompt_file)
        assert isinstance(prompt_dict, dict)


        for key in prompt_dict.keys():
            if prompt_dict[key]['single_row']:
                continue # Compact test_prompt.json, in this case single_row prompts are not allowed
            prompt = ""
            for pr in prompt_dict[key]['prompts']:
                prompt+=pr+"\n"
            # Get nn-dataset codes
            print("Preparing Data(s)...")
            if prompt_dict[key]['task']=="all":
                data = lemur.data(only_best_accuracy=True).query("epoch==1")
            elif prompt_dict[key]['task']=="":
                data = None
            else:
                data = lemur.data(only_best_accuracy=True,task=prompt_dict[key]['task']).query("epoch==1")
            print("Data accquration complete")
            # Get addon nn-dataset codes
            if prompt_dict[key]['addon_task']=="all":
                addon_data = lemur.data(only_best_accuracy=True).query("epoch==1")
            elif prompt_dict[key]['addon_task']=="":
                addon_data = None
            elif prompt_dict[key]['addon_task']==prompt_dict[key]['task']:
                addon_data = data # When they are the same, avoid sampling twice
            else:
                addon_data = lemur.data(only_best_accuracy=True,task=prompt_dict[key]['addon_task']).query("epoch==1")
            print("Addon-Data accquration complete")
            if data is None:
                assert ValueError("Task must be specified (or set to 'all')")
            else:
                for _, row in data.iterrows():
                    para_dict = dict()
                    for it in prompt_dict[key]["input_list"]:
                        para_dict[it['para']]=row[it['value']]
                    if not (addon_data is None):
                        ## Apply non-repeat filter if exists:
                        filter = f"nn.str.split('-',n=1).str[0]=='{row['nn'].split('-')[0]}'&nn!='{row['nn']}'" # Default addon filter in this case should be models with same prefix
                        if 'no_repeat' in prompt_dict[key]: # Compact test_prompt.json, items in prm is not supported
                            for filter_it in prompt_dict[key]['no_repeat']:
                                if isinstance(row[filter_it],str):
                                    filter += f"&{filter_it}!='{row[filter_it]}'"
                                else:
                                    filter += f"&{filter_it}!={row[filter_it]}"
                        if 'keep_same' in prompt_dict[key]:
                            for filter_it in prompt_dict[key]['keep_same']:
                                if isinstance(row[filter_it],str):
                                    filter += f"&{filter_it}=='{row[filter_it]}'"
                                else:
                                    filter += f"&{filter_it}=={row[filter_it]}"
                        addon_row = addon_data.query(filter)
                        if len(addon_row)>0:
                            addon_row = addon_row.sample(n=1).iloc[0]
                        else:
                            continue # The case no result matches requirement
                        for it in prompt_dict[key]["addon_list"]:
                            para_dict[it['para']]=addon_row[it['value']]
                    inst = prompt.format(**para_dict)
                    # Having the same name prefix before '-'
                    response = "```\n"+addon_row['nn_code']+"\n```"
                    text = self.tokenizer.apply_chat_template(
                        [
                            {"role": "user", "content": inst},
                            {"role": "assistant", "content": response}
                        ], tokenize=False
                    )

                    dataframe.loc[len(dataframe)] = [inst, "", response, "", text]

        return dataframe
