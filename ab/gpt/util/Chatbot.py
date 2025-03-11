import re

from transformers import PreTrainedTokenizer, PreTrainedModel, pipeline

extra_instructions = (
    " Use PyTorch for the implementation. Keep the code short. Name the main class of the model \"Net\"."
    " The model code must include default parameters for initialization in the constructor. "
    "Provide only the code. Don't provide any explanation. Remove any text from this reply. "
    "Don't include comments in the code."
)

example_prompt = (
        "Write PyTorch code for an efficient classification model that includes self-attention blocks."
        + extra_instructions
)


class ChatBot:
    def __init__(self, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, keep_memory=False):
        self.model = model
        self.tokenizer = tokenizer
        self.__pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )
        self.__keep_memory = keep_memory
        if self.__keep_memory:
            self.__messages = []

    def chat(self, prompt: str, max_len=None, max_words=None, engineer_prompt=True, code_only=True) -> str:
        if engineer_prompt:
            prompt += extra_instructions

        if self.__keep_memory:
            self.__messages.append({"role": "user", "content": prompt})
            in_next = self.__messages
        else:
            in_next = [{"role": "user", "content": prompt}]

        out = self.__pipeline(
            in_next,
            max_new_tokens=max_words,
            max_len=max_len
        )[0]["generated_text"][-1]['content']
        assert isinstance(out, str)

        if self.__keep_memory:
            self.__messages.append({"role": "assistant", "content": out})

        # if code_only:
        #     x = re.search("```((.|\s)*?)```", out)
        #     if x:
        #         out = x.group()
        #         out = out.replace("```python", "")
        #         out = out.replace("```", "")

        # return out
        if code_only:
            # 改进的代码提取逻辑
            code_blocks = []
            lines = out.split('\n')
            in_code_block = False
            current_block = []
            
            for line in lines:
                if line.strip().startswith('```'):
                    if in_code_block:  # 结束代码块
                        in_code_block = False
                        if current_block:
                            code_blocks.append('\n'.join(current_block))
                            current_block = []
                    else:  # 开始代码块
                        in_code_block = True
                        # 去除可能的语言标记
                        if len(line.strip()) > 3:
                            continue
                elif in_code_block:
                    current_block.append(line)
            
            # 如果找到代码块，返回第一个代码块
            if code_blocks:
                return code_blocks[0]
            else:
                # 如果没有找到代码块，尝试使用正则表达式作为后备方案
                # 使用超时保护防止卡住
                import time
                start_time = time.time()
                timeout = 15  # 2秒超时
                
            try:
                # 使用普通的re模块和超时保护
                start_time = time.time()
                timeout = 1.0  # 1秒超时
                
                try:
                    # 标准的re模块没有timeout参数，所以我们用时间判断
                    x = re.search(r"```((.|\s)*?)```", out)
                    # 检查是否超时
                    if time.time() - start_time > timeout:
                        print("Regular expression took too long, using fallback method")
                        return out
                    
                    if x:
                        code = x.group(1)
                        if code.strip().startswith('python'):
                            code = code[6:].strip()
                        return code.strip()
                    else:
                        return out
                except Exception as e:
                    print(f"Error in regex: {e}")
                    return out
            except Exception:
                # 如果所有方法都失败，返回原始输出
                return out
        
        return out
