"""
This file is largely borrowed from OpenR (https://github.com/openreasoner/openr)
"""

from dataclasses import dataclass
from typing import List, Optional, Union

from transformers import AutoTokenizer

from reason.inference.text_generation import ConcatedLMGenResult, _generate_fastchat, _generate_sgl


@dataclass
class LMCallingConfig:
    n: int = 1
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1  # -1 for vllm by default
    max_new_tokens: int = 512
    stop_token_ids: Optional[List[int]] = None
    stop_str: Optional[Union[str, List[str]]] = None
    include_stop_str_in_output: bool = False
    first_generation: bool = False


class LanguageModelCallingFunction:

    def __init__(self, llm_step_tag: str = None):
        self.llm_step_tag = llm_step_tag

    def __call__(self, messages: List, config: LMCallingConfig) -> ConcatedLMGenResult:
        raise NotImplementedError


class VLLMRemoteCaller(LanguageModelCallingFunction):

    def __init__(
        self,
        model_name,
        model_path,
        controller_addr="http://0.0.0.0:28777",
        llm_step_tag: str = None,
        apply_chat_template: bool = False,
        multi_gpu: bool = False,
        serve_type: str = "fastchat",
        double_line_break: int = 0,
    ):
        self.model_name = model_name
        self.model_path = model_path
        self.controller_addr = controller_addr
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.apply_chat_template = apply_chat_template
        self.multi_gpu = multi_gpu
        self.serve_type = serve_type
        self.double_line_break = double_line_break
        super().__init__(llm_step_tag)

    def __call__(self, messages: str, config: LMCallingConfig) -> ConcatedLMGenResult:
        if self.serve_type == "fastchat":
            return _generate_fastchat(
                messages=messages,
                model_name=self.model_name,
                n=config.n,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                max_new_tokens=config.max_new_tokens,
                stop_token_ids=config.stop_token_ids,
                stop_str=config.stop_str,
                controller_addr=self.controller_addr,
                include_stop_str_in_output=config.include_stop_str_in_output,
                tokenizer=self.tokenizer,
                apply_chat_template=self.apply_chat_template,
                multi_gpu=self.multi_gpu,
                double_line_break=self.double_line_break,
                first_generation=config.first_generation,
            )
        elif self.serve_type == "sgl_api":
            return _generate_sgl(
                messages=messages,
                model_name=self.model_name,
                n=config.n,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                max_new_tokens=config.max_new_tokens,
                stop_token_ids=config.stop_token_ids,
                stop_str=config.stop_str,
                controller_addr=self.controller_addr,
                include_stop_str_in_output=config.include_stop_str_in_output,
                tokenizer=self.tokenizer,
                apply_chat_template=self.apply_chat_template,
                multi_gpu=self.multi_gpu,
                double_line_break=self.double_line_break,
                first_generation=config.first_generation,
            )
