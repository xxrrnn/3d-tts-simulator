"""
This file is largely borrowed from OpenR (https://github.com/openreasoner/openr)
"""

import traceback
from dataclasses import dataclass
from typing import List

import requests
from transformers import AutoTokenizer


@dataclass
class ConcatedLMGenResult:
    text: List[str]
    prompt_tokens: List[int]
    num_tokens: List[int]
    cumulative_logprob: List[float]
    logp_avg_by_len: List[float]
    finish_reason: List[str]

    # post init compute number of completion_tokens
    def __post_init__(self):
        self.completion_tokens = sum(self.num_tokens)


def process_prompt(prompt: str, tokenizer: AutoTokenizer, model_name, double_line_break=0, first_generation=False):
    eos_token = tokenizer.eos_token

    if prompt.endswith(f"{eos_token}\n"):
        prompt = prompt[:-len(f"{eos_token}\n")]
    elif prompt.endswith(eos_token):
        prompt = prompt[:-len(eos_token)]

    if double_line_break == 1:
        if 'llama-3' in model_name.lower() and 'meta-llama' in model_name.lower():
            if not prompt.endswith('\n\n'):
                prompt += '\n\n'
    elif double_line_break == 2:
        if 'llama-3' in model_name.lower() and 'meta-llama' in model_name.lower():
            if not prompt.endswith('\n\n'):
                prompt += '\n\n'

    return prompt


def _generate_fastchat(
    messages,
    model_name,
    n,
    temperature,
    top_p,
    top_k,
    max_new_tokens,
    stop_token_ids,
    stop_str,
    include_stop_str_in_output,
    controller_addr,
    tokenizer,
    apply_chat_template=False,
    worker_addr="",
    multi_gpu=False,
    double_line_break=0,
    first_generation=False,
) -> ConcatedLMGenResult:
    if multi_gpu:
        ret = requests.post(controller_addr + "/get_worker_address", json={"model": model_name})  # get worker address by model name
        worker_addr = ret.json()["address"]
        if not worker_addr:
            raise ValueError("Language Model name {} does not exist.".format(model_name))
    else:
        worker_addr = "http://0.0.0.0:10082"

    if apply_chat_template:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        prompt = process_prompt(prompt, tokenizer, model_name, double_line_break, first_generation)
    else:
        prompt = messages

    headers = {"User-Agent": "FastChat Client"}
    gen_params = {
        "model": model_name,
        "prompt": prompt,
        "temperature": temperature,
        "n": n,
        "top_p": top_p,
        "top_k": top_k,
        "stop_token_ids": stop_token_ids,
        "max_new_tokens": max_new_tokens,
        "stop": stop_str,
        "echo": False,
        "include_stop_str_in_output": include_stop_str_in_output,
    }

    try:
        response = requests.post(worker_addr + "/worker_generate", headers=headers, json=gen_params, stream=True)
        results = response.json()
    except Exception as e:
        print(f'Error in _generate_fastchat: {e}')

    output_token_lens = results["output_token_len"]
    cum_logps = results["cumulative_logprob"]
    avg_len_logps = [clp / max(1, otl) for clp, otl in zip(cum_logps, output_token_lens)]

    return ConcatedLMGenResult(
        text=results["text"],
        prompt_tokens=results["usage"]["prompt_tokens"],
        num_tokens=results["output_token_len"],
        cumulative_logprob=cum_logps,
        logp_avg_by_len=avg_len_logps,
        finish_reason=results["finish_reason"],
    )


def _generate_sgl(
    messages,
    model_name,
    n,
    temperature,
    top_p,
    top_k,
    max_new_tokens,
    stop_token_ids,
    stop_str,
    include_stop_str_in_output,
    controller_addr,
    tokenizer,
    apply_chat_template=False,
    worker_addr="",
    multi_gpu=False,
    double_line_break=0,
    first_generation=False,
) -> ConcatedLMGenResult:

    if multi_gpu:
        raise ValueError("Multi-GPU is not supported for SGlang model.")
    else:
        worker_addr = "http://0.0.0.0:10082"

    if apply_chat_template:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        prompt = process_prompt(prompt, tokenizer, model_name, double_line_break, first_generation)
    else:
        prompt = messages

    gen_params = {
        "text": prompt,
        "sampling_params": {
            "n": n,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "stop": stop_str,
            "stop_token_ids": stop_token_ids,
        },
        # "return_logprob": True,
    }

    try:
        response = requests.post(worker_addr + "/generate", json=gen_params, stream=True)
        results = response.json()
        if n == 1:
            results = [results]
    except Exception as e:
        print(f'Error in _generate_fastchat: {e}')

    text_list, prompt_tokens_list, num_tokens_list, finish_reason_list = [], [], [], []

    for output in results:
        text = output['text']
        prompt_tokens_list.append(output['meta_info']['prompt_tokens'])
        num_tokens_list.append(output['meta_info']['completion_tokens'])
        finish_reason = output['meta_info']['finish_reason']
        finish_reason_list.append(finish_reason['type'])
        if include_stop_str_in_output and finish_reason['type'] == 'stop' and 'matched' in finish_reason.keys():
            try:
                if isinstance(finish_reason['matched'], int):
                    pass
                elif isinstance(finish_reason['matched'], str):
                    if isinstance(stop_str, list):
                        assert finish_reason['matched'] in stop_str
                    else:
                        assert finish_reason['matched'] == stop_str
                    text += finish_reason['matched']
            except Exception as e:
                traceback.print_exc()
                print(f'Error in _generate_fastchat processing finish_reason: {e}')
        text_list.append(text)

    cum_logps = [0.0] * len(text_list)
    avg_len_logps = [0.0] * len(text_list)

    return ConcatedLMGenResult(
        text=text_list,
        prompt_tokens=prompt_tokens_list,
        num_tokens=num_tokens_list,
        cumulative_logprob=cum_logps,
        logp_avg_by_len=avg_len_logps,
        finish_reason=finish_reason_list,
    )
