"""
This file is largely borrowed from OpenR (https://github.com/openreasoner/openr)
"""

import abc
from typing import Dict, List, Optional
import numpy as np
import copy
import pdb
import torch
from utils import print_with_rank
from transformers import PreTrainedTokenizer
from reason.inference.lm_call import LMCallingConfig, ConcatedLMGenResult

INVALID_ANS = "[invalid]"


class NoLegalActionException(Exception):
    pass


class ResetException(Exception):
    pass


class BaseEnv(abc.ABC):
    """Basic environment to use for MCTS"""

    @abc.abstractmethod
    def reset(self, update_legal_action: bool):
        raise NotImplementedError

    @abc.abstractmethod
    def step(self, action, update_legal_action=True):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def legal_actions(self):
        raise NotImplementedError

    @abc.abstractmethod
    def copy(self):
        raise NotImplementedError

    @staticmethod
    def build_query_str(
        cot_task_desc: Optional[str],
        cot_examples: Optional[str],
        problem_format_str: str,
        problem_input: str,
        is_few_shot: bool = False,
        model_names = [],
    ):
        """a wrap function that wrap the problem text with certrain format
        e.g. prompt_str = "Input: " + join_numbers(" ", xs) + "\nSteps:\n"
        >>> query_str = Game24Env.build_query_str("1 1 1 1")
        >>> print(query_str)
        >>> Use numbers and basic arithmetic operations (+ - * /) to obtain 24. Each step, you are only allowed to choose two of the remaining numbers to obtain a new number.
        Input: 1 1 1 1
        Steps:

        >>>
        """

        messages = []
        problem_format_str = problem_format_str.format(question=problem_input)
        if 'deepseek-r1' in model_names[0].lower():
            messages.append({"role": "user", "content": problem_format_str + '\n' + cot_task_desc})
        else:
            if cot_task_desc:
                messages.append({"role": "system", "content": cot_task_desc})
            if is_few_shot:
                for example in cot_examples:
                    messages.append({"role": "user", "content": example["question"]})
                    messages.append({"role": "assistant", "content": example["answer"]})
            messages.append({"role": "user", "content": problem_format_str})

        return messages

    @staticmethod
    def build_response_str(answer_str: str, tokenizer: PreTrainedTokenizer, add_eos_token: bool):
        raise NotImplementedError


class CoTEnv(BaseEnv):
    """The basic environment for solving natural language problems using CoT"""

    def _is_correct(self, completion) -> bool:
        raise NotImplementedError

    def get_reward(self):
        """To implement based on learned reward model"""
        raise NotImplementedError

    def __init__(
        self,
        config,
        math_problems,
        llm_gen_fns,
        rm_call,
        task_desc_str: str,
        cot_example_str: str,
        problem_format_str: str,
        reset=True,
        sep=None,
        model_names=[],
        update_legal_action=True,
    ):
        self.config = config
        self.mcts_mode = "play_with_bot_mode"
        self.math_problems = math_problems
        self.llm_gen_fns = llm_gen_fns
        self.rm_call = rm_call
        self.action_history = None
        self.reward_history = None
        self.token_history = None
        self.prob_history = None
        self.model_history = None
        self.math_problem = None
        self._legal_actions = None
        self.is_few_shot = config.get("is_few_shot", False)
        self.add_step_prompt = config.get("add_step_prompt", False)
        self.direct_io = config.get("direct_io", 0)
        self.double_line_break = config.get("double_line_break", 0)
        # self.prm_step_tag = rm_call.prm_step_tag  # "ки\n"
        self.prm_step_tag = "ки\n"
        self.sep = sep
        self.model_names = model_names

        if config.get("cot_prompt", ""):
            task_desc_str = config["cot_prompt"]
        self._task_desc_str = task_desc_str
        self._cot_example_str = cot_example_str
        self._problem_format_str = problem_format_str

        prefixes = []
        if self._task_desc_str is not None:
            prefixes.append(self._task_desc_str)
        if self.is_few_shot:
            prefixes.append(self._cot_example_str)
        if len(prefixes) > 0:
            self.task_prefix = "\n".join(prefixes)
        else:
            self.task_prefix = None

        if reset:
            self.reset(update_legal_action=update_legal_action)

    def reset(self, update_legal_action=True):
        # reset environment to problem idx
        self.set_problem(idx=0)
        self.action_history = []
        self.reward_history = []
        self.token_history = []
        self.prob_history = []
        self.model_history = []
        self._init_query = self.build_query_str(
            cot_examples=self._cot_example_str,
            cot_task_desc=self._task_desc_str,
            problem_format_str=self._problem_format_str,
            problem_input=self.math_problem["question"],
            is_few_shot=self.is_few_shot,
            model_names=self.model_names,
        )
        if update_legal_action:
            cnt = 0
            max_try = 1
            while cnt < max_try:
                cnt += 1
                try:
                    self._legal_actions, api_completion_token = self.update_legal_actions(initial=True)
                    break
                except Exception as e:
                    if cnt == max_try:
                        self._legal_actions, api_completion_token = self.update_legal_actions(initial=True, force_update=True)
                        print("Force update legal actions:", self._legal_actions)
        else:
            api_completion_token = 0
        info = {"api_completion_token": api_completion_token}
        return self.get_state(model_name='raw'), info

    def step(self, action, update_legal_action=True, model_name="", custom_n=0, reward=0.0, num_token=0, prob=0.0):
        self.action_history.append(action)
        self.reward_history.append(reward)
        self.token_history.append(num_token)
        self.prob_history.append(prob)
        if model_name:
            self.model_history.append(model_name)
        state = self.get_state(model_name=model_name)
        reward = self.get_reward()
        terminated, truncated, info = self.get_done_and_info()

        if not (terminated or truncated) and update_legal_action:  # update legal actions
            cnt = 0
            while cnt < 3:
                cnt += 1
                try:
                    self._legal_actions, api_completion_token = self.update_legal_actions(custom_n=custom_n)
                    info["api_completion_token"] = api_completion_token
                    break
                except NoLegalActionException as e:
                    if cnt == 3:
                        terminated = True
                        reward = 0
                        self._legal_actions = None
                        info["winner"] = 2
                        info["api_completion_token"] = 0
                    else:
                        pass
        else:
            self._legal_actions = None
            if info["winner"] == 1:
                reward = 1.0
            info["api_completion_token"] = 0
        return state, reward, terminated, truncated, info

    def get_state(self, model_name='other', add_step_prompt=False):
        messages = copy.deepcopy(self._init_query)
        messages.append({"role": "assistant", "content": "".join(self.action_history)})

        if add_step_prompt and self.direct_io != 2:
            if 'llama-3' in self.model_names[0].lower():  # TODO: Check llama
                sep = "## Step"
            else:
                sep = "Step"
            if not self.double_line_break:  # TODO: Check double
                messages[-1]["content"] += f"{sep} {len(self.action_history) + 1}: "
        if model_name == 'raw':
            ret = ""
            for idx, mess in enumerate(messages):
                ret += f'{mess["role"]}: {mess["content"]}'
                if idx < len(messages) - 1:
                    ret += '\n'
            return ret
        return messages

    def post_process_act(self, action: str):
        # This step may change the token count
        return action

    def update_legal_actions(self, initial=False, force_update=False, custom_n=0):
        if len(self.llm_gen_fns) == 1:
            if initial:
                n = self.config["max_actions"]
            elif custom_n:
                n = custom_n
            else:
                n = self.config["max_actions"] // self.config["beam_size"]
            if self.direct_io:
                stop_str, include_stop_str_in_output = None, False
            else:
                stop_str, include_stop_str_in_output = self.sep, True
            first_generation = len(self.action_history) == 0
            messages = self.get_state(self.llm_gen_fns[0].model_name, add_step_prompt=self.add_step_prompt)
            result: ConcatedLMGenResult = self.llm_gen_fns[0](
                messages=messages,
                config=LMCallingConfig(
                    n=n,
                    stop_str=stop_str,  # '\n\n' for Qwen-2.5-Math-1.5B-Instruct
                    include_stop_str_in_output=include_stop_str_in_output,
                    first_generation=first_generation,
                    **self.config["generation_config"],
                ),
            )
            texts = result.text  # [text1, text2]
            logps_avg_by_len = result.logp_avg_by_len  # [-0.10557132510029904, -0.23053854329903292]
            token_len = result.num_tokens  # [212, 192]
            temp_model_names = [self.llm_gen_fns[0].model_name] * len(texts)
            temp_model_ids = [0] * len(texts)
            finish_reason_list = []
            if isinstance(result.finish_reason, list):
                finish_reason_list.extend(result.finish_reason)
            else:
                raise ValueError("finish_reason should be a list")
        else:
            raise NotImplementedError

        text_list, prob_list, num_token_list = [], [], []
        model_names, model_ids = [], []
        next_state_terminated = {}
        raw_text_list = []

        for i in range(len(texts)):
            if self.direct_io:
                terminated = True
            else:
                if isinstance(self.sep, str):
                    terminated = not texts[i].endswith(self.sep)
                elif isinstance(self.sep, list):
                    terminated = True
                    for sep in self.sep:
                        if texts[i].endswith(sep):
                            terminated = False
                            break
            processed_act = self.post_process_act(texts[i])
            finish_reason = finish_reason_list[i]
            if not self.double_line_break:
                temp_act = processed_act.replace("## Step ", "Step ")
                is_double_line_break = temp_act.endswith("\n\n") and temp_act.startswith("Step ") and (len(temp_act) == len("Step 1: \n\n") or len(temp_act) == len("Step 10: \n\n"))
                if is_double_line_break:
                    finish_reason = "length"
            if len(processed_act) > 0 and processed_act not in text_list and finish_reason == "stop":
                # only stop is valid, otherwise the output action is truncated actually
                text_list.append(processed_act)
                raw_text_list.append(texts[i])
                prob_list.append(logps_avg_by_len[i])
                num_token_list.append(token_len[i])
                next_state_terminated[processed_act] = terminated
                model_names.append(temp_model_names[i])
                model_ids.append(temp_model_ids[i])
            elif force_update or self.direct_io:
                text_list.append(processed_act)
                raw_text_list.append(texts[i])
                prob_list.append(logps_avg_by_len[i])
                num_token_list.append(token_len[i])
                next_state_terminated[processed_act] = terminated
                model_names.append(temp_model_names[i])
                model_ids.append(temp_model_ids[i])

        if len(prob_list) == 0:
            print_with_rank("state: {}".format(self.get_state(model_name='raw')))
            if len(self.llm_gen_fns) == 1:
                print_with_rank("gen_result: {}".format(result))
            raise NoLegalActionException("No possible action have been generated.")

        prob_list = np.exp(prob_list)
        prob_list = list(prob_list)
        # prob_list = np.array(prob_list)
        # prob_list = prob_list / np.sum(prob_list)  # normalize probability

        _legal_actions = [{
            "action": action,
            "prob": prob,
            "num_token": n_token,
            "finish_reason": finish_reason,
            "model_name": model_name,
            "model_id": model_id,
            "messages": messages,
            "stop_str": stop_str,
            "raw_action": raw_action,
        } for action, prob, n_token, finish_reason, model_name, model_id, raw_action in zip(text_list, prob_list, num_token_list,
            finish_reason_list, model_names, model_ids, raw_text_list)]

        if len(self.llm_gen_fns) == 1:
            completion_tokens = result.completion_tokens
        self._next_state_terminated = next_state_terminated

        return _legal_actions, completion_tokens

    def set_problem(self, idx):
        self.math_problem = self.math_problems[idx]

    @property
    def query(self):
        return self._init_query

    @property
    def question(self) -> str:
        return self.math_problem["question"]

    @property
    def answer(self):
        if len(self.action_history) == 0:
            return ""
        elif self.direct_io == 2:
            assert len(self.action_history) == 1
            return self.action_history[0]
        elif self.direct_io == 1:
            assert len(self.action_history) == 1
            steps = self.action_history[0].split("\n\n")
            answer = ""
            for step in steps:
                if step.strip() == "":
                    continue
                answer += step.strip() + f" {self.prm_step_tag}"
            return answer
        else:
            answer = ""
            for action in self.action_history:
                answer += action.strip() + f" {self.prm_step_tag}"
            return answer

    def check_stop_by_answer(self):
        if isinstance(self._stop_str, str) and self._stop_str in self.action_history[-1]:
            terminated = True
        elif isinstance(self._stop_str, list):
            terminated = True
            for stop_str in self._stop_str:
                if stop_str not in self.action_history[-1]:
                    terminated = False
        return terminated

    def check_stop_by_sep(self):
        if isinstance(self.sep, str):
            return self.sep not in self.action_history[-1]
        elif isinstance(self.sep, list):
            for sep in self.sep:
                if sep in self.action_history[-1]:
                    return False
        return False

    def get_done_and_info(self):
        info = {"winner": 0}
        # done when reaches maximum length or LLM generates stop words
        if self._stop_str is not None and self.check_stop_by_answer():
            terminated = True
        elif self._next_state_terminated[self.action_history[-1]]:
            terminated = True
        else:
            terminated = self.check_stop_by_sep()

        if self.config["max_length"] > 1:
            truncated = len(self.action_history) >= self.config["max_length"]
            assert len(self.action_history) <= self.config["max_length"]
        else:
            truncated = False
        if terminated or truncated:
            if self._is_correct(self.action_history[-1]):
                info["winner"] = 1
            else:
                info["winner"] = 2
            return terminated, truncated, info
        return terminated, truncated, info

    def copy(self):
        env = self.__class__(
            self.config,
            self.math_problems,
            self.llm_gen_fns,
            self.rm_call,
            self._task_desc_str,
            self._cot_example_str,
            self._problem_format_str,
            reset=False,
        )
        env.math_problem = copy.deepcopy(self.math_problem)
        env._legal_actions = copy.deepcopy(self._legal_actions)
        env.action_history = copy.deepcopy(self.action_history)
        env.reward_history = copy.deepcopy(self.reward_history)
        env.token_history = copy.deepcopy(self.token_history)
        env.prob_history = copy.deepcopy(self.prob_history)
        env.model_history = copy.deepcopy(self.model_history)
        env._init_query = copy.deepcopy(self._init_query)
        env._next_state_terminated = copy.deepcopy(self._next_state_terminated)
        return env

    @property
    def legal_actions(self):
        return self._legal_actions
