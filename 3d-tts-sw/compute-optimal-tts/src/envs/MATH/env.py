"""
This file is largely borrowed from OpenR (https://github.com/openreasoner/openr)
"""

import copy
import re
from typing import List, Optional
import numpy as np
from envs.base_env import CoTEnv, NoLegalActionException, INVALID_ANS
from .prompt import COT_EXAMPLES, COT_TASK_DESC, PROBLEM_FORMAT_STR
# from .verify_utils import extract_answer as extract_fn, grade_answer
from .parse_utils_qwen import extract_answer as extract_fn, parse_ground_truth
from .grader import math_equal

ANS_RE = None
STOP_STR = None


def extract_answer(answer_str: str) -> str:
    return extract_fn(answer_str, data_name='math')


def extract_groundtruth(groundtruth_str: str) -> str:
    return parse_ground_truth(groundtruth_str, data_name='math')


def judge_correct(problem_str: str, extracted_groundtruth: Optional[str], answer: str) -> bool:
    result = math_equal(answer, extracted_groundtruth)
    return result


class Env(CoTEnv):
    def __init__(
        self,
        config,
        math_problems,
        llm_gen_fns,
        rm_call,
        task_desc_str: str = COT_TASK_DESC,
        cot_example_str: str = COT_EXAMPLES,
        problem_format_str: str = PROBLEM_FORMAT_STR,
        reset=True,
        update_legal_action=True,
    ):
        super().__init__(
            config,
            math_problems,
            llm_gen_fns,
            rm_call,
            task_desc_str if not config.get("cot_prompt", False) else config["cot_prompt"],
            cot_example_str,
            problem_format_str,
            reset,
            sep=config.get("sep", None),
            model_names=config["model_names"],
            update_legal_action=update_legal_action,
        )
        self._stop_str = config.get("stop_str", None)
        self.sep = config.get("sep", None)
        self.model_names = config["model_names"]
        self.double_line_break = config.get("double_line_break", 0)

    def post_process_act(self, action: str):
        action = action.strip()
        if self.direct_io == 2:
            return action
        elif self.direct_io == 1:
            if self.double_line_break:
                return action.strip()
            if 'llama-3' in self.model_names[0].lower():  # TODO: Check llama
                sep = "## Step "
                if self.add_step_prompt and not action.startswith(sep):
                    action = f"{sep}{len(self.action_history) + 1}: " + action
                splits = action.split(f"{sep} ")
                splits = [s.strip() for s in splits if s.strip() != ""]
                if len(splits) == 1:
                    return action.strip() + f" {self.prm_step_tag}"
                elif len(splits) > 1:
                    processed_action = f"{sep} " + splits[0] + f" {self.prm_step_tag}"
                    for i in range(1, len(splits)):
                        s = splits[i]
                        try:
                            colon_idx = s.index(":")
                            s = s[:(colon_idx + 1)] + " " + s[(colon_idx + 1):].strip()
                        except:
                            pass
                        processed_action += f"{sep} {s}" + f" {self.prm_step_tag}"
                    return processed_action
            else:
                sep = "Step "
                if self.add_step_prompt and not action.startswith(sep):
                    action = f"{sep}{len(self.action_history) + 1}: " + action
                splits = action.split("\nStep ")
                splits = [s.strip() for s in splits]
                if len(splits) == 1:
                    return splits[0] + f" {self.prm_step_tag}"
                elif len(splits) > 1:
                    processed_action = splits[0] + f" {self.prm_step_tag}"
                    for i in range(1, len(splits)):
                        s = splits[i]
                        try:
                            colon_idx = s.index(":")
                            s = s[:(colon_idx + 1)] + " " + s[(colon_idx + 1):].strip()
                        except:
                            pass
                        processed_action += f"Step {s}" + f" {self.prm_step_tag}"
                    return processed_action
        else:
            if self.double_line_break == 1:
                return action.strip() + "\n\n"
            elif self.double_line_break == 2:
                return action.strip() + "\n\n"
            if 'llama-3' in self.model_names[0].lower():  # TODO: Check llama
                sep = "## Step "
                # sep = "Step"
                if self.add_step_prompt and not action.startswith(sep):
                    action = f"{sep}{len(self.action_history) + 1}: " + action
                if action.endswith(sep):
                    action = action[:-len(sep)]
                if action.endswith("## Step"):
                    action = action[:-len("## Step")]
                if action.endswith(f"\nStep "):
                    action = action[:-len(f"\nStep ")]
                if action.endswith(f"\nStep"):
                    action = action[:-len(f"\nStep")]
                action = action.strip()
                if not action.endswith("\n\n"):
                    if action.endswith("\n"):
                        action += "\n"
                    else:
                        action += "\n\n"
            else:
                sep = "Step "
                if self.add_step_prompt and not action.startswith(sep):
                    action = f"{sep}{len(self.action_history) + 1}: " + action
                if action.endswith(f"\n{sep}"):
                    action = action[:-len(f"\n{sep}")]
                if action.endswith(f"\nStep"):
                    action = action[:-len(f"\nStep")]
                action = action.strip()
                if not action.endswith("\n\n"):
                    if action.endswith("\n"):
                        action += "\n"
                    else:
                        action += "\n\n"

        return action

    def _is_correct(self, completion):
        extracted_answer = extract_answer(completion)
        return judge_correct(self.math_problem["question"], self.math_problem["answer"], extracted_answer)

    def get_reward(self):
        """To implement based on learned reward model"""
        return 0
