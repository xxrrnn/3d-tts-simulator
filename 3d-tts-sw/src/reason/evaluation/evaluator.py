"""
This file is largely borrowed from OpenR (https://github.com/openreasoner/openr)
"""

import copy
import importlib
import jsonlines
import time
import traceback
from dataclasses import dataclass
from typing import Callable, Dict, List, Union

import numpy as np
import os
import ray

from envs import get_default_query_str_builder, get_env_datasets
from envs.base_env import INVALID_ANS
from reason.inference.lm_call import LanguageModelCallingFunction, LMCallingConfig, ConcatedLMGenResult
from reason.inference.rm_call import RewardModelCallingFunction
from reason.reranking.vote_utils import (
    MAJORITY_VOTE,
    PRM_MIN_MAX,
    PRM_MIN_VOTE,
    PRM_LAST_MAX,
    PRM_LAST_VOTE,
    PRM_AVG_MAX,
    PRM_AVG_VOTE,
    AGG_FN_MAP,
)
from utils import get_step_cnt, to_raw_string, load_jsonl


class Task:
    def __init__(self, task_name: str, is_few_shot: bool = False, model_names=[]):
        if task_name == "AMC23" or "AIME24":
            task_name = "MATH"
        self.task_name = task_name
        task_module = importlib.import_module(f"envs.{task_name}")
        if task_name == "MATH" or "rstar":
            self.extract_answer = task_module.extract_answer
            self.extract_groundtruth = task_module.extract_groundtruth
            self.judge_correct = task_module.judge_correct
        else:
            raise NotImplementedError(f"Task {task_name} is not supported")

        self._is_few_shot = is_few_shot
        self.model_names = model_names
        self.env_fn = task_module.Env

    def prompt_fn(self, problem_input: str):
        return get_default_query_str_builder(self.task_name)(problem_input, is_few_shot=self._is_few_shot, model_names=self.model_names)

    def test_ds(self, task_name):
        return get_env_datasets(task_name)[1]


CHOSEN_AGGR_METHODS = [
    MAJORITY_VOTE,
    PRM_MIN_MAX,
    PRM_MIN_VOTE,
    PRM_LAST_MAX,
    PRM_LAST_VOTE,
    PRM_AVG_MAX,
    PRM_AVG_VOTE,
]


def judge_ans(
    problem_str: str,
    extracted_groundtruth: str,
    extracted_answers: List[str],
    v_list: List[float],
    aggration_mode: str,
    judge_correct_fn,
    normalize=False,
):
    valid_ans_list, valid_v_list = [], []
    for i, ans in enumerate(extracted_answers):
        if ans != INVALID_ANS:
            valid_ans_list.append(ans)
            valid_v_list.append(v_list[i])
    if len(valid_ans_list) == 0:
        return 0

    if "orm" in aggration_mode and normalize:
        # score_normalization: this is only necessary for [-1, 1] values
        valid_v_list = np.array(valid_v_list)
        valid_v_list -= valid_v_list.min()
        valid_v_list /= valid_v_list.max() + 1e-3
        valid_v_list = valid_v_list.tolist()
    aggregated_ans = AGG_FN_MAP[aggration_mode](valid_ans_list, valid_v_list)

    return 1 if judge_correct_fn(problem_str, extracted_groundtruth, aggregated_ans) else 0


@dataclass
class SolutionOutput:
    solutions: List[str]
    completion_tokens: List[int]


@dataclass
class TreeSearchSolutionOutput(SolutionOutput):
    tree_completion_tokens: List[int]
    reward_history: List[float]
    token_history: List[int]
    prob_history: List[float]
    model_history: List[str]


class MathEvaluator:

    def __init__(
        self, task: Union[str, Task], lm_calls: List[LanguageModelCallingFunction], rm_call: RewardModelCallingFunction, direct_io=False
    ):
        if isinstance(task, str):
            self._task = Task(task_name=task)
        else:
            assert isinstance(task, Task)
            self._task = task
        self.lm_calls = lm_calls
        self.rm_call = rm_call
        self.direct_io = direct_io

    def evaluate_problem(self, problem_inst: Dict[str, str], solver_fn: Callable) -> List[str]:
        solution: SolutionOutput = solver_fn(problem_inst, self.lm_calls, self.rm_call)
        reward_history = solution.reward_history
        token_history = solution.token_history
        prob_history = solution.prob_history
        if isinstance(solution, TreeSearchSolutionOutput):
            model_history = [[model.split('/')[-1] for model in traj] for traj in solution.model_history]
        else:
            model_history = [[]] * len(solution.solutions)
        result, output = self.analyze_output(problem_inst, solution.solutions, reward_history, token_history, prob_history, model_history)
        total_completion_token = 0
        for i, o in enumerate(output):
            o["completion_tokens"] = solution.completion_tokens[i]
            if isinstance(solution, TreeSearchSolutionOutput):
                o["tree_completion_tokens"] = solution.tree_completion_tokens[i]
            # We define the completion_tokens as the tokens consumed between two generated answers, therefore we need to take sum here.
            total_completion_token += solution.completion_tokens[i]
        result["total_completion_tokens"] = total_completion_token
        return problem_inst, result, output

    def analyze_output(
        self, problem_inst: Dict[str, str], gen_answers: List[str], reward_history, token_history, prob_history, model_history=None
    ):
        if 'extracted_groundtruth' in problem_inst:
            extracted_groundtruth = problem_inst['extracted_groundtruth']
        else:
            extracted_groundtruth = self._task.extract_groundtruth(problem_inst["answer"])

        if self.direct_io == 1:  # BoN
            input_list = [(problem_inst["question"], txt) for txt in gen_answers]
            for i in range(2):
                try:
                    value_list = self.rm_call(input_list)
                    break
                except Exception as e:
                    import traceback
                    print(f"Error in computing reward: {e}")
                    traceback.print_exc()
                    value_list = [[0.0]] * len(gen_answers)
            reward_history = value_list
        else:
            value_list = reward_history

        extracted_answers = [self._task.extract_answer(txt) for txt in gen_answers]
        output_list = [
            {
                "path_idx": i, "text": txt, "value": v, "extracted_answer": extracted_answer, "reward_history": reward,
                "token_history": token, "prob_history": prob, "model_history": model
            }
            for i, (txt, v, extracted_answer, reward, token, prob, model) in
            enumerate(zip(gen_answers, value_list, extracted_answers, reward_history, token_history, prob_history, model_history))
        ]
        res = {
            agg_method:
                judge_ans(
                    problem_inst["question"],
                    extracted_groundtruth,
                    extracted_answers,
                    value_list,
                    agg_method,
                    self._task.judge_correct,
                ) for agg_method in (CHOSEN_AGGR_METHODS if len(gen_answers) > 1 else [MAJORITY_VOTE])
        }
        return res, output_list


@ray.remote
class RemoteMathEvaluator(MathEvaluator):
    def __init__(
        self, task: str, lm_calls: List[LanguageModelCallingFunction], rm_call: RewardModelCallingFunction, direct_io=False
    ):
        super().__init__(task, lm_calls, rm_call, direct_io)
