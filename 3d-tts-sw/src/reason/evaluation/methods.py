"""
This file is largely borrowed from OpenR (https://github.com/openreasoner/openr)
"""

from dataclasses import dataclass
import functools
from typing import Dict, List
from reason.inference.lm_call import LMCallingConfig, LanguageModelCallingFunction
from reason.inference.rm_call import RewardModelCallingFunction
from reason.evaluation.evaluator import SolutionOutput, Task, TreeSearchSolutionOutput
from reason.guided_search.tree import SearchTree


@dataclass
class BasicConfig:
    task_name: str


@dataclass
class TreeSearchConfig(BasicConfig):
    tree_max_width: int = 10
    tree_max_depth: int = 10
    init_critic_value: bool = True

    model_names: List[str] = None
    llm_names: List[str] = None
    is_few_shot: bool = False
    add_step_prompt: bool = False
    cot_prompt: str = ""
    stop_str: List[str] = list
    sep: List[str] = list
    direct_io: int = 0
    double_line_break: bool = False

    def __post_init__(self):
        assert self.tree_max_width > 0, "Tree width must be greater than 0"
        if self.stop_str is not None:
            assert self.tree_max_depth > 0, "Tree depth must be greater than 0"


@dataclass
class BeamSearchConfig(TreeSearchConfig):
    beam_size: int = 1

    def __post_init__(self):
        super().__post_init__()
        assert self.beam_size > 0, "Beam size must be greater than 0"
        assert self.init_critic_value, "BeamSearch should set init_critic_value to True"


def beam_search(
    config: BeamSearchConfig,
    gen_config: LMCallingConfig,
    problem_inst: Dict[str, str],
    lm_calls: List[LanguageModelCallingFunction],
    rm_call: RewardModelCallingFunction,
) -> SolutionOutput:
    task = Task(task_name=config.task_name)
    env = task.env_fn(
        config={
            "max_actions": config.tree_max_width,
            "max_length": config.tree_max_depth,
            "beam_size": config.beam_size,
            "cot_prompt": config.cot_prompt,
            "stop_str": config.stop_str,
            "sep": config.sep,
            "generation_config": {
                "max_new_tokens": gen_config.max_new_tokens,
                "temperature": gen_config.temperature,
                "top_p": gen_config.top_p,
                "top_k": gen_config.top_k,
            },
            "is_few_shot": config.is_few_shot,
            "add_step_prompt": config.add_step_prompt,
            "direct_io": config.direct_io,
            "double_line_break": config.double_line_break,
            "model_names": config.model_names,
        },
        math_problems=[{
            "question": problem_inst["question"],
            "answer": problem_inst["extracted_groundtruth"] if "extracted_groundtruth" in problem_inst else task.extract_groundtruth(problem_inst["answer"]),
        }],
        llm_gen_fns=lm_calls,
        rm_call=rm_call,
        update_legal_action=False,
    )

    search_tree = SearchTree(cfg={"model_names": config.model_names, "direct_io": config.direct_io, "max_actions": config.tree_max_width})
    traj_list = search_tree.beam_search(env, config.beam_size, config.tree_max_depth, rm_call)

    return TreeSearchSolutionOutput(
        solutions=[t["text"] for t in traj_list],
        completion_tokens=[t["api_completion_tokens"] for t in traj_list],
        tree_completion_tokens=[t["tree_completion_tokens"] for t in traj_list],
        reward_history=[t["reward_history"] for t in traj_list],
        token_history=[t["token_history"] for t in traj_list],
        prob_history=[t["prob_history"] for t in traj_list],
        model_history=[t["model_history"] for t in traj_list],
    )
