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
    # 写入 straggler / eval 日志用的短名（与 output 目录命名一致）
    eval_log_policy_model: str = ""
    eval_log_reward_model: str = ""
    straggler_prune_enabled: bool = False
    straggler_length_ratio: float = 1.5
    straggler_min_tokens: int = 80
    straggler_prune_other_reward_gate: bool = False
    straggler_prune_other_reward_threshold: float = 0.0
    eval_seed: int = 0
    # n>1 时拆成多次 n=1（各带不同派生 seed），避免同请求多样本雷同后被去重成单分支
    split_lm_n_for_seeds: bool = True
    # 并行线程上限（HTTP 到同一 worker）；<=0 表示 min(n, 256)
    split_lm_parallel_workers: int = 32

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
            "eval_seed": config.eval_seed,
            "split_lm_n_for_seeds": config.split_lm_n_for_seeds,
            "split_lm_parallel_workers": config.split_lm_parallel_workers,
        },
        math_problems=[{
            "question": problem_inst["question"],
            "answer": problem_inst["extracted_groundtruth"] if "extracted_groundtruth" in problem_inst else task.extract_groundtruth(problem_inst["answer"]),
        }],
        llm_gen_fns=lm_calls,
        rm_call=rm_call,
        update_legal_action=False,
    )

    search_tree = SearchTree(
        cfg={
            "model_names": config.model_names,
            "direct_io": config.direct_io,
            "max_actions": config.tree_max_width,
            "eval_log_task_name": config.task_name,
            "eval_log_policy_model": config.eval_log_policy_model,
            "eval_log_reward_model": config.eval_log_reward_model,
            "eval_log_tree_max_depth": config.tree_max_depth,
            "eval_log_beam_size": config.beam_size,
            "straggler_prune_enabled": config.straggler_prune_enabled,
            "straggler_length_ratio": config.straggler_length_ratio,
            "straggler_min_tokens": config.straggler_min_tokens,
            "straggler_prune_other_reward_gate": config.straggler_prune_other_reward_gate,
            "straggler_prune_other_reward_threshold": config.straggler_prune_other_reward_threshold,
        }
    )
    traj_list = search_tree.beam_search(env, config.beam_size, config.tree_max_depth, rm_call)

    ### beam search start
    straggler_log = traj_list[-1].get("straggler_log") if traj_list else None
    return TreeSearchSolutionOutput(
        solutions=[t["text"] for t in traj_list],
        completion_tokens=[t["api_completion_tokens"] for t in traj_list],
        tree_completion_tokens=[t["tree_completion_tokens"] for t in traj_list],
        reward_history=[t["reward_history"] for t in traj_list],
        token_history=[t["token_history"] for t in traj_list],
        prob_history=[t["prob_history"] for t in traj_list],
        token_prob_history=[t["token_prob_history"] for t in traj_list],
        model_history=[t["model_history"] for t in traj_list],
        detailed_beam_search_logs=[t.get("detailed_beam_search_log") for t in traj_list],
        final_path_infos=[t.get("final_path_info") for t in traj_list],
        timing_infos=[t.get("timing_info") for t in traj_list],
        straggler_log=straggler_log,
    )
    ### beam search end
