"""
The Node and MCTS class for AlphaZero.
This file is largely borrowed from OpenR (https://github.com/openreasoner/openr)
"""

import copy
import json
import math
import os
import traceback
import time

import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional, Tuple, Union, Callable, Type
from utils import print_rank_0, print_with_rank, get_model_name
from envs.base_env import CoTEnv
import heapq
from loguru import logger


class Node(object):
    """
    Overview:
        The node base class for tree_search.
    """

    def __init__(self, parent: "Node" = None, prior_p: float = 1.0, initial_value: float = 0.0, parent_value: float = 0.0) -> None:
        self._parent = parent
        self._children = {}
        self._visit_count = 0
        self._value_sum = 0
        self.prior_p = prior_p
        self.prior_p_ori = prior_p

        self._initial_value = initial_value
        self._parent_value = parent_value
        self._terminated = False

    def __lt__(self, other):
        return self._initial_value < other._initial_value

    @property
    def terminated(self):
        return self._terminated

    def set_as_terminate_node(self):
        self._terminated = True

    @property
    def value(self) -> float:
        """
        Overview:
            The value of the current node.
        Returns:
            - output (:obj:`Int`): Current value, used to compute ucb score.
        """
        if self._visit_count == 0:
            # if not visited, return the initial value
            return self._initial_value
        return self._value_sum / self._visit_count

    def update(self, value: float) -> None:
        """
        Overview:
            Update the current node information, such as visit_count and value_sum.
        Arguments:
            - value (:obj:`Int`): The value of the node.
        """
        self._visit_count += 1
        self._value_sum += value

    def update_recursive(self, leaf_value: float, mcts_mode: str) -> None:
        """
        Overview:
            Update node information recursively.
        Arguments:
            - leaf_value (:obj:`Int`): The value of the node.
        """
        if mcts_mode == "self_play_mode":
            self.update(leaf_value)
            if self.is_root():
                return
            self._parent.update_recursive(-leaf_value, mcts_mode)
        if mcts_mode == "play_with_bot_mode":
            self.update(leaf_value)
            if self.is_root():
                return
            self._parent.update_recursive(leaf_value, mcts_mode)

    def is_leaf(self) -> bool:
        """
        Overview:
            Check if the current node is a leaf node or not.
        Returns:
            - output (:obj:`Dict`): Dict type children node.
        """
        return self._children == {}

    def is_root(self) -> bool:
        """
        Overview:
            Check if the current node is a root node or not.
        Returns:
            - output (:obj:`Bool`): Whether it is the parent node.
        """
        return self._parent is None

    @property
    def parent(self) -> None:
        return self._parent

    @property
    def children(self) -> None:
        return self._children

    @property
    def visit_count(self) -> None:
        return self._visit_count

    def get_info(self):
        # return [
        #     "visit_cnt: {}, value: {:.6f}, prior: {:.6f}".format(
        #         self.visit_count, self.value, self.prior_p)
        # ]
        return {
            "visit_cnt": self.visit_count,
            "value": self.value,
            "prior_p": float(self.prior_p_ori),
            "initial_value": self._initial_value,
            "terminated": self.terminated,
        }

    def clear(self):
        self._visit_count = 0
        self._value_sum = 0
        self.prior_p = self.prior_p_ori

    def to_json(self):
        childrens = {}
        for name, child_node in self.children.items():
            childrens[name] = child_node.to_json()

        rets = {"children": childrens, "info": self.get_info()}
        return rets

    def __str__(self) -> str:
        if self.is_root():
            return "root"
        else:
            return "child: value: {:.3f}, prior: {:.3f}".format(self.last_action, self.value, self.prior_p)


class LanguageNode(Node):
    text_state: Optional[str] = None
    last_action: Optional[str] = None
    num_generated_token: Optional[int] = None

    def __init__(
        self,
        parent: Node = None,
        prior_p: float = 1.0,
        prm_value: Optional[float] = None,
        text_state: Optional[str] = None,
        last_action: Optional[str] = None,
        initial_value: float = 0.0,
        parent_value: float = 0.0,
        num_generated_token: Optional[int] = None,
        model_name: str = "",
        token_prob_list: Optional[List[float]] = None,
        prm_step_scores: Optional[List[float]] = None,
        token_topk_logprobs_list: Optional[List[dict]] = None,
    ) -> None:
        super().__init__(parent, prior_p, initial_value, parent_value)
        self.text_state = text_state
        self.last_action = last_action
        self.prm_value = prm_value
        self.prm_step_scores = prm_step_scores

        self.num_generated_token = num_generated_token
        self.has_collected_token_num = False

        self.model_name = model_name
        self.token_prob_list = token_prob_list if token_prob_list is not None else []
        self.token_topk_logprobs_list = token_topk_logprobs_list if token_topk_logprobs_list is not None else []

    def get_path(self):
        ans = []
        node = self
        while not node.is_root():
            ans.append(node.last_action)
            node = node.parent
        return "\n".join(reversed(ans))

    def get_info(self):
        info_dict = super().get_info()
        if not self.is_root():
            info_dict["last_action"] = self.last_action
            info_dict["prm_value"] = self.prm_value
            info_dict["prm_step_scores"] = self.prm_step_scores
            # 添加token概率信息用于计算logit entropy
            if hasattr(self, 'token_topk_logprobs_list'):
                info_dict["token_topk_logprobs_list"] = self.token_topk_logprobs_list
        else:
            info_dict["text_state"] = self.text_state
        return info_dict

    def __str__(self):
        if self.is_root():
            return "root: {}".format(self.text_state)
        else:
            return "action: {}, value: {:.3f}, prior: {:.3f}".format(self.last_action, self.value, self.prior_p)


def get_root(node: Node):
    while not node.is_root():
        node = node.parent
    return node


class SearchTree:
    """
    Overview:
        MCTS search process.
    """

    def __init__(self, cfg) -> None:
        self._cfg = cfg

        self._num_simulations = self._cfg.get("num_simulations", 20)

        # UCB formula
        self._pb_c_base = self._cfg.get("pb_c_base", 19652)  # 19652
        self._pb_c_init = self._cfg.get("pb_c_init", 1.25)  # 1.25

        # Root prior exploration noise.
        self._root_dirichlet_alpha = self._cfg.get("root_dirichlet_alpha", 0.3)  # 0.3  # for chess, 0.03 for Go and 0.15 for shogi.
        self._root_noise_weight = self._cfg.get("root_noise_weight", 0.25)  # 0.25

        self.root = None

        self.answers = set()
        self.wrong_answers = set()
        self.visited_paths = None

        self.no_terminal_reward = self._cfg.get("no_terminal_reward", True)
        self.mask_non_terminal_node_value = self._cfg.get("mask_non_terminal_node_value", False)

        self._init_critic_value = self._cfg.get("init_critic_value", True)

        self._completion_tokens = 0

        self.model_names = self._cfg.get("model_names", [])
        self.direct_io = self._cfg.get("direct_io", 0)
        self.max_actions = self._cfg.get("max_actions", 0)

        self._eval_log_task_name = self._cfg.get("eval_log_task_name", "")
        self._eval_log_policy_model = self._cfg.get("eval_log_policy_model", "")
        self._eval_log_reward_model = self._cfg.get("eval_log_reward_model", "")
        self._eval_log_tree_max_depth = int(self._cfg.get("eval_log_tree_max_depth", 0))
        self._eval_log_beam_size = int(self._cfg.get("eval_log_beam_size", 0))
        if not self._eval_log_policy_model and self.model_names:
            self._eval_log_policy_model = get_model_name(self.model_names[0])

        # 0405：增加straggler判定相关的参数 
        # straggler: unusually long sibling branch gets PRM aggregate reward zeroed (beam / MCTS expansion).
        self._straggler_prune_enabled = self._cfg.get("straggler_prune_enabled", False) #是否启用straggler检测 
        self._straggler_length_ratio = float(self._cfg.get("straggler_length_ratio", 1.5)) #straggler判定的倍率值
        self._straggler_min_tokens = int(self._cfg.get("straggler_min_tokens", 80))  # straggler的长度最低阈值
        self._straggler_prune_events: List[Dict[str, Any]] = []

    def _straggler_run_context_dict(self) -> Dict[str, Any]:
        return {
            "task": self._eval_log_task_name,
            "policy": self._eval_log_policy_model,
            "reward_model": self._eval_log_reward_model,
            "tree_max_width": self.max_actions,
            "tree_max_depth": self._eval_log_tree_max_depth,
            "beam_size": self._eval_log_beam_size,
        }

    def straggler_log_payload(self) -> Dict[str, Any]:
        """写入 record jsonl 的 straggler 配置 + 每次剪枝事件（与 beam 轨迹一并落盘）。"""
        return {
            "prune_enabled": self._straggler_prune_enabled,
            "length_ratio": self._straggler_length_ratio,
            "min_tokens": self._straggler_min_tokens,
            "init_critic_value": self._init_critic_value,
            "context": self._straggler_run_context_dict(),
            "prune_events": list(self._straggler_prune_events),
        }

    @property
    def num_generated_token(self):
        return self._completion_tokens

    def clear_node(self, node):
        assert node is not None
        node.clear()
        for child in node.children.values():
            self.clear_node(child)

    def beam_search(
        self,
        simulate_env: CoTEnv,
        beam_size: int,
        max_step: int,
        reward_model_fn: Optional[Callable] = None,
    ) -> List[Dict]:
        """Beam Search implementation
        Args:
            simulate_env: The environment to simulate the search.
            beam_size: beam_size
            max_step: The maximum number of steps to search.
            reward_model_fn: The reward model function to evaluate the state.
        """
        ### beam search start
        self._straggler_prune_events = []
        # 检查是否启用详细日志记录
        enable_detailed_log = os.environ.get('BEAM_SEARCH_DETAILED_LOG', '0') == '1'
        
        if max_step == 1:
            assert self.direct_io
        search_start_time = time.perf_counter()
        api_call_completion_tokens = 0
        _, info = simulate_env.reset(update_legal_action=True)
        api_call_completion_tokens += info["api_completion_token"]
        if self.root is None:
            root = LanguageNode(text_state=simulate_env.get_state(model_name='raw'))
            self._expand_leaf_node(root, simulate_env, reward_model_fn)
            self.root = root

        end_nodes, top_k_nodes = [], [(-self.root._initial_value, -self.root._initial_value, -self.root._parent_value, self.root, simulate_env.copy())]
        k = copy.deepcopy(beam_size)
        
        # 初始化搜索过程统计信息和详细记录
        search_step_stats = []
        detailed_beam_logs = None
        step_branch_token_probs = []
        
        if enable_detailed_log:
            detailed_beam_logs = {
                "beam_size": beam_size,
                "max_step": max_step,
                "initial_root_value": self.root._initial_value,
                "step_details": []
            }

        for i in range(max_step + 1):
            step_start_time = time.perf_counter()
            step_detail = None
            current_step_branch_token_probs = []
            if enable_detailed_log:
                step_detail = {
                    "step": i,
                    "current_nodes": [],
                    "selection_process": {
                        "available_branches": 0,
                        "selected_branches": [],
                        "terminated_count": len(end_nodes)
                    },
                    "expansion_results": []
                }
            
            step_info = {
                "step": i,
                "nodes_to_expand": len(top_k_nodes),
                "terminated_nodes": len(end_nodes),
                "beam_width_used": min(k, len(top_k_nodes))
            }
            
            cur_nodes_to_search = top_k_nodes
            top_k_nodes = []
            
            # 记录当前搜索的节点信息
            if enable_detailed_log and step_detail:
                for idx, (cur_neg_q_plus_a, cur_neg_v, cur_neg_parent_v, cur_node, cur_env) in enumerate(cur_nodes_to_search):
                    node_info = {
                        "node_index": idx,
                        "q_value": -cur_neg_q_plus_a,
                        "value": -cur_neg_v,
                        "parent_value": -cur_neg_parent_v,
                        "is_terminated": cur_node.terminated,
                        "path_content": cur_node.get_path() if hasattr(cur_node, 'get_path') else str(cur_node.last_action),
                        "current_state": cur_env.answer if hasattr(cur_env, 'answer') else ""
                    }
                    step_detail["current_nodes"].append(node_info)
            
            for cur_neg_q_plus_a, cur_neg_v, cur_neg_parent_v, cur_node, cur_env in cur_nodes_to_search:
                if cur_node.terminated:
                    end_nodes.append((cur_neg_q_plus_a, cur_neg_v, cur_neg_parent_v, cur_node, cur_env))
                    if len(end_nodes) == beam_size:
                        break
                elif len(end_nodes) < beam_size:
                    # select at most topk children add push to heap
                    assert (len(cur_node.children) > 0), "in beam search you should expand this non-terminal node at first."

                    if enable_detailed_log and step_detail:
                        step_detail["selection_process"]["available_branches"] += len(cur_node.children)
                    
                    branch_candidates = []
                    
                    if self.direct_io:
                        ps = {child_idx: copy.deepcopy(child.prior_p) for child_idx, child in cur_node.children.items()}
                        num_tokens = {child_idx: copy.deepcopy(child.num_generated_token) for child_idx, child in cur_node.children.items()}
                        normalized_ps = {child_idx: p ** (1 / max(1, num_tokens[child_idx])) for child_idx, p in ps.items()}
                        values = {child_idx: copy.deepcopy(child._initial_value) for child_idx, child in cur_node.children.items()}
                        parent_values = {child_idx: copy.deepcopy(child._parent_value) for child_idx, child in cur_node.children.items()}
                        q_plus_alpha_a = values

                        k = beam_size - len(end_nodes)
                        top_k_children = sorted(
                            [(child_idx, child, q_plus_alpha_a[child_idx], values[child_idx], parent_values[child_idx]) for child_idx, child in
                             cur_node.children.items()],
                            key=lambda x: x[2], reverse=True,
                        )[:k]

                        if enable_detailed_log:
                            # 按 step 收集每个分支的 token-level probs
                            for _, child in cur_node.children.items():
                                current_step_branch_token_probs.append(child.token_prob_list if child.token_prob_list is not None else [])
                        
                        # 记录所有候选分支信息，保持原始顺序
                        if enable_detailed_log:
                            # 先按奖励排序找出被选中的分支
                            sorted_candidates = sorted(
                                [(child_idx, child, q_plus_alpha_a[child_idx], values[child_idx], parent_values[child_idx]) for child_idx, child in cur_node.children.items()],
                                key=lambda x: x[2], reverse=True
                            )
                            selected_indices = set(c_idx for c_idx, _, _, _, _ in sorted_candidates[:k])
                            
                            # 按照原始顺序记录所有分支
                            for child_idx, child in cur_node.children.items():
                                c_q_plus_a = q_plus_alpha_a[child_idx]
                                c_value = values[child_idx]
                                c_parent_value = parent_values[child_idx]
                                branch_info = {
                                    "child_index": child_idx,
                                    "reward_score": c_value,
                                    "prm_step_scores": child.prm_step_scores,
                                    "q_plus_a": c_q_plus_a,
                                    "parent_value": c_parent_value,
                                    "prior_prob": ps[child_idx],
                                    "num_tokens": num_tokens[child_idx],
                                    "token_probs": child.token_prob_list,
                                    "token_topk_logprobs": child.token_topk_logprobs_list,
                                    "selected": child_idx in selected_indices,
                                    "branch_content": child.last_action if child.last_action else "",
                                    "full_path": child.get_path() if hasattr(child, 'get_path') else ""
                                }
                                branch_candidates.append(branch_info)
                        
                        for c_idx, c_node, c_q_plus_a, c_value, c_parent_value in top_k_children:
                            new_env = cur_env.copy()
                            heapq.heappush(top_k_nodes, (-c_q_plus_a, -c_value, -c_parent_value, c_node, new_env))
                    else:
                        ps = {action: copy.deepcopy(child.prior_p) for action, child in cur_node.children.items()}
                        num_tokens = {action: copy.deepcopy(child.num_generated_token) for action, child in cur_node.children.items()}
                        normalized_ps = {action: p ** (1 / max(1, num_tokens[action])) for action, p in ps.items()}
                        values = {action: copy.deepcopy(child._initial_value) for action, child in cur_node.children.items()}
                        parent_values = {action: copy.deepcopy(child._parent_value) for action, child in cur_node.children.items()}
                        q_plus_alpha_a = values

                        k = beam_size - len(end_nodes)
                        top_k_children = sorted(
                            [(action, child, q_plus_alpha_a[action], values[action], parent_values[action]) for action, child in cur_node.children.items()],
                            key=lambda x: x[2], reverse=True,
                        )[:k]

                        if enable_detailed_log:
                            # 按 step 收集每个分支的 token-level probs
                            for _, child in cur_node.children.items():
                                current_step_branch_token_probs.append(child.token_prob_list if child.token_prob_list is not None else [])
                        
                        # 记录所有候选分支信息，保持原始顺序
                        if enable_detailed_log:
                            # 先按奖励排序找出被选中的分支
                            sorted_candidates = sorted(
                                [(action, child, q_plus_alpha_a[action], values[action], parent_values[action]) for action, child in cur_node.children.items()],
                                key=lambda x: x[2], reverse=True
                            )
                            selected_actions = set(c_act for c_act, _, _, _, _ in sorted_candidates[:k])
                            
                            # 按照原始顺序记录所有分支
                            for action, child in cur_node.children.items():
                                c_q_plus_a = q_plus_alpha_a[action]
                                c_value = values[action] 
                                c_parent_value = parent_values[action]
                                branch_info = {
                                    "action": action,
                                    "reward_score": c_value,
                                    "prm_step_scores": child.prm_step_scores,
                                    "q_plus_a": c_q_plus_a,
                                    "parent_value": c_parent_value,
                                    "prior_prob": ps[action],
                                    "num_tokens": num_tokens[action],
                                    "token_probs": child.token_prob_list,
                                    "token_topk_logprobs": child.token_topk_logprobs_list,
                                    "selected": action in selected_actions,
                                    "branch_content": str(action),
                                    "full_path": child.get_path() if hasattr(child, 'get_path') else str(action)
                                }
                                branch_candidates.append(branch_info)
                        
                        for c_act, c_node, c_q_plus_a, c_value, c_parent_value in top_k_children:
                            new_env = cur_env.copy()
                            heapq.heappush(top_k_nodes, (-c_q_plus_a, -c_value, -c_parent_value, c_node, new_env))
                    
                    if enable_detailed_log and step_detail:
                        step_detail["selection_process"]["selected_branches"].extend(branch_candidates)
            
            if enable_detailed_log:
                step_branch_token_probs.append(current_step_branch_token_probs)
                    
            top_k_nodes = heapq.nsmallest(k, top_k_nodes)  # nsmallest since we negate the value

            # expand selected nodes and record expansion results
            for node_idx, (q_plus_a, value, parent_value, node, new_env) in enumerate(top_k_nodes):
                expansion_result = None
                if enable_detailed_log:
                    expansion_result = {
                        "node_index": node_idx,
                        "pre_expansion_value": -value,
                        "pre_expansion_state": new_env.answer if hasattr(new_env, 'answer') else "",
                        "action_taken": node.last_action if node.last_action else ""
                    }
                
                _, _, terminated, truncated, info = new_env.step(
                    node.last_action, update_legal_action=self.direct_io == 0, model_name=node.model_name,
                    reward=node._initial_value, num_token=node.num_generated_token, prob=node.prior_p,
                    token_probs=node.token_prob_list,
                )
                api_call_completion_tokens += info["api_completion_token"]
                
                if enable_detailed_log and expansion_result:
                    expansion_result.update({
                        "post_expansion_state": new_env.answer if hasattr(new_env, 'answer') else "",
                        "terminated": terminated,
                        "truncated": truncated,
                        "api_completion_tokens": info["api_completion_token"]
                    })
                
                if terminated or truncated:
                    node.set_as_terminate_node()
                    if enable_detailed_log and expansion_result:
                        expansion_result["final_status"] = "terminated"
                else:
                    self._expand_leaf_node(node, new_env, reward_model_fn)
                    if enable_detailed_log and expansion_result:
                        expansion_result["final_status"] = "expanded"
                        expansion_result["num_new_children"] = len(node.children)
                
                if enable_detailed_log and expansion_result and step_detail:
                    step_detail["expansion_results"].append(expansion_result)
            
            # 记录这一步的统计信息
            step_elapsed_sec = time.perf_counter() - step_start_time
            step_info["nodes_expanded"] = len(top_k_nodes)
            step_info["final_terminated_nodes"] = len(end_nodes)
            step_info["elapsed_sec"] = step_elapsed_sec
            search_step_stats.append(step_info)
            
            # 添加到详细日志
            if enable_detailed_log and step_detail and detailed_beam_logs:
                step_detail["summary"] = step_info
                detailed_beam_logs["step_details"].append(step_detail)

            if len(end_nodes) == beam_size:
                break
        ### beam search end

        ### beam search start
        traj_list = []
        # 添加搜索过程的统计信息
        total_search_elapsed_sec = time.perf_counter() - search_start_time
        per_step_elapsed_sec = [s.get("elapsed_sec", 0.0) for s in search_step_stats]
        search_process_stats = {
            "total_search_steps": max_step + 1,
            "actual_steps_used": 0,
            "nodes_expanded_per_step": [],
            "beam_width_per_step": [],
            "terminated_early": len(end_nodes) == beam_size and max_step > 0,
            "step_by_step_details": search_step_stats,
            "total_elapsed_sec": total_search_elapsed_sec,
            "per_step_elapsed_sec": per_step_elapsed_sec,
        }
        
        # 汇总详细日志信息（仅在启用时）
        if enable_detailed_log and detailed_beam_logs:
            detailed_beam_logs["search_summary"] = {
                "total_end_nodes": len(end_nodes),
                "actual_beam_size_used": len(end_nodes),
                "total_api_tokens": api_call_completion_tokens,
                "total_tree_tokens": self._completion_tokens
            }
        
        for i, (neg_e_q_plus_a, neg_e_v, neg_e_parent_v, e_node, e_env) in enumerate(end_nodes):
            # 计算路径深度
            path_depth = len(e_env.reward_history)
            search_process_stats["actual_steps_used"] = max(search_process_stats["actual_steps_used"], path_depth)
            
            # 为每个路径添加最终节点的完整信息（仅在启用时）
            final_path_info = None
            if enable_detailed_log:
                final_path_info = {
                    "final_node_value": -neg_e_v,
                    "final_q_plus_a": -neg_e_q_plus_a,
                    "final_parent_value": -neg_e_parent_v,
                    "is_terminated": e_node.terminated,
                    "final_content": e_env.answer if hasattr(e_env, 'answer') else "",
                    "path_trace": e_node.get_path() if hasattr(e_node, 'get_path') else ""
                }
            
            traj_dict = {
                "path_idx": i,
                "text": e_env.answer,
                "value": -neg_e_v,
                "parent_value": -neg_e_parent_v,
                "q_plus_a": -neg_e_q_plus_a,
                "api_completion_tokens": 0,
                "tree_completion_tokens": 0,
                "reward_history": e_env.reward_history,
                "token_history": e_env.token_history,
                "prob_history": e_env.prob_history,
                "token_prob_history": e_env.token_prob_history,
                "model_history": e_env.model_history,
                "path_depth": path_depth,
                "path_efficiency": sum(e_env.reward_history) / len(e_env.reward_history) if e_env.reward_history else 0,
                "timing_info": {
                    "total_elapsed_sec": total_search_elapsed_sec,
                    "per_step_elapsed_sec": per_step_elapsed_sec,
                    "step_count": len(per_step_elapsed_sec),
                },
                # num_generated_token is hard to compute for each single answer
            }
            
            # 只在启用详细日志时添加这些字段
            if enable_detailed_log:
                if final_path_info:
                    traj_dict["final_path_info"] = final_path_info
                traj_dict["step_branch_token_probs"] = step_branch_token_probs
            
            traj_list.append(traj_dict)
        
        # 添加搜索过程统计信息到最后一个路径中
        if traj_list:
            traj_list[-1]["tree_completion_tokens"] = self._completion_tokens
            traj_list[-1]["api_completion_tokens"] = api_call_completion_tokens
            traj_list[-1]["search_process_stats"] = search_process_stats
            # 只在启用详细日志时添加详细日志
            if enable_detailed_log and detailed_beam_logs:
                traj_list[-1]["detailed_beam_search_log"] = detailed_beam_logs
            traj_list[-1]["straggler_log"] = self.straggler_log_payload()

        return traj_list
        ### beam search end

    def _select_child(self, node: LanguageNode, simulate_env: CoTEnv) -> Tuple[Union[int, float], Node]:
        """
        Overview:
            Select the child with the highest UCB score.
        Arguments:
            - node (:obj:`Class Node`): Current node.
        Returns:
            - action (:obj:`Int`): choose the action with the highest ucb score.
            - child (:obj:`Node`): the child node reached by executing the action with the highest ucb score.
        """

        action = None
        child = None
        best_score = -9999999

        for action_tmp, child_tmp in node.children.items():
            ucb_score = self._ucb_score(node, child_tmp)
            score = ucb_score
            if score > best_score:
                best_score = score
                action = action_tmp
                child = child_tmp

        if child is None:
            child = node  # child==None, node is leaf node in play_with_bot_mode.

        return action, child

    def _select_by_prior(self, node: Node, simulate_env: CoTEnv):
        data_tmp = [(x_action, x_node.prior_p) for x_action, x_node in node.children.items()]
        action_list, prior_list = list(zip(*data_tmp))
        chosen_action = np.random.choice(action_list, p=np.array(prior_list))
        chosen_node = node.children[chosen_action]

        return chosen_action, chosen_node

    def _expand_leaf_node(
        self,
        node: Node,
        simulate_env: CoTEnv,
        rm_call: Optional[Callable] = None,
    ) -> float:
        """
        Overview:
            expand the node with the rm_call.
        Arguments:
            - node (:obj:`Class Node`): current node when performing mcts search.
            - simulate_env (:obj:`Class BaseGameEnv`): the class of simulate env.
            - rm_call (:obj:`Function`): the Callable to compute the state value.
        Returns:
            - leaf_value (:obj:`Bool`): the leaf node's value.
        """
        """
        action_probs_dict, leaf_value = rm_call(simulate_env)
        for action, prior_p in action_probs_dict.items():
            if action in simulate_env.legal_actions:
                node.children[action] = Node(parent=node, prior_p=prior_p)
        """

        text_state = simulate_env.get_state(model_name='raw')
        if not self._init_critic_value:
            leaf_value = rm_call(text_state)
        else:
            leaf_value = node._initial_value
            assert len(simulate_env.legal_actions) > 0
            if self.direct_io:
                prms = [[0.0] for _ in simulate_env.legal_actions]
            else:
                prm_inputs = [(simulate_env.question, simulate_env.answer + x["action"]) for x in simulate_env.legal_actions]
                for i in range(2):
                    try:
                        prms = rm_call(prm_inputs)
                        break
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        if i == 1:  # 最后一次重试失败，使用默认值
                            prms = [[0.0] for _ in simulate_env.legal_actions]
            child_values = []
            prm_scores_per_action: List[Optional[List[float]]] = []
            for act, rs in zip(simulate_env.legal_actions, prms):
                if len(simulate_env.action_history) + 1 != len(rs):
                    logger.warning(f"PRM value length not match with action history. len(prm)={len(rs)}, "
                                   f"len(action_history)={len(simulate_env.action_history)}\ns:\n{text_state}\na:\n{act}\nrs:{rs}")
                    try:
                        prm = rm_call([(simulate_env.question, simulate_env.answer + x["action"]) for x in [act]], verbose=True, legal_action=[act])
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                    child_values.append(0.0)
                    prm_scores_per_action.append([float(x) for x in rs] if rs is not None else [])
                elif len(rs) == 0:
                    logger.warning(f"Empty PRM value for: \nState: \n{text_state} \naction: \n{act}, will be set to 0.0")
                    child_values.append(0.0)
                    prm_scores_per_action.append([])
                else:
                    # prm-last
                    child_values.append(rs[-1])  # PRM get last r as single reward, [0.9783847332000732, 0.9621075391769409]
                    prm_scores_per_action.append([float(x) for x in rs])
                    # # prm-min
                    # child_values.append(min(rs))
                    # # prob-prm
                    # child_values.append(act['prob'])
            
            #=================== 0405： 增加straggler判定，将其reward分数设置为0=============
            if self._straggler_prune_enabled and len(simulate_env.legal_actions) >= 2: #只有一个分支则不用考虑
                token_lens = [int(x["num_token"]) for x in simulate_env.legal_actions]
                n = len(token_lens)
                for si in range(n):
                    li = token_lens[si]
                    if li <= self._straggler_min_tokens:
                        continue
                    max_other = max(token_lens[j] for j in range(n) if j != si)
                    if max_other <= 0:
                        continue
                    if li > self._straggler_length_ratio * max_other:
                        prev_r = child_values[si]
                        child_values[si] = 0.0
                        ev: Dict[str, Any] = {
                            "branch_idx": si,
                            "num_token": li,
                            "max_other_siblings": max_other,
                            "ratio_threshold": self._straggler_length_ratio,
                            "min_tokens_threshold": self._straggler_min_tokens,
                            "prm_reward_before": float(prev_r) if prev_r is not None else None,
                            "prm_reward_after": 0.0,
                            "action": "beam_value_set_to_zero",
                        }
                        ev.update(self._straggler_run_context_dict())
                        self._straggler_prune_events.append(ev)
            #==========================================================================

        assert len(node.children) == 0

        # 为每个legal_actions建子节点
        for i, action_dict in enumerate(simulate_env.legal_actions):
            action, prob = action_dict["action"], action_dict["prob"]
            model_name = action_dict["model_name"]

            if self._init_critic_value:
                child_value = child_values[i]
            else:
                # XXX(ziyu): consider turn off this branch, i.e. always assume
                #  `self._init_critic=True`, since with LLM
                child_value = 0.0

            step_prm = prm_scores_per_action[i] if self._init_critic_value else None

            if self.direct_io:
                node.children[i] = LanguageNode(
                    parent=node,
                    prior_p=prob,
                    # prm_value=prm_value,
                    text_state=text_state,
                    last_action=action,
                    initial_value=child_value,
                    parent_value=leaf_value,
                    num_generated_token=action_dict["num_token"],
                    model_name=model_name,
                    token_prob_list=action_dict.get("token_probs", []),
                    prm_step_scores=step_prm,
                    token_topk_logprobs_list=action_dict.get("token_topk_logprobs", []),
                )
            else:
                node.children[action] = LanguageNode(
                    parent=node,
                    prior_p=prob,
                    # prm_value=prm_value,
                    text_state=text_state,
                    last_action=action,
                    initial_value=child_value,
                    parent_value=leaf_value,
                    num_generated_token=action_dict["num_token"],
                    model_name=model_name,
                    token_prob_list=action_dict.get("token_probs", []),
                    prm_step_scores=step_prm,
                    token_topk_logprobs_list=action_dict.get("token_topk_logprobs", []),
                )
            # set terminal node here
            if simulate_env._next_state_terminated[action]:
                if self.direct_io:
                    node.children[i].set_as_terminate_node()
                else:
                    node.children[action].set_as_terminate_node()
        if len(node.children) == 0:
            print_rank_0("Prune all current children at node {}".format(node.last_action))

        # collect num tokens
        if not node.has_collected_token_num:
            self._completion_tokens += sum(c.num_generated_token for c in node.children.values())
            node.has_collected_token_num = True
        else:
            raise RuntimeError("Token number has been collected again.")

        return leaf_value

    def _ucb_score(self, parent: Node, child: Node) -> float:
        """
        Overview:
            Compute UCB score. The score for a node is based on its value, plus an exploration bonus based on the prior.
        Arguments:
            - parent (:obj:`Class Node`): Current node.
            - child (:obj:`Class Node`): Current node's child.
        Returns:
            - score (:obj:`Bool`): The UCB score.
        """
        pb_c = (math.log((parent.visit_count + self._pb_c_base + 1) / self._pb_c_base) + self._pb_c_init)
        pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

        prior_score = pb_c * child.prior_p
        value_score = child.value

        return prior_score + value_score
        # return value_score

    def reset_prior(self, node: Node) -> None:
        """
        Overview:
            Reset prior probability
        Arguments:
            - node (:obj:`Class Node`): Current node.
        """
        for a in node.children.keys():
            node.children[a].prior_p = node.children[a].prior_p_ori

    def _add_exploration_noise(self, node: Node) -> None:
        """
        Overview:
            Add exploration noise.
        Arguments:
            - node (:obj:`Class Node`): Current node.
        """
        # Get a list of actions corresponding to the child nodes.
        actions = list(node.children.keys())
        # Create a list of alpha values for Dirichlet noise.
        alpha = [self._root_dirichlet_alpha] * len(actions)
        # Generate Dirichlet noise using the alpha values.
        noise = np.random.dirichlet(alpha)
        # Compute the weight of the exploration noise.
        frac = self._root_noise_weight
        # Update the prior probability of each child node with the exploration noise.
        for a, n in zip(actions, noise):
            node.children[a].prior_p = node.children[a].prior_p * (1 - frac) + n * frac

    @classmethod
    def from_json(cls, cfg: dict, json_path: str, reset_visit_info: bool):
        tree_json = json.load(open(json_path, "r"))

        def build_tree(tree_dict: dict) -> Node:
            node_info = tree_dict["info"]
            current_node = LanguageNode(
                text_state=node_info.get("text_state", None),
                last_action=node_info.get("last_action", None),
                prior_p=node_info["prior_p"],
                prm_value=node_info.get("prm_value", None),
                initial_value=node_info.get("initial_value", 0.0),
                prm_step_scores=node_info.get("prm_step_scores", None),
            )

            if not reset_visit_info:
                current_node._visit_count = node_info["visit_cnt"]
                current_node._value_sum = node_info["value"] * current_node.visit_count
            if node_info.get("terminated", False):
                current_node.set_as_terminate_node()

            for name, child_dict in tree_dict["children"].items():
                child_node = build_tree(child_dict)
                current_node._children[name] = child_node
                child_node._parent = current_node

            return current_node

        root_node = build_tree(tree_dict=tree_json)

        obj = cls(cfg)
        obj.root = root_node
        return obj

    def draw_tree(self):
        # Not tested yet
        root = self.root
        assert root, 'Root node is None'

        def draw_node(node, depth):
            print('|' + '-' * depth + str(node))
            for child in node.children.values():
                draw_node(child, depth + 1)

        print(f"\n---------Expanded Tree---------")
        draw_node(self.root, 0)
