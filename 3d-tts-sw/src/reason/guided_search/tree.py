"""
The Node and MCTS class for AlphaZero.
This file is largely borrowed from OpenR (https://github.com/openreasoner/openr)
"""

import copy
import json
import math
import traceback

import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Any, Optional, Tuple, Union, Callable, Type
from utils import print_rank_0, print_with_rank
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
    ) -> None:
        super().__init__(parent, prior_p, initial_value, parent_value)
        self.text_state = text_state
        self.last_action = last_action
        self.prm_value = prm_value

        self.num_generated_token = num_generated_token
        self.has_collected_token_num = False

        self.model_name = model_name

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
        if max_step == 1:
            assert self.direct_io
        api_call_completion_tokens = 0
        _, info = simulate_env.reset(update_legal_action=True)
        api_call_completion_tokens += info["api_completion_token"]
        if self.root is None:
            root = LanguageNode(text_state=simulate_env.get_state(model_name='raw'))
            self._expand_leaf_node(root, simulate_env, reward_model_fn)
            self.root = root

        end_nodes, top_k_nodes = [], [(-root._initial_value, -root._initial_value, -root._parent_value, root, simulate_env.copy())]
        k = copy.deepcopy(beam_size)

        for i in range(max_step + 1):
            cur_nodes_to_search = top_k_nodes
            top_k_nodes = []
            for cur_neg_q_plus_a, cur_neg_v, cur_neg_parent_v, cur_node, cur_env in cur_nodes_to_search:
                if cur_node.terminated:
                    end_nodes.append((cur_neg_q_plus_a, cur_neg_v, cur_neg_parent_v, cur_node, cur_env))
                    if len(end_nodes) == beam_size:
                        break
                elif len(end_nodes) < beam_size:
                    # select at most topk children add push to heap
                    assert (len(cur_node.children) > 0), "in beam search you should expand this non-terminal node at first."

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
                        for c_act, c_node, c_q_plus_a, c_value, c_parent_value in top_k_children:
                            new_env = cur_env.copy()
                            heapq.heappush(top_k_nodes, (-c_q_plus_a, -c_value, -c_parent_value, c_node, new_env))
            top_k_nodes = heapq.nsmallest(k, top_k_nodes)  # nsmallest since we negate the value

            # expand selected nodes
            for q_plus_a, value, parent_value, node, new_env in top_k_nodes:
                _, _, terminated, truncated, info = new_env.step(
                    node.last_action, update_legal_action=self.direct_io == 0, model_name=node.model_name,
                    reward=node._initial_value, num_token=node.num_generated_token, prob=node.prior_p,
                )
                api_call_completion_tokens += info["api_completion_token"]
                if terminated or truncated:
                    node.set_as_terminate_node()
                else:
                    self._expand_leaf_node(node, new_env, reward_model_fn)

            if len(end_nodes) == beam_size:
                break

        traj_list = []
        for i, (neg_e_q_plus_a, neg_e_v, neg_e_parent_v, e_node, e_env) in enumerate(end_nodes):
            traj_list.append({
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
                "model_history": e_env.model_history,
                # num_generated_token is hard to compute for each single answer
            })
        traj_list[-1]["tree_completion_tokens"] = self._completion_tokens
        traj_list[-1]["api_completion_tokens"] = api_call_completion_tokens
        return traj_list

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
                        # prms = [[0.0] for _ in simulate_env.legal_actions]
            child_values = []
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
                elif len(rs) == 0:
                    logger.warning(f"Empty PRM value for: \nState: \n{text_state} \naction: \n{act}, will be set to 0.0")
                    child_values.append(0.0)
                else:
                    # prm-last
                    child_values.append(rs[-1])  # PRM get last r as single reward, [0.9783847332000732, 0.9621075391769409]
                    # # prm-min
                    # child_values.append(min(rs))
                    # # prob-prm
                    # child_values.append(act['prob'])

        assert len(node.children) == 0
        for i, action_dict in enumerate(simulate_env.legal_actions):
            action, prob = action_dict["action"], action_dict["prob"]
            model_name = action_dict["model_name"]

            if self._init_critic_value:
                child_value = child_values[i]
            else:
                # XXX(ziyu): consider turn off this branch, i.e. always assume
                #  `self._init_critic=True`, since with LLM
                child_value = 0.0

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
