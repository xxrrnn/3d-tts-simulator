"""
This file is largely borrowed from OpenR (https://github.com/openreasoner/openr)
"""

import os
import copy
import json
import sys
import time
from argparse import ArgumentParser
from datetime import datetime
from functools import partial

import jsonlines
import numpy as np
import tree
from transformers import AutoTokenizer
import ray
from ray.util.actor_pool import ActorPool

from reason.evaluation.evaluator import RemoteMathEvaluator
from reason.evaluation.methods import BeamSearchConfig, beam_search, Task
from reason.inference.lm_call import LMCallingConfig, VLLMRemoteCaller
from reason.inference.rm_call import (
    RMRemoteCaller,
    DummyRewardModelCaller,
    RemoteRewardModelConfig,
    get_prm_special_tokens,
)
from utils import check_process_cnt, assign_tasks, get_model_name, setup_seed, check_lock_timeout

cot_prompt_dict = {
    'llama_official': """Solve the following math problem efficiently and clearly:\n\n- For simple problems (2 steps or fewer):\nProvide a concise solution with minimal explanation.\n\n- For complex problems (3 steps or more):\nUse this step-by-step format:\n\n## Step 1: [Concise description]\n[Brief explanation and calculations]\n\n## Step 2: [Concise description]\n[Brief explanation and calculations]\n\n...\n\nRegardless of the approach, always conclude with:\n\nTherefore, the final answer is: $\\boxed{answer}$. I hope it is correct.\n\nWhere [answer] is just the final number or expression that solves the problem.""",
    'qwen': """Please reason step by step, and put your final answer within \\boxed{}.""",
    'default': """Please reason step by step, and put your final answer within \\boxed{}.""",
}

llm_step_tag_dict = {
    'llama': "## Step ",
    'qwen': "\nStep ",
    'default': "\nStep ",
}

sep_dict = {
    'llama': ["## Step"],
    'qwen': ["\nStep"],
    'default': ["\nStep"],
}

stop_str_dict = {
    'llama': ["\\boxed"],
    'qwen': ["\\boxed"],
    'default': ["\\boxed"],
}

if __name__ == "__main__":
    parser = ArgumentParser()
    # LLM config
    parser.add_argument("--LM", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--top_p", type=float, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--add_step_prompt", action="store_true")
    parser.add_argument("--serve_type", type=str, default="fastchat", choices=["fastchat"])
    parser.add_argument("--cot_prompt", type=str, default="")
    parser.add_argument("--llm_step_tag", type=str, default="")
    parser.add_argument("--stop_str", default=[])
    parser.add_argument("--sep", default=[])
    parser.add_argument("--double_line_break", type=int, default=0)
    # RM config
    parser.add_argument("--RM", type=str, default="dummy")
    parser.add_argument("--rm_device", type=str, default="cuda")
    parser.add_argument("--good_tag", type=str, default="+")
    parser.add_argument("--bad_tag", type=str, default="-")
    parser.add_argument("--prm_step_tag", type=str, default="ки\n")
    parser.add_argument("--prm_format_str", type=str, default="{question} {answer}")
    parser.add_argument("--rm_serve_type", type=str, default="fastchat", choices=["fastchat"])
    # method config
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--num_sequence", type=int, default=1)
    parser.add_argument("--tree_max_depth", type=int, default=None)
    parser.add_argument("--tree_max_width", type=int, default=None)
    # other config
    parser.add_argument("--task_name", type=str, default="MATH")
    parser.add_argument("--is_few_shot", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    #0405： 增加straggler剪枝参数
    parser.add_argument(
        "--straggler_prune",
        type=int,
        default=0,
        choices=[0, 1],
        help="1=启用 beam 展开时 straggler 剪枝（PRM 分置 0），0=关闭",
    )
    parser.add_argument("--straggler_length_ratio", type=float, default=1.5)
    parser.add_argument("--straggler_min_tokens", type=int, default=80)
    parser.add_argument(
        "--straggler_prune_other_reward_gate",
        type=int,
        default=0,
        choices=[0, 1],
        help="1=仅当非 straggler 兄弟分支 PRM 分最大值>阈值时才剪枝；0=仅按长度倍率判定",
    )
    parser.add_argument(
        "--straggler_prune_other_reward_threshold",
        type=float,
        default=0.0,
        help="与 --straggler_prune_other_reward_gate=1 配合：兄弟分支 PRM 最大分须严格大于该值才剪 straggler",
    )
    parser.add_argument(
        "--straggler_deferred_prune",
        type=int,
        default=0,
        choices=[0, 1],
        help="1=启用跨 step 延迟 straggler 剪枝（须与 --straggler_prune=1 同时开启）；0=仅即时长度剪枝",
    )
    parser.add_argument(
        "--straggler_predictor_enabled",
        type=int,
        default=0,
        choices=[0, 1],
        help="1=启用 MLP 预测器进行 straggler 检测；0=关闭",
    )
    parser.add_argument(
        "--straggler_predictor_weights",
        type=str,
        default="",
        help="MLP 预测器权重文件路径",
    )
    parser.add_argument(
        "--straggler_predictor_priors",
        type=str,
        default="",
        help="MLP 预测器先验文件路径",
    )
    parser.add_argument(
        "--active_branch_gate",
        type=int,
        default=2,
        choices=[1, 2],
        help="活跃分支数阈值，当 n_active_branches <= 此值时调用 MLP 预测器",
    )
    parser.add_argument(
        "--straggler_budget_on",
        type=int,
        default=0,
        choices=[0, 1],
        help="1=启用 straggler budget（前 N 步不剪枝）；0=关闭",
    )
    parser.add_argument(
        "--straggler_budget",
        type=int,
        default=2,
        help="前 N 步不进行 straggler 剪枝（仅 --straggler_budget_on=1 时生效）",
    )
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--controller_addr", type=str, default="http://localhost:10014")
    parser.add_argument("--num_worker", type=int, default=8)
    parser.add_argument("--local", type=int, default=0)
    parser.add_argument("--question_parallel_num", type=int, default=0)
    parser.add_argument("--question_max_num", type=int, default=0)
    parser.add_argument("--lock_dir", type=str, default="lock_dir")
    parser.add_argument("--multi_gpu", action="store_true")
    parser.add_argument("--batch_size", type=int, default=0)
    parser.add_argument("--max_time", type=int, default=0)
    parser.add_argument(
        "--deterministic",
        type=int,
        default=0,
        choices=[0, 1],
        help="1=严格确定性模式：串行发送 LLM 请求，消除 batch 组合导致的浮点非确定性；会降低吞吐",
    )

    args = parser.parse_args()

    if 'llama-3' in args.LM.lower():
        args.cot_prompt = cot_prompt_dict['llama_official']
        args.llm_step_tag = llm_step_tag_dict['llama']
        args.sep = sep_dict['llama']
        args.stop_str = stop_str_dict['llama']
    elif 'qwen' in args.LM.lower():
        args.cot_prompt = cot_prompt_dict['qwen']
        args.llm_step_tag = llm_step_tag_dict['qwen']
        args.sep = sep_dict['qwen']
        args.stop_str = stop_str_dict['qwen']
    elif 'deepseek-r1' in args.LM.lower():
        args.cot_prompt = cot_prompt_dict['default']
        args.llm_step_tag = llm_step_tag_dict['default']
        args.sep = sep_dict['default']
        args.stop_str = stop_str_dict['default']
    else:
        args.cot_prompt = cot_prompt_dict['default']
        args.llm_step_tag = llm_step_tag_dict['default']
        args.sep = sep_dict['default']
        args.stop_str = stop_str_dict['default']

    if args.double_line_break == 1:
        args.sep = ["\n\n"]

    os.makedirs(args.save_dir, exist_ok=True)

    if "dummy" in args.RM:
        assert args.method in ["cot", "best_of_n"]
    if args.tree_max_depth is not None and args.tree_max_width is not None:
        assert args.tree_max_width % args.num_sequence == 0

    args.LM = args.LM.split(',')

    setup_seed(args.seed)
    if args.local:
        print("run in pure local mode for debug only")
        args.num_worker = 1
        ray.init(local_mode=True)
    else:
        ray.init()

    if args.RM.endswith("/"):
        args.RM = args.RM[:-1]
    rm_model_name = args.RM
    rm_model_path = args.RM
    if "dummy" in args.RM:
        rm_config = RemoteRewardModelConfig(
            prm_step_tag=args.prm_step_tag, format_str=args.prm_format_str, model_name=args.RM, controller_addr=args.controller_addr,
            step_tag_id=None, returned_token_ids=None, rm_serve_type=args.rm_serve_type, multi_gpu=args.multi_gpu
        )
        rm_call = DummyRewardModelCaller(rm_config)
    else:
        if args.rm_serve_type == "fastchat":
            tokenizer = AutoTokenizer.from_pretrained(rm_model_path, trust_remote_code=True)
            step_tag_id, returned_token_ids = get_prm_special_tokens(rm_model_name, tokenizer)
            if 'pqm' in args.RM:
                prm_format_str = "{question}\n{answer}"
            else:
                prm_format_str = "{question} {answer}"
            rm_config = RemoteRewardModelConfig(
                prm_step_tag=args.prm_step_tag, format_str=prm_format_str, model_name=args.RM, controller_addr=args.controller_addr,
                step_tag_id=step_tag_id, returned_token_ids=returned_token_ids, rm_serve_type=args.rm_serve_type, multi_gpu=args.multi_gpu,
            )
            rm_call = RMRemoteCaller(rm_config, tokenizer=tokenizer)
        else:
            raise NotImplementedError

    llm_step_tags = [args.llm_step_tag for _ in args.LM]

    def build_llm_gen_fns(actor_index: int):
        return [
            VLLMRemoteCaller(
                lm,
                lm,
                args.controller_addr,
                args.llm_step_tag,
                apply_chat_template=True,
                multi_gpu=args.multi_gpu,
                serve_type=args.serve_type,
                double_line_break=args.double_line_break,
                # generation_seed=int(args.seed) + int(actor_index),
                generation_seed=int(args.seed),  # 后备默认值；beam_search 中会被 _derive_lm_branch_seed 覆盖 尝试一下这样写，看有无多个branch、是否可复现
            )
            for lm in args.LM
        ]

    rm_call = partial(rm_call, model_names=args.LM)

    task = Task(task_name=args.task_name, is_few_shot=args.is_few_shot, model_names=args.LM)


    def parallel_evaluate_test_dataset(actor_pool, raw_test_ds, method_name, solver_fn, save_dir, question_parallel_num):
        results = []
        question2id = {problem_inst["question"]: i for i, problem_inst in enumerate(raw_test_ds)}

        test_ds, _ = assign_tasks(
            raw_test_ds, question_parallel_num, args.num_sequence, save_dir, args.lock_dir, args.batch_size, args.max_time
        )

        res_q = actor_pool.map_unordered(lambda p, x: p.evaluate_problem.remote(x, solver_fn), test_ds)
        start_time = time.time()
        last_time = start_time

        for i, (problem_inst, result, output) in enumerate(res_q):
            if len(output) == 0:
                continue
            obj = {"question": problem_inst["question"], "groundtruth": problem_inst["answer"], "result": result, "output": output}
            q_idx = question2id[problem_inst["question"]]
            question_path = os.path.join(save_dir, f"question_{q_idx}")
            os.makedirs(question_path, exist_ok=True)

            file_path = problem_inst["file_path"]
            idx = int(file_path.split("_")[-1].split(".")[0])
            try:
                record_writer = jsonlines.open(file_path, mode="w", flush=True)
                record_writer.write(obj)
            except Exception as e:
                print(f"Save Error: {e}")

            temp_time = time.time()
            delta_time = temp_time - last_time
            total_time = temp_time - start_time
            time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")[2:]
            last_time = temp_time

            cnt = check_process_cnt(raw_test_ds, question_parallel_num, save_dir)
            total = len(raw_test_ds) * question_parallel_num if question_parallel_num else len(raw_test_ds)
            print(
                f"[{time_str}]   Cnt: {i + 1:>3} / {len(test_ds):>3}  |  Q: {q_idx:>3}  |  Idx: {idx:>3}  |  "
                f"Del T: {delta_time:>6.1f}s  |  Tot T: {total_time:>7.1f}s  |  Avg T: {total_time / (i + 1):>6.1f}s/it  |  "
                f"Pct: {cnt:>5} / {total:>5} = {cnt / total * 100:.2f}%"
            )

            if not question_parallel_num:
                results.append(result)

        if not question_parallel_num:
            try:
                avg_res = (tree.map_structure(lambda *xs: np.mean(xs), *results),)
                json.dump(avg_res, open(os.path.join(save_dir, f"avg_result.json"), "w"))
                print("Method: {}. Average result: {}".format(method_name, avg_res))
            except Exception as e:
                pass

        return results


    if 'deepseek-r1' in args.LM[0].lower():
        args.temperature = 0.6
        args.top_p = 0.95
        args.max_new_tokens = 32768

    cfg_dict_record = dict()
    gen_config = LMCallingConfig(
        n=args.num_sequence,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
    )
    cfg_dict_record["gen_config"] = gen_config.__dict__

    if args.method == "cot":
        direct_io = 2
    elif args.method == "best_of_n":
        direct_io = 1
    else:
        direct_io = 0

    if args.method in ["cot", "best_of_n"]:
        args.tree_max_depth = 1
        if args.method == "cot":
            args.num_sequence = 1
            args.tree_max_width = 1
            if 'deepseek-r1' in args.LM[0].lower():
                top_p = 0.95
            else:
                top_p = 1.0
            gen_config = LMCallingConfig(
                n=1,
                temperature=args.temperature,
                top_k=-1,
                top_p=top_p,
                max_new_tokens=gen_config.max_new_tokens,
            )
        elif args.method == "best_of_n":
            args.num_sequence = args.tree_max_width
        method_config = BeamSearchConfig(
            task_name=args.task_name,
            tree_max_depth=1,
            tree_max_width=args.tree_max_width,
            beam_size=args.num_sequence,
            model_names=args.LM,
            is_few_shot=args.is_few_shot,
            add_step_prompt=args.add_step_prompt,
            cot_prompt=args.cot_prompt,
            stop_str=None,
            sep=args.sep,
            direct_io=direct_io,
            double_line_break=args.double_line_break,
            eval_log_policy_model=get_model_name(args.LM[0]),
            eval_log_reward_model=get_model_name(args.RM),
            straggler_prune_enabled=bool(args.straggler_prune),
            straggler_length_ratio=args.straggler_length_ratio,
            straggler_min_tokens=args.straggler_min_tokens,
            straggler_prune_other_reward_gate=bool(args.straggler_prune_other_reward_gate),
            straggler_prune_other_reward_threshold=args.straggler_prune_other_reward_threshold,
            straggler_deferred_prune_enabled=bool(args.straggler_deferred_prune),
            straggler_predictor_enabled=bool(args.straggler_predictor_enabled),
            straggler_predictor_weights=args.straggler_predictor_weights,
            straggler_predictor_priors=args.straggler_predictor_priors,
            active_branch_gate=args.active_branch_gate,
            straggler_budget_on=bool(args.straggler_budget_on),
            straggler_budget=args.straggler_budget,
            eval_seed=int(args.seed),
            deterministic=bool(args.deterministic),
        )
        solver_fn = partial(beam_search, method_config, gen_config)
    elif "beam_search" in args.method:
        method_config = BeamSearchConfig(
            task_name=args.task_name,
            tree_max_depth=args.tree_max_depth,
            tree_max_width=args.tree_max_width,
            beam_size=args.num_sequence,
            model_names=args.LM,
            is_few_shot=args.is_few_shot,
            add_step_prompt=args.add_step_prompt,
            cot_prompt=args.cot_prompt,
            stop_str=args.stop_str,
            sep=args.sep,
            direct_io=direct_io,
            double_line_break=args.double_line_break,
            eval_log_policy_model=get_model_name(args.LM[0]),
            eval_log_reward_model=get_model_name(args.RM),
            straggler_prune_enabled=bool(args.straggler_prune),
            straggler_length_ratio=args.straggler_length_ratio,
            straggler_min_tokens=args.straggler_min_tokens,
            straggler_prune_other_reward_gate=bool(args.straggler_prune_other_reward_gate),
            straggler_prune_other_reward_threshold=args.straggler_prune_other_reward_threshold,
            straggler_deferred_prune_enabled=bool(args.straggler_deferred_prune),
            straggler_predictor_enabled=bool(args.straggler_predictor_enabled),
            straggler_predictor_weights=args.straggler_predictor_weights,
            straggler_predictor_priors=args.straggler_predictor_priors,
            active_branch_gate=args.active_branch_gate,
            straggler_budget_on=bool(args.straggler_budget_on),
            straggler_budget=args.straggler_budget,
            eval_seed=int(args.seed),
            deterministic=bool(args.deterministic),
        )
        solver_fn = partial(beam_search, method_config, gen_config)
    else:
        raise ValueError(f"Unknown method: {args.method}")

    cfg_dict_record["method"] = args.method
    cfg_dict_record["method_config"] = method_config.__dict__

    params = f'{args.tree_max_depth}_{args.tree_max_width}_{args.num_sequence}'
    model_name = get_model_name(args.LM[0])
    rm_model_name = get_model_name(args.RM)
    save_dir = f'{args.save_dir}/{args.task_name}_{args.method}/{model_name}/{rm_model_name}/{params}'
    print(f"Auto set dir as {save_dir}")

    try:
        os.makedirs(save_dir, exist_ok=True)
        if args.lock_dir:
            os.makedirs(os.path.join(save_dir, args.lock_dir), exist_ok=True)
    except Exception as e:
        print(f"Error: {e}")

    cfg_dict_record["llm_step_tags"] = llm_step_tags
    cfg_dict_record["seed"] = args.seed
    cfg_dict_record["straggler_prune"] = args.straggler_prune
    cfg_dict_record["straggler_length_ratio"] = args.straggler_length_ratio
    cfg_dict_record["straggler_min_tokens"] = args.straggler_min_tokens
    cfg_dict_record["straggler_prune_other_reward_gate"] = args.straggler_prune_other_reward_gate
    cfg_dict_record["straggler_prune_other_reward_threshold"] = args.straggler_prune_other_reward_threshold
    cfg_dict_record["straggler_deferred_prune"] = args.straggler_deferred_prune
    cfg_dict_record["straggler_predictor_enabled"] = args.straggler_predictor_enabled
    cfg_dict_record["straggler_predictor_weights"] = args.straggler_predictor_weights
    cfg_dict_record["straggler_predictor_priors"] = args.straggler_predictor_priors
    cfg_dict_record["straggler_budget_on"] = args.straggler_budget_on
    cfg_dict_record["straggler_budget"] = args.straggler_budget
    cfg_dict_record["active_branch_gate"] = args.active_branch_gate
    cfg_dict_record["prm_step_tag"] = args.prm_step_tag
    cfg_dict_record["good_tag"] = args.good_tag
    cfg_dict_record["bad_tag"] = args.bad_tag
    cfg_dict_record["stop_str"] = args.stop_str
    cfg_dict_record["sep"] = args.sep
    cfg_dict_record["LM"] = args.LM
    cfg_dict_record["RM"] = args.RM
    try:
        json.dump(cfg_dict_record, open(os.path.join(save_dir, f"config.json"), "w"))
    except Exception as e:
        print(f"Error: {e}")

    actor_pool = ActorPool(
        [
            RemoteMathEvaluator.remote(
                args.task_name,
                build_llm_gen_fns(actor_index),
                rm_call,
                direct_io=direct_io,
                seed=args.seed,
                actor_index=actor_index,
            )
            for actor_index in range(args.num_worker)
        ]
    )

    test_ds = task.test_ds(args.task_name)

    returned_temp = parallel_evaluate_test_dataset(
        actor_pool, test_ds, args.method, solver_fn, save_dir, question_parallel_num=args.question_parallel_num,
    )
    check_lock_timeout(test_ds, args.question_parallel_num, save_dir, args.lock_dir, args.max_time)
