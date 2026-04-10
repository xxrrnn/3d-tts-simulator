#!/usr/bin/env python3
"""
最终简化工作负载生成器（仅 beam=1 的配置目录）

保留：record 顶层的 result（含 majority_vote、total_completion_tokens），
以及 prefill 的 KV cache、decode 的 branch 路径。

decode / prefill **仅**从 ``detailed_beam_search_log`` 解析（需 ``BEAM_SEARCH_DETAILED_LOG=1``），无则 steps 为空、kv 为 0。

可选字段（命令行）：

- ``--include-reward``：将每分支 ``reward_score`` 以 ``branch_rewards`` 写入各 step（默认不写）。
- ``--no-token-topk-logprobs``：不写 ``branch_token_topk_logprobs``（默认写）。
- ``--no-token-probs``：不写 ``branch_token_probs``（对应 detailed 中 ``token_probs``，默认写）。

**仅处理** ``num_sequence=1`` 的配置目录：

- 标准名：以 ``_1`` 结尾（如 ``16384_8_1``、``16384_4_1``）
- Straggler 重命名：``{depth}_{width}_1_straggler_*``（如 ``16384_4_1_straggler_0_0_0``、
  ``16384_4_1_straggler_1_1.5_100``）

跳过 ``16384_8_4``、``16384_4_2`` 等 ``*_2``、``*_4`` 目录。

usage::

    python gen_workload_beam_1.py --input ../../src/output/AMC23_beam_search
    # 默认：有 branch_token_probs、branch_token_topk_logprobs，无 branch_rewards

    python gen_workload_beam_1.py --input ... --include-reward
    python gen_workload_beam_1.py --input ... --no-token-topk-logprobs
    python gen_workload_beam_1.py --input ... --no-token-probs
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

_WORKLOAD_ROOT = Path(__file__).resolve().parent / "model_workloads"


def _is_beam1_config_dir(name: str) -> bool:
    """True 当且仅当配置为 tree_max_depth × tree_max_width × num_seq 且 num_seq==1。"""
    if name.endswith("_1"):
        return True
    if "_straggler_" in name:
        base, _, _ = name.partition("_straggler_")
        return bool(base) and base.endswith("_1")
    return False


def _extract_result(data: Dict[str, Any]) -> Dict[str, Any]:
    result = data.get("result")
    if result is None:
        return {}
    return dict(result)


def _extract_prefill_kv_cache(data: Dict[str, Any]) -> int:
    if not data.get("output"):
        return 0
    first_output = data["output"][0]
    detailed_log = first_output.get("detailed_beam_search_log") or {}
    step_details = detailed_log.get("step_details") or []
    if not step_details:
        return 0
    first_step = step_details[0]
    expansion_results = first_step.get("expansion_results") or []
    if not expansion_results:
        return 0
    first_expansion = expansion_results[0]
    initial_tokens = first_expansion.get("api_completion_tokens", 0)
    question_tokens = len((data.get("question") or "").split())
    return question_tokens + initial_tokens


def _decode_steps_from_detailed(
    data: Dict[str, Any],
    *,
    include_reward: bool = False,
    include_token_probs: bool = True,
    include_token_topk_logprobs: bool = True,
) -> List[Dict[str, Any]]:
    steps: List[Dict[str, Any]] = []
    if not data.get("output"):
        return steps
    first_output = data["output"][0]
    detailed_log = first_output.get("detailed_beam_search_log") or {}
    if not detailed_log:
        return steps
    for step_info in detailed_log.get("step_details", []):
        step = step_info.get("step", 0)
        selected_branches = (step_info.get("selection_process") or {}).get("selected_branches") or []
        branch_tokens = []
        branch_rewards: List[Any] = []
        branch_token_probs: List[Any] = []
        branch_token_topk_logprobs: List[Any] = []
        selected_index = -1
        for i, branch in enumerate(selected_branches):
            # 使用 token_topk_logprobs 的子列表长度作为分支长度
            branch_tokens.append(len(branch.get("token_topk_logprobs") or []))
            if include_reward:
                branch_rewards.append(branch.get("reward_score"))
            if include_token_probs:
                branch_token_probs.append(branch.get("token_probs") or [])
            if include_token_topk_logprobs:
                branch_token_topk_logprobs.append(branch.get("token_topk_logprobs") or [])
            if branch.get("selected", False):
                selected_index = i
        row: Dict[str, Any] = {
            "step": step,
            "branch_count": len(selected_branches),
            "branch_tokens": branch_tokens,
            "selected_branch_index": selected_index,
        }
        if include_reward:
            row["branch_rewards"] = branch_rewards
        if include_token_probs:
            row["branch_token_probs"] = branch_token_probs
        if include_token_topk_logprobs:
            row["branch_token_topk_logprobs"] = branch_token_topk_logprobs
        steps.append(row)
    return steps


def load_question_data(record_path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(record_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line.strip())
                if "output" in data and len(data["output"]) > 0:
                    return data
        return None
    except Exception as e:
        logger.error(f"Error loading {record_path}: {e}")
        return None


def generate_workload_for_question(
    question_dir: Path,
    output_dir: Path,
    *,
    include_reward: bool = False,
    include_token_probs: bool = True,
    include_token_topk_logprobs: bool = True,
) -> bool:
    record_file = question_dir / "record_0.jsonl"
    if not record_file.exists():
        logger.warning(f"Record file not found: {record_file}")
        return False
    data = load_question_data(record_file)
    if not data:
        logger.error(f"Failed to load data from {record_file}")
        return False

    decode_steps = _decode_steps_from_detailed(
        data,
        include_reward=include_reward,
        include_token_probs=include_token_probs,
        include_token_topk_logprobs=include_token_topk_logprobs,
    )
    if not decode_steps:
        logger.warning(
            "%s: decode.steps 为空（无 detailed_beam_search_log 或未记录 step_details）",
            record_file,
        )

    workload: Dict[str, Any] = {
        "result": _extract_result(data),
        "question_id": question_dir.name,
        "prefill": {"kv_cache_count": _extract_prefill_kv_cache(data)},
        "decode": {"steps": decode_steps},
    }

    output_file = output_dir / f"{question_dir.name}_workload.json"
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(workload, f, indent=2, ensure_ascii=False)
        logger.info(f"Generated workload: {output_file}")
        return True
    except Exception as e:
        logger.error(f"Error saving workload to {output_file}: {e}")
        return False


def generate_all_workloads(
    input_dir: Path,
    dataset_name: str,
    *,
    include_reward: bool = False,
    include_token_probs: bool = True,
    include_token_topk_logprobs: bool = True,
) -> None:
    logger.info(f"Starting workload generation for dataset: {dataset_name}")
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        return
    generated_count = 0
    total_questions = 0
    for policy_dir in sorted(input_dir.iterdir(), key=lambda p: p.name):
        if not policy_dir.is_dir():
            continue
        logger.info(f"Processing policy model: {policy_dir.name}")
        for reward_dir in sorted(policy_dir.iterdir(), key=lambda p: p.name):
            if not reward_dir.is_dir():
                continue
            logger.info(f"  Processing reward model: {reward_dir.name}")
            for config_dir in sorted(reward_dir.iterdir(), key=lambda p: p.name):
                if not config_dir.is_dir():
                    continue
                if not _is_beam1_config_dir(config_dir.name):
                    logger.debug(f"    Skip config (not num_seq=1 / *_1[_straggler_*]): {config_dir.name}")
                    continue
                logger.info(f"    Processing config: {config_dir.name}")
                output_base = _WORKLOAD_ROOT / dataset_name / policy_dir.name / reward_dir.name / config_dir.name
                output_base.mkdir(parents=True, exist_ok=True)
                for question_dir in sorted(config_dir.iterdir(), key=lambda p: p.name):
                    if not question_dir.is_dir() or not question_dir.name.startswith("question_"):
                        continue
                    total_questions += 1
                    if generate_workload_for_question(
                        question_dir,
                        output_base,
                        include_reward=include_reward,
                        include_token_probs=include_token_probs,
                        include_token_topk_logprobs=include_token_topk_logprobs,
                    ):
                        generated_count += 1
    logger.info(f"Generated {generated_count}/{total_questions} workloads")


def main() -> None:
    parser = argparse.ArgumentParser(description="最简化工作负载生成器（num_seq=1：*_1 或 *_1_straggler_*）")
    parser.add_argument("--input", required=True, help="输入目录路径")
    parser.add_argument("--dataset", help="数据集名称")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细输出")
    parser.add_argument(
        "--include-reward",
        action="store_true",
        help="将每分支 reward（detailed 中 reward_score）写入 decode.steps[].branch_rewards；默认不写",
    )
    parser.add_argument(
        "--no-token-topk-logprobs",
        action="store_true",
        help="不写 decode.steps[].branch_token_topk_logprobs；默认写入",
    )
    parser.add_argument(
        "--no-token-probs",
        action="store_true",
        help="不写 decode.steps[].branch_token_probs（detailed 中 token_probs）；默认写入",
    )
    args = parser.parse_args()
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    input_path = Path(args.input)
    dataset_name = args.dataset or input_path.name
    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Input: {input_path}")
    logger.info(f"Output root: {_WORKLOAD_ROOT / dataset_name}")
    include_token_probs = not args.no_token_probs
    include_token_topk_logprobs = not args.no_token_topk_logprobs
    logger.info(
        "workload 可选字段: include_reward=%s, include_token_probs=%s, include_token_topk_logprobs=%s",
        args.include_reward,
        include_token_probs,
        include_token_topk_logprobs,
    )
    generate_all_workloads(
        input_path,
        dataset_name,
        include_reward=args.include_reward,
        include_token_probs=include_token_probs,
        include_token_topk_logprobs=include_token_topk_logprobs,
    )


if __name__ == "__main__":
    main()
