#!/usr/bin/env python3
"""
从 src/output 下的所有 jsonl 生成 speculation workload。

目标：
1) 保留 workload 风格信息（prefill/decode steps）
2) 额外提供每个 step 每个 branch 的 reward 与 probability
3) 输出到 process/speculation/output，并保持输入目录结构
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_first_jsonl_record(file_path: Path) -> Optional[Dict[str, Any]]:
    """读取 jsonl 的第一条记录。"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                return json.loads(line)
    except Exception as e:
        logger.error("Failed to load %s: %s", file_path, e)
    return None


def extract_prefill_kv_cache(record: Dict[str, Any]) -> int:
    """提取 prefill 的 KV cache 近似值。"""
    outputs = record.get("output", [])
    if not outputs:
        return 0
    detailed = outputs[0].get("detailed_beam_search_log", {})
    step_details = detailed.get("step_details", [])
    if not step_details:
        return 0
    first_expansions = step_details[0].get("expansion_results", [])
    if not first_expansions:
        return 0
    initial_tokens = int(first_expansions[0].get("api_completion_tokens", 0))
    question_tokens = len(record.get("question", "").split())
    return question_tokens + initial_tokens


def extract_step_branch_metrics(one_output: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    提取每个 step 的分支信息：
    - reward_score
    - probability (prior_prob)
    - token_probs
    """
    detailed = one_output.get("detailed_beam_search_log", {})
    step_details = detailed.get("step_details", [])
    all_steps: List[Dict[str, Any]] = []

    for step_info in step_details:
        step = int(step_info.get("step", 0))
        selection = step_info.get("selection_process", {})
        selected_branches = selection.get("selected_branches", [])

        branches: List[Dict[str, Any]] = []
        selected_index = -1

        for idx, branch in enumerate(selected_branches):
            if bool(branch.get("selected", False)) and selected_index < 0:
                selected_index = idx
            branches.append(
                {
                    "branch_index": idx,
                    "reward": float(branch.get("reward_score", 0.0)),
                    "probability": float(branch.get("prior_prob", 0.0)),
                    "num_tokens": int(branch.get("num_tokens", 0)),
                    "token_probs": branch.get("token_probs", []),
                }
            )

        all_steps.append(
            {
                "step": step,
                "branch_count": len(branches),
                "selected_branch_index": selected_index,
                "branches": branches,
            }
        )

    return all_steps


def build_workload(record: Dict[str, Any], source_file: Path) -> Dict[str, Any]:
    """构造输出 workload JSON。"""
    outputs = record.get("output", [])
    output_entries: List[Dict[str, Any]] = []

    for one_output in outputs:
        output_entries.append(
            {
                "path_idx": one_output.get("path_idx", 0),
                "reward_history": one_output.get("reward_history", []),
                "prob_history": one_output.get("prob_history", []),
                "token_history": one_output.get("token_history", []),
                "step_branch_metrics": extract_step_branch_metrics(one_output),
            }
        )

    return {
        "source_file": str(source_file),
        "question": record.get("question", ""),
        "prefill": {"kv_cache_count": extract_prefill_kv_cache(record)},
        "decode": {"outputs": output_entries},
    }


def convert_one_file(input_file: Path, input_root: Path, output_root: Path) -> bool:
    record = load_first_jsonl_record(input_file)
    if record is None:
        return False

    workload = build_workload(record, input_file)
    relative = input_file.relative_to(input_root)
    output_file = (output_root / relative).with_suffix(".json")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(workload, f, indent=2, ensure_ascii=False)
        logger.info("Generated: %s", output_file)
        return True
    except Exception as e:
        logger.error("Failed to save %s: %s", output_file, e)
        return False


def convert_all(input_root: Path, output_root: Path) -> None:
    if not input_root.exists():
        logger.error("Input root does not exist: %s", input_root)
        return

    all_jsonl = sorted(input_root.rglob("*.jsonl"))
    total = len(all_jsonl)
    success = 0

    logger.info("Found %d jsonl files under %s", total, input_root)
    for fp in all_jsonl:
        if convert_one_file(fp, input_root, output_root):
            success += 1

    logger.info("Done. Success: %d / %d", success, total)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate speculation workloads from src/output jsonl files")
    parser.add_argument(
        "--input",
        default="/DISK1/data/rnxu_24/Paper/3d-tts-simulator/3d-tts-sw/compute-optimal-tts/src/output",
        help="Input root directory (default: src/output)",
    )
    parser.add_argument(
        "--output",
        default="/DISK1/data/rnxu_24/Paper/3d-tts-simulator/3d-tts-sw/compute-optimal-tts/process/speculation/output",
        help="Output root directory (default: process/speculation/output)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logs")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    input_root = Path(args.input)
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    logger.info("Input root: %s", input_root)
    logger.info("Output root: %s", output_root)
    convert_all(input_root, output_root)


if __name__ == "__main__":
    main()

