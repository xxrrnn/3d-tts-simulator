#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _pick_branch_text(branch: Dict[str, Any]) -> str:
    for key in ("action", "branch_content", "full_path", "q_plus_a"):
        value = branch.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _extract_step_list(output_item: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    按 step 组织：每个 step 包含 branch_count、selected_index、reward_score_list 和 branches 列表。
    参考 gen_workload_beam_1.py 的结构。
    """
    detailed = output_item.get("detailed_beam_search_log", {})
    step_details = detailed.get("step_details", [])
    step_list: List[Dict[str, Any]] = []

    for step_info in step_details:
        step_id = step_info.get("step")
        selection = step_info.get("selection_process", {})
        selected_branches = selection.get("selected_branches", [])

        branches = []
        reward_score_list = []
        branch_len = []
        selected_index = -1

        for idx, branch in enumerate(selected_branches):
            text = _pick_branch_text(branch)
            topk_prob = branch.get("token_topk_logprobs", [])
            reward_score = branch.get("reward_score")
            length = len(topk_prob) if isinstance(topk_prob, list) else 0

            branches.append(
                {
                    "text": text,
                    "reward_score": reward_score,
                    "length": length,
                }
            )
            reward_score_list.append(reward_score)
            branch_len.append(length)

            if branch.get("selected", False):
                selected_index = idx

        step_list.append(
            {
                "step": step_id,
                "branch_count": len(selected_branches),
                "branch_len": branch_len,
                "selected_index": selected_index,
                "reward_score_list": reward_score_list,
                "branches": branches,
            }
        )

    # 如果最后一个 step 的 selected_index 是 -1，删除它
    if step_list and step_list[-1]["selected_index"] == -1:
        step_list.pop()

    return step_list


def _compact_record(obj: Dict[str, Any]) -> Dict[str, Any]:
    compact_outputs = []
    for out in obj.get("output", []):
        final_text = out.get("text", "")
        step_list = _extract_step_list(out)

        compact_outputs.append(
            {
                "path_idx": out.get("path_idx"),
                "extracted_answer": out.get("extracted_answer"),
                "final_text": final_text,
                "final_reward": out.get("reward_history", [])[-1] if out.get("reward_history") else None,
                "step_list": step_list,
            }
        )

    return {
        "question": obj.get("question"),
        "groundtruth": obj.get("groundtruth"),
        "result": obj.get("result"),
        "output_compact": compact_outputs,
    }


def _load_jsonl_first_line(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        first_line = f.readline().strip()
    if not first_line:
        raise ValueError(f"Empty file: {path}")
    return json.loads(first_line)


def _default_out_path(in_path: Path) -> Path:
    return in_path.with_name(f"{in_path.stem}.formatted.json")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert record_0.jsonl to a compact readable JSON. "
            "Keep per-branch text/reward/length and deduplicate repeated text."
        )
    )
    parser.add_argument("input_path", help="Path to record_0.jsonl")
    parser.add_argument(
        "--output",
        help="Output JSON path. Default: same directory, record_0.formatted.json",
        default=None,
    )
    args = parser.parse_args()

    in_path = Path(args.input_path).expanduser().resolve()
    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}")

    out_path = Path(args.output).expanduser().resolve() if args.output else _default_out_path(in_path)
    obj = _load_jsonl_first_line(in_path)
    compact = _compact_record(obj)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(compact, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"Input : {in_path}")
    print(f"Output: {out_path}")
    total_steps = sum(len(o.get('step_list', [])) for o in compact.get('output_compact', []))
    print(f"Total steps: {total_steps}")


if __name__ == "__main__":
    main()
