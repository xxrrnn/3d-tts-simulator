#!/usr/bin/env python3
"""
0409：cursor基于indetify_straggler.py改的脚本，考虑两个straggler的情况。
从 ``*_workload.json`` 中识别 straggler branch（仅考察「最长的两个分支」）。

与 ``identify_straggler.py`` 的判定式相同：某分支长度 ``L`` 满足
``L > STRAGGLER_RATIO * max(其余分支长度)`` 且 ``L > STRAGGLER_MIN_TOKENS``。
区别：每个 step 只按 token 数从高到低取前两个 **不同下标** 的分支做上述判断；
不再对其余较短分支做 straggler 判定。

usage::

    python identify_straggler_top2_longest.py
    python identify_straggler_top2_longest.py --root ./model_workloads/...
    python identify_straggler_top2_longest.py -o ./straggler/custom_top2.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from identify_straggler import iter_workload_json_files, load_workload

# 与 identify_straggler.py 第 43–45 行同名常量保持一致；仅改本文件即可单独调参
STRAGGLER_RATIO = 1.5
STRAGGLER_MIN_TOKENS = 90


def is_straggler_branch(my_tokens: int, max_other_tokens: int) -> bool:
    return my_tokens > STRAGGLER_RATIO * max_other_tokens and my_tokens > STRAGGLER_MIN_TOKENS


# 与 gen_workload_beam_1._WORKLOAD_ROOT 一致：workload/model_workloads
_WORKLOAD_ROOT = Path(__file__).resolve().parent / "model_workloads"
_STRAGGLER_DIR = Path(__file__).resolve().parent / "straggler"
_DEFAULT_OUTPUT_JSON = _STRAGGLER_DIR / "stragglers_top2_longest.json"


def two_longest_branch_indices(branch_tokens: List[Any]) -> List[int]:
    """按 token 降序、同长按下标升序，返回最长的两个分支下标（长度不足 2 时返回能取到的全部）。"""
    pairs: List[tuple[int, int]] = []
    for i, t in enumerate(branch_tokens):
        try:
            pairs.append((int(t), i))
        except (TypeError, ValueError):
            pairs.append((0, i))
    pairs.sort(key=lambda x: (-x[0], x[1]))
    if len(pairs) <= 1:
        return []
    return [pairs[0][1], pairs[1][1]]


def find_stragglers_in_workload_top2_longest(
    workload_path: Path,
    workload: Dict[str, Any],
    *,
    include_token_topk: bool = True,
) -> List[Dict[str, Any]]:
    question_id = workload.get("question_id")
    decode = workload.get("decode") or {}
    steps = decode.get("steps") or []

    out: List[Dict[str, Any]] = []
    for step_idx, row in enumerate(steps):
        branch_tokens = row.get("branch_tokens") or []
        n = len(branch_tokens)
        if n <= 1:
            continue
        step_total_tokens = int(sum(int(t) for t in branch_tokens))
        raw_rewards = row.get("branch_rewards")
        if isinstance(raw_rewards, list):
            step_branch_rewards: Optional[List[Any]] = list(raw_rewards)
        else:
            step_branch_rewards = None
        sel_raw = row.get("selected_branch_index")
        try:
            selected_branch_index = int(sel_raw) if sel_raw is not None else -1
        except (TypeError, ValueError):
            selected_branch_index = -1

        candidate_indices = two_longest_branch_indices(branch_tokens)
        for branch_idx in candidate_indices:
            my_tokens = branch_tokens[branch_idx]
            others = branch_tokens[:branch_idx] + branch_tokens[branch_idx + 1 :]
            max_other = max(int(t) for t in others) if others else 0
            if not is_straggler_branch(int(my_tokens), int(max_other)):
                continue
            my_i = int(my_tokens)
            if step_total_tokens > 0:
                straggler_token_share = my_i / step_total_tokens
            else:
                straggler_token_share = None
            straggler_selected = selected_branch_index >= 0 and branch_idx == selected_branch_index
            if step_branch_rewards is not None and 0 <= branch_idx < len(step_branch_rewards):
                straggler_reward: Any = step_branch_rewards[branch_idx]
            else:
                straggler_reward = None
            rec: Dict[str, Any] = {
                "workload_path": str(workload_path.resolve()),
                "question_id": question_id,
                "majority_vote": (workload.get("result") or {}).get("majority_vote"),
                "step_index": step_idx,
                "branch_index": branch_idx,
                "straggler_tokens": my_i,
                "max_other_tokens": int(max_other),
                "branch_tokens": [int(t) for t in branch_tokens],
                "step_total_tokens": step_total_tokens,
                "straggler_token_share": straggler_token_share,
                "branch_reward": step_branch_rewards,
                "straggler_reward": straggler_reward,
                "selected_branch_index": selected_branch_index,
                "straggler_selected": straggler_selected,
            }
            if include_token_topk:
                topk_list = row.get("branch_token_topk_logprobs")
                if isinstance(topk_list, list) and 0 <= branch_idx < len(topk_list):
                    st_topk = topk_list[branch_idx]
                else:
                    st_topk = None
                rec["straggler_token_topk_logprobs"] = st_topk
            out.append(rec)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="识别 workload JSON 中的 straggler（仅最长的两个分支候选）"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=_WORKLOAD_ROOT,
        help=(
            "目录（递归扫描其下全部 *_workload.json）或单个 *_workload.json；"
            f"默认: {_WORKLOAD_ROOT}"
        ),
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=_DEFAULT_OUTPUT_JSON,
        help=f"结果 JSON 路径，默认: {_DEFAULT_OUTPUT_JSON}",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="除写入 JSON 外，在终端打印每条 straggler 摘要",
    )
    parser.add_argument(
        "--no-straggler-token-topk",
        action="store_true",
        help="不写入 straggler_token_topk_logprobs",
    )
    args = parser.parse_args()
    root: Path = args.root
    output_path: Path = args.output
    include_topk = not bool(args.no_straggler_token_topk)

    if not root.exists():
        print(f"路径不存在: {root}", file=sys.stderr)
        sys.exit(1)

    all_records: List[Dict[str, Any]] = []
    workload_files = 0
    for path in iter_workload_json_files(root):
        data = load_workload(path)
        if data is None:
            continue
        workload_files += 1
        all_records.extend(
            find_stragglers_in_workload_top2_longest(path, data, include_token_topk=include_topk)
        )

    report: Dict[str, Any] = {
        "generated_by": "identify_straggler_top2_longest.py",
        "mode": "top2_longest_branches_only",
        "scan_root": str(root.resolve()),
        "straggler_ratio": STRAGGLER_RATIO,
        "straggler_min_tokens": STRAGGLER_MIN_TOKENS,
        "include_straggler_token_topk": include_topk,
        "workload_files_scanned": workload_files,
        "record_count": len(all_records),
        "records": all_records,
    }

    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
    except OSError as e:
        print(f"无法写入 {output_path}: {e}", file=sys.stderr)
        sys.exit(1)

    print(
        f"已扫描 {workload_files} 个 workload 文件，检出 {len(all_records)} 条 straggler（仅最长两分支候选），"
        f"已写入 {output_path}"
    )

    if args.verbose:
        for rec in all_records:
            share = rec["straggler_token_share"]
            share_s = f"{share:.4f}" if isinstance(share, float) else str(share)
            print(
                f"  {rec['workload_path']}\n"
                f"    question_id={rec['question_id']} decode.steps[{rec['step_index']}] "
                f"branch[{rec['branch_index']}] straggler_tokens={rec['straggler_tokens']} "
                f"max_other={rec['max_other_tokens']} step_total={rec['step_total_tokens']} "
                f"straggler_share={share_s}\n"
                f"    branch_tokens={rec['branch_tokens']} branch_reward={rec['branch_reward']} "
                f"straggler_reward={rec['straggler_reward']} "
                f"selected_branch_index={rec['selected_branch_index']} straggler_selected={rec['straggler_selected']}\n"
                f"    majority_vote={rec['majority_vote']}"
            )


if __name__ == "__main__":
    main()
