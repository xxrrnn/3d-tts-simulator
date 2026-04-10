#!/usr/bin/env python3
"""
从 ``*_workload.json`` 中按固定规则识别 straggler branch。

定义（与生成 workload 的 ``decode.steps[].branch_tokens`` 一致）：

1. 该 step 上 branch 数 > 1。
2. 某分支长度 ``L`` 满足：``L > 1.5 * max(其余分支长度)`` 且 ``L > 80``。

对每条 straggler 记录输出：本分支长度、同 step 全部分支长度、该 step 总 token、
straggler 占该 step token 比例、``branch_reward``（数组，当前 step 各分支得分，与 workload
``branch_rewards`` 顺序一致；无则为 ``null``）、``straggler_reward``（掉队分支单独得分；
无对应项时为 ``null``）、``selected_branch_index``、straggler 是否被选中、``step_index``、
workload 路径、``majority_vote``。

默认在每条记录中写入掉队分支的 ``straggler_token_topk_logprobs``（与 workload 中
``decode.steps[].branch_token_topk_logprobs`` 同序，取掉队分支那一项；workload 若未含该字段则为
``null``）。若需减小输出体积，可加 ``--no-straggler-token-topk``。

usage::

    # 默认扫描 workload/model_workloads，结果写入 workload/straggler/stragglers.json（含 topk）
    python identify_straggler.py
    python identify_straggler.py --no-straggler-token-topk   # 不写 topk，文件更小
    python identify_straggler.py --root ./model_workloads/AIME24_beam_search/...
    python identify_straggler.py --output ./straggler/custom.json
    python identify_straggler.py -v   # 同时在终端打印每条记录摘要
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

# 与 gen_workload_beam_1._WORKLOAD_ROOT 一致：workload/model_workloads
_WORKLOAD_ROOT = Path(__file__).resolve().parent / "model_workloads"
_STRAGGLER_DIR = Path(__file__).resolve().parent / "straggler"
_DEFAULT_OUTPUT_JSON = _STRAGGLER_DIR / "stragglers.json"

STRAGGLER_RATIO = 1.5
STRAGGLER_MIN_TOKENS = 100


def is_straggler_branch(my_tokens: int, max_other_tokens: int) -> bool:
    return my_tokens > STRAGGLER_RATIO * max_other_tokens and my_tokens > STRAGGLER_MIN_TOKENS


def find_stragglers_in_workload(
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

        for branch_idx, my_tokens in enumerate(branch_tokens):
            others = branch_tokens[:branch_idx] + branch_tokens[branch_idx + 1 :]
            max_other = max(others)
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


def iter_workload_json_files(root: Path) -> Iterator[Path]:
    if root.is_file() and root.suffix == ".json" and root.name.endswith("_workload.json"):
        yield root
        return
    yield from sorted(root.rglob("*_workload.json"))


def load_workload(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="识别 workload JSON 中的 straggler branch")
    parser.add_argument(
        "--root",
        type=Path,
        default=_WORKLOAD_ROOT,
        help=(
            "目录（递归扫描其下全部 *_workload.json）或单个 *_workload.json；"
            f"默认: {_WORKLOAD_ROOT}（即本仓库 workload/model_workloads）"
        ),
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=_DEFAULT_OUTPUT_JSON,
        help=f"结果 JSON 路径，默认: {_DEFAULT_OUTPUT_JSON}（会自动创建目录）",
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
        help="不写入 straggler_token_topk_logprobs（默认写入，便于缩小 JSON）",
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
            find_stragglers_in_workload(path, data, include_token_topk=include_topk)
        )

    report: Dict[str, Any] = {
        "generated_by": "identify_straggler.py",
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
        f"已扫描 {workload_files} 个 workload 文件，检出 {len(all_records)} 条 straggler，"
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
