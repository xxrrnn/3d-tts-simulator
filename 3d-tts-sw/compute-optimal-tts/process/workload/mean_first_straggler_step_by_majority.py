#!/usr/bin/env python3
"""
对 ``identify_straggler.py`` 生成的 straggler JSON，按题目正误分别统计「首次 straggler 所在 step」的均值。

定义（每个 question 只计一次）：

- 该题在 ``records`` 中所有 straggler 记录的 ``step_index`` 里取 **最小值**，视为该题 **首次出现 straggler 的 step**。
- **做对**：``majority_vote == 1`` 的题目集合上，对上述最小 step 求 **算术平均**（各题最小 step 之和 / 该集合题目数）。
- **做错**：``majority_vote != 1`` 的题目集合同理。

另：**首次被选择的 straggler** — 仅看 ``straggler_selected == true`` 的记录，每题取这些记录里 ``step_index`` 的 **最小值**（即首次出现「被选中的掉队分支」的 step），再在对做/做错子集上求均值。  
若无任何 ``straggler_selected`` 为 true 的题目，则该项为 ``null`` / 终端显示「无题目」。

说明：「首次 straggler」仅含「至少出现过一次 straggler」的题目；「首次被选中的 straggler」仅含「至少有一次被选中的 straggler」的题目。

usage::

    python mean_first_straggler_step_by_majority.py straggler/stragglers_AIME_1.5b_7b_40_2_1.json
    python mean_first_straggler_step_by_majority.py --json straggler/stragglers_AIME_1.5b_7b_40_2_1.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _first_step_per_question(
    records: List[Dict[str, Any]],
    *,
    correct: bool,
    selected_only: bool = False,
) -> Dict[str, int]:
    """question_id -> min(step_index)。

    ``selected_only``：为 True 时只统计 ``straggler_selected is True`` 的记录。
    """
    by_q: Dict[str, List[int]] = defaultdict(list)
    for r in records:
        if selected_only and r.get("straggler_selected") is not True:
            continue
        mv = r.get("majority_vote")
        if correct:
            if mv != 1:
                continue
        else:
            if mv == 1:
                continue
        qid = r.get("question_id")
        if not qid:
            continue
        si = r.get("step_index")
        if si is None:
            continue
        by_q[str(qid)].append(int(si))
    return {q: min(steps) for q, steps in by_q.items() if steps}


def _mean_stats(first_by_q: Dict[str, int]) -> Tuple[int, Optional[float], int]:
    """题目数、均值、各题首次 step 之和。"""
    if not first_by_q:
        return 0, None, 0
    firsts = list(first_by_q.values())
    s = sum(firsts)
    n = len(firsts)
    return n, s / n, s


def summarize_file(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    records = data.get("records")
    if not isinstance(records, list):
        raise ValueError(f"{path}: 缺少 records 数组")

    correct_first = _first_step_per_question(records, correct=True)
    wrong_first = _first_step_per_question(records, correct=False)

    correct_sel = _first_step_per_question(
        records, correct=True, selected_only=True
    )
    wrong_sel = _first_step_per_question(
        records, correct=False, selected_only=True
    )

    cn, cmean, csum = _mean_stats(correct_first)
    wn, wmean, wsum = _mean_stats(wrong_first)

    csn, csmean, cssum = _mean_stats(correct_sel)
    wsn, wsmean, wssum = _mean_stats(wrong_sel)

    def _block(
        n: int,
        sum_s: int,
        mean_v: Optional[float],
        n_sel: int,
        sum_sel: int,
        mean_sel: Optional[float],
    ) -> Dict[str, Any]:
        return {
            "question_count_with_straggler": n,
            "sum_first_straggler_step_index": sum_s,
            "mean_first_straggler_step_index": round(mean_v, 6)
            if mean_v is not None
            else None,
            "question_count_with_selected_straggler": n_sel,
            "sum_first_selected_straggler_step_index": sum_sel,
            "mean_first_selected_straggler_step_index": round(mean_sel, 6)
            if mean_sel is not None
            else None,
        }

    return {
        "file": str(path.resolve()),
        "generated_by": data.get("generated_by"),
        "workload_files_scanned": data.get("workload_files_scanned"),
        "record_count": data.get("record_count"),
        "correct_majority_vote_1": _block(cn, csum, cmean, csn, cssum, csmean),
        "wrong_majority_vote_not_1": _block(wn, wsum, wmean, wsn, wssum, wsmean),
    }


def _print_human(s: Dict[str, Any]) -> None:
    print(f"文件: {s['file']}")
    if s.get("generated_by"):
        print(f"来源: {s['generated_by']}")
    for label, key in (
        ("做对 (majority_vote=1)，且至少有一次 straggler 的题目", "correct_majority_vote_1"),
        ("做错 (majority_vote≠1)，且至少有一次 straggler 的题目", "wrong_majority_vote_not_1"),
    ):
        b = s[key]
        print(f"\n{label}")
        print(f"  题目数量(任一次 straggler): {b['question_count_with_straggler']}")
        if b["mean_first_straggler_step_index"] is None:
            print("  首次 straggler step 均值: (无题目)")
        else:
            print(
                f"  各题首次 straggler 的 step 之和: {b['sum_first_straggler_step_index']}"
            )
            print(
                f"  首次 straggler step 均值: {b['mean_first_straggler_step_index']}"
            )
        print(
            f"  题目数量(含被选中的 straggler): {b['question_count_with_selected_straggler']}"
        )
        if b["mean_first_selected_straggler_step_index"] is None:
            print("  首次「被选中」straggler 的 step 均值: (无题目)")
        else:
            print(
                f"  各题首次被选中 straggler 的 step 之和: {b['sum_first_selected_straggler_step_index']}"
            )
            print(
                f"  首次「被选中」straggler 的 step 均值: {b['mean_first_selected_straggler_step_index']}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="按 majority_vote 分组，统计每题首次 straggler / 首次被选中 straggler 的 step 均值",
    )
    parser.add_argument(
        "json_files",
        nargs="+",
        type=Path,
        help="stragglers_*.json 路径",
    )
    parser.add_argument("--json", action="store_true", help="只输出 JSON")
    args = parser.parse_args()

    out_list: List[Dict[str, Any]] = []
    for p in args.json_files:
        p = p.resolve()
        if not p.is_file():
            print(f"跳过（非文件）: {p}", file=sys.stderr)
            continue
        try:
            out_list.append(summarize_file(p))
        except (OSError, json.JSONDecodeError, ValueError) as e:
            print(f"错误 {p}: {e}", file=sys.stderr)
            sys.exit(1)

    if not out_list:
        print("没有可处理的文件", file=sys.stderr)
        sys.exit(1)

    if args.json:
        payload = out_list[0] if len(out_list) == 1 else out_list
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    for i, s in enumerate(out_list):
        if i:
            print()
        _print_human(s)


if __name__ == "__main__":
    main()
