#!/usr/bin/env python3
"""
对 ``identify_straggler.py`` 生成的 straggler JSON（含 ``records`` 数组）按题目正误分组统计：

- **做对**：``majority_vote == 1``
- **做错**：``majority_vote != 1``（含 0 与缺失）

每组输出：

- straggler 记录总数（每条 record 即一次 straggler 检出）
- ``straggler_selected == true`` 的数量
- 未被选中的数量（``straggler_selected`` 不为 true，通常为 false）

usage::

    python summarize_straggler_by_majority.py straggler/stragglers_AMC_1.5b_7b_40_8_1.json
    python summarize_straggler_by_majority.py straggler/*.json
    python summarize_straggler_by_majority.py --json straggler/stragglers_AMC_1.5b_7b_40_8_1.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _is_correct(mv: Any) -> bool:
    return mv == 1


def _split_records(
    records: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    correct: List[Dict[str, Any]] = []
    wrong: List[Dict[str, Any]] = []
    for r in records:
        mv = r.get("majority_vote")
        if _is_correct(mv):
            correct.append(r)
        else:
            wrong.append(r)
    return correct, wrong


def _stats(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    total = len(rows)
    selected = sum(1 for r in rows if r.get("straggler_selected") is True)
    not_selected = total - selected
    return {
        "straggler_total": total,
        "straggler_selected_true": selected,
        "straggler_not_selected": not_selected,
    }


def summarize_file(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    records = data.get("records")
    if not isinstance(records, list):
        raise ValueError(f"{path}: 缺少 records 数组")

    correct_rows, wrong_rows = _split_records(records)
    return {
        "file": str(path.resolve()),
        "generated_by": data.get("generated_by"),
        "record_count_header": data.get("record_count"),
        "correct_majority_vote_1": _stats(correct_rows),
        "wrong_majority_vote_not_1": _stats(wrong_rows),
    }


def _print_human(summary: Dict[str, Any]) -> None:
    print(f"文件: {summary['file']}")
    if summary.get("generated_by"):
        print(f"来源: {summary['generated_by']}")
    for label, key in (
        ("做对 (majority_vote=1)", "correct_majority_vote_1"),
        ("做错 (majority_vote≠1)", "wrong_majority_vote_not_1"),
    ):
        s = summary[key]
        print(f"\n{label}")
        print(f"  straggler 总数:           {s['straggler_total']}")
        print(f"  被选中 (selected=true):   {s['straggler_selected_true']}")
        print(f"  未被选中:                 {s['straggler_not_selected']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="按 majority_vote 分组统计 straggler 条数与被选中情况",
    )
    parser.add_argument(
        "json_files",
        nargs="+",
        type=Path,
        help="stragglers_*.json 路径",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="只输出 JSON（每个文件一个对象；多文件时为数组）",
    )
    args = parser.parse_args()

    summaries: List[Dict[str, Any]] = []
    for p in args.json_files:
        p = p.resolve()
        if not p.is_file():
            print(f"跳过（非文件）: {p}", file=sys.stderr)
            continue
        try:
            summaries.append(summarize_file(p))
        except (OSError, json.JSONDecodeError, ValueError) as e:
            print(f"错误 {p}: {e}", file=sys.stderr)
            sys.exit(1)

    if not summaries:
        print("没有可处理的文件", file=sys.stderr)
        sys.exit(1)

    if args.json:
        out = summaries[0] if len(summaries) == 1 else summaries
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return

    for i, s in enumerate(summaries):
        if i:
            print()
        _print_human(s)


if __name__ == "__main__":
    main()
