#!/usr/bin/env python3
"""
对 ``identify_straggler.py`` 生成的 straggler JSON，给定 ``BREAK_POINT``，只考虑 ``step_index < BREAK_POINT`` 的记录，计算：

    被选中的 straggler 条数 / 该 step 范围内 straggler 总条数

即（所有题目中，满足 ``step_index < BREAK_POINT`` 且 ``straggler_selected is True`` 的记录数）
除以（所有题目中，满足 ``step_index < BREAK_POINT`` 的记录数）。

说明：JSON 中每条 ``records`` 即一次 straggler 事件；分母为「早期 step 上检出的 straggler 次数」之和。

默认 ``BREAK_POINT=2`` 表示只统计 ``step_index`` 为 0、1 的记录。

usage::

    python straggler_selected_ratio_below_step.py straggler/stragglers_AIME_1.5b_1.5b_40_2_1.json
    python straggler_selected_ratio_below_step.py --break-point 3 straggler/stragglers_AIME_1.5b_1.5b_40_2_1.json
    python straggler_selected_ratio_below_step.py --json straggler/stragglers_AIME_1.5b_1.5b_40_2_1.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def summarize_file(path: Path, break_point: int) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    records = data.get("records")
    if not isinstance(records, list):
        raise ValueError(f"{path}: 缺少 records 数组")

    below_total = 0
    below_selected = 0
    for r in records:
        si = r.get("step_index")
        if si is None:
            continue
        try:
            step = int(si)
        except (TypeError, ValueError):
            continue
        if step >= break_point:
            continue
        below_total += 1
        if r.get("straggler_selected") is True:
            below_selected += 1

    ratio: Optional[float]
    if below_total == 0:
        ratio = None
    else:
        ratio = below_selected / below_total

    return {
        "file": str(path.resolve()),
        "generated_by": data.get("generated_by"),
        "break_point": break_point,
        "step_condition": f"step_index < {break_point}",
        "straggler_count_below_breakpoint": below_total,
        "selected_straggler_count_below_breakpoint": below_selected,
        "ratio_selected_over_all_stragglers": round(ratio, 10) if ratio is not None else None,
    }


def _print_human(s: Dict[str, Any]) -> None:
    print(f"文件: {s['file']}")
    if s.get("generated_by"):
        print(f"来源: {s['generated_by']}")
    print(f"BREAK_POINT: {s['break_point']}（{s['step_condition']}）")
    print(f"step 范围内的 straggler 条数(分母): {s['straggler_count_below_breakpoint']}")
    print(
        f"其中被选中的 straggler 条数(分子): {s['selected_straggler_count_below_breakpoint']}"
    )
    r = s["ratio_selected_over_all_stragglers"]
    if r is None:
        print("比值: (分母为 0，无符合条件的记录)")
    else:
        print(f"比值 (选中/全部): {r}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="step < BREAK_POINT 时，被选中 straggler 数 / 该区域 straggler 总数",
    )
    parser.add_argument(
        "json_files",
        nargs="+",
        type=Path,
        help="stragglers_*.json",
    )
    parser.add_argument(
        "--break-point",
        type=int,
        default=2,
        metavar="N",
        help="只统计 step_index < N，默认 2",
    )
    parser.add_argument("--json", action="store_true", help="只输出 JSON")
    args = parser.parse_args()

    if args.break_point < 0:
        print("BREAK_POINT 须 >= 0", file=sys.stderr)
        sys.exit(1)

    out_list: List[Dict[str, Any]] = []
    for p in args.json_files:
        p = p.resolve()
        if not p.is_file():
            print(f"跳过（非文件）: {p}", file=sys.stderr)
            continue
        try:
            out_list.append(summarize_file(p, args.break_point))
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
