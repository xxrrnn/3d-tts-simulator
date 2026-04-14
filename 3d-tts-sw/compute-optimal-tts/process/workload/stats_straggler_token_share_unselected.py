#!/usr/bin/env python3
"""
对 ``identify_straggler.py`` 生成的 straggler JSON，在 **straggler 未被选中**
（``straggler_selected == false``）的记录上，统计 ``straggler_token_share`` 的最小值、最大值、均值、**中位数**。

每条 ``records`` 项对应一次 straggler 检出（某题某 step 某分支）；均值对所有满足条件的项求算术平均。

usage::

    python stats_straggler_token_share_unselected.py straggler/stragglers_AMC_1.5b_7b_40_2_1.json
    python stats_straggler_token_share_unselected.py --json straggler/stragglers_AMC_1.5b_7b_40_2_1.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _collect_shares(records: List[Dict[str, Any]]) -> List[float]:
    out: List[float] = []
    for r in records:
        if r.get("straggler_selected") is not False:
            continue
        v = r.get("straggler_token_share")
        if v is None:
            continue
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            continue
    return out


def _min_max_mean_median(
    values: List[float],
) -> Tuple[int, Optional[float], Optional[float], Optional[float], Optional[float]]:
    n = len(values)
    if n == 0:
        return 0, None, None, None, None
    return (
        n,
        min(values),
        max(values),
        sum(values) / n,
        statistics.median(values),
    )


def summarize_file(path: Path) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    records = data.get("records")
    if not isinstance(records, list):
        raise ValueError(f"{path}: 缺少 records 数组")

    shares = _collect_shares(records)
    n, vmin, vmax, vmean, vmedian = _min_max_mean_median(shares)

    return {
        "file": str(path.resolve()),
        "generated_by": data.get("generated_by"),
        "record_count": data.get("record_count"),
        "filter": "straggler_selected is false",
        "matching_step_count": n,
        "straggler_token_share_min": vmin,
        "straggler_token_share_max": vmax,
        "straggler_token_share_mean": round(vmean, 10) if vmean is not None else None,
        "straggler_token_share_median": round(vmedian, 10)
        if vmedian is not None
        else None,
    }


def _print_human(s: Dict[str, Any]) -> None:
    print(f"文件: {s['file']}")
    if s.get("generated_by"):
        print(f"来源: {s['generated_by']}")
    print(f"条件: {s['filter']}")
    print(f"满足条件的 step 条数: {s['matching_step_count']}")
    if s["matching_step_count"] == 0:
        print("straggler_token_share: min / max / mean / median = (无数据)")
        return
    print(
        f"straggler_token_share 最小值: {s['straggler_token_share_min']}"
    )
    print(
        f"straggler_token_share 最大值: {s['straggler_token_share_max']}"
    )
    print(
        f"straggler_token_share 均值:   {s['straggler_token_share_mean']}"
    )
    print(
        f"straggler_token_share 中位数: {s['straggler_token_share_median']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="未被选中的 straggler 上 straggler_token_share 的 min/max/mean/median",
    )
    parser.add_argument("json_files", nargs="+", type=Path, help="stragglers_*.json")
    parser.add_argument("--json", action="store_true", help="只输出 JSON")
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
