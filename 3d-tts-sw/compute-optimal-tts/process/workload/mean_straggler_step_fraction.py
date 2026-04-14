#!/usr/bin/env python3
"""
对 ``identify_straggler.py`` 生成的 straggler JSON，按题目统计：

- **出现 straggler 的 step 数**：该题所有 straggler 记录里 **不同** ``step_index`` 的个数。
- **整道题 decode step 数**：对应 ``*_workload.json`` 中 ``decode.steps`` 的长度。
- **每题比例**：``steps_with_straggler / total_decode_steps``。

再对所有题目求 **比例的算术平均**。

默认若 JSON 含 ``scan_root`` 且目录存在，则对该目录下所有 ``*_workload.json`` 逐题计算（无 straggler 记录的题目比例为 0）；
否则仅对 ``records`` 中出现过的 ``question_id`` 求均值（见 ``--only-recorded-questions`` 说明）。

usage::

    python mean_straggler_step_fraction.py straggler/stragglers_AMC_1.5b_7b_40_2_1.json
    python mean_straggler_step_fraction.py --json straggler/stragglers_AMC_1.5b_7b_40_2_1.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


def _decode_step_count(workload: Dict[str, Any]) -> int:
    dec = workload.get("decode") or {}
    steps = dec.get("steps")
    if not isinstance(steps, list):
        return 0
    return len(steps)


def _load_workload(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _question_id_from_workload_name(name: str) -> Optional[str]:
    # question_0_workload.json -> question_0
    suffix = "_workload.json"
    if not name.endswith(suffix):
        return None
    return name[: -len(suffix)]


def _straggler_steps_by_question(
    records: List[Dict[str, Any]],
) -> Tuple[Dict[str, Set[int]], Dict[str, Path]]:
    """question_id -> set(step_index)；question_id -> 任意一条记录中的 workload 路径。"""
    by_steps: Dict[str, Set[int]] = defaultdict(set)
    path_by_q: Dict[str, Path] = {}
    for r in records:
        qid = r.get("question_id")
        if not qid:
            continue
        qid = str(qid)
        si = r.get("step_index")
        if si is not None:
            try:
                by_steps[qid].add(int(si))
            except (TypeError, ValueError):
                pass
        wp = r.get("workload_path")
        if wp and qid not in path_by_q:
            path_by_q[qid] = Path(str(wp))
    return dict(by_steps), path_by_q


def summarize_file(
    path: Path,
    *,
    all_questions_from_scan_root: bool = True,
) -> Dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    records = data.get("records")
    if not isinstance(records, list):
        raise ValueError(f"{path}: 缺少 records 数组")

    by_steps, path_by_q = _straggler_steps_by_question(records)
    scan_root = data.get("scan_root")
    scan_path: Optional[Path] = None
    if scan_root:
        scan_path = Path(str(scan_root))
        if not scan_path.is_dir():
            scan_path = None

    per_question: List[Dict[str, Any]] = []
    workload_paths: List[Path] = []

    if all_questions_from_scan_root and scan_path is not None:
        workload_paths = sorted(scan_path.glob("*_workload.json"))
        for wlp in workload_paths:
            qid = _question_id_from_workload_name(wlp.name)
            if not qid:
                continue
            wl = _load_workload(wlp)
            if wl is None:
                per_question.append(
                    {
                        "question_id": qid,
                        "error": "failed_to_load_workload",
                        "workload_path": str(wlp.resolve()),
                    }
                )
                continue
            total = _decode_step_count(wl)
            n_str = len(by_steps.get(qid, set()))
            ratio = (n_str / total) if total > 0 else None
            per_question.append(
                {
                    "question_id": qid,
                    "steps_with_straggler": n_str,
                    "total_decode_steps": total,
                    "fraction": round(ratio, 10) if ratio is not None else None,
                    "workload_path": str(wlp.resolve()),
                }
            )
        mode = "all_workloads_under_scan_root"
    else:
        for qid in sorted(by_steps.keys()):
            wlp = path_by_q.get(qid)
            if wlp is None:
                continue
            wl = _load_workload(wlp)
            if wl is None:
                per_question.append(
                    {
                        "question_id": qid,
                        "error": "failed_to_load_workload",
                        "workload_path": str(wlp),
                    }
                )
                continue
            total = _decode_step_count(wl)
            n_str = len(by_steps[qid])
            ratio = (n_str / total) if total > 0 else None
            per_question.append(
                {
                    "question_id": qid,
                    "steps_with_straggler": n_str,
                    "total_decode_steps": total,
                    "fraction": round(ratio, 10) if ratio is not None else None,
                    "workload_path": str(wlp.resolve()),
                }
            )
        mode = "only_questions_with_straggler_records"

    valid_fracs: List[float] = []
    for row in per_question:
        f = row.get("fraction")
        if isinstance(f, (int, float)):
            valid_fracs.append(float(f))

    mean_frac = sum(valid_fracs) / len(valid_fracs) if valid_fracs else None

    return {
        "file": str(path.resolve()),
        "generated_by": data.get("generated_by"),
        "scan_root": str(scan_path.resolve()) if scan_path else data.get("scan_root"),
        "mode": mode,
        "question_count": len(per_question),
        "mean_fraction_straggler_steps": round(mean_frac, 10)
        if mean_frac is not None
        else None,
        "per_question": per_question,
    }


def _print_human(s: Dict[str, Any]) -> None:
    print(f"文件: {s['file']}")
    print(f"模式: {s['mode']}")
    if s.get("scan_root"):
        print(f"scan_root: {s['scan_root']}")
    print(f"题目数: {s['question_count']}")
    print(
        f"各题「含 straggler 的 step 数 / decode 总 step 数」的均值: {s['mean_fraction_straggler_steps']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="每题 straggler 出现 step 占比，再对题目求平均",
    )
    parser.add_argument("json_files", nargs="+", type=Path, help="stragglers_*.json")
    parser.add_argument("--json", action="store_true", help="输出 JSON（含 per_question）")
    parser.add_argument(
        "--only-recorded-questions",
        action="store_true",
        help="仅对 records 中出现过的题目求均值（忽略 scan_root 下无 straggler 的题）",
    )
    args = parser.parse_args()

    summaries: List[Dict[str, Any]] = []
    for p in args.json_files:
        p = p.resolve()
        if not p.is_file():
            print(f"跳过（非文件）: {p}", file=sys.stderr)
            continue
        try:
            summaries.append(
                summarize_file(
                    p,
                    all_questions_from_scan_root=not args.only_recorded_questions,
                )
            )
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
        slim = {k: v for k, v in s.items() if k != "per_question"}
        _print_human(slim)
        print("(使用 --json 可查看每题明细 per_question)")


if __name__ == "__main__":
    main()
