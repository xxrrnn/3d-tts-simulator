#!/usr/bin/env python3
"""
0409： cursor写的脚本，用来统计worklaod中一种配置下做对/做错的题目，run1、run2、run3取并集。
从 ``model_workloads`` 下 ``40_*_*_straggler_*_run{1,2,3}`` 目录中的 ``*_workload.json`` 统计
「做对 / 做错」的题目序号；对 run1、run2、run3 分别读入后，对「对」「错」各自取题目序号的并集。

判定：``result.majority_vote`` — ``1`` 为做对，``0`` 为做错；其它值记入 ``ambiguous``。

分组键：同一数据集、同一 LLM、同一 PRM、同一配置前缀（去掉目录名末尾 ``_run[123]``），
即 ``40_x_y_straggler_...`` 在三种 run 下合并统计。

路径约定与 ``identify_straggler.py`` 一致：默认 ``--root`` 为脚本同目录下 ``model_workloads``。

usage::

    python union_straggler_correctness.py
    python union_straggler_correctness.py --root ./model_workloads/AIME24_beam_search
    python union_straggler_correctness.py -o ./straggler/correctness_union.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple

_WORKLOAD_ROOT = Path(__file__).resolve().parent / "model_workloads"
_DEFAULT_OUTPUT = Path(__file__).resolve().parent / "straggler" / "correctness_union.json"

# 例如 40_4_1_straggler_1_1.5_100_0_0_def1_run2
_CONFIG_DIR_RE = re.compile(
    r"^40_\d+_\d+_straggler_.+_run(?P<run>[123])$"
)


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


def parse_question_index(workload: Dict[str, Any], path: Path) -> Optional[int]:
    qid = workload.get("question_id")
    if isinstance(qid, str) and qid.startswith("question_"):
        try:
            return int(qid.split("_", 1)[1])
        except (IndexError, ValueError):
            pass
    m = re.match(r"^question_(\d+)_workload\.json$", path.name)
    if m:
        return int(m.group(1))
    return None


def resolve_relative_to_preferred(file_path: Path, preferred_root: Path, fallback_root: Path) -> Path:
    resolved = file_path.resolve()
    for base in (preferred_root.resolve(), fallback_root.resolve()):
        try:
            return resolved.relative_to(base)
        except ValueError:
            continue
    raise ValueError(f"无法将 {file_path} 相对于 workload 根或扫描根解析")


def extract_group(
    file_path: Path,
    *,
    workload_root: Path,
    scan_root: Path,
) -> Optional[Tuple[str, str, str, str, int]]:
    """
    返回 (dataset, llm, prm, config_base, run) ；不满足过滤条件时返回 None。
    """
    rel = resolve_relative_to_preferred(file_path, workload_root, scan_root)
    parts = rel.parts
    if len(parts) < 5:
        return None
    dataset, llm, prm, config_dir = parts[0], parts[1], parts[2], parts[3]
    m = _CONFIG_DIR_RE.match(config_dir)
    if not m:
        return None
    run = int(m.group("run"))
    # 目录名以 _run1 / _run2 / _run3 结尾（各 5 字符）
    config_base = config_dir[:-5]
    return dataset, llm, prm, config_base, run


def majority_label(workload: Dict[str, Any]) -> Optional[int]:
    result = workload.get("result") or {}
    mv = result.get("majority_vote")
    if mv is None:
        return None
    if mv in (0, 1, True, False):
        return 1 if bool(mv) else 0
    try:
        v = int(mv)
        if v in (0, 1):
            return v
    except (TypeError, ValueError):
        pass
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="统计 40_*_straggler_* 各 run 做对/做错题目序号的并集",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=_WORKLOAD_ROOT,
        help=f"扫描目录或单个 *_workload.json；默认: {_WORKLOAD_ROOT}",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help=f"输出 JSON；默认: {_DEFAULT_OUTPUT}",
    )
    args = parser.parse_args()
    scan_root: Path = args.root
    output_path: Path = args.output

    if not scan_root.exists():
        print(f"路径不存在: {scan_root}", file=sys.stderr)
        sys.exit(1)

    # key -> { "correct": set, "wrong": set, "ambiguous": set, "runs": set }
    groups: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}

    files_seen = 0
    skipped_filter = 0
    for path in iter_workload_json_files(scan_root):
        ctx = extract_group(path, workload_root=_WORKLOAD_ROOT, scan_root=scan_root)
        if ctx is None:
            skipped_filter += 1
            continue
        dataset, llm, prm, config_base, run = ctx
        data = load_workload(path)
        if data is None:
            continue
        files_seen += 1
        qidx = parse_question_index(data, path)
        if qidx is None:
            continue
        label = majority_label(data)
        key = (dataset, llm, prm, config_base)
        if key not in groups:
            groups[key] = {
                "correct": set(),
                "wrong": set(),
                "ambiguous": set(),
                "runs": set(),
            }
        g = groups[key]
        g["runs"].add(run)
        if label == 1:
            g["correct"].add(qidx)
        elif label == 0:
            g["wrong"].add(qidx)
        else:
            g["ambiguous"].add(qidx)

    out_groups: List[Dict[str, Any]] = []
    for (dataset, llm, prm, config_base), g in sorted(groups.items()):
        correct: Set[int] = g["correct"]
        wrong: Set[int] = g["wrong"]
        ambiguous: Set[int] = g["ambiguous"]
        both = sorted(correct & wrong)
        out_groups.append(
            {
                "dataset": dataset,
                "llm": llm,
                "prm": prm,
                "config_base": config_base,
                "runs_present": sorted(g["runs"]),
                "correct_question_indices": sorted(correct),
                "wrong_question_indices": sorted(wrong),
                "ambiguous_question_indices": sorted(ambiguous),
                "counts": {
                    "correct_union": len(correct),
                    "wrong_union": len(wrong),
                    "ambiguous": len(ambiguous),
                    "in_both_correct_and_wrong_union": len(both),
                },
                "overlap_correct_and_wrong_indices": both,
            }
        )

    report: Dict[str, Any] = {
        "generated_by": "union_straggler_correctness.py",
        "scan_root": str(scan_root.resolve()),
        "workload_root": str(_WORKLOAD_ROOT.resolve()),
        "workload_files_matched": files_seen,
        "paths_skipped_by_name_pattern": skipped_filter,
        "group_count": len(out_groups),
        "groups": out_groups,
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
        f"已匹配 {files_seen} 个 workload 文件，"
        f"{len(out_groups)} 个 (数据集, 模型组合, 40_*_straggler_* 配置) 分组，"
        f"已写入 {output_path}"
    )


if __name__ == "__main__":
    main()
