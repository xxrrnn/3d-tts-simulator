#!/usr/bin/env python3
"""
从 ``straggler/*.json``（``identify_straggler.py`` 输出）中按题号与 ``majority_vote`` 筛选记录，
并只保留指定字段（不含 ``straggler_token_topk_logprobs`` 等大字段）。

默认：题号属于 ``{0,1,3,6,8,14,16}`` 且 ``majority_vote == 1``。

JSON 内 ``workload_path`` 等为数据自带绝对路径，本脚本只读写 straggler 汇总 JSON，不修改这些字段。

usage::

    # 默认读同目录下 straggler/straggler_AIME_1.5_100.json，筛后写入 straggler/ 下 *_filtered.json
    python filter_straggler_records.py

    python filter_straggler_records.py \
        --input ./straggler/straggler_AIME_1.5_100.json \
        -o ./straggler/straggler_AIME_1.5_100_filtered.json
    python filter_straggler_records.py --question-ids 0,1,3 --input /path/to/stragglers.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

_SCRIPT_DIR = Path(__file__).resolve().parent
_STRAGGLER_DIR = _SCRIPT_DIR / "straggler"
_DEFAULT_INPUT_JSON = _STRAGGLER_DIR / "straggler_AIME_1.5_100.json"

_DEFAULT_QUESTION_IDS = (0, 1, 3, 6, 8, 14, 16)

_KEEP_KEYS = (
    "workload_path",
    "question_id",
    "majority_vote",
    "step_index",
    "branch_index",
    "straggler_tokens",
    "max_other_tokens",
    "branch_tokens",
    "step_total_tokens",
    "straggler_token_share",
    "branch_reward",
    "straggler_reward",
    "selected_branch_index",
    "straggler_selected",
)


def parse_question_index(question_id: Any) -> Optional[int]:
    if not isinstance(question_id, str):
        return None
    m = re.match(r"^question_(\d+)$", question_id.strip())
    if not m:
        return None
    return int(m.group(1))


def project_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    return {k: rec[k] for k in _KEEP_KEYS if k in rec}


def main() -> None:
    parser = argparse.ArgumentParser(description="按题号与 majority_vote 筛选 straggler JSON 记录")
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=_DEFAULT_INPUT_JSON,
        help=(
            "identify_straggler.py 生成的 JSON；"
            f"默认: {_DEFAULT_INPUT_JSON}（即本脚本所在目录下的 straggler/）"
        ),
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="输出 JSON（默认：与输入同目录，文件名为 <input stem>_filtered.json）",
    )
    parser.add_argument(
        "--question-ids",
        type=str,
        default=",".join(str(x) for x in _DEFAULT_QUESTION_IDS),
        help=f"逗号分隔题号，默认: {','.join(map(str, _DEFAULT_QUESTION_IDS))}",
    )
    args = parser.parse_args()
    inp: Path = args.input
    out: Path = args.output if args.output else inp.with_name(f"{inp.stem}_filtered.json")

    id_set: Set[int] = set()
    for part in args.question_ids.split(","):
        part = part.strip()
        if not part:
            continue
        id_set.add(int(part))
    if not id_set:
        print("题号集合为空", file=sys.stderr)
        sys.exit(1)

    if not inp.exists():
        print(
            f"文件不存在: {inp}\n"
            f"请用 --input 指定 straggler 目录下的 JSON（例如 {_STRAGGLER_DIR}/straggler_AIME_1.5_100.json）。",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        with open(inp, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"无法读取 JSON: {e}", file=sys.stderr)
        sys.exit(1)

    records = data.get("records")
    if not isinstance(records, list):
        print("顶层缺少 records 数组", file=sys.stderr)
        sys.exit(1)

    out_recs: List[Dict[str, Any]] = []
    for rec in records:
        if not isinstance(rec, dict):
            continue
        qidx = parse_question_index(rec.get("question_id"))
        if qidx is None or qidx not in id_set:
            continue
        mv = rec.get("majority_vote")
        if mv != 1:
            continue
        out_recs.append(project_record(rec))

    report: Dict[str, Any] = {
        "generated_by": "filter_straggler_records.py",
        "source": str(inp.resolve()),
        "filter": {
            "question_indices": sorted(id_set),
            "majority_vote": 1,
            "kept_keys": list(_KEEP_KEYS),
        },
        "input_record_count": len(records),
        "output_record_count": len(out_recs),
        "records": out_recs,
    }

    out = out.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
    except OSError as e:
        print(f"无法写入 {out}: {e}", file=sys.stderr)
        sys.exit(1)

    print(
        f"从 {len(records)} 条输入中筛出 {len(out_recs)} 条，"
        f"已写入 {out}"
    )


if __name__ == "__main__":
    main()
