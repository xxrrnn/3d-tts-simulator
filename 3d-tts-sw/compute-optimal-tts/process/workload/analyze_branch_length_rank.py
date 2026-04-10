#!/usr/bin/env python3
"""
0410 cursor写的脚本：
从 workload JSON 分析：各 step 被选中的 branch 在「按 branch_tokens 长度排序」中的名次，
以及相邻两个 step（均满足 branch_count >= 3）上，当前步与上一步选中分支长度名次是否有关联。

长度名次（1-based，并列取平均秩，与 scipy rankdata average 一致）：
- rank_asc：按长度升序，1=最短，B=最长
- norm_rank：归一化到 [0,1]，(rank_asc - 1) / (B - 1)，B=1 时记为 0.5

用法::

    python analyze_branch_length_rank.py --input ../model_workloads
    python analyze_branch_length_rank.py --input /path/to/AIME24_beam_search/.../40_8_1_straggler_0_0_0_0_0_def0
    python analyze_branch_length_rank.py --input question_8_workload.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

_DEFAULT_ROOT = Path(__file__).resolve().parent / "model_workloads"


def _rankdata_average_1based(values: Sequence[float]) -> List[float]:
    """Average ranks for ties, 1-based, same idea as scipy.stats.rankdata(..., method='average')."""
    n = len(values)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        v0 = values[order[i]]
        while j + 1 < n and values[order[j + 1]] == v0:
            j += 1
        # positions i..j in sorted order share the same length
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _length_rank_for_step(step: Dict[str, Any]) -> Optional[Tuple[float, float, int]]:
    """
    Returns (rank_asc_1based, norm_rank_in_0_1, branch_count) or None if invalid.
    norm_rank: 0=最短, 1=最长；单分支为 0.5。
    """
    tokens = step.get("branch_tokens")
    if not isinstance(tokens, list) or len(tokens) == 0:
        return None
    sel = step.get("selected_branch_index")
    if not isinstance(sel, int) or sel < 0 or sel >= len(tokens):
        return None
    b = len(tokens)
    nums = [float(x) for x in tokens]
    ranks = _rankdata_average_1based(nums)
    r = ranks[sel]
    if b <= 1:
        norm = 0.5
    else:
        norm = (r - 1.0) / (b - 1.0)
    return (r, norm, b)


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    n = len(xs)
    if n != len(ys) or n < 2:
        return float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:
        return float("nan")
    sxy = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    return sxy / math.sqrt(sxx * syy)


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Spearman = Pearson on rank-transformed data (average ranks for ties)."""
    if len(xs) != len(ys) or len(xs) < 2:
        return float("nan")
    rx = _rankdata_average_1based(list(xs))
    ry = _rankdata_average_1based(list(ys))
    return _pearson(rx, ry)


def _iter_workload_files(root: Path) -> Iterable[Path]:
    if root.is_file():
        if root.name.endswith("_workload.json"):
            yield root
        return
    yield from sorted(root.rglob("*_workload.json"))


def analyze_file(path: Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    decode = data.get("decode") or {}
    steps = decode.get("steps") or []
    out: Dict[str, Any] = {
        "path": str(path),
        "steps": [],
        "pairs_ge3": [],
    }
    prev: Optional[Dict[str, Any]] = None
    for st in steps:
        bc = int(st.get("branch_count", 0) or 0)
        lr = _length_rank_for_step(st)
        if lr is None:
            continue
        rank_asc, norm, b = lr
        step_idx = st.get("step")
        rec = {
            "step_index": step_idx,
            "branch_count": b,
            "rank_asc": rank_asc,
            "norm_rank": norm,
        }
        out["steps"].append(rec)
        if prev is not None and prev["branch_count"] >= 3 and b >= 3:
            out["pairs_ge3"].append(
                {
                    "prev_step": prev["step_index"],
                    "curr_step": step_idx,
                    "prev_norm": prev["norm_rank"],
                    "curr_norm": norm,
                    "prev_rank_asc": prev["rank_asc"],
                    "curr_rank_asc": rank_asc,
                    "prev_B": prev["branch_count"],
                    "curr_B": b,
                }
            )
        prev = rec
    return out


def _quantiles(sorted_vals: List[float], qs: Tuple[float, ...]) -> Dict[str, float]:
    if not sorted_vals:
        return {f"p{int(q*100)}": float("nan") for q in qs}
    n = len(sorted_vals)
    out = {}
    for q in qs:
        pos = q * (n - 1)
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            v = sorted_vals[lo]
        else:
            v = sorted_vals[lo] * (hi - pos) + sorted_vals[hi] * (pos - lo)
        out[f"p{int(q*100)}"] = v
    return out


def _print_report(
    per_file: List[Dict[str, Any]],
    min_branches: int,
) -> None:
    # Aggregate: all steps with branch_count >= min_branches
    norms: List[float] = []
    by_step_idx: Dict[Any, List[float]] = defaultdict(list)
    bucket_bc: Dict[str, List[float]] = defaultdict(list)

    pairs_prev: List[float] = []
    pairs_curr: List[float] = []

    for fd in per_file:
        for s in fd["steps"]:
            if s["branch_count"] < min_branches:
                continue
            norms.append(s["norm_rank"])
            by_step_idx[s["step_index"]].append(s["norm_rank"])
            b = s["branch_count"]
            key = f"B={b}"
            bucket_bc[key].append(s["norm_rank"])
        for p in fd["pairs_ge3"]:
            pairs_prev.append(p["prev_norm"])
            pairs_curr.append(p["curr_norm"])

    print("=== 选中分支在「按长度升序」中的归一化名次 norm_rank（0=最短，1=最长）===\n")
    print(
        f"统计条件：branch_count >= {min_branches}；"
        f"并列长度使用平均秩；文件数={len(per_file)}，"
        f"符合条件 step 数={len(norms)}\n"
    )

    if norms:
        sn = sorted(norms)
        qd = _quantiles(sn, (0.0, 0.25, 0.5, 0.75, 1.0))
        print("整体分布（norm_rank）：")
        print(f"  min={sn[0]:.4f}  {'  '.join(f'{k}={v:.4f}' for k, v in qd.items())}  max={sn[-1]:.4f}")
        print(f"  mean={sum(norms)/len(norms):.4f}")
        # 粗分桶：短/中/长
        short = sum(1 for x in norms if x <= 1.0 / 3.0)
        mid = sum(1 for x in norms if 1.0 / 3.0 < x < 2.0 / 3.0)
        long_ = sum(1 for x in norms if x >= 2.0 / 3.0)
        tot = len(norms)
        print(
            f"  三分位粗占比：偏短 [0,1/3] {short/tot*100:.1f}%  "
            f"中 (1/3,2/3) {mid/tot*100:.1f}%  偏长 [2/3,1] {long_/tot*100:.1f}%"
        )
    else:
        print("（无符合条件 step）")

    print("\n--- 按 decode 内 step 序号（字段 step）分组的平均 norm_rank（样本少时仅供参考）---\n")
    for si in sorted(by_step_idx.keys(), key=lambda x: (x is None, x)):
        xs = by_step_idx[si]
        m = sum(xs) / len(xs)
        print(f"  step={si!r}: n={len(xs):5d}  mean_norm={m:.4f}")

    print("\n--- 按 branch 数分桶的 mean norm_rank ---\n")
    for key in sorted(bucket_bc.keys(), key=lambda s: int(s.split("=")[1]) if "=" in s else 0):
        xs = bucket_bc[key]
        m = sum(xs) / len(xs)
        print(f"  {key}: n={len(xs):5d}  mean_norm={m:.4f}")

    print(
        "\n=== 相邻 step 关联（仅上一 step 与当前 step 均 branch_count>=3；pairs_ge3）===\n"
    )
    n_pairs = len(pairs_prev)
    print(f"相邻对数量：{n_pairs}\n")
    if n_pairs >= 2:
        sp = _spearman(pairs_prev, pairs_curr)
        pe = _pearson(pairs_prev, pairs_curr)
        print(f"Spearman(上一步 norm_rank, 当前步 norm_rank) = {sp:.4f}")
        print(f"Pearson(上一步 norm_rank, 当前步 norm_rank)  = {pe:.4f}")
        # 离散联合：把 norm 分成 5 箱
        bins = 5

        def bin5(x: float) -> int:
            return min(int(x * bins), bins - 1)

        joint = Counter()
        for a, b in zip(pairs_prev, pairs_curr):
            joint[(bin5(a), bin5(b))] += 1
        print(f"\n联合分布（norm_rank 各分 {bins} 档，索引 0=最短区 … 4=最长区），行=上一步，列=当前步：")
        header = "       " + "".join(f"  c{j}" for j in range(bins))
        print(header)
        for i in range(bins):
            row = f"  r{i}  "
            for j in range(bins):
                row += f" {joint[(i, j)]:5d}"
            print(row)
    elif n_pairs == 1:
        print("仅 1 对相邻，无法计算相关系数。")
    else:
        print("无符合条件的相邻对。")

    print()


def main() -> int:
    parser = argparse.ArgumentParser(description="分析 workload 中选中分支的长度排序与相邻步关联")
    parser.add_argument(
        "--input",
        type=Path,
        default=_DEFAULT_ROOT,
        help=f"model_workloads 根目录、子目录或单个 *_workload.json（默认：{_DEFAULT_ROOT}）",
    )
    parser.add_argument(
        "--min-branches",
        type=int,
        default=3,
        help="单步分布统计时最少分支数（默认 3，与「>=3 分支」分析一致）",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="可选：将聚合后的原始列表写入 JSON",
    )
    args = parser.parse_args()
    root = args.input.resolve()
    if not root.exists():
        logger.error("路径不存在：%s", root)
        return 1

    files = list(_iter_workload_files(root))
    if not files:
        logger.error("未找到 *_workload.json：%s", root)
        return 1

    per_file: List[Dict[str, Any]] = []
    for p in files:
        try:
            per_file.append(analyze_file(p))
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("跳过 %s：%s", p, e)

    _print_report(per_file, min_branches=args.min_branches)

    if args.json_out:
        payload = {
            "input": str(root),
            "min_branches": args.min_branches,
            "files": per_file,
        }
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        logger.info("已写入 %s", args.json_out)

    return 0


if __name__ == "__main__":
    sys.exit(main())
