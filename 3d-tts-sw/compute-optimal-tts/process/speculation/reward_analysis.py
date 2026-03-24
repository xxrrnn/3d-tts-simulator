#!/usr/bin/env python3
"""
分析分支选择 index 的历史偏置（基于 reward workload 结果）。

关注点：
1) baseline: 每个 index=n 被选中的总体概率
2) streak bias: 连续之前都选中 index=n 时，当前继续选中 n 的概率是否更高
3) all-same history: 前 k 个 eligible step 全是 index=n 条件下，当前继续选 n 的概率
4) 一阶转移：P(curr=n | prev=n) 与 P(curr=n | prev!=n)

注：仅统计 branch_count > 1 的 step。
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning("Failed to read %s: %s", path, e)
        return None


def rate(a: int, b: int) -> float:
    return float(a / b) if b else 0.0


def _init_counter() -> Dict[str, int]:
    return {"total": 0, "hit": 0}


def analyze(files: List[Path], max_k: int) -> Dict[str, Any]:
    skipped_single_branch = 0

    # 全局 eligible step
    total_steps = 0

    # 按 index 汇总
    index_counts: Dict[int, int] = {}
    transition_same: Dict[int, Dict[str, int]] = {}
    transition_diff: Dict[int, Dict[str, int]] = {}
    streak_stats: Dict[int, Dict[int, Dict[str, int]]] = {}
    all_same_k: Dict[int, Dict[int, Dict[str, int]]] = {}

    for fp in files:
        obj = load_json(fp)
        if obj is None:
            continue

        outputs = obj.get("decode", {}).get("outputs", [])
        for out in outputs:
            steps = out.get("step_branch_metrics", [])
            eligible_selected: List[int] = []

            for s in steps:
                if int(s.get("branch_count", 0)) <= 1:
                    skipped_single_branch += 1
                    continue

                sel = int(s.get("selected_branch_index", -1))
                if sel < 0:
                    continue

                total_steps += 1
                index_counts[sel] = index_counts.get(sel, 0) + 1

                # 当前 step 出现的分支 index（作为候选 n）
                branch_indices = [int(b.get("branch_index", i)) for i, b in enumerate(s.get("branches", []))]

                # 一阶转移：对每个 n 统计 P(curr=n | prev=n) 和 P(curr=n | prev!=n)
                if eligible_selected:
                    prev = eligible_selected[-1]
                    for n in branch_indices:
                        if n not in transition_same:
                            transition_same[n] = _init_counter()
                        if n not in transition_diff:
                            transition_diff[n] = _init_counter()
                        if prev == n:
                            transition_same[n]["total"] += 1
                            if sel == n:
                                transition_same[n]["hit"] += 1
                        else:
                            transition_diff[n]["total"] += 1
                            if sel == n:
                                transition_diff[n]["hit"] += 1

                # streak / all-same-k：对每个 n 分别统计
                for n in branch_indices:
                    # 连续前序 n 的长度
                    streak = 0
                    i = len(eligible_selected) - 1
                    while i >= 0 and eligible_selected[i] == n:
                        streak += 1
                        i -= 1

                    if n not in streak_stats:
                        streak_stats[n] = {}
                    if streak not in streak_stats[n]:
                        streak_stats[n][streak] = _init_counter()
                    streak_stats[n][streak]["total"] += 1
                    if sel == n:
                        streak_stats[n][streak]["hit"] += 1

                    if n not in all_same_k:
                        all_same_k[n] = {k: _init_counter() for k in range(1, max_k + 1)}
                    for k in range(1, max_k + 1):
                        if len(eligible_selected) >= k and all(x == n for x in eligible_selected[-k:]):
                            all_same_k[n][k]["total"] += 1
                            if sel == n:
                                all_same_k[n][k]["hit"] += 1

                eligible_selected.append(sel)

    index_summary: Dict[str, Any] = {}
    all_seen_indices = sorted(set(index_counts.keys()) | set(transition_same.keys()) | set(streak_stats.keys()) | set(all_same_k.keys()))
    for n in all_seen_indices:
        baseline_cnt = index_counts.get(n, 0)
        same = transition_same.get(n, _init_counter())
        diff = transition_diff.get(n, _init_counter())

        streak_out = {}
        for k_streak, v in sorted(streak_stats.get(n, {}).items(), key=lambda x: x[0]):
            streak_out[str(k_streak)] = {
                "total": v["total"],
                "hit": v["hit"],
                "p_curr_eq_n": rate(v["hit"], v["total"]),
            }

        all_same_out = {}
        for k, v in sorted(all_same_k.get(n, {}).items(), key=lambda x: x[0]):
            all_same_out[str(k)] = {
                "total": v["total"],
                "hit": v["hit"],
                "p_curr_eq_n_given_prev_k_all_n": rate(v["hit"], v["total"]),
            }

        index_summary[str(n)] = {
            "baseline_count": baseline_cnt,
            "baseline_p_selected_n": rate(baseline_cnt, total_steps),
            "transition_p_curr_n_given_prev_n": rate(same["hit"], same["total"]),
            "transition_p_curr_n_given_prev_not_n": rate(diff["hit"], diff["total"]),
            "transition_count_prev_n": same["total"],
            "transition_count_prev_not_n": diff["total"],
            "streak_stats": streak_out,
            "all_same_history_stats": all_same_out,
        }

    summary = {
        "eligible_steps_total": total_steps,
        "skipped_single_branch_steps": skipped_single_branch,
        "index_summary": index_summary,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze reward/index selection bias from speculation outputs")
    parser.add_argument(
        "--input",
        default="/DISK1/data/rnxu_24/Paper/3d-tts-simulator/3d-tts-sw/compute-optimal-tts/process/speculation/output",
        help="Directory containing speculation output json files",
    )
    parser.add_argument("--max-k", type=int, default=6, help="Max history length k for all-zero condition")
    parser.add_argument(
        "--output",
        default="/DISK1/data/rnxu_24/Paper/3d-tts-simulator/3d-tts-sw/compute-optimal-tts/process/speculation/output/reward_analysis_result.json",
        help="Output analysis json file",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    in_root = Path(args.input)
    files = sorted(p for p in in_root.rglob("record_0.json") if p.is_file())
    logger.info("Found %d workload files (record_0.json)", len(files))

    result = analyze(files, max_k=args.max_k)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    logger.info("Saved analysis to %s", out_path)
    logger.info("Saved index-wise reward selection analysis.")


if __name__ == "__main__":
    main()

