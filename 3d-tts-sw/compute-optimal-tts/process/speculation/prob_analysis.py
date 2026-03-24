#!/usr/bin/env python3
"""
分析分支 token probability pattern 与“分支是否被选中”的相关性。

核心思路：
1) 对每个 step（仅 branch_count > 1）取所有分支的 token_probs
2) 去掉所有分支共同前缀 token（common prefix）
3) 计算每个分支特征：
   - next_n_mean_prob: 去前缀后前 n 个 token 概率均值
   - last_prob: 该分支最后一个 token 概率
4) 用 top-1 命中率评估这些特征是否能预测 selected 分支
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning("Failed to read %s: %s", path, e)
        return None


def common_prefix_len(seqs: List[List[float]]) -> int:
    if not seqs:
        return 0
    min_len = min(len(s) for s in seqs)
    pref = 0
    for i in range(min_len):
        v = seqs[0][i]
        if all(len(s) > i and s[i] == v for s in seqs[1:]):
            pref += 1
        else:
            break
    return pref


def safe_mean(values: List[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def top1_match(scores: List[float], selected_idx: int) -> bool:
    if not scores or selected_idx < 0 or selected_idx >= len(scores):
        return False
    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    return best_idx == selected_idx


def analyze_steps(files: List[Path], n_tokens: int) -> Dict[str, Any]:
    total_steps = 0
    skipped_single_branch = 0
    valid_selected_steps = 0

    hit_next_n = 0
    hit_last = 0
    hit_combined = 0

    selected_next_n_vals: List[float] = []
    non_selected_next_n_vals: List[float] = []
    selected_last_vals: List[float] = []
    non_selected_last_vals: List[float] = []

    per_step_records: List[Dict[str, Any]] = []

    for fp in files:
        obj = load_json(fp)
        if obj is None:
            continue
        outputs = obj.get("decode", {}).get("outputs", [])
        for out in outputs:
            steps = out.get("step_branch_metrics", [])
            for s in steps:
                branch_count = int(s.get("branch_count", 0))
                if branch_count <= 1:
                    skipped_single_branch += 1
                    continue
                total_steps += 1

                selected_idx = int(s.get("selected_branch_index", -1))
                branches = s.get("branches", [])
                token_probs_lists = [b.get("token_probs", []) for b in branches]
                pref_len = common_prefix_len(token_probs_lists)

                next_n_scores: List[float] = []
                last_scores: List[float] = []
                combined_scores: List[float] = []

                for b in branches:
                    tps = b.get("token_probs", [])
                    trimmed = tps[pref_len:] if pref_len < len(tps) else []
                    head = trimmed[:n_tokens] if n_tokens > 0 else trimmed
                    next_n_mean = safe_mean(head)
                    last_prob = float(tps[-1]) if tps else 0.0
                    next_n_scores.append(next_n_mean)
                    last_scores.append(last_prob)
                    combined_scores.append((next_n_mean + last_prob) / 2.0)

                if 0 <= selected_idx < len(branches):
                    valid_selected_steps += 1
                    hit_next_n += int(top1_match(next_n_scores, selected_idx))
                    hit_last += int(top1_match(last_scores, selected_idx))
                    hit_combined += int(top1_match(combined_scores, selected_idx))

                    for i, (x_next, x_last) in enumerate(zip(next_n_scores, last_scores)):
                        if i == selected_idx:
                            selected_next_n_vals.append(x_next)
                            selected_last_vals.append(x_last)
                        else:
                            non_selected_next_n_vals.append(x_next)
                            non_selected_last_vals.append(x_last)

                per_step_records.append(
                    {
                        "file": str(fp),
                        "path_idx": out.get("path_idx", 0),
                        "step": int(s.get("step", -1)),
                        "branch_count": branch_count,
                        "selected_branch_index": selected_idx,
                        "common_prefix_len": pref_len,
                        "next_n_scores": next_n_scores,
                        "last_scores": last_scores,
                    }
                )

    def rate(x: int, y: int) -> float:
        return float(x / y) if y else 0.0

    summary = {
        "n_tokens": n_tokens,
        "total_multi_branch_steps": total_steps,
        "skipped_single_branch_steps": skipped_single_branch,
        "valid_selected_steps": valid_selected_steps,
        "top1_accuracy_next_n_mean": rate(hit_next_n, valid_selected_steps),
        "top1_accuracy_last_prob": rate(hit_last, valid_selected_steps),
        "top1_accuracy_combined_mean": rate(hit_combined, valid_selected_steps),
        "mean_next_n_selected": safe_mean(selected_next_n_vals),
        "mean_next_n_non_selected": safe_mean(non_selected_next_n_vals),
        "mean_last_selected": safe_mean(selected_last_vals),
        "mean_last_non_selected": safe_mean(non_selected_last_vals),
        "delta_next_n": safe_mean(selected_next_n_vals) - safe_mean(non_selected_next_n_vals),
        "delta_last": safe_mean(selected_last_vals) - safe_mean(non_selected_last_vals),
    }
    return {"summary": summary, "per_step_records": per_step_records}


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze probability patterns for branch selection")
    parser.add_argument(
        "--input",
        default="/DISK1/data/rnxu_24/Paper/3d-tts-simulator/3d-tts-sw/compute-optimal-tts/process/speculation/output",
        help="Directory containing generated speculation json files",
    )
    parser.add_argument("--n", type=int, default=8, help="Number of tokens after common prefix")
    parser.add_argument(
        "--output",
        default="/DISK1/data/rnxu_24/Paper/3d-tts-simulator/3d-tts-sw/compute-optimal-tts/process/speculation/output/prob_analysis_result.json",
        help="Output analysis json file",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    in_root = Path(args.input)
    files = sorted(p for p in in_root.rglob("record_0.json") if p.is_file())
    logger.info("Found %d workload files (record_0.json)", len(files))

    result = analyze_steps(files, n_tokens=args.n)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    logger.info("Saved analysis to %s", out_path)
    logger.info("Summary: %s", result["summary"])


if __name__ == "__main__":
    main()

