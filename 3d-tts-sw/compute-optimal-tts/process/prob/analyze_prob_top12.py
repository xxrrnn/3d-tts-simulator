#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import math
import argparse
from collections import Counter, defaultdict
from statistics import mean
from typing import List, Dict, Any, Optional, Tuple

# Prob+长度惩罚的 beta 候选
PRIOR_LEN_BETAS = [0.05, 0.1, 0.2]


def rankdata_desc(values: List[float]) -> List[float]:
    """给定数值列表，按降序做平均秩（1=最大）"""
    n = len(values)
    # (value, idx) 按 value 降序
    sorted_pairs = sorted([(v, i) for i, v in enumerate(values)], key=lambda x: x[0], reverse=True)
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_pairs[j + 1][0] == sorted_pairs[i][0]:
            j += 1
        # 平均名次（1-indexed）
        avg_rank = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[sorted_pairs[k][1]] = avg_rank
        i = j + 1
    return ranks


def pearson(x: List[float], y: List[float]) -> Optional[float]:
    if len(x) != len(y) or len(x) < 2:
        return None
    mx, my = mean(x), mean(y)
    vx = sum((xi - mx) ** 2 for xi in x)
    vy = sum((yi - my) ** 2 for yi in y)
    if vx == 0 or vy == 0:
        return None
    cov = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    return cov / math.sqrt(vx * vy)


def spearman(x: List[float], y: List[float]) -> Optional[float]:
    if len(x) != len(y) or len(x) < 2:
        return None
    rx = rankdata_desc(x)  # 降序秩
    ry = rankdata_desc(y)
    return pearson(rx, ry)


def safe_float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return default


def zscore(values: List[float]) -> List[float]:
    """步内标准化；方差为0时返回全0，避免除0。"""
    if not values:
        return []
    m = sum(values) / len(values)
    var = sum((x - m) ** 2 for x in values) / len(values)
    if var <= 0:
        return [0.0] * len(values)
    std = math.sqrt(var)
    return [(x - m) / std for x in values]


def read_first_json_line(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            line = f.readline().strip()
            if not line:
                return None
            return json.loads(line)
    except Exception:
        return None


def extract_steps(record_obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    """从一个 record json 中提取所有 step 的 selected_branches"""
    out_steps = []
    outputs = record_obj.get("output", [])
    for o in outputs:
        dlog = o.get("detailed_beam_search_log", {})
        for step in dlog.get("step_details", []):
            sel = step.get("selection_process", {})
            branches = sel.get("selected_branches", [])
            if branches:
                out_steps.append({
                    "step": step.get("step", None),
                    "branches": branches
                })
    return out_steps


def analyze_step(branches: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    branches 每个元素应包含:
      - prior_prob
      - selected (bool)
    """
    probs = [safe_float(b.get("prior_prob", 0.0)) for b in branches]
    if len(probs) < 2:
        return None

    selected_idx = None
    for i, b in enumerate(branches):
        if bool(b.get("selected", False)):
            selected_idx = i
            break

    if selected_idx is None:
        return None

    # 按 prob 降序排序，保留原 idx
    order = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
    top1_idx = order[0]
    top2_idx = order[1] if len(order) > 1 else None

    # selected 的名次（1-based）
    rank_map = {idx: r + 1 for r, idx in enumerate(order)}
    selected_rank = rank_map[selected_idx]

    selected_prob = probs[selected_idx]
    top1_prob = probs[top1_idx]
    top2_prob = probs[top2_idx] if top2_idx is not None else float("nan")

    # 先验-only 分数：
    # 1) Prob-only: z(logp)
    # 2) Prob + 长度惩罚: z(logp) - beta * z(len), beta in PRIOR_LEN_BETAS
    # logp 使用 ln(prior_prob)，对非正概率做裁剪
    eps = 1e-12
    logps = [math.log(max(p, eps)) for p in probs]
    lens = [safe_float(b.get("num_tokens", 0.0)) for b in branches]

    z_logp = zscore(logps)
    z_len = zscore(lens)

    # Prob-only
    prob_only_scores = z_logp
    prob_only_order = sorted(range(len(prob_only_scores)), key=lambda i: prob_only_scores[i], reverse=True)
    prob_only_top1_idx = prob_only_order[0]
    prob_only_top2_idx = prob_only_order[1] if len(prob_only_order) > 1 else None

    # Prob + len penalty
    prior_len_scores_by_beta = {}
    prior_len_orders_by_beta = {}
    for beta in PRIOR_LEN_BETAS:
        sc = [z_logp[i] - beta * z_len[i] for i in range(len(branches))]
        od = sorted(range(len(sc)), key=lambda i: sc[i], reverse=True)
        prior_len_scores_by_beta[beta] = sc
        prior_len_orders_by_beta[beta] = od

    ret = {
        "n_branches": len(probs),
        "selected_idx": selected_idx,
        "selected_prob": selected_prob,
        "top1_idx": top1_idx,
        "top1_prob": top1_prob,
        "top2_idx": top2_idx,
        "top2_prob": top2_prob,
        "selected_is_top1": int(selected_idx == top1_idx),
        "selected_in_top2": int(selected_idx in set(order[:2])),
        "selected_rank": selected_rank,
        "delta_sel_top1": selected_prob - top1_prob,
        "delta_sel_top2": selected_prob - top2_prob if top2_idx is not None else float("nan"),
        "selected_is_prob_only_top1": int(selected_idx == prob_only_top1_idx),
        "selected_in_prob_only_top2": int(selected_idx in set(prob_only_order[:2])),
        "prob_only_top1_idx": prob_only_top1_idx,
        "prob_only_top2_idx": prob_only_top2_idx,
        "prob_only_top1_score": prob_only_scores[prob_only_top1_idx],
        "prob_only_selected_score": prob_only_scores[selected_idx],
        "prob_only_top1_is_prob_top1": int(prob_only_top1_idx == top1_idx),
        # 相关性：selected(0/1) vs prior_prob
        "spearman_selected_vs_prob": spearman(
            [1.0 if i == selected_idx else 0.0 for i in range(len(probs))],
            probs
        ),
        # 相关性：selected(0/1) vs prob-only score
        "spearman_selected_vs_prob_only": spearman(
            [1.0 if i == selected_idx else 0.0 for i in range(len(prob_only_scores))],
            prob_only_scores
        ),
    }

    # 动态追加每个 beta 的结果
    for beta in PRIOR_LEN_BETAS:
        od = prior_len_orders_by_beta[beta]
        sc = prior_len_scores_by_beta[beta]
        beta_tag = f"{beta:.2f}".replace(".", "")
        ret[f"selected_is_prob_len_b{beta_tag}_top1"] = int(selected_idx == od[0])
        ret[f"selected_in_prob_len_b{beta_tag}_top2"] = int(selected_idx in set(od[:2]))
        ret[f"prob_len_b{beta_tag}_top1_idx"] = od[0]
        ret[f"prob_len_b{beta_tag}_top2_idx"] = od[1] if len(od) > 1 else None
        ret[f"prob_len_b{beta_tag}_top1_score"] = sc[od[0]]
        ret[f"prob_len_b{beta_tag}_selected_score"] = sc[selected_idx]
        ret[f"prob_len_b{beta_tag}_top1_is_prob_top1"] = int(od[0] == top1_idx)
        ret[f"spearman_selected_vs_prob_len_b{beta_tag}"] = spearman(
            [1.0 if i == selected_idx else 0.0 for i in range(len(sc))],
            sc
        )

    return ret


def walk_records(input_dir: str) -> List[str]:
    recs = []
    for root, _, files in os.walk(input_dir):
        for fn in files:
            if fn.endswith(".jsonl") and fn.startswith("record_"):
                recs.append(os.path.join(root, fn))
    return recs


def summarize(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        return {}

    hit_top1 = sum(r["selected_is_top1"] for r in results) / len(results)
    hit_top2 = sum(r["selected_in_top2"] for r in results) / len(results)

    rank_counter = Counter(r["selected_rank"] for r in results)

    deltas1 = [r["delta_sel_top1"] for r in results]
    deltas2 = [r["delta_sel_top2"] for r in results if not math.isnan(r["delta_sel_top2"])]

    spears = [r["spearman_selected_vs_prob"] for r in results if r["spearman_selected_vs_prob"] is not None]
    spears_prob_only = [
        r["spearman_selected_vs_prob_only"] for r in results
        if r["spearman_selected_vs_prob_only"] is not None
    ]

    prob_only_top1_hit = sum(r["selected_is_prob_only_top1"] for r in results) / len(results)
    prob_only_top2_hit = sum(r["selected_in_prob_only_top2"] for r in results) / len(results)
    prob_only_top1_eq_prob_top1 = sum(r["prob_only_top1_is_prob_top1"] for r in results) / len(results)

    # Prob+len(beta) 汇总
    prob_len_summary = {}
    for beta in PRIOR_LEN_BETAS:
        beta_tag = f"{beta:.2f}".replace(".", "")
        top1_key = f"selected_is_prob_len_b{beta_tag}_top1"
        top2_key = f"selected_in_prob_len_b{beta_tag}_top2"
        eq_key = f"prob_len_b{beta_tag}_top1_is_prob_top1"
        spear_key = f"spearman_selected_vs_prob_len_b{beta_tag}"
        spears_beta = [r[spear_key] for r in results if r[spear_key] is not None]
        prob_len_summary[f"b{beta_tag}"] = {
            "top1_hit_rate": sum(r[top1_key] for r in results) / len(results),
            "top2_hit_rate": sum(r[top2_key] for r in results) / len(results),
            "top1_equals_prob_top1_rate": sum(r[eq_key] for r in results) / len(results),
            "avg_spearman_selected_vs_score": sum(spears_beta) / len(spears_beta) if spears_beta else None,
        }

    # 每个 selected index 的全局被选中概率
    selected_idx_counter = Counter(r["selected_idx"] for r in results)
    selected_index_probability_overall = {
        idx: cnt / len(results) for idx, cnt in sorted(selected_idx_counter.items())
    }

    # 按分支数分桶看命中率
    bucket = defaultdict(list)
    for r in results:
        bucket[r["n_branches"]].append(r)

    bucket_stats = {}
    for k, rs in sorted(bucket.items()):
        idx_counter = Counter(x["selected_idx"] for x in rs)
        idx_prob = {idx: cnt / len(rs) for idx, cnt in sorted(idx_counter.items())}
        bucket_info = {
            "count": len(rs),
            "top1_hit": sum(x["selected_is_top1"] for x in rs) / len(rs),
            "top2_hit": sum(x["selected_in_top2"] for x in rs) / len(rs),
            "prob_only_top1_hit": sum(x["selected_is_prob_only_top1"] for x in rs) / len(rs),
            "prob_only_top2_hit": sum(x["selected_in_prob_only_top2"] for x in rs) / len(rs),
            "avg_selected_rank": sum(x["selected_rank"] for x in rs) / len(rs),
            "selected_index_probability": idx_prob,
        }
        for beta in PRIOR_LEN_BETAS:
            beta_tag = f"{beta:.2f}".replace(".", "")
            bucket_info[f"prob_len_b{beta_tag}_top1_hit"] = sum(
                x[f"selected_is_prob_len_b{beta_tag}_top1"] for x in rs
            ) / len(rs)
            bucket_info[f"prob_len_b{beta_tag}_top2_hit"] = sum(
                x[f"selected_in_prob_len_b{beta_tag}_top2"] for x in rs
            ) / len(rs)
        bucket_stats[k] = bucket_info

    return {
        "num_steps": len(results),
        "top1_hit_rate": hit_top1,
        "top2_hit_rate": hit_top2,
        "selected_rank_distribution": dict(sorted(rank_counter.items())),
        "avg_selected_rank": sum(r["selected_rank"] for r in results) / len(results),
        "avg_delta_sel_top1": sum(deltas1) / len(deltas1) if deltas1 else None,
        "avg_delta_sel_top2": sum(deltas2) / len(deltas2) if deltas2 else None,
        "avg_spearman_selected_vs_prob": sum(spears) / len(spears) if spears else None,
        "prob_only_top1_hit_rate": prob_only_top1_hit,
        "prob_only_top2_hit_rate": prob_only_top2_hit,
        "prob_only_top1_equals_prob_top1_rate": prob_only_top1_eq_prob_top1,
        "avg_spearman_selected_vs_prob_only": sum(spears_prob_only) / len(spears_prob_only) if spears_prob_only else None,
        "prob_len_summary": prob_len_summary,
        "selected_index_probability_overall": selected_index_probability_overall,
        "by_branch_count": bucket_stats,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="root dir, e.g. .../AMC23_beam_search")
    parser.add_argument("--save-step-csv", default="", help="optional step-level csv output path")
    parser.add_argument("--save-json", default="", help="optional summary json output path")
    args = parser.parse_args()

    record_files = walk_records(args.input)
    if not record_files:
        print(f"[WARN] no record_*.jsonl found under: {args.input}")
        return

    step_results = []
    n_records = 0
    n_records_with_steps = 0

    for p in record_files:
        obj = read_first_json_line(p)
        if obj is None:
            continue
        n_records += 1
        steps = extract_steps(obj)
        if steps:
            n_records_with_steps += 1
        for st in steps:
            an = analyze_step(st["branches"])
            if an is None:
                continue
            an["record_path"] = p
            an["step"] = st["step"]
            step_results.append(an)

    summary = summarize(step_results)

    print("========== Correlation Analysis: selected branch vs top1/top2 prob ==========")
    print(f"input: {args.input}")
    print(f"record files found: {len(record_files)}")
    print(f"record files parsed: {n_records}")
    print(f"record files with valid step logs: {n_records_with_steps}")
    print(f"valid analyzed steps: {summary.get('num_steps', 0)}")
    if not summary:
        print("[WARN] no valid steps")
        return

    print(f"\nTop1 hit rate (selected == prob top1): {summary['top1_hit_rate']:.4f}")
    print(f"Top2 hit rate (selected in prob top2): {summary['top2_hit_rate']:.4f}")
    print(f"Avg selected rank (1=best): {summary['avg_selected_rank']:.4f}")
    print(f"Avg delta(selected - top1): {summary['avg_delta_sel_top1']:.6f}")
    print(f"Avg delta(selected - top2): {summary['avg_delta_sel_top2']:.6f}")
    print(f"Avg Spearman(selected indicator vs prob): {summary['avg_spearman_selected_vs_prob']}")
    print(f"\nProb-only top1 hit rate (selected == prob-only top1): {summary['prob_only_top1_hit_rate']:.4f}")
    print(f"Prob-only top2 hit rate (selected in prob-only top2): {summary['prob_only_top2_hit_rate']:.4f}")
    print(f"Prob-only top1 == prob top1 rate: {summary['prob_only_top1_equals_prob_top1_rate']:.4f}")
    print(f"Avg Spearman(selected indicator vs prob-only): {summary['avg_spearman_selected_vs_prob_only']}")

    print("\nProb + length penalty summary:")
    for k, st in summary["prob_len_summary"].items():
        beta = k[1:]
        print(f"  beta=0.{beta}: top1_hit={st['top1_hit_rate']:.4f}, top2_hit={st['top2_hit_rate']:.4f}, "
              f"top1==prob_top1={st['top1_equals_prob_top1_rate']:.4f}, "
              f"avg_spearman={st['avg_spearman_selected_vs_score']}")

    print("\nSelected index probability (overall, excluding branch_count=1):")
    for idx, p in summary["selected_index_probability_overall"].items():
        print(f"  index {idx}: {p:.4f}")

    print("\nSelected rank distribution:")
    for k, v in summary["selected_rank_distribution"].items():
        print(f"  rank {k}: {v}")

    print("\nBy branch count:")
    for bc, st in summary["by_branch_count"].items():
        print(f"  branches={bc}: count={st['count']}, top1_hit={st['top1_hit']:.4f}, "
              f"top2_hit={st['top2_hit']:.4f}, prob_only_top1_hit={st['prob_only_top1_hit']:.4f}, "
              f"prob_only_top2_hit={st['prob_only_top2_hit']:.4f}, avg_rank={st['avg_selected_rank']:.4f}")
        for beta in PRIOR_LEN_BETAS:
            beta_tag = f"{beta:.2f}".replace(".", "")
            print(f"    prob_len(beta={beta:.2f}): top1_hit={st[f'prob_len_b{beta_tag}_top1_hit']:.4f}, "
                  f"top2_hit={st[f'prob_len_b{beta_tag}_top2_hit']:.4f}")
        idx_prob_str = ", ".join([f"idx{idx}:{p:.4f}" for idx, p in st["selected_index_probability"].items()])
        print(f"    selected_index_probability: {idx_prob_str}")

    if args.save_json:
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"\n[OK] summary json saved: {args.save_json}")

    if args.save_step_csv:
        # 手写简单 CSV，避免额外依赖
        cols = [
            "record_path", "step", "n_branches", "selected_idx", "selected_prob",
            "top1_idx", "top1_prob", "top2_idx", "top2_prob",
            "selected_is_top1", "selected_in_top2", "selected_rank",
            "delta_sel_top1", "delta_sel_top2", "spearman_selected_vs_prob",
            "selected_is_prob_only_top1", "selected_in_prob_only_top2",
            "prob_only_top1_idx", "prob_only_top2_idx",
            "prob_only_top1_score", "prob_only_selected_score",
            "prob_only_top1_is_prob_top1", "spearman_selected_vs_prob_only",
        ]
        for beta in PRIOR_LEN_BETAS:
            beta_tag = f"{beta:.2f}".replace(".", "")
            cols.extend([
                f"selected_is_prob_len_b{beta_tag}_top1",
                f"selected_in_prob_len_b{beta_tag}_top2",
                f"prob_len_b{beta_tag}_top1_idx",
                f"prob_len_b{beta_tag}_top2_idx",
                f"prob_len_b{beta_tag}_top1_score",
                f"prob_len_b{beta_tag}_selected_score",
                f"prob_len_b{beta_tag}_top1_is_prob_top1",
                f"spearman_selected_vs_prob_len_b{beta_tag}",
            ])
        with open(args.save_step_csv, "w", encoding="utf-8") as f:
            f.write(",".join(cols) + "\n")
            for r in step_results:
                row = [str(r.get(c, "")) for c in cols]
                f.write(",".join(row) + "\n")
        print(f"[OK] step-level csv saved: {args.save_step_csv}")


if __name__ == "__main__":
    main()