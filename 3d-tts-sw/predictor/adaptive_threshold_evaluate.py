"""
predictor.adaptive_threshold_evaluate
=====================================
Evaluation helpers for the runtime adaptive threshold predictor.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np

from .adaptive_threshold_model import BaseAdaptiveThresholdPredictor


def evaluate_adaptive_threshold_predictor(
    predictor: BaseAdaptiveThresholdPredictor,
    X: np.ndarray,
    y: np.ndarray,
    budgets: np.ndarray,
    meta: List[Dict[str, Any]],
    *,
    active_branch_gate: int = 2,
) -> Dict[str, Any]:
    preds = []
    threshold_gaps = []
    saved_tokens = []
    negative_headroom = []
    per_group = defaultdict(lambda: {
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "tn": 0,
        "saved_tokens": 0.0,
        "threshold_gaps": [],
    })

    for i in range(len(meta)):
        group_key = meta[i]["group_key"]
        max_other = int(meta[i]["max_other"])
        current_tokens = int(meta[i]["my_tokens"])
        pred, threshold = predictor.predict_label(
            X[i],
            current_tokens=current_tokens,
            max_other_tokens=max_other,
            n_active_branches=int(meta[i]["n_active_branches"]),
            active_branch_gate=active_branch_gate,
            group_key=group_key,
        )
        pred_int = int(pred)
        preds.append(pred_int)

        gap = float(threshold - max_other)
        threshold_gaps.append(gap)
        per_group[group_key]["threshold_gaps"].append(gap)

        saved = max(current_tokens - math.ceil(threshold), 0) if pred else 0
        saved_tokens.append(saved)
        per_group[group_key]["saved_tokens"] += float(saved)

        if y[i] == 0:
            negative_headroom.append(float(threshold - current_tokens))

        key = per_group[group_key]
        if pred_int == 1 and y[i] == 1:
            key["tp"] += 1
        elif pred_int == 1 and y[i] == 0:
            key["fp"] += 1
        elif pred_int == 0 and y[i] == 1:
            key["fn"] += 1
        else:
            key["tn"] += 1

    preds_arr = np.asarray(preds, dtype=np.int32)
    tp = int(((preds_arr == 1) & (y == 1)).sum())
    fp = int(((preds_arr == 1) & (y == 0)).sum())
    fn = int(((preds_arr == 0) & (y == 1)).sum())
    tn = int(((preds_arr == 0) & (y == 0)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    per_group_report = {}
    for group_key, stats in sorted(per_group.items()):
        group_tp = stats["tp"]
        group_fp = stats["fp"]
        group_fn = stats["fn"]
        group_tn = stats["tn"]
        group_precision = group_tp / max(group_tp + group_fp, 1)
        group_recall = group_tp / max(group_tp + group_fn, 1)
        per_group_report[group_key] = {
            "tp": group_tp,
            "fp": group_fp,
            "fn": group_fn,
            "tn": group_tn,
            "precision": round(group_precision, 4),
            "recall": round(group_recall, 4),
            "f1": round(
                2 * group_precision * group_recall / max(group_precision + group_recall, 1e-8),
                4,
            ),
            "saved_tokens": round(float(stats["saved_tokens"]), 2),
            "avg_threshold_gap": round(
                float(np.mean(stats["threshold_gaps"])) if stats["threshold_gaps"] else 0.0,
                3,
            ),
        }

    tp_saved = [saved_tokens[i] for i in range(len(saved_tokens)) if preds[i] == 1 and y[i] == 1]
    return {
        "n_samples": int(len(y)),
        "accuracy": round(float((preds_arr == y).mean()), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "avg_threshold_gap": round(float(np.mean(threshold_gaps)) if threshold_gaps else 0.0, 3),
        "median_threshold_gap": round(float(np.median(threshold_gaps)) if threshold_gaps else 0.0, 3),
        "avg_negative_headroom": round(
            float(np.mean(negative_headroom)) if negative_headroom else 0.0,
            3,
        ),
        "total_saved_tokens": round(float(sum(saved_tokens)), 2),
        "tp_saved_tokens": round(float(sum(tp_saved)), 2),
        "avg_saved_tokens_tp": round(float(np.mean(tp_saved)) if tp_saved else 0.0, 3),
        "per_group": per_group_report,
    }
