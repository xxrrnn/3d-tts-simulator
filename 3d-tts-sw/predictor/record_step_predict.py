"""
predictor.record_step_predict
=============================
Predict adaptive-threshold outcomes for one ``record_0.jsonl`` step.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from .adaptive_threshold_dataset import (
    _config_key,
    _policy_beam_key,
    is_adaptive_straggler,
)
from .adaptive_threshold_features import extract_adaptive_threshold_features
from .adaptive_threshold_model import (
    HeuristicAdaptiveThresholdPredictor,
    MLPAdaptiveThresholdPredictor,
)
from .record_utils import load_record, parse_config_from_path
from .config import ACTIVE_BRANCH_GATE_DEFAULT


def _load_predictor(weights_path: str):
    if weights_path and Path(weights_path).exists():
        return MLPAdaptiveThresholdPredictor.load(weights_path)
    return HeuristicAdaptiveThresholdPredictor()


def _load_priors(priors_path: str) -> Dict[str, float]:
    if priors_path and Path(priors_path).exists():
        with open(priors_path, encoding="utf-8") as f:
            obj = json.load(f)
        return {str(k): float(v) for k, v in obj.items()}
    return {}


def predict_record_step(
    record_path: str,
    *,
    step_idx: int,
    weights_path: str = "predictor/adaptive_threshold_weights.json",
    priors_path: str = "predictor/adaptive_threshold_priors.json",
    active_branch_gate: int = ACTIVE_BRANCH_GATE_DEFAULT,
) -> Dict[str, Any]:
    """Run the adaptive threshold predictor on one step from one record."""
    meta_cfg = parse_config_from_path(record_path)
    if not meta_cfg:
        raise ValueError(f"Cannot parse config from path: {record_path}")

    record = load_record(record_path)
    outputs = record.get("output") or []
    if not outputs:
        raise ValueError(f"Record has no output payload: {record_path}")

    log = outputs[0].get("detailed_beam_search_log") or {}
    steps = log.get("step_details") or []
    if step_idx < 0 or step_idx >= len(steps):
        raise IndexError(
            f"step_idx={step_idx} out of range for record with {len(steps)} steps"
        )

    predictor = _load_predictor(weights_path)
    priors = _load_priors(priors_path)

    config_key = _config_key(
        meta_cfg["policy_model"],
        meta_cfg["prm_name"],
        meta_cfg["beam_width"],
    )
    group_key = _policy_beam_key(meta_cfg["policy_model"], meta_cfg["beam_width"])
    config_prior = priors.get(config_key, 0.5)
    total_steps = len(steps)

    prior_step_total_tokens = 0
    for prev_step in steps[:step_idx]:
        prev_branches = (prev_step.get("selection_process") or {}).get("selected_branches") or []
        prior_step_total_tokens += sum(int(b.get("num_tokens", 0)) for b in prev_branches)

    step = steps[step_idx]
    branches = (step.get("selection_process") or {}).get("selected_branches") or []
    token_counts = [int(b.get("num_tokens", 0)) for b in branches]
    step_full_total = sum(token_counts)

    branch_reports: List[Dict[str, Any]] = []
    triggered = []

    for branch_idx, branch in enumerate(branches):
        my_tokens = token_counts[branch_idx]
        other_tokens = [t for j, t in enumerate(token_counts) if j != branch_idx]
        max_other = max(other_tokens) if other_tokens else 0
        obs_point = max_other
        n_active_branches = sum(1 for t in token_counts if t >= obs_point)
        gate_ok = n_active_branches <= active_branch_gate
        eligible = len(token_counts) > 1 and my_tokens >= max_other and gate_ok

        report: Dict[str, Any] = {
            "branch_idx": branch_idx,
            "num_tokens": my_tokens,
            "max_other": max_other,
            "eligible_for_prediction": bool(eligible),
            "active_branch_gate": int(active_branch_gate),
            "gate_ok": bool(gate_ok),
            "threshold": None,
            "budget": None,
            "threshold_fired": False,
            "predicted_straggler": False,
            "true_straggler": bool(
                is_adaptive_straggler(my_tokens, max_other, len(token_counts))
            ),
        }

        if eligible:
            sibling_visible_counts = [min(t, obs_point) for t in other_tokens]
            step_total_tokens = sum(min(t, obs_point) for t in token_counts)

            features = extract_adaptive_threshold_features(
                token_probs=branch.get("token_probs") or [],
                token_topk_logprobs=branch.get("token_topk_logprobs"),
                obs_point=obs_point,
                max_other_tokens=max_other,
                sibling_token_counts=sibling_visible_counts,
                n_branches=len(token_counts),
                n_active_branches=n_active_branches,
                step_idx=step_idx,
                total_steps=total_steps,
                step_total_tokens=step_total_tokens,
                prior_step_total_tokens=prior_step_total_tokens,
                beam_width=meta_cfg["beam_width"],
                config_prior=config_prior,
            )
            budget = predictor.predict_budget(features, group_key=group_key)
            fired, threshold = predictor.predict_label(
                features,
                current_tokens=my_tokens,
                max_other_tokens=max_other,
                n_active_branches=n_active_branches,
                active_branch_gate=active_branch_gate,
                group_key=group_key,
            )
            report.update({
                "budget": float(budget),
                "threshold": float(threshold),
                "threshold_fired": bool(fired),
                "predicted_straggler": bool(fired),
                "obs_point": obs_point,
                "n_active_branches": n_active_branches,
                "step_total_tokens_visible": step_total_tokens,
            })
            if fired:
                triggered.append(branch_idx)

        branch_reports.append(report)

    return {
        "record_path": record_path,
        "question_id": meta_cfg["question_id"],
        "benchmark": meta_cfg["benchmark"],
        "policy_model": meta_cfg["policy_model"],
        "prm_name": meta_cfg["prm_name"],
        "beam_width": meta_cfg["beam_width"],
        "config_str": meta_cfg["config_str"],
        "step_idx": step_idx,
        "total_steps": total_steps,
        "n_branches": len(branches),
        "prior_step_total_tokens": prior_step_total_tokens,
        "step_total_tokens_full": step_full_total,
        "group_key": group_key,
        "config_prior": config_prior,
        "active_branch_gate": int(active_branch_gate),
        "has_triggered_branch": bool(triggered),
        "triggered_branch_indices": triggered,
        "branches": branch_reports,
    }
