"""
predictor.adaptive_threshold_dataset
====================================
Build datasets for the runtime-aware adaptive threshold predictor.

Each sample corresponds to a branch that is still alive when the longest
visible sibling reaches ``max_other`` tokens.  The predictor observes only
that prefix and outputs a safe extra budget for normal branches.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .adaptive_threshold_features import (
    FEATURE_DIM,
    FEATURE_NAMES,
    extract_adaptive_threshold_features,
)
from .config import (
    ADAPTIVE_STRAGGLER_MIN_BRANCH_TOKENS,
    ADAPTIVE_STRAGGLER_RATIO_THRESHOLD,
    ACTIVE_BRANCH_GATE_DEFAULT,
    TEST_FRACTION,
    VAL_FRACTION,
    SPLIT_RANDOM_SEED,
)
from .record_utils import find_record_files, load_record, parse_config_from_path


def is_adaptive_straggler(
    my_tokens: int,
    max_other_tokens: int,
    n_branches: int,
) -> bool:
    """User-provided straggler definition for the adaptive threshold path."""
    return bool(
        n_branches > 1
        and my_tokens > ADAPTIVE_STRAGGLER_MIN_BRANCH_TOKENS
        and my_tokens > max_other_tokens * ADAPTIVE_STRAGGLER_RATIO_THRESHOLD
    )


def _policy_beam_key(policy_model: str, beam_width: int) -> str:
    return f"{policy_model}|beam={beam_width}"


def _config_key(policy_model: str, prm_name: str, beam_width: int) -> str:
    return f"{policy_model}|{prm_name}|beam={beam_width}"


def _extract_adaptive_samples_from_record(
    record_path: str,
    config_priors: Optional[Dict[str, float]] = None,
    active_branch_gate: int = ACTIVE_BRANCH_GATE_DEFAULT,
) -> List[Dict[str, Any]]:
    meta_cfg = parse_config_from_path(record_path)
    if not meta_cfg:
        return []

    try:
        record = load_record(record_path)
    except Exception:
        return []

    outputs = record.get("output", [])
    if not outputs:
        return []

    log = outputs[0].get("detailed_beam_search_log") or {}
    steps = log.get("step_details", [])
    if not steps:
        return []

    policy_model = meta_cfg["policy_model"]
    prm_name = meta_cfg["prm_name"]
    beam_width = meta_cfg["beam_width"]
    question_id = meta_cfg["question_id"]
    group_key = _policy_beam_key(policy_model, beam_width)
    config_key = _config_key(policy_model, prm_name, beam_width)
    config_prior = (config_priors or {}).get(config_key, 0.5)

    samples: List[Dict[str, Any]] = []
    prior_step_total_tokens = 0

    for step_idx, step in enumerate(steps):
        branches = (step.get("selection_process") or {}).get("selected_branches") or []
        step_full_total = sum(int(b.get("num_tokens", 0)) for b in branches)
        total_steps = len(steps)

        if len(branches) > 1:
            token_counts = [int(b.get("num_tokens", 0)) for b in branches]
            n_branches = len(token_counts)

            for branch_idx, branch in enumerate(branches):
                my_tokens = token_counts[branch_idx]
                other_tokens = [t for j, t in enumerate(token_counts) if j != branch_idx]
                max_other = max(other_tokens)
                if my_tokens < max_other:
                    continue

                obs_point = max_other
                sibling_visible_counts = [min(t, obs_point) for t in other_tokens]
                n_active_branches = sum(1 for t in token_counts if t >= obs_point)
                if n_active_branches > active_branch_gate:
                    continue
                step_total_tokens = sum(min(t, obs_point) for t in token_counts)

                feats = extract_adaptive_threshold_features(
                    token_probs=branch.get("token_probs") or [],
                    token_topk_logprobs=branch.get("token_topk_logprobs"),
                    obs_point=obs_point,
                    max_other_tokens=max_other,
                    sibling_token_counts=sibling_visible_counts,
                    n_branches=n_branches,
                    n_active_branches=n_active_branches,
                    step_idx=step_idx,
                    total_steps=total_steps,
                    step_total_tokens=step_total_tokens,
                    prior_step_total_tokens=prior_step_total_tokens,
                    beam_width=beam_width,
                    config_prior=config_prior,
                )

                label = int(is_adaptive_straggler(my_tokens, max_other, n_branches))
                budget_target = max(my_tokens - max_other, 0)

                samples.append({
                    "features": feats,
                    "label": label,
                    "budget_target": float(budget_target),
                    "meta": {
                        "question_id": question_id,
                        "step": step_idx,
                        "branch_idx": branch_idx,
                        "my_tokens": my_tokens,
                        "max_other": max_other,
                        "budget_target": budget_target,
                        "n_branches": n_branches,
                        "n_active_branches": n_active_branches,
                        "step_total_tokens": step_total_tokens,
                        "prior_step_total_tokens": prior_step_total_tokens,
                        "beam_width": beam_width,
                        "prm_name": prm_name,
                        "policy_model": policy_model,
                        "benchmark": meta_cfg["benchmark"],
                        "config_str": meta_cfg["config_str"],
                        "config_key": config_key,
                        "group_key": group_key,
                        "record_path": record_path,
                    },
                })

        prior_step_total_tokens += step_full_total

    return samples


def compute_adaptive_config_priors(
    record_paths: Sequence[str],
    active_branch_gate: int = ACTIVE_BRANCH_GATE_DEFAULT,
) -> Dict[str, float]:
    """Compute train-only positive-rate priors per policy/prm/beam config."""
    counts: Dict[str, List[int]] = defaultdict(list)

    for record_path in record_paths:
        meta_cfg = parse_config_from_path(record_path)
        if not meta_cfg:
            continue

        try:
            record = load_record(record_path)
        except Exception:
            continue

        outputs = record.get("output", [])
        if not outputs:
            continue
        log = outputs[0].get("detailed_beam_search_log") or {}
        for step in log.get("step_details", []):
            branches = (step.get("selection_process") or {}).get("selected_branches") or []
            if len(branches) <= 1:
                continue
            token_counts = [int(b.get("num_tokens", 0)) for b in branches]
            key = _config_key(
                meta_cfg["policy_model"],
                meta_cfg["prm_name"],
                meta_cfg["beam_width"],
            )
            for i, my_tokens in enumerate(token_counts):
                other_tokens = [t for j, t in enumerate(token_counts) if j != i]
                max_other = max(other_tokens)
                if my_tokens < max_other:
                    continue
                n_active_branches = sum(1 for t in token_counts if t >= max_other)
                if n_active_branches > active_branch_gate:
                    continue
                counts[key].append(int(is_adaptive_straggler(my_tokens, max_other, len(token_counts))))

    priors = {}
    for key, values in counts.items():
        priors[key] = float(sum(values) / len(values)) if values else 0.5
    return priors


def build_adaptive_threshold_dataset(
    output_dir: str,
    *,
    active_branch_gate: int = ACTIVE_BRANCH_GATE_DEFAULT,
    test_fraction: float = TEST_FRACTION,
    val_fraction: float = VAL_FRACTION,
    split_seed: int = SPLIT_RANDOM_SEED,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Build train/val/test splits for the adaptive threshold predictor."""
    record_paths = find_record_files(output_dir)
    if verbose:
        print(f"Found {len(record_paths)} record files under {output_dir}")

    rng = np.random.RandomState(split_seed)
    shuffled_paths = list(record_paths)
    rng.shuffle(shuffled_paths)

    n_total_records = len(shuffled_paths)
    n_test = int(n_total_records * test_fraction)
    n_test = min(max(n_test, 1), max(n_total_records - 2, 1))
    test_paths = shuffled_paths[:n_test]
    train_val_paths = shuffled_paths[n_test:]

    n_val = int(len(train_val_paths) * val_fraction)
    n_val = min(max(n_val, 1), max(len(train_val_paths) - 1, 1))
    val_paths = train_val_paths[:n_val]
    train_paths = train_val_paths[n_val:]

    config_priors = compute_adaptive_config_priors(
        train_paths,
        active_branch_gate=active_branch_gate,
    )
    if verbose:
        print(
            f"Train records: {len(train_paths)}, "
            f"Val records: {len(val_paths)}, "
            f"Test records: {len(test_paths)}"
        )
        print(f"Adaptive config priors: {len(config_priors)}")
        print(f"Active-branch gate: <= {active_branch_gate}")

    train_samples: List[Dict[str, Any]] = []
    for record_path in train_paths:
        train_samples.extend(
            _extract_adaptive_samples_from_record(
                record_path,
                config_priors,
                active_branch_gate=active_branch_gate,
            )
        )

    val_samples: List[Dict[str, Any]] = []
    for record_path in val_paths:
        val_samples.extend(
            _extract_adaptive_samples_from_record(
                record_path,
                config_priors,
                active_branch_gate=active_branch_gate,
            )
        )

    test_samples: List[Dict[str, Any]] = []
    for record_path in test_paths:
        test_samples.extend(
            _extract_adaptive_samples_from_record(
                record_path,
                config_priors,
                active_branch_gate=active_branch_gate,
            )
        )

    def _pack(samples: List[Dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        if not samples:
            return (
                np.empty((0, FEATURE_DIM), dtype=np.float32),
                np.empty((0,), dtype=np.int32),
                np.empty((0,), dtype=np.float32),
                [],
            )
        X = np.stack([s["features"] for s in samples]).astype(np.float32)
        y = np.asarray([s["label"] for s in samples], dtype=np.int32)
        budgets = np.asarray([s["budget_target"] for s in samples], dtype=np.float32)
        meta = [s["meta"] for s in samples]
        return X, y, budgets, meta

    X_train, y_train, budget_train, meta_train = _pack(train_samples)
    X_val, y_val, budget_val, meta_val = _pack(val_samples)
    X_test, y_test, budget_test, meta_test = _pack(test_samples)

    if verbose:
        for split_name, labels in [("train", y_train), ("val", y_val), ("test", y_test)]:
            pos = int(labels.sum())
            total = len(labels)
            print(
                f"  {split_name}: {total} samples, {pos} stragglers "
                f"({100.0 * pos / max(total, 1):.2f}%)"
            )
        neg_train = int((y_train == 0).sum())
        print(
            f"  train negatives={neg_train}"
        )

    return {
        "X_train": X_train,
        "y_train": y_train,
        "budget_train": budget_train,
        "meta_train": meta_train,
        "X_val": X_val,
        "y_val": y_val,
        "budget_val": budget_val,
        "meta_val": meta_val,
        "X_test": X_test,
        "y_test": y_test,
        "budget_test": budget_test,
        "meta_test": meta_test,
        "config_priors": config_priors,
        "feature_names": FEATURE_NAMES,
        "active_branch_gate": int(active_branch_gate),
    }
