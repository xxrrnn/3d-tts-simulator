"""
predictor.adaptive_threshold_features
=====================================
Runtime-aware causal features for the adaptive threshold predictor.

The predictor is invoked when a branch reaches the current ``max_other``
reference length.  Features therefore use only prefix information up to the
observation point plus runtime context that is already known at that time.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .config import ADAPTIVE_TRACKED_TOKEN_IDS, RECENT_WINDOW


FEATURE_NAMES = [
    "prob_mean",
    "prob_min",
    "prob_recent_mean",
    "low_prob_frac_05",
    "low_prob_frac_02",
    "topk_entropy_recent",
    "topk_margin_recent",
    "tracked_token_recent_rate",
    "max_other_tokens",
    "n_active_branches",
    "prior_step_total_tokens",
    "config_prior",
]

FEATURE_DIM = len(FEATURE_NAMES)


def extract_adaptive_threshold_features(
    token_probs: Sequence[float],
    token_topk_logprobs: Optional[Sequence[Sequence[float]]],
    *,
    obs_point: int,
    max_other_tokens: int,
    sibling_token_counts: Sequence[int],
    n_branches: int,
    n_active_branches: int,
    step_idx: int,
    total_steps: int,
    step_total_tokens: int,
    prior_step_total_tokens: int,
    beam_width: int = 0,
    config_prior: float = 0.5,
    tracked_token_ids: Optional[Sequence[str]] = None,
) -> np.ndarray:
    """Return a runtime-aware feature vector for threshold prediction."""
    horizon = max(int(obs_point), 0)
    probs = np.asarray(token_probs[:horizon], dtype=np.float64)
    if probs.size == 0:
        return np.zeros(FEATURE_DIM, dtype=np.float32)

    tracked_ids = tuple(str(tid) for tid in (tracked_token_ids or ADAPTIVE_TRACKED_TOKEN_IDS))
    T = len(probs)
    rw = min(RECENT_WINDOW, T)
    recent = probs[-rw:]

    prob_mean = float(probs.mean())
    prob_min = float(probs.min())
    prob_recent_mean = float(recent.mean())
    low_prob_frac_05 = float(np.mean(probs <= 0.50))
    low_prob_frac_02 = float(np.mean(probs <= 0.20))

    entropies = []
    margins = []
    tracked_hits = []
    if token_topk_logprobs is not None:
        for item in token_topk_logprobs[:horizon]:
            if not item:
                continue
            vals, has_tracked = _extract_topk_values_and_hit(item, tracked_ids)
            if vals.size == 0:
                continue
            entropies.append(_entropy_from_logprobs(vals))
            margins.append(_topk_margin(vals))
            tracked_hits.append(has_tracked)

    if entropies:
        ents = np.asarray(entropies, dtype=np.float64)
        topk_entropy_recent = float(ents[-min(rw, len(ents)):].mean())
    else:
        topk_entropy_recent = 0.0

    if margins:
        mgs = np.asarray(margins, dtype=np.float64)
        topk_margin_recent = float(mgs[-min(rw, len(mgs)):].mean())
    else:
        topk_margin_recent = 0.0

    if tracked_hits:
        hits = np.asarray(tracked_hits, dtype=np.float64)
        tracked_token_recent_rate = float(hits[-min(rw, len(hits)):].mean())
    else:
        tracked_token_recent_rate = 0.0

    _ = sibling_token_counts
    _ = step_idx
    _ = total_steps
    _ = step_total_tokens
    _ = beam_width

    vec = np.array([
        prob_mean,
        prob_min,
        prob_recent_mean,
        low_prob_frac_05,
        low_prob_frac_02,
        topk_entropy_recent,
        topk_margin_recent,
        tracked_token_recent_rate,
        max_other_tokens / 256.0,
        n_active_branches / 8.0,
        prior_step_total_tokens / 8192.0,
        config_prior,
    ], dtype=np.float32)
    assert vec.shape == (FEATURE_DIM,)
    return vec


def _extract_topk_values_and_hit(item, tracked_ids: Sequence[str]) -> tuple[np.ndarray, float]:
    if isinstance(item, dict):
        vals = np.asarray(list(item.values()), dtype=np.float64)
        hit = 1.0 if any(tok in item for tok in tracked_ids) else 0.0
        return vals, hit

    vals = np.asarray(list(item), dtype=np.float64)
    return vals, 0.0


def _entropy_from_logprobs(logprobs: np.ndarray) -> float:
    probs = np.exp(logprobs - np.max(logprobs))
    total = probs.sum()
    if total <= 0:
        return 0.0
    probs = probs / total
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log(probs + 1e-12)))


def _topk_margin(logprobs: np.ndarray) -> float:
    if logprobs.size == 0:
        return 0.0
    arr = np.sort(logprobs)[::-1]
    if arr.size == 1:
        return 0.0
    return float(arr[0] - arr[1])
