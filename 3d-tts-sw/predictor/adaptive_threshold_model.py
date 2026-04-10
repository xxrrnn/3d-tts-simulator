"""
predictor.adaptive_threshold_model
==================================
Inference-only predictors for runtime adaptive thresholds.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from .adaptive_threshold_features import FEATURE_DIM, FEATURE_NAMES
from .config import (
    ADAPTIVE_BUDGET_MAX,
    ADAPTIVE_CALIBRATION_MARGIN,
    ACTIVE_BRANCH_GATE_DEFAULT,
)


class BaseAdaptiveThresholdPredictor:
    """Common inference API for runtime threshold predictors."""

    def predict_budget(self, features: np.ndarray, group_key: str = "") -> float:
        raise NotImplementedError

    def predict_threshold(
        self,
        features: np.ndarray,
        *,
        max_other_tokens: int,
        group_key: str = "",
    ) -> float:
        budget = self.predict_budget(features, group_key=group_key)
        return float(max_other_tokens + budget)

    def predict_label(
        self,
        features: np.ndarray,
        *,
        current_tokens: int,
        max_other_tokens: int,
        n_active_branches: Optional[int] = None,
        active_branch_gate: int = ACTIVE_BRANCH_GATE_DEFAULT,
        group_key: str = "",
    ) -> Tuple[bool, float]:
        threshold = self.predict_threshold(
            features,
            max_other_tokens=max_other_tokens,
            group_key=group_key,
        )
        gate_ok = (
            n_active_branches is None
            or int(n_active_branches) <= int(active_branch_gate)
        )
        fired = bool(gate_ok and current_tokens > threshold)
        return fired, threshold


class HeuristicAdaptiveThresholdPredictor(BaseAdaptiveThresholdPredictor):
    """Zero-parameter fallback with group baselines and lightweight penalties."""

    def __init__(
        self,
        *,
        group_budgets: Optional[Dict[str, float]] = None,
        group_slack: Optional[Dict[str, float]] = None,
        global_budget: float = 24.0,
        global_slack: float = 0.0,
        budget_max: float = ADAPTIVE_BUDGET_MAX,
    ):
        self.group_budgets = dict(group_budgets or {})
        self.group_slack = dict(group_slack or {})
        self.global_budget = float(global_budget)
        self.global_slack = float(global_slack)
        self.budget_max = float(budget_max)

    def predict_budget(self, features: np.ndarray, group_key: str = "") -> float:
        f = {name: float(features[i]) for i, name in enumerate(FEATURE_NAMES)}
        budget = self.group_budgets.get(group_key, self.global_budget)

        # Lower-confidence branches get a tighter threshold.
        budget -= 28.0 * f["low_prob_frac_05"]
        budget -= 60.0 * f["low_prob_frac_02"]
        budget -= 22.0 * f["topk_entropy_recent"]
        budget -= 18.0 * max(1.0 - f["topk_margin_recent"], 0.0)
        budget -= 42.0 * f["tracked_token_recent_rate"]

        budget += self.group_slack.get(group_key, self.global_slack)
        return float(np.clip(budget, 0.0, self.budget_max))


class MLPAdaptiveThresholdPredictor(BaseAdaptiveThresholdPredictor):
    """Tiny MLP regressor exported as JSON and evaluated with NumPy only."""

    def __init__(self, weights: Dict):
        self.layers: List[Tuple[np.ndarray, np.ndarray]] = []
        for layer in weights["layers"]:
            W = np.asarray(layer["W"], dtype=np.float32)
            b = np.asarray(layer["b"], dtype=np.float32)
            self.layers.append((W, b))

        self.feature_mean = np.asarray(
            weights.get("feature_mean", [0.0] * FEATURE_DIM),
            dtype=np.float32,
        )
        self.feature_std = np.asarray(
            weights.get("feature_std", [1.0] * FEATURE_DIM),
            dtype=np.float32,
        )
        self.feature_std[self.feature_std < 1e-8] = 1.0

        self.group_slack = {
            str(k): float(v) for k, v in weights.get("group_slack", {}).items()
        }
        self.global_slack = float(weights.get("global_slack", 0.0))
        self.group_baselines = {
            str(k): float(v) for k, v in weights.get("group_baselines", {}).items()
        }
        self.global_baseline = float(weights.get("global_baseline", 0.0))
        self.budget_max = float(weights.get("budget_max", ADAPTIVE_BUDGET_MAX))
        self.quantile = float(weights.get("quantile", 0.0))
        self.calibration_margin = float(
            weights.get("calibration_margin", ADAPTIVE_CALIBRATION_MARGIN)
        )

    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(x, 0.0)

    @staticmethod
    def _softplus(x: np.ndarray) -> np.ndarray:
        return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)

    def _forward(self, features: np.ndarray) -> float:
        h = (features - self.feature_mean) / self.feature_std
        for idx, (W, b) in enumerate(self.layers):
            h = h @ W.T + b
            if idx < len(self.layers) - 1:
                h = self._relu(h)
        return float(self._softplus(np.asarray(h)).squeeze())

    def predict_budget(self, features: np.ndarray, group_key: str = "") -> float:
        budget = self._forward(features)
        budget += self.group_slack.get(group_key, self.global_slack)
        return float(np.clip(budget, 0.0, self.budget_max))

    def save(self, path: Union[str, Path]) -> None:
        obj = {
            "layers": [{"W": W.tolist(), "b": b.tolist()} for W, b in self.layers],
            "feature_mean": self.feature_mean.tolist(),
            "feature_std": self.feature_std.tolist(),
            "group_slack": self.group_slack,
            "global_slack": self.global_slack,
            "group_baselines": self.group_baselines,
            "global_baseline": self.global_baseline,
            "budget_max": self.budget_max,
            "quantile": self.quantile,
            "calibration_margin": self.calibration_margin,
        }
        Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=1))

    @classmethod
    def load(cls, path: Union[str, Path]) -> "MLPAdaptiveThresholdPredictor":
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f))
