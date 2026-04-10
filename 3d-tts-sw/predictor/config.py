"""
predictor.config
================
Shared constants for the standalone adaptive-threshold predictor package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


ADAPTIVE_STRAGGLER_RATIO_THRESHOLD: float = 1.5
ADAPTIVE_STRAGGLER_MIN_BRANCH_TOKENS: int = 100

ADAPTIVE_TRACKED_TOKEN_IDS: List[str] = ["272"]
ADAPTIVE_BUDGET_QUANTILE_DEFAULT: float = 0.97
ADAPTIVE_BUDGET_MAX: float = 512.0
ADAPTIVE_CALIBRATION_MARGIN: float = 1.0
ACTIVE_BRANCH_GATE_DEFAULT: int = 2

RECENT_WINDOW: int = 10

TEST_FRACTION: float = 0.15
VAL_FRACTION: float = 0.15
SPLIT_RANDOM_SEED: int = 42


@dataclass
class AdaptiveThresholdConfig:
    """Quantile-regression MLP used by the adaptive threshold predictor."""

    input_dim: int = 0
    hidden_dims: List[int] = field(default_factory=lambda: [32, 16])
    dropout: float = 0.05
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 60
    batch_size: int = 256
    patience: int = 10
    quantile: float = ADAPTIVE_BUDGET_QUANTILE_DEFAULT
