"""
predictor.adaptive_threshold_trainer
====================================
Train the runtime adaptive threshold predictor.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .adaptive_threshold_dataset import build_adaptive_threshold_dataset
from .adaptive_threshold_evaluate import evaluate_adaptive_threshold_predictor
from .adaptive_threshold_features import FEATURE_DIM
from .adaptive_threshold_model import (
    HeuristicAdaptiveThresholdPredictor,
    MLPAdaptiveThresholdPredictor,
)
from .config import AdaptiveThresholdConfig, ADAPTIVE_BUDGET_MAX, ADAPTIVE_CALIBRATION_MARGIN


def _pinball_loss_torch(pred, target, quantile: float):
    import torch

    err = target - pred
    return torch.maximum(quantile * err, (quantile - 1.0) * err).mean()


def _train_quantile_mlp_torch(
    X_train: np.ndarray,
    budget_train: np.ndarray,
    X_val: np.ndarray,
    budget_val: np.ndarray,
    cfg: AdaptiveThresholdConfig,
    verbose: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    feat_mean = X_train.mean(axis=0)
    feat_std = X_train.std(axis=0)
    feat_std[feat_std < 1e-8] = 1.0

    X_tr_n = (X_train - feat_mean) / feat_std
    X_val_n = (X_val - feat_mean) / feat_std

    dims = [FEATURE_DIM] + list(cfg.hidden_dims) + [1]
    layers: List[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.ReLU())
            if cfg.dropout > 0:
                layers.append(nn.Dropout(cfg.dropout))
    model = nn.Sequential(*layers)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.epochs,
        eta_min=cfg.lr * 0.01,
    )

    tr_ds = TensorDataset(
        torch.from_numpy(X_tr_n).float(),
        torch.from_numpy(budget_train).float(),
    )
    tr_dl = DataLoader(
        tr_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_X_t = torch.from_numpy(X_val_n).float()
    val_budget_t = torch.from_numpy(budget_val).float()

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(cfg.epochs):
        model.train()
        epoch_loss = 0.0
        for xb, yb in tr_dl:
            optimizer.zero_grad()
            raw = model(xb).squeeze(-1)
            pred = torch.nn.functional.softplus(raw)
            loss = _pinball_loss_torch(pred, yb, cfg.quantile)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        epoch_loss /= max(len(tr_ds), 1)
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_raw = model(val_X_t).squeeze(-1)
            val_pred = torch.nn.functional.softplus(val_raw)
            val_loss = _pinball_loss_torch(val_pred, val_budget_t, cfg.quantile).item()
            mae = torch.mean(torch.abs(val_pred - val_budget_t)).item()

        if verbose and (epoch + 1) % 10 == 0:
            print(
                f"  Epoch {epoch+1:3d}/{cfg.epochs}: "
                f"train_loss={epoch_loss:.4f} val_loss={val_loss:.4f} val_mae={mae:.3f}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch + 1}")
                break

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        val_raw = model(val_X_t).squeeze(-1)
        val_pred = torch.nn.functional.softplus(val_raw).cpu().numpy()

    sd = model.state_dict()
    weight_layers = []
    idx = 0
    while True:
        wk = f"{idx}.weight"
        bk = f"{idx}.bias"
        if wk not in sd:
            break
        weight_layers.append({
            "W": sd[wk].cpu().numpy().tolist(),
            "b": sd[bk].cpu().numpy().tolist(),
        })
        idx += 1
        while f"{idx}.weight" not in sd and idx < 32:
            idx += 1

    metrics = {
        "val_pinball_loss": float(best_val_loss),
        "val_mae": float(np.mean(np.abs(val_pred - budget_val))),
        "n_train_regression": int(len(X_train)),
        "n_val_regression": int(len(X_val)),
        "quantile": float(cfg.quantile),
    }
    weights = {
        "layers": weight_layers,
        "feature_mean": feat_mean.tolist(),
        "feature_std": feat_std.tolist(),
        "budget_max": ADAPTIVE_BUDGET_MAX,
        "quantile": float(cfg.quantile),
        "calibration_margin": ADAPTIVE_CALIBRATION_MARGIN,
    }
    return weights, metrics


def _compute_group_baselines(
    budgets: np.ndarray,
    meta: List[Dict[str, Any]],
    y: np.ndarray,
    quantile: float,
) -> Tuple[Dict[str, float], float]:
    group_values: Dict[str, List[float]] = {}
    all_values: List[float] = []
    for i, label in enumerate(y):
        if label != 0:
            continue
        group_key = meta[i]["group_key"]
        group_values.setdefault(group_key, []).append(float(budgets[i]))
        all_values.append(float(budgets[i]))

    global_baseline = float(np.quantile(all_values, quantile)) if all_values else 0.0
    baselines = {}
    for group_key, values in group_values.items():
        baselines[group_key] = float(np.quantile(values, quantile)) if values else global_baseline
    return baselines, global_baseline


def _compute_group_slack(
    predictor: MLPAdaptiveThresholdPredictor,
    X_val: np.ndarray,
    budget_val: np.ndarray,
    y_val: np.ndarray,
    meta_val: List[Dict[str, Any]],
    *,
    coverage: float = 1.0,
    min_group_negatives: int = 8,
    margin: float = ADAPTIVE_CALIBRATION_MARGIN,
) -> Tuple[Dict[str, float], float]:
    residuals_by_group: Dict[str, List[float]] = {}
    global_residuals: List[float] = []

    for i, label in enumerate(y_val):
        if label != 0:
            continue
        group_key = meta_val[i]["group_key"]
        raw_budget = predictor._forward(X_val[i])
        residual = float(budget_val[i] - raw_budget)
        residuals_by_group.setdefault(group_key, []).append(residual)
        global_residuals.append(residual)

    if not global_residuals:
        return {}, 0.0

    global_slack = max(
        float(np.quantile(global_residuals, coverage)) + margin,
        0.0,
    )
    group_slack = {}
    for group_key, residuals in residuals_by_group.items():
        if len(residuals) < min_group_negatives:
            continue
        group_slack[group_key] = max(
            float(np.quantile(residuals, coverage)) + margin,
            0.0,
        )
    return group_slack, global_slack


def train_and_save(
    output_dir: str,
    save_path: str,
    *,
    mlp_cfg: Optional[AdaptiveThresholdConfig] = None,
    coverage: float = 1.0,
    min_group_negatives: int = 8,
    active_branch_gate: int = 2,
    test_fraction: float = 0.15,
    val_fraction: float = 0.15,
    split_seed: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    cfg = mlp_cfg or AdaptiveThresholdConfig()
    data = build_adaptive_threshold_dataset(
        output_dir,
        active_branch_gate=active_branch_gate,
        test_fraction=test_fraction,
        val_fraction=val_fraction,
        split_seed=split_seed,
        verbose=verbose,
    )

    X_train = data["X_train"]
    y_train = data["y_train"]
    budget_train = data["budget_train"]
    meta_train = data["meta_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    budget_val = data["budget_val"]
    meta_val = data["meta_val"]
    X_test = data["X_test"]
    y_test = data["y_test"]
    budget_test = data["budget_test"]
    meta_test = data["meta_test"]

    neg_train = y_train == 0
    neg_val = y_val == 0
    if neg_train.sum() == 0 or neg_val.sum() == 0:
        raise RuntimeError("Adaptive threshold training requires negative samples in train and val.")

    if verbose:
        print(
            f"\nTraining adaptive threshold MLP "
            f"(input={FEATURE_DIM}, hidden={cfg.hidden_dims}, quantile={cfg.quantile:.2f}) ..."
        )

    weights, train_metrics = _train_quantile_mlp_torch(
        X_train[neg_train],
        budget_train[neg_train],
        X_val[neg_val],
        budget_val[neg_val],
        cfg,
        verbose=verbose,
    )

    group_baselines, global_baseline = _compute_group_baselines(
        budget_train,
        meta_train,
        y_train,
        cfg.quantile,
    )
    weights["group_baselines"] = group_baselines
    weights["global_baseline"] = global_baseline

    predictor = MLPAdaptiveThresholdPredictor(weights)
    group_slack, global_slack = _compute_group_slack(
        predictor,
        X_val,
        budget_val,
        y_val,
        meta_val,
        coverage=coverage,
        min_group_negatives=min_group_negatives,
    )
    weights["group_slack"] = group_slack
    weights["global_slack"] = global_slack

    predictor = MLPAdaptiveThresholdPredictor(weights)
    heuristic = HeuristicAdaptiveThresholdPredictor(
        group_budgets=group_baselines,
        group_slack=group_slack,
        global_budget=global_baseline,
        global_slack=global_slack,
        budget_max=weights["budget_max"],
    )

    train_metrics_eval = evaluate_adaptive_threshold_predictor(
        predictor,
        X_train,
        y_train,
        budget_train,
        meta_train,
        active_branch_gate=active_branch_gate,
    )
    val_metrics = evaluate_adaptive_threshold_predictor(
        predictor,
        X_val,
        y_val,
        budget_val,
        meta_val,
        active_branch_gate=active_branch_gate,
    )
    test_metrics = evaluate_adaptive_threshold_predictor(
        predictor,
        X_test,
        y_test,
        budget_test,
        meta_test,
        active_branch_gate=active_branch_gate,
    )
    heuristic_test_metrics = evaluate_adaptive_threshold_predictor(
        heuristic,
        X_test,
        y_test,
        budget_test,
        meta_test,
        active_branch_gate=active_branch_gate,
    )

    metrics = {
        **train_metrics,
        "coverage": float(coverage),
        "min_group_negatives": int(min_group_negatives),
        "group_count": int(len(group_baselines)),
        "global_baseline": float(global_baseline),
        "global_slack": float(global_slack),
        "active_branch_gate": int(active_branch_gate),
        "split_seed": int(split_seed),
        "test_fraction": float(test_fraction),
        "val_fraction": float(val_fraction),
        "train": train_metrics_eval,
        "val": val_metrics,
        "test": test_metrics,
        "heuristic_test": heuristic_test_metrics,
        "n_train_total": int(len(X_train)),
        "n_val_total": int(len(X_val)),
        "n_test_total": int(len(X_test)),
    }

    save_file = Path(save_path)
    save_file.parent.mkdir(parents=True, exist_ok=True)
    save_file.write_text(json.dumps(weights, ensure_ascii=False, indent=1))
    save_file.with_suffix(".metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2)
    )
    save_file.with_name("adaptive_threshold_priors.json").write_text(
        json.dumps(data["config_priors"], ensure_ascii=False, indent=2)
    )
    if verbose:
        print(f"\nAdaptive threshold weights saved to {save_file}")
        print(f"Metrics saved to {save_file.with_suffix('.metrics.json')}")

    return metrics
