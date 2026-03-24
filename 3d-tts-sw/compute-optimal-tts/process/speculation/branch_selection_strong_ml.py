#!/usr/bin/env python3
"""
更强预测器（不使用 reward 特征）：
- MLPClassifier (神经网络)
- HistGradientBoostingClassifier

任务：预测 branch 是否被选中，并以 step-level top1 accuracy 为主指标。
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning("Failed to load %s: %s", path, e)
        return None


def mean_or_zero(xs: List[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0


def std_or_zero(xs: List[float]) -> float:
    if not xs:
        return 0.0
    arr = np.asarray(xs, dtype=float)
    return float(arr.std())


def safe_last(xs: List[float]) -> float:
    return float(xs[-1]) if xs else 0.0


def common_prefix_len(all_token_probs: List[List[float]]) -> int:
    if not all_token_probs:
        return 0
    m = min(len(x) for x in all_token_probs)
    pref = 0
    for i in range(m):
        v = all_token_probs[0][i]
        if all(x[i] == v for x in all_token_probs[1:]):
            pref += 1
        else:
            break
    return pref


def build_rows(files: List[Path], n_head_tokens: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    step_id = 0
    for fp in files:
        obj = load_json(fp)
        if obj is None:
            continue
        for out in obj.get("decode", {}).get("outputs", []):
            hist_selected: List[int] = []
            for step in out.get("step_branch_metrics", []):
                branch_count = int(step.get("branch_count", 0))
                if branch_count <= 1:
                    continue
                selected_idx = int(step.get("selected_branch_index", -1))
                branches = step.get("branches", [])
                all_probs = [b.get("token_probs", []) for b in branches]
                pref = common_prefix_len(all_probs)

                probs = [float(b.get("probability", 0.0)) for b in branches]
                prob_order = np.argsort(-np.asarray(probs))
                prob_rank = {int(idx): int(r) for r, idx in enumerate(prob_order)}

                prev_sel = hist_selected[-1] if hist_selected else -1
                prev2_sel = hist_selected[-2] if len(hist_selected) >= 2 else -1
                streak = 0
                i = len(hist_selected) - 1
                while i >= 0 and hist_selected[i] == prev_sel and prev_sel >= 0:
                    streak += 1
                    i -= 1

                for b in branches:
                    idx = int(b.get("branch_index", 0))
                    tps = b.get("token_probs", [])
                    trimmed = tps[pref:] if pref < len(tps) else []
                    head = trimmed[:n_head_tokens] if n_head_tokens > 0 else trimmed
                    rows.append(
                        {
                            "step_id": step_id,
                            "label": 1 if idx == selected_idx else 0,
                            "branch_index": idx,
                            "branch_count": branch_count,
                            "step_num": int(step.get("step", 0)),
                            "probability": float(b.get("probability", 0.0)),
                            "num_tokens": int(b.get("num_tokens", 0)),
                            "token_mean": mean_or_zero(tps),
                            "token_std": std_or_zero(tps),
                            "token_last": safe_last(tps),
                            "trimmed_head_mean": mean_or_zero(head),
                            "trimmed_head_last": safe_last(head),
                            "common_prefix_len": pref,
                            "prob_rank": prob_rank.get(idx, branch_count),
                            "is_prev_selected_same_idx": 1 if idx == prev_sel else 0,
                            "is_prev2_selected_same_idx": 1 if idx == prev2_sel else 0,
                            "prev_selected_streak": streak if idx == prev_sel else 0,
                        }
                    )

                if selected_idx >= 0:
                    hist_selected.append(selected_idx)
                step_id += 1
    return rows


def group_split(step_ids: np.ndarray, test_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    uniq = np.unique(step_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n_test = max(1, int(len(uniq) * test_ratio))
    test_steps = set(uniq[:n_test].tolist())
    test_mask = np.array([sid in test_steps for sid in step_ids], dtype=bool)
    return ~test_mask, test_mask


def top1_acc(step_ids: np.ndarray, y_true: np.ndarray, y_score: np.ndarray) -> float:
    ok = 0
    tot = 0
    for sid in np.unique(step_ids):
        idx = np.where(step_ids == sid)[0]
        if len(idx) <= 1:
            continue
        pick = idx[np.argmax(y_score[idx])]
        ok += int(y_true[pick] == 1)
        tot += 1
    return float(ok / tot) if tot else 0.0


def train_eval(rows: List[Dict[str, Any]], test_ratio: float, seed: int) -> Dict[str, Any]:
    features = [
        "branch_index",
        "branch_count",
        "step_num",
        "probability",
        "num_tokens",
        "token_mean",
        "token_std",
        "token_last",
        "trimmed_head_mean",
        "trimmed_head_last",
        "common_prefix_len",
        "prob_rank",
        "is_prev_selected_same_idx",
        "is_prev2_selected_same_idx",
        "prev_selected_streak",
    ]

    X = np.asarray([[float(r[f]) for f in features] for r in rows], dtype=float)
    y = np.asarray([int(r["label"]) for r in rows], dtype=int)
    step_ids = np.asarray([int(r["step_id"]) for r in rows], dtype=int)
    tr, te = group_split(step_ids, test_ratio=test_ratio, seed=seed)

    X_train, y_train = X[tr], y[tr]
    X_test, y_test = X[te], y[te]
    steps_test = step_ids[te]

    # 基线
    prob_baseline = X_test[:, features.index("probability")]

    mlp = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "clf",
                MLPClassifier(
                    hidden_layer_sizes=(128, 64),
                    activation="relu",
                    alpha=1e-4,
                    learning_rate_init=1e-3,
                    max_iter=300,
                    random_state=seed,
                    early_stopping=True,
                    validation_fraction=0.1,
                ),
            ),
        ]
    )
    mlp.fit(X_train, y_train)
    mlp_score = mlp.predict_proba(X_test)[:, 1]

    hgb = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "clf",
                HistGradientBoostingClassifier(
                    max_depth=6,
                    learning_rate=0.05,
                    max_iter=500,
                    min_samples_leaf=30,
                    l2_regularization=1e-2,
                    random_state=seed,
                ),
            ),
        ]
    )
    hgb.fit(X_train, y_train)
    hgb_score = hgb.predict_proba(X_test)[:, 1]

    metrics = {
        "sample_count_train": int(len(y_train)),
        "sample_count_test": int(len(y_test)),
        "positive_rate_train": float(y_train.mean()),
        "positive_rate_test": float(y_test.mean()),
        "mlp_auc": float(roc_auc_score(y_test, mlp_score)),
        "mlp_ap": float(average_precision_score(y_test, mlp_score)),
        "mlp_top1_acc": top1_acc(steps_test, y_test, mlp_score),
        "hgb_auc": float(roc_auc_score(y_test, hgb_score)),
        "hgb_ap": float(average_precision_score(y_test, hgb_score)),
        "hgb_top1_acc": top1_acc(steps_test, y_test, hgb_score),
        "prob_baseline_top1_acc": top1_acc(steps_test, y_test, prob_baseline),
    }
    return {"metrics": metrics, "feature_names": features}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train stronger no-reward branch selector")
    parser.add_argument(
        "--input",
        default="/DISK1/data/rnxu_24/Paper/3d-tts-simulator/3d-tts-sw/compute-optimal-tts/process/speculation/output",
        help="Input directory with record_0.json files",
    )
    parser.add_argument(
        "--output",
        default="/DISK1/data/rnxu_24/Paper/3d-tts-simulator/3d-tts-sw/compute-optimal-tts/process/speculation/output/branch_selection_strong_ml_result.json",
        help="Output json path",
    )
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    files = sorted(Path(args.input).rglob("record_0.json"))
    logger.info("Found %d workload files", len(files))
    rows = build_rows(files, n_head_tokens=args.n_head)
    logger.info("Built %d samples", len(rows))

    result = train_eval(rows, test_ratio=args.test_ratio, seed=args.seed)
    result["config"] = {
        "input_files": len(files),
        "rows": len(rows),
        "n_head": args.n_head,
        "test_ratio": args.test_ratio,
        "seed": args.seed,
        "reward_features_used": False,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info("Saved result to %s", out)
    logger.info("Metrics: %s", result["metrics"])


if __name__ == "__main__":
    main()

