#!/usr/bin/env python3
"""
利用已有 speculation workload 数据，训练小模型预测 branch 是否被选中。

数据来源:
  process/speculation/output/**/record_0.json

样本定义:
  每个 multi-branch step 里的每个 branch 是一个样本
  label = 1 表示该 branch 被选中，否则 0

评估:
  1) branch-level AUC / AP
  2) step-level top1 accuracy（每个 step 取预测分最高的 branch，看是否命中 selected）
  3) 与简单基线（按 reward / probability 取最大）比较
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
except Exception:  # pragma: no cover
    ColumnTransformer = None
    GradientBoostingClassifier = None
    SimpleImputer = None
    LogisticRegression = None
    average_precision_score = None
    roc_auc_score = None
    MLPClassifier = None
    Pipeline = None
    StandardScaler = None


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
    arr = np.array(xs, dtype=float)
    return float(arr.std())


def safe_last(xs: List[float]) -> float:
    return float(xs[-1]) if xs else 0.0


def weighted_head_mean(xs: List[float], n: int) -> float:
    """位置加权均值：越靠前权重越大。"""
    if not xs:
        return 0.0
    head = xs[:n] if n > 0 else xs
    if not head:
        return 0.0
    w = np.arange(len(head), 0, -1, dtype=float)
    arr = np.asarray(head, dtype=float)
    return float((arr * w).sum() / w.sum())


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


def build_dataset(files: List[Path], n_head_tokens: int) -> Dict[str, Any]:
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

                rewards = [float(b.get("reward", 0.0)) for b in branches]
                probs = [float(b.get("probability", 0.0)) for b in branches]
                tok_lens = [int(b.get("num_tokens", 0)) for b in branches]
                probs_arr = np.asarray(probs, dtype=float)
                lens_arr = np.asarray(tok_lens, dtype=float)
                probs_mean = float(probs_arr.mean()) if len(probs_arr) else 0.0
                probs_max = float(probs_arr.max()) if len(probs_arr) else 0.0
                probs_std = float(probs_arr.std()) if len(probs_arr) else 0.0
                lens_mean = float(lens_arr.mean()) if len(lens_arr) else 0.0

                # 简单排序特征（step 内相对强弱）
                reward_order = np.argsort(-np.array(rewards))
                prob_order = np.argsort(-np.array(probs))
                reward_rank = {int(idx): int(r) for r, idx in enumerate(reward_order)}
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
                    token_mean = mean_or_zero(tps)
                    token_std = std_or_zero(tps)
                    token_last = safe_last(tps)
                    trimmed_head_mean = mean_or_zero(head)
                    trimmed_head_last = safe_last(head)
                    trimmed_head_wmean = weighted_head_mean(trimmed, n_head_tokens)
                    prob = float(b.get("probability", 0.0))
                    n_tok = int(b.get("num_tokens", 0))

                    row = {
                        "file": str(fp),
                        "step_id": step_id,
                        "label": 1 if idx == selected_idx else 0,
                        "branch_index": idx,
                        "branch_count": branch_count,
                        "step_num": int(step.get("step", 0)),
                        "reward": float(b.get("reward", 0.0)),
                        "probability": prob,
                        "num_tokens": n_tok,
                        "token_mean": token_mean,
                        "token_std": token_std,
                        "token_last": token_last,
                        "trimmed_head_mean": trimmed_head_mean,
                        "trimmed_head_last": trimmed_head_last,
                        "trimmed_head_wmean": trimmed_head_wmean,
                        "common_prefix_len": pref,
                        "reward_rank": reward_rank.get(idx, branch_count),
                        "prob_rank": prob_rank.get(idx, branch_count),
                        # step 内相对特征
                        "prob_minus_step_mean": prob - probs_mean,
                        "prob_gap_to_step_max": probs_max - prob,
                        "prob_step_std": probs_std,
                        "len_minus_step_mean": n_tok - lens_mean,
                        # 一些交互特征
                        "prob_x_trimmed_head_wmean": prob * trimmed_head_wmean,
                        "prob_x_token_last": prob * token_last,
                        "prob_div_num_tokens": prob / max(1, n_tok),
                        "is_prev_selected_same_idx": 1 if idx == prev_sel else 0,
                        "is_prev2_selected_same_idx": 1 if idx == prev2_sel else 0,
                        "prev_selected_streak": streak if idx == prev_sel else 0,
                    }
                    rows.append(row)

                if selected_idx >= 0:
                    hist_selected.append(selected_idx)
                step_id += 1

    return {"rows": rows}


def group_train_test_split(step_ids: np.ndarray, test_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    uniq = np.unique(step_ids)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n_test = max(1, int(len(uniq) * test_ratio))
    test_steps = set(uniq[:n_test].tolist())
    test_mask = np.array([sid in test_steps for sid in step_ids], dtype=bool)
    train_mask = ~test_mask
    return train_mask, test_mask


def top1_accuracy(step_ids: np.ndarray, y_true: np.ndarray, y_score: np.ndarray) -> float:
    ok = 0
    total = 0
    for sid in np.unique(step_ids):
        idx = np.where(step_ids == sid)[0]
        if len(idx) <= 1:
            continue
        pick = idx[np.argmax(y_score[idx])]
        if y_true[pick] == 1:
            ok += 1
        total += 1
    return float(ok / total) if total else 0.0


def run_training(
    rows: List[Dict[str, Any]],
    test_ratio: float,
    seed: int,
    exclude_reward_features: bool = False,
) -> Dict[str, Any]:
    if not rows:
        raise RuntimeError("No training rows built from input data.")
    if LogisticRegression is None:
        raise RuntimeError("scikit-learn is not available. Please install sklearn first.")

    feature_names = [
        "branch_index",
        "branch_count",
        "step_num",
        "reward",
        "probability",
        "num_tokens",
        "token_mean",
        "token_std",
        "token_last",
        "trimmed_head_mean",
        "trimmed_head_last",
        "trimmed_head_wmean",
        "common_prefix_len",
        "reward_rank",
        "prob_rank",
        "prob_minus_step_mean",
        "prob_gap_to_step_max",
        "prob_step_std",
        "len_minus_step_mean",
        "prob_x_trimmed_head_wmean",
        "prob_x_token_last",
        "prob_div_num_tokens",
        "is_prev_selected_same_idx",
        "is_prev2_selected_same_idx",
        "prev_selected_streak",
    ]
    if exclude_reward_features:
        feature_names = [f for f in feature_names if f not in {"reward", "reward_rank"}]

    X = np.array([[float(r[n]) for n in feature_names] for r in rows], dtype=float)
    y = np.array([int(r["label"]) for r in rows], dtype=int)
    step_ids = np.array([int(r["step_id"]) for r in rows], dtype=int)

    train_mask, test_mask = group_train_test_split(step_ids, test_ratio=test_ratio, seed=seed)
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    steps_test = step_ids[test_mask]

    # 模型1: 逻辑回归
    lr = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )
    lr.fit(X_train, y_train)
    lr_score = lr.predict_proba(X_test)[:, 1]

    # 模型2: 小型 GBDT
    gbdt = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", GradientBoostingClassifier(random_state=seed)),
        ]
    )
    gbdt.fit(X_train, y_train)
    gbdt_score = gbdt.predict_proba(X_test)[:, 1]

    # 模型3: MLP（对非线性特征组合更友好）
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
                    learning_rate_init=8e-4,
                    max_iter=400,
                    random_state=seed,
                    early_stopping=True,
                    validation_fraction=0.1,
                ),
            ),
        ]
    )
    mlp.fit(X_train, y_train)
    mlp_score = mlp.predict_proba(X_test)[:, 1]

    # 基线：按 reward / prob 直接排序
    reward_baseline = None
    if "reward" in feature_names:
        reward_baseline = X_test[:, feature_names.index("reward")]
    prob_baseline = X_test[:, feature_names.index("probability")] if "probability" in feature_names else np.zeros(len(X_test))

    metrics = {
        "sample_count_train": int(len(y_train)),
        "sample_count_test": int(len(y_test)),
        "positive_rate_train": float(y_train.mean()),
        "positive_rate_test": float(y_test.mean()),
        "lr_auc": float(roc_auc_score(y_test, lr_score)),
        "lr_ap": float(average_precision_score(y_test, lr_score)),
        "lr_top1_acc": top1_accuracy(steps_test, y_test, lr_score),
        "gbdt_auc": float(roc_auc_score(y_test, gbdt_score)),
        "gbdt_ap": float(average_precision_score(y_test, gbdt_score)),
        "gbdt_top1_acc": top1_accuracy(steps_test, y_test, gbdt_score),
        "mlp_auc": float(roc_auc_score(y_test, mlp_score)),
        "mlp_ap": float(average_precision_score(y_test, mlp_score)),
        "mlp_top1_acc": top1_accuracy(steps_test, y_test, mlp_score),
        "reward_baseline_top1_acc": top1_accuracy(steps_test, y_test, reward_baseline) if reward_baseline is not None else None,
        "prob_baseline_top1_acc": top1_accuracy(steps_test, y_test, prob_baseline),
    }

    lr_coef = lr.named_steps["clf"].coef_[0]
    lr_feat_importance = sorted(
        [{"feature": n, "coef": float(c), "abs_coef": float(abs(c))} for n, c in zip(feature_names, lr_coef)],
        key=lambda x: x["abs_coef"],
        reverse=True,
    )

    gbdt_importance = gbdt.named_steps["clf"].feature_importances_
    gbdt_feat_importance = sorted(
        [{"feature": n, "importance": float(v)} for n, v in zip(feature_names, gbdt_importance)],
        key=lambda x: x["importance"],
        reverse=True,
    )

    mlp_first_layer = mlp.named_steps["clf"].coefs_[0]
    mlp_input_importance = np.mean(np.abs(mlp_first_layer), axis=1)
    mlp_feat_importance = sorted(
        [{"feature": n, "importance": float(v)} for n, v in zip(feature_names, mlp_input_importance)],
        key=lambda x: x["importance"],
        reverse=True,
    )

    return {
        "metrics": metrics,
        "feature_names": feature_names,
        "lr_feature_importance": lr_feat_importance,
        "gbdt_feature_importance": gbdt_feat_importance,
        "mlp_feature_importance": mlp_feat_importance,
    }


def run_training_by_branch_count(
    rows: List[Dict[str, Any]],
    test_ratio: float,
    seed: int,
    exclude_reward_features: bool = False,
    min_steps: int = 50,
) -> Dict[str, Any]:
    """
    按 branch_count 分组分别训练/评估。
    只有当分组里的 step 数 >= min_steps 时才训练，避免样本太少不稳定。
    """
    groups: Dict[int, List[Dict[str, Any]]] = {}
    for r in rows:
        bc = int(r.get("branch_count", 0))
        groups.setdefault(bc, []).append(r)

    out: Dict[str, Any] = {}
    for bc, sub_rows in sorted(groups.items(), key=lambda x: x[0]):
        step_ids = {int(x["step_id"]) for x in sub_rows}
        if len(step_ids) < min_steps:
            out[str(bc)] = {
                "skipped": True,
                "reason": f"insufficient steps: {len(step_ids)} < {min_steps}",
                "row_count": len(sub_rows),
                "step_count": len(step_ids),
            }
            continue
        try:
            result = run_training(
                rows=sub_rows,
                test_ratio=test_ratio,
                seed=seed,
                exclude_reward_features=exclude_reward_features,
            )
            out[str(bc)] = {
                "skipped": False,
                "row_count": len(sub_rows),
                "step_count": len(step_ids),
                "metrics": result["metrics"],
                "feature_names": result["feature_names"],
                "lr_feature_importance": result["lr_feature_importance"],
                "gbdt_feature_importance": result["gbdt_feature_importance"],
                "mlp_feature_importance": result["mlp_feature_importance"],
            }
        except Exception as e:
            out[str(bc)] = {
                "skipped": True,
                "reason": f"train_failed: {e}",
                "row_count": len(sub_rows),
                "step_count": len(step_ids),
            }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Train small ML models to predict selected branch")
    parser.add_argument(
        "--input",
        default="/DISK1/data/rnxu_24/Paper/3d-tts-simulator/3d-tts-sw/compute-optimal-tts/process/speculation/output",
        help="Input folder with speculation workload json files",
    )
    parser.add_argument(
        "--output",
        default="/DISK1/data/rnxu_24/Paper/3d-tts-simulator/3d-tts-sw/compute-optimal-tts/process/speculation/output/branch_selection_ml_result.json",
        help="Output result json path",
    )
    parser.add_argument("--n-head", type=int, default=8, help="Head token count after common prefix")
    parser.add_argument("--test-ratio", type=float, default=0.2, help="Step-group test split ratio")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--exclude-reward-features",
        action="store_true",
        help="Exclude reward/reward_rank to avoid target leakage and test non-reward predictability",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument(
        "--split-by-branch-count",
        action="store_true",
        help="Train/evaluate separately for each branch_count",
    )
    parser.add_argument(
        "--min-steps-per-group",
        type=int,
        default=50,
        help="Minimum number of steps required for a branch_count group",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    in_root = Path(args.input)
    files = sorted(p for p in in_root.rglob("record_0.json") if p.is_file())
    logger.info("Found %d workload files", len(files))

    ds = build_dataset(files, n_head_tokens=args.n_head)
    logger.info("Built %d branch-level samples", len(ds["rows"]))

    result = run_training(
        ds["rows"],
        test_ratio=args.test_ratio,
        seed=args.seed,
        exclude_reward_features=args.exclude_reward_features,
    )
    by_branch_count = None
    if args.split_by_branch_count:
        by_branch_count = run_training_by_branch_count(
            ds["rows"],
            test_ratio=args.test_ratio,
            seed=args.seed,
            exclude_reward_features=args.exclude_reward_features,
            min_steps=args.min_steps_per_group,
        )

    result["config"] = {
        "n_head": args.n_head,
        "test_ratio": args.test_ratio,
        "seed": args.seed,
        "exclude_reward_features": args.exclude_reward_features,
        "split_by_branch_count": args.split_by_branch_count,
        "min_steps_per_group": args.min_steps_per_group,
        "input_files": len(files),
        "rows": len(ds["rows"]),
    }
    if by_branch_count is not None:
        result["by_branch_count"] = by_branch_count

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    logger.info("Saved ML result to %s", out)
    logger.info("Metrics: %s", result["metrics"])


if __name__ == "__main__":
    main()

