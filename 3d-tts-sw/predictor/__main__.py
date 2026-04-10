"""
CLI for the standalone adaptive-threshold predictor package.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def cmd_train(args) -> None:
    from .adaptive_threshold_trainer import train_and_save
    from .config import AdaptiveThresholdConfig

    cfg = AdaptiveThresholdConfig(
        hidden_dims=[int(x) for x in args.hidden.split(",") if x.strip()],
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        patience=args.patience,
        dropout=args.dropout,
        quantile=args.quantile,
    )
    metrics = train_and_save(
        args.output_dir,
        args.save,
        mlp_cfg=cfg,
        coverage=args.coverage,
        min_group_negatives=args.min_group_negatives,
        active_branch_gate=args.active_branch_gate,
        test_fraction=args.test_fraction,
        val_fraction=args.val_fraction,
        split_seed=args.split_seed,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def cmd_evaluate(args) -> None:
    from .adaptive_threshold_dataset import build_adaptive_threshold_dataset
    from .adaptive_threshold_evaluate import evaluate_adaptive_threshold_predictor
    from .adaptive_threshold_model import (
        HeuristicAdaptiveThresholdPredictor,
        MLPAdaptiveThresholdPredictor,
    )

    if args.weights and Path(args.weights).exists():
        predictor = MLPAdaptiveThresholdPredictor.load(args.weights)
        print(f"Loaded adaptive threshold weights from {args.weights}")
    else:
        predictor = HeuristicAdaptiveThresholdPredictor()
        print("No weights found, using heuristic adaptive threshold predictor")

    data = build_adaptive_threshold_dataset(
        args.output_dir,
        active_branch_gate=args.active_branch_gate,
        test_fraction=args.test_fraction,
        val_fraction=args.val_fraction,
        split_seed=args.split_seed,
        verbose=True,
    )
    report = {
        "train": evaluate_adaptive_threshold_predictor(
            predictor,
            data["X_train"],
            data["y_train"],
            data["budget_train"],
            data["meta_train"],
            active_branch_gate=args.active_branch_gate,
        ),
        "val": evaluate_adaptive_threshold_predictor(
            predictor,
            data["X_val"],
            data["y_val"],
            data["budget_val"],
            data["meta_val"],
            active_branch_gate=args.active_branch_gate,
        ),
        "test": evaluate_adaptive_threshold_predictor(
            predictor,
            data["X_test"],
            data["y_test"],
            data["budget_test"],
            data["meta_test"],
            active_branch_gate=args.active_branch_gate,
        ),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.save_report:
        Path(args.save_report).write_text(json.dumps(report, ensure_ascii=False, indent=2))


def cmd_predict(args) -> None:
    from .adaptive_threshold_features import extract_adaptive_threshold_features
    from .adaptive_threshold_model import (
        HeuristicAdaptiveThresholdPredictor,
        MLPAdaptiveThresholdPredictor,
    )

    if args.weights and Path(args.weights).exists():
        predictor = MLPAdaptiveThresholdPredictor.load(args.weights)
    else:
        predictor = HeuristicAdaptiveThresholdPredictor()

    probs = json.loads(args.token_probs)
    topk = json.loads(args.topk) if args.topk else None
    sibling_counts = json.loads(args.sibling_counts)
    obs_point = args.obs_point if args.obs_point > 0 else len(probs)
    current_tokens = args.current_tokens if args.current_tokens > 0 else obs_point
    step_total_tokens = (
        args.step_total_tokens if args.step_total_tokens > 0 else obs_point + sum(sibling_counts)
    )
    n_active_branches = (
        args.n_active_branches
        if args.n_active_branches > 0
        else 1 + sum(1 for count in sibling_counts if count >= obs_point)
    )
    group_key = args.group_key
    if not group_key and args.policy_model and args.beam_width > 0:
        group_key = f"{args.policy_model}|beam={args.beam_width}"

    features = extract_adaptive_threshold_features(
        token_probs=probs,
        token_topk_logprobs=topk,
        obs_point=obs_point,
        max_other_tokens=args.max_other,
        sibling_token_counts=sibling_counts,
        n_branches=args.n_branches,
        n_active_branches=n_active_branches,
        step_idx=args.step_idx,
        total_steps=args.total_steps,
        step_total_tokens=step_total_tokens,
        prior_step_total_tokens=args.prior_step_total_tokens,
        beam_width=args.beam_width,
        config_prior=args.config_prior,
    )
    budget = predictor.predict_budget(features, group_key=group_key)
    fired, threshold = predictor.predict_label(
        features,
        current_tokens=current_tokens,
        max_other_tokens=args.max_other,
        n_active_branches=n_active_branches,
        active_branch_gate=args.active_branch_gate,
        group_key=group_key,
    )
    print(json.dumps({
        "group_key": group_key,
        "budget": round(float(budget), 3),
        "threshold": round(float(threshold), 3),
        "current_tokens": current_tokens,
        "n_active_branches": n_active_branches,
        "active_branch_gate": args.active_branch_gate,
        "fired": bool(fired),
    }))


def cmd_predict_record_step(args) -> None:
    from .record_step_predict import predict_record_step

    report = predict_record_step(
        args.record_path,
        step_idx=args.step_idx,
        weights_path=args.weights,
        priors_path=args.priors,
        active_branch_gate=args.active_branch_gate,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m predictor",
        description="Standalone adaptive-threshold predictor",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    tr = sub.add_parser("train", help="Train adaptive threshold predictor")
    tr.add_argument("--output-dir", type=str, required=True)
    tr.add_argument("--save", type=str, default="predictor/adaptive_threshold_weights.json")
    tr.add_argument("--hidden", type=str, default="32,16")
    tr.add_argument("--epochs", type=int, default=60)
    tr.add_argument("--lr", type=float, default=1e-3)
    tr.add_argument("--batch-size", type=int, default=256)
    tr.add_argument("--patience", type=int, default=10)
    tr.add_argument("--dropout", type=float, default=0.05)
    tr.add_argument("--quantile", type=float, default=0.97)
    tr.add_argument("--coverage", type=float, default=1.0)
    tr.add_argument("--min-group-negatives", type=int, default=8)
    tr.add_argument("--active-branch-gate", type=int, choices=[1, 2], default=2)
    tr.add_argument("--test-fraction", type=float, default=0.15)
    tr.add_argument("--val-fraction", type=float, default=0.15)
    tr.add_argument("--split-seed", type=int, default=42)

    ev = sub.add_parser("evaluate", help="Evaluate adaptive threshold predictor")
    ev.add_argument("--output-dir", type=str, required=True)
    ev.add_argument("--weights", type=str, default="predictor/adaptive_threshold_weights.json")
    ev.add_argument("--save-report", type=str, default="")
    ev.add_argument("--active-branch-gate", type=int, choices=[1, 2], default=2)
    ev.add_argument("--test-fraction", type=float, default=0.15)
    ev.add_argument("--val-fraction", type=float, default=0.15)
    ev.add_argument("--split-seed", type=int, default=42)

    pr = sub.add_parser("predict", help="Predict adaptive threshold for one branch")
    pr.add_argument("--weights", type=str, default="predictor/adaptive_threshold_weights.json")
    pr.add_argument("--token-probs", type=str, required=True)
    pr.add_argument("--topk", type=str, default="")
    pr.add_argument("--sibling-counts", type=str, required=True)
    pr.add_argument("--max-other", type=int, required=True)
    pr.add_argument("--n-branches", type=int, required=True)
    pr.add_argument("--n-active-branches", type=int, default=0)
    pr.add_argument("--step-idx", type=int, default=0)
    pr.add_argument("--total-steps", type=int, default=1)
    pr.add_argument("--step-total-tokens", type=int, default=0)
    pr.add_argument("--prior-step-total-tokens", type=int, default=0)
    pr.add_argument("--beam-width", type=int, default=0)
    pr.add_argument("--config-prior", type=float, default=0.5)
    pr.add_argument("--obs-point", type=int, default=0)
    pr.add_argument("--current-tokens", type=int, default=0)
    pr.add_argument("--policy-model", type=str, default="")
    pr.add_argument("--group-key", type=str, default="")
    pr.add_argument("--active-branch-gate", type=int, choices=[1, 2], default=2)

    prs = sub.add_parser(
        "predict-record-step",
        help="Predict threshold firing for all branches in one record step",
    )
    prs.add_argument("--record-path", type=str, required=True)
    prs.add_argument("--step-idx", type=int, required=True)
    prs.add_argument("--weights", type=str, default="predictor/adaptive_threshold_weights.json")
    prs.add_argument("--priors", type=str, default="predictor/adaptive_threshold_priors.json")
    prs.add_argument("--active-branch-gate", type=int, choices=[1, 2], default=2)

    args = parser.parse_args()
    if args.command == "train":
        cmd_train(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "predict":
        cmd_predict(args)
    elif args.command == "predict-record-step":
        cmd_predict_record_step(args)


if __name__ == "__main__":
    main()
