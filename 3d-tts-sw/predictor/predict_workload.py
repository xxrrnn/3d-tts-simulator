"""
predict_workload.py
===================
Replay recorded workload files through the MLP straggler predictor.

For every step in every question, the script checks each branch that
*exceeds* the longest sibling (max_other_tokens).  Features are extracted
causally (prefix-only, no future info), and the predictor decides
whether the branch should have been terminated.

Ground-truth straggler labels are computed from the ratio / min-tokens
heuristic so we can measure Precision / Recall / F1 and token savings.

Usage
-----
    ``python predictor.predict_workload`` is **invalid** (Python treats that as a
    file path).

    **Important:** ``python -m predictor.predict_workload`` only works when the
    current directory is the **parent** of the ``predictor/`` package
    (e.g. ``3d-tts-sw``). If your shell is **inside** ``predictor/``, Python
    cannot resolve the top-level package name ``predictor`` — use
    ``python -m predict_workload`` (no ``predictor.`` prefix) or
    ``python predict_workload.py`` instead.

    From ``3d-tts-sw``::

        python -m predictor.predict_workload \\
            --workload-root <model_workloads_...> \\
            --weights <weights.json> --priors <priors.json> \\
            --active-branch-gate 1

    From ``3d-tts-sw/predictor`` (this directory)::

        python -m predict_workload \\
            --workload-root <model_workloads_...> \\
            --weights <weights.json> --priors <priors.json> \\
            --active-branch-gate 1

        # equivalent:
        python predict_workload.py ...

    From any cwd (wrapper adds ``3d-tts-sw`` to ``sys.path``)::

        python /path/to/3d-tts-sw/run_predict_workload.py \\
            --workload-root <model_workloads_...> ...
"""

from __future__ import annotations

import argparse
import contextlib
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Imports from the predictor package (sibling modules)
# ---------------------------------------------------------------------------
_PREDICTOR_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _PREDICTOR_DIR.parent          # e.g. .../3d-tts-sw
# Put the parent of predictor/ at the front of sys.path, and remove
# predictor/ itself so that `import predictor` resolves to the package
# under _PROJECT_ROOT rather than the cwd when run from inside predictor/.
sys.path.insert(0, str(_PROJECT_ROOT))
for _shadow in (str(_PREDICTOR_DIR), "."):
    while _shadow in sys.path:
        sys.path.remove(_shadow)

from predictor.adaptive_threshold_features import extract_adaptive_threshold_features
from predictor.adaptive_threshold_model import MLPAdaptiveThresholdPredictor
from predictor.config import (
    ADAPTIVE_STRAGGLER_MIN_BRANCH_TOKENS,
    ADAPTIVE_STRAGGLER_RATIO_THRESHOLD,
)

_RESULT_DIR = _PREDICTOR_DIR / "result"


class _TeeStdout:
    """Write to the original stdout and a text file."""

    def __init__(self, console, log) -> None:
        self._console = console
        self._log = log

    def write(self, s: str) -> int:
        self._console.write(s)
        self._log.write(s)
        return len(s)

    def flush(self) -> None:
        self._console.flush()
        self._log.flush()

    # argparse / help may call isatty()
    def isatty(self) -> bool:
        return self._console.isatty()

    @property
    def encoding(self) -> str:  # type: ignore[override]
        return getattr(self._console, "encoding", "utf-8")


@contextlib.contextmanager
def _tee_stdout_to_file(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as logf:
        old = sys.stdout
        sys.stdout = _TeeStdout(old, logf)
        try:
            yield path
        finally:
            sys.stdout = old


def _safe_result_txt_stem(resolved: Path) -> str:
    """Filename stem from --workload-root / --workload-dir (last path component)."""
    name = resolved.name or "workload"
    for c in '<>:"/\\|?*':
        name = name.replace(c, "_")
    return name or "workload"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_priors(path: str) -> Dict[str, float]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _parse_beam_width_from_dirname(dirname: str) -> int:
    """Extract beam width from config dir name like '40_4_1_straggler_...'."""
    parts = dirname.split("_")
    if len(parts) >= 2:
        try:
            return int(parts[1])
        except ValueError:
            pass
    return 4  # fallback


def _parse_policy_model_from_path(config_dir: Path) -> str:
    """Walk up to find the policy-model directory name."""
    # structure: .../TASK_beam_search/POLICY/REWARD/CONFIG/
    reward_dir = config_dir.parent        # REWARD
    policy_dir = reward_dir.parent        # POLICY
    return policy_dir.name


def _get_config_prior(
    priors: Dict[str, float],
    policy_model: str,
    reward_model: str,
    beam_width: int,
) -> float:
    key = f"{policy_model}|{reward_model}|beam={beam_width}"
    return priors.get(key, 0.5)


def _get_group_key(policy_model: str, beam_width: int) -> str:
    return f"{policy_model}|beam={beam_width}"


# ---------------------------------------------------------------------------
# Core prediction logic (per question)
# ---------------------------------------------------------------------------

def predict_one_question(
    workload_path: Path,
    predictor: MLPAdaptiveThresholdPredictor,
    *,
    config_prior: float,
    beam_width: int,
    active_branch_gate: int,
    group_key: str,
    straggler_ratio: float = ADAPTIVE_STRAGGLER_RATIO_THRESHOLD,
    straggler_min_tokens: int = ADAPTIVE_STRAGGLER_MIN_BRANCH_TOKENS,
) -> List[dict]:
    with open(workload_path, encoding="utf-8") as f:
        data = json.load(f)

    steps = data["decode"]["steps"]
    question_id = data.get("question_id", workload_path.stem)
    records: List[dict] = []

    for step_idx, step in enumerate(steps):
        branch_count = step["branch_count"]
        branch_tokens: List[int] = step["branch_tokens"]
        branch_probs = step.get("branch_token_probs", [])
        branch_topk = step.get("branch_token_topk_logprobs", [])

        if branch_count <= 1:
            continue

        # prior-step total tokens
        prior_step_total = (
            sum(steps[step_idx - 1]["branch_tokens"]) if step_idx > 0 else 0
        )

        for b in range(branch_count):
            cur_tokens = branch_tokens[b]
            other_tokens = [branch_tokens[j] for j in range(branch_count) if j != b]
            max_other = max(other_tokens)

            if cur_tokens <= max_other:
                continue  # predictor not invoked for this branch

            # --- ground truth (ratio + min-tokens heuristic) ---------------
            gt_straggler = (
                cur_tokens / max_other > straggler_ratio
                and cur_tokens > straggler_min_tokens
            )

            # --- causal feature extraction ---------------------------------
            # obs_point = max_other: only use the first max_other token probs
            obs_point = max_other
            probs_b = branch_probs[b] if b < len(branch_probs) else []
            topk_b = branch_topk[b] if b < len(branch_topk) else None

            # n_active = branches still generating when the max_other branch
            # just finished (those with more tokens than max_other)
            n_active = sum(1 for t in branch_tokens if t > max_other)

            features = extract_adaptive_threshold_features(
                token_probs=probs_b,
                token_topk_logprobs=topk_b,
                obs_point=obs_point,
                max_other_tokens=max_other,
                sibling_token_counts=[
                    branch_tokens[j] for j in range(branch_count) if j != b
                ],
                n_branches=branch_count,
                n_active_branches=n_active,
                step_idx=step_idx,
                total_steps=len(steps),
                step_total_tokens=sum(branch_tokens),
                prior_step_total_tokens=prior_step_total,
                beam_width=beam_width,
                config_prior=config_prior,
            )

            fired, threshold = predictor.predict_label(
                features,
                current_tokens=cur_tokens,
                max_other_tokens=max_other,
                n_active_branches=n_active,
                active_branch_gate=active_branch_gate,
                group_key=group_key,
            )

            saved_tokens = max(cur_tokens - int(threshold), 0) if fired else 0

            records.append(
                {
                    "question_id": question_id,
                    "step": step_idx,
                    "branch": b,
                    "branch_count": branch_count,
                    "cur_tokens": cur_tokens,
                    "max_other": max_other,
                    "ratio": round(cur_tokens / max_other, 3),
                    "n_active": n_active,
                    "threshold": round(threshold, 1),
                    "predicted_fire": fired,
                    "gt_straggler": gt_straggler,
                    "saved_tokens": saved_tokens,
                }
            )

    return records


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------

def compute_metrics(records: List[dict]) -> dict:
    if not records:
        return {"total": 0}

    tp = sum(1 for r in records if r["predicted_fire"] and r["gt_straggler"])
    fp = sum(1 for r in records if r["predicted_fire"] and not r["gt_straggler"])
    fn = sum(1 for r in records if not r["predicted_fire"] and r["gt_straggler"])
    tn = sum(1 for r in records if not r["predicted_fire"] and not r["gt_straggler"])

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    total_straggler_tokens = sum(r["cur_tokens"] for r in records if r["gt_straggler"])
    saved_tokens = sum(r["saved_tokens"] for r in records)

    return {
        "total": len(records),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "total_straggler_tokens": total_straggler_tokens,
        "saved_tokens": saved_tokens,
    }


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------

def print_per_step_table(records: List[dict]) -> None:
    """Print per-branch prediction results.

    Columns
    -------
    question : question id (e.g. question_0)
    step     : beam-search step index within this question
    br       : branch index within the step
    cur      : actual total tokens this branch generated in this step
    max_o    : 其他branch的最大长度
    ratio    : cur / max_o，ratio > 1.5 且 cur > 100 即为 ground-truth straggler
    thr      : MLP 预测的阈值 = max_o + predicted_budget；cur > thr 时 predictor 触发剪枝
    pred     : predictor 是否触发（True = 会剪掉该 branch）
    gt       : ground-truth 是否为 straggler (ratio > 1.5 AND cur > 100)
    saved    : 若触发剪枝可节省的 token 数 = max(cur - thr, 0)；未触发则 0
    (suffix) : TP 正确识别 / FP 误判 / FN 漏判
    """
    if not records:
        return
    print(
        f"{'question':<14} {'step':>4} {'br':>3} {'cur':>5} {'max_o':>5} "
        f"{'ratio':>6} {'thr':>7} {'pred':>5} {'gt':>5} {'saved':>6}"
    )
    print("-" * 72)
    for r in records:
        mark = ""
        if r["predicted_fire"] and r["gt_straggler"]:
            mark = "TP"   # True Positive:  correctly predicted straggler
        elif r["predicted_fire"] and not r["gt_straggler"]:
            mark = "FP"   # False Positive: falsely predicted straggler
        elif not r["predicted_fire"] and r["gt_straggler"]:
            mark = "FN"   # False Negative: missed a real straggler
        print(
            f"{r['question_id']:<14} {r['step']:>4} {r['branch']:>3} "
            f"{r['cur_tokens']:>5} {r['max_other']:>5} {r['ratio']:>6.2f} "
            f"{r['threshold']:>7.1f} {str(r['predicted_fire']):>5} "
            f"{str(r['gt_straggler']):>5} {r['saved_tokens']:>6}  {mark}"
        )


def print_metrics(metrics: dict, label: str = "") -> None:
    if metrics["total"] == 0:
        print(f"[{label}] No candidate branches found (no branch exceeded max_other).")
        return
    print(f"\n{'=' * 60}")
    if label:
        print(f"  {label}")
        print(f"{'=' * 60}")
    print(f"  Candidates (branch > max_other):  {metrics['total']}")
    print(
        f"  TP={metrics['tp']}  FP={metrics['fp']}  "
        f"FN={metrics['fn']}  TN={metrics['tn']}"
    )
    print(
        f"  Precision={metrics['precision']:.4f}  "
        f"Recall={metrics['recall']:.4f}  F1={metrics['f1']:.4f}"
    )
    print(
        f"  Total straggler tokens: {metrics['total_straggler_tokens']}"
        f"  |  Saved tokens: {metrics['saved_tokens']}"
    )
    print(f"{'=' * 60}\n")


# ---------------------------------------------------------------------------
# Directory scanners
# ---------------------------------------------------------------------------

def process_config_dir(
    config_dir: Path,
    predictor: MLPAdaptiveThresholdPredictor,
    priors: Dict[str, float],
    *,
    active_branch_gate: int,
    policy_model_override: Optional[str] = None,
    verbose: bool = True,
) -> List[dict]:
    """Process all question_*_workload.json under one config directory."""
    beam_width = _parse_beam_width_from_dirname(config_dir.name)
    policy_model = policy_model_override or _parse_policy_model_from_path(config_dir)
    # Fix known typo in directory names
    policy_model_clean = policy_model.replace("Mtah", "Math")
    reward_model = config_dir.parent.name

    config_prior = _get_config_prior(priors, policy_model_clean, reward_model, beam_width)
    group_key = _get_group_key(policy_model_clean, beam_width)

    workload_files = sorted(config_dir.glob("question_*_workload.json"))
    if not workload_files:
        print(f"  No workload files in {config_dir}")
        return []

    all_records: List[dict] = []
    for wf in workload_files:
        recs = predict_one_question(
            wf,
            predictor,
            config_prior=config_prior,
            beam_width=beam_width,
            active_branch_gate=active_branch_gate,
            group_key=group_key,
        )
        all_records.extend(recs)

    label = (
        f"{config_dir.parent.parent.parent.name} | "
        f"policy={policy_model_clean} | reward={reward_model} | "
        f"beam={beam_width} | gate={active_branch_gate} | "
        f"prior={config_prior:.4f} | files={len(workload_files)}"
    )
    if verbose:
        print_per_step_table(all_records)
    print_metrics(compute_metrics(all_records), label=label)

    return all_records


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict straggler branches in workload data (causal, no peeking).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Invocation (cwd matters for -m):\n"
            "  cd …/3d-tts-sw     && python -m predictor.predict_workload …\n"
            "  cd …/3d-tts-sw/predictor && python -m predict_workload …   # not predictor.predict_workload\n"
            "  cd …/3d-tts-sw/predictor && python predict_workload.py …\n"
            "  python …/3d-tts-sw/run_predict_workload.py …"
        ),
    )
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument(
        "--workload-dir",
        type=str,
        help="Single config directory containing question_*_workload.json files.",
    )
    grp.add_argument(
        "--workload-root",
        type=str,
        help="Root directory; scan all TASK/POLICY/REWARD/CONFIG sub-trees.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=str(_PREDICTOR_DIR / "adaptive_threshold_weights_low_fp.json"),
        help="Path to MLP weight JSON.",
    )
    parser.add_argument(
        "--priors",
        type=str,
        default=str(_PREDICTOR_DIR / "adaptive_threshold_priors.json"),
        help="Path to priors JSON.",
    )
    parser.add_argument("--beam-width", type=int, default=0,
                        help="Override beam width (0 = auto-detect from dir name).")
    parser.add_argument("--policy-model", type=str, default=None,
                        help="Override policy model name for prior / group-key lookup.")
    parser.add_argument("--active-branch-gate", type=int, default=1)
    parser.add_argument("--straggler-ratio", type=float,
                        default=ADAPTIVE_STRAGGLER_RATIO_THRESHOLD)
    parser.add_argument("--straggler-min-tokens", type=int,
                        default=ADAPTIVE_STRAGGLER_MIN_BRANCH_TOKENS)
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="Hide per-step table, show only metrics.")
    args = parser.parse_args()

    predictor = MLPAdaptiveThresholdPredictor.load(args.weights)
    priors = _load_priors(args.priors)

    if args.workload_dir:
        path_for_name = Path(args.workload_dir).resolve()
    else:
        path_for_name = Path(args.workload_root).resolve()
    result_txt = _RESULT_DIR / f"{_safe_result_txt_stem(path_for_name)}.txt"

    if args.workload_dir:
        config_dir = Path(args.workload_dir)
        if not config_dir.is_dir():
            print(f"Error: {config_dir} is not a directory", file=sys.stderr)
            sys.exit(1)
        with _tee_stdout_to_file(result_txt):
            process_config_dir(
                config_dir,
                predictor,
                priors,
                active_branch_gate=args.active_branch_gate,
                policy_model_override=args.policy_model,
                verbose=not args.quiet,
            )
    else:
        root = Path(args.workload_root)
        # Find all directories that contain question_*_workload.json
        config_dirs = sorted(
            {p.parent for p in root.rglob("question_*_workload.json")}
        )
        if not config_dirs:
            print(f"No workload files found under {root}", file=sys.stderr)
            sys.exit(1)

        with _tee_stdout_to_file(result_txt):
            grand_records: List[dict] = []
            for cd in config_dirs:
                print(f"\n>>> {cd.relative_to(root)}")
                recs = process_config_dir(
                    cd,
                    predictor,
                    priors,
                    active_branch_gate=args.active_branch_gate,
                    policy_model_override=args.policy_model,
                    verbose=not args.quiet,
                )
                grand_records.extend(recs)

            if len(config_dirs) > 1:
                print_metrics(compute_metrics(grand_records), label="GRAND TOTAL")

    print(f"Transcript saved to {result_txt}", file=sys.stderr)


if __name__ == "__main__":
    main()
