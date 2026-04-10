#!/usr/bin/env python3
"""
统计 model_workloads 下：同一 dataset / policy / reward、同一 beam 配置（16384_{width}_{num_seq}），
不剪枝目录（*straggler_0_0_0）与剪枝目录（*straggler_1_*）的 token 对比。

每题 token：对 decode.steps 中每一步，只取被选分支的续写长度
``branch_tokens[selected_branch_index]``，再对所有 step 求和（不含 prefill）。
每题推理 step 数：``len(decode.steps)``（与 token 统计使用同一批 question 文件）。

配置级：对上述每题指标在目录内求 sum（所有题相加）与 mean（每题平均）。

用法:
    python stats_straggler_workload_tokens.py
    python stats_straggler_workload_tokens.py --root ./model_workloads --csv out.csv
    python stats_straggler_workload_tokens.py --root ./model_workloads --csv out.csv --report out.md
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple


@dataclass
class TokenAgg:
    total: int = 0
    count: int = 0

    def add(self, v: int) -> None:
        self.total += int(v)
        self.count += 1

    @property
    def mean(self) -> float:
        return self.total / self.count if self.count else 0.0


def _parse_straggler_dir(name: str) -> Optional[Tuple[str, str]]:
    """
    返回 (depth_width_numseq_prefix, straggler_suffix_rest) 或 None。
    例如 16384_4_1_straggler_0_0_0 -> ("16384_4_1", "0_0_0")
    """
    if "_straggler_" not in name:
        return None
    base, _, rest = name.partition("_straggler_")
    if not base or not rest:
        return None
    return base, rest


def _selected_branch_path_token_sum(data: dict) -> Optional[int]:
    """
    每题：各 step 仅累加最终选中分支的 branch_tokens 项；无 decode.steps 则无法统计。
    """
    steps = (data.get("decode") or {}).get("steps")
    if not steps:
        return None
    total = 0
    for s in steps:
        bt = s.get("branch_tokens")
        if not isinstance(bt, list):
            continue
        idx = s.get("selected_branch_index")
        if not isinstance(idx, int) or idx < 0 or idx >= len(bt):
            continue
        v = bt[idx]
        if v is None:
            continue
        try:
            total += int(v)
        except (TypeError, ValueError):
            continue
    return total


class DirAgg(NamedTuple):
    tokens: TokenAgg
    decode_steps: TokenAgg


def _collect_stats(config_dir: Path) -> DirAgg:
    tok_agg = TokenAgg()
    step_agg = TokenAgg()
    for p in sorted(config_dir.glob("question_*_workload.json")):
        try:
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
            tok = _selected_branch_path_token_sum(data)
            if tok is not None:
                steps = (data.get("decode") or {}).get("steps") or []
                tok_agg.add(tok)
                step_agg.add(len(steps))
            else:
                print(f"WARN skip (no usable decode.steps): {p}", file=sys.stderr)
        except (json.JSONDecodeError, OSError) as e:
            print(f"WARN skip {p}: {e}", file=sys.stderr)
    return DirAgg(tokens=tok_agg, decode_steps=step_agg)


def scan_pairs(root: Path) -> List[dict]:
    rows: List[dict] = []

    if not root.is_dir():
        print(f"ERROR: root not a directory: {root}", file=sys.stderr)
        return rows

    for dataset_dir in sorted(root.iterdir(), key=lambda p: p.name):
        if not dataset_dir.is_dir():
            continue
        for policy_dir in sorted(dataset_dir.iterdir(), key=lambda p: p.name):
            if not policy_dir.is_dir():
                continue
            for reward_dir in sorted(policy_dir.iterdir(), key=lambda p: p.name):
                if not reward_dir.is_dir():
                    continue

                # base_key -> {"no_prune": Path, "prune": {suffix_rest: Path}}
                grouped: Dict[str, Dict] = defaultdict(lambda: {"no_prune": None, "prune": {}})

                for cfg in sorted(reward_dir.iterdir(), key=lambda p: p.name):
                    if not cfg.is_dir():
                        continue
                    parsed = _parse_straggler_dir(cfg.name)
                    if not parsed:
                        continue
                    base, rest = parsed
                    if rest == "0_0_0":
                        grouped[base]["no_prune"] = cfg
                    elif rest.startswith("1_"):
                        grouped[base]["prune"][rest] = cfg
                    # 其它后缀（若有）忽略

                for base, g in sorted(grouped.items(), key=lambda x: x[0]):
                    no_path: Optional[Path] = g["no_prune"]
                    if no_path is None:
                        continue
                    prune_map: Dict[str, Path] = g["prune"]
                    if not prune_map:
                        continue

                    agg_no = _collect_stats(no_path)

                    for prune_suffix, pr_path in sorted(prune_map.items(), key=lambda x: x[0]):
                        agg_pr = _collect_stats(pr_path)
                        tn, sn = agg_no.tokens, agg_no.decode_steps
                        tp, sp = agg_pr.tokens, agg_pr.decode_steps
                        rows.append(
                            {
                                "dataset": dataset_dir.name,
                                "policy": policy_dir.name,
                                "reward": reward_dir.name,
                                "branch_numseq": base,
                                "dir_no_prune": no_path.name,
                                "dir_prune": pr_path.name,
                                "straggler_prune_params": prune_suffix,
                                "n_no_prune": tn.count,
                                "n_prune": tp.count,
                                "sum_selected_path_tokens_no_prune": tn.total,
                                "mean_selected_path_tokens_no_prune": round(tn.mean, 4),
                                "sum_selected_path_tokens_prune": tp.total,
                                "mean_selected_path_tokens_prune": round(tp.mean, 4),
                                "mean_diff_prune_minus_no": round(tp.mean - tn.mean, 4)
                                if tn.count and tp.count
                                else None,
                                "sum_decode_steps_no_prune": sn.total,
                                "mean_decode_steps_no_prune": round(sn.mean, 4),
                                "sum_decode_steps_prune": sp.total,
                                "mean_decode_steps_prune": round(sp.mean, 4),
                                "mean_diff_decode_steps_prune_minus_no": round(sp.mean - sn.mean, 4)
                                if sn.count and sp.count
                                else None,
                            }
                        )

    return rows


# CSV 列顺序：先标识与目录名，再样本数，再 token，再 step，最后差分
CSV_COLUMNS: List[str] = [
    "dataset",
    "policy",
    "reward",
    "branch_numseq",
    "straggler_prune_params",
    "dir_no_prune",
    "dir_prune",
    "n_no_prune",
    "n_prune",
    "sum_selected_path_tokens_no_prune",
    "mean_selected_path_tokens_no_prune",
    "sum_selected_path_tokens_prune",
    "mean_selected_path_tokens_prune",
    "mean_diff_prune_minus_no",
    "sum_decode_steps_no_prune",
    "mean_decode_steps_no_prune",
    "sum_decode_steps_prune",
    "mean_decode_steps_prune",
    "mean_diff_decode_steps_prune_minus_no",
]


def write_csv(rows: List[dict], path: Path) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _md_cell(v) -> str:
    if v is None:
        return ""
    s = str(v).replace("|", "\\|").replace("\n", " ")
    return s


def write_markdown_report(rows: List[dict], path: Path, root: Path) -> None:
    """分组 Markdown：含指标说明 + 按 dataset/policy/reward 分节的宽表。"""
    if not rows:
        return
    now = datetime.datetime.now().isoformat(timespec="seconds")
    lines = [
        "# Straggler workload 统计报告",
        "",
        f"- **生成时间**: {now}",
        f"- **扫描根目录**: `{root}`",
        "",
        "## 指标说明",
        "",
        "- **选中路径 token（每题）**: 每个 `decode.steps` 步上取 `branch_tokens[selected_branch_index]` 之和。",
        "- **decode 步数（每题）**: `len(decode.steps)`。",
        "- **sum_***: 该配置目录下所有纳入统计的题目的上述值之和。",
        "- **mean_***: 对应 sum 除以题目数 `n`。",
        "- **mean_diff_*_prune_minus_no**: 剪枝侧均值 − 不剪枝侧均值。",
        "",
    ]

    headers_zh = [
        "beam配置",
        "剪枝后缀",
        "目录(不剪枝)",
        "目录(剪枝)",
        "n(不剪)",
        "n(剪)",
        "tok总和(不剪)",
        "tok均值(不剪)",
        "tok总和(剪)",
        "tok均值(剪)",
        "Δtok均值",
        "步总和(不剪)",
        "步均值(不剪)",
        "步总和(剪)",
        "步均值(剪)",
        "Δ步均值",
    ]

    key_cols = [
        "branch_numseq",
        "straggler_prune_params",
        "dir_no_prune",
        "dir_prune",
        "n_no_prune",
        "n_prune",
        "sum_selected_path_tokens_no_prune",
        "mean_selected_path_tokens_no_prune",
        "sum_selected_path_tokens_prune",
        "mean_selected_path_tokens_prune",
        "mean_diff_prune_minus_no",
        "sum_decode_steps_no_prune",
        "mean_decode_steps_no_prune",
        "sum_decode_steps_prune",
        "mean_decode_steps_prune",
        "mean_diff_decode_steps_prune_minus_no",
    ]

    groups: Dict[Tuple[str, str, str], List[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["dataset"], r["policy"], r["reward"])].append(r)

    for (ds, pol, rew), grp in sorted(groups.items(), key=lambda x: x[0]):
        lines.append(f"## `{ds}` / `{pol}` / `{rew}`")
        lines.append("")
        head = "| " + " | ".join(headers_zh) + " |"
        sep = "|" + "|".join([" --- "] * len(headers_zh)) + "|"
        lines.extend([head, sep])
        for r in sorted(grp, key=lambda x: (x["branch_numseq"], x["straggler_prune_params"])):
            cells = [_md_cell(r.get(k)) for k in key_cols]
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="对比 straggler_0_0_0 与 straggler_1_*：每题对选中分支的 branch_tokens 按 step 求和，再对题求和/均值"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent / "model_workloads",
        help="model_workloads 根目录",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="写入 CSV（UTF-8 BOM，列顺序固定，便于 Excel 打开）",
    )
    parser.add_argument(
        "--report",
        "-r",
        type=Path,
        default=None,
        help="可选：写入 Markdown 报告（中文表头、按 dataset/policy/reward 分节）",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    rows = scan_pairs(root)
    # 人类可读打印（完整字段名）
    if rows:
        print(
            "dataset / policy / reward / branch_numseq | "
            "no_prune vs prune | n | selected-path tokens sum/mean | decode steps sum/mean | diffs(pr-no)\n"
        )
        for r in rows:
            print(
                f"{r['dataset']} | {r['policy']} | {r['reward']} | {r['branch_numseq']}\n"
                f"  no:  {r['dir_no_prune']}  n={r['n_no_prune']}  "
                f"tok sum={r['sum_selected_path_tokens_no_prune']}  tok mean={r['mean_selected_path_tokens_no_prune']}  "
                f"step sum={r['sum_decode_steps_no_prune']}  step mean={r['mean_decode_steps_no_prune']}\n"
                f"  pr:  {r['dir_prune']}  n={r['n_prune']}  "
                f"tok sum={r['sum_selected_path_tokens_prune']}  tok mean={r['mean_selected_path_tokens_prune']}  "
                f"step sum={r['sum_decode_steps_prune']}  step mean={r['mean_decode_steps_prune']}\n"
                f"  mean_diff tok(pr-no)={r['mean_diff_prune_minus_no']}  "
                f"mean_diff step(pr-no)={r['mean_diff_decode_steps_prune_minus_no']}\n"
            )
    else:
        print("未找到成对目录。")

    if args.csv:
        write_csv(rows, args.csv.resolve())
        print(f"Wrote {len(rows)} rows -> {args.csv} (UTF-8 BOM, ordered columns)")
    if args.report:
        write_markdown_report(rows, args.report.resolve(), root)
        print(f"Wrote readable report -> {args.report}")


if __name__ == "__main__":
    main()
