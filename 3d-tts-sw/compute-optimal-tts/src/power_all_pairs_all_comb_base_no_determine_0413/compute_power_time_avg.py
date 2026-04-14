#!/usr/bin/env python3
"""
从 power_samples.csv 对所有采样点的 power_draw_w 求算术平均值，并写入同目录 power_summary.json。
# 处理 power 下所有子目录（每个含 power_samples.csv 的文件夹）
cd .../src/power && python3 compute_power_time_avg.py

# 只处理指定目录
python3 compute_power_time_avg.py AMC23_..._bud0

# 指定扫描根目录
python3 compute_power_time_avg.py --root /path/to/power

"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import datetime
from pathlib import Path


def load_power_values(csv_path: Path) -> tuple[list[float], set[int]]:
    powers: list[float] = []
    gpu_indices: set[int] = set()
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            p_raw = row.get("power_draw_w")
            if p_raw is None or not str(p_raw).strip():
                continue
            gi_raw = row.get("gpu_index")
            if gi_raw is not None and str(gi_raw).strip():
                gpu_indices.add(int(str(gi_raw).strip()))
            powers.append(float(str(p_raw).strip()))
    return powers, gpu_indices


def parse_metadata_from_dirname(config_name: str) -> tuple[str, int]:
    task = ""
    m_task = re.match(r"^(AMC23|AIME24)_", config_name)
    if m_task:
        task = m_task.group(1)
    batch_size = 0
    m_batch = re.search(r"_(\d+)_2_1_straggler", config_name)
    if m_batch:
        batch_size = int(m_batch.group(1))
    return task, batch_size


def merge_summary(
    config_dir: Path,
    config_name: str,
    avg_power_w: float,
    gpu_indices: list[int],
) -> dict:
    summary_path = config_dir / "power_summary.json"
    base: dict = {}
    if summary_path.is_file():
        try:
            base = json.loads(summary_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            base = {}

    task, batch_from_name = parse_metadata_from_dirname(config_name)
    task = base.get("task") or task
    batch_size = base.get("batch_size")
    if batch_size is None:
        batch_size = batch_from_name
    gpu_ids = base.get("gpu_ids")
    if gpu_ids is None and gpu_indices:
        gpu_ids = ",".join(str(g) for g in sorted(gpu_indices))

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return {
        "config_name": config_name,
        "task": task,
        "batch_size": batch_size,
        "avg_power_w": round(avg_power_w, 2),
        "timestamp": ts,
        "gpu_ids": gpu_ids if gpu_ids is not None else "0",
    }


def process_one_dir(config_dir: Path) -> bool:
    csv_path = config_dir / "power_samples.csv"
    if not csv_path.is_file():
        return False

    powers, gpu_set = load_power_values(csv_path)
    if not powers:
        print(f"跳过（无有效采样）: {config_dir}", file=sys.stderr)
        return False

    overall = sum(powers) / len(powers)
    config_name = config_dir.name
    gpu_indices = sorted(gpu_set)
    summary = merge_summary(config_dir, config_name, overall, gpu_indices)

    out_path = config_dir / "power_summary.json"
    out_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=4) + "\n", encoding="utf-8"
    )
    print(f"已写入 {out_path}  avg_power_w={summary['avg_power_w']} W")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="从 power_samples.csv 计算采样点算术平均功耗并写入 power_summary.json"
    )
    parser.add_argument(
        "dirs",
        nargs="*",
        type=Path,
        help="含 power_samples.csv 的配置目录；缺省则处理脚本所在目录下所有子目录",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="扫描根目录（仅在不传 dirs 时生效），默认为本脚本所在目录",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    root = args.root or script_dir

    if args.dirs:
        targets = [d.resolve() for d in args.dirs]
    else:
        targets = sorted(p for p in root.iterdir() if p.is_dir())

    ok = 0
    for d in targets:
        if process_one_dir(d):
            ok += 1

    print(f"完成: {ok}/{len(targets)} 个目录")


if __name__ == "__main__":
    main()
