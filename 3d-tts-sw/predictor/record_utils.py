"""
predictor.record_utils
======================
Helpers for locating and parsing ``record_0.jsonl`` files.
"""

from __future__ import annotations

import glob
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List


def parse_config_from_path(record_path: str) -> Dict[str, Any]:
    """Extract benchmark / policy / prm / beam / question_id from path."""
    path = Path(record_path).resolve()
    parts = path.parts
    question_id = None
    q_idx = None
    for i, part in enumerate(parts):
        match = re.match(r"question_(\d+)$", part)
        if match:
            q_idx = i
            question_id = int(match.group(1))
            break

    if q_idx is None or question_id is None:
        return {}

    config_str = parts[q_idx - 1]
    prm_name = parts[q_idx - 2]
    policy_model = parts[q_idx - 3]
    benchmark = parts[q_idx - 4] if q_idx >= 4 else ""
    config_parts = config_str.split("_")
    beam_width = int(config_parts[1]) if len(config_parts) >= 2 else 2

    return {
        "benchmark": benchmark,
        "policy_model": policy_model,
        "prm_name": prm_name,
        "config_str": config_str,
        "beam_width": beam_width,
        "question_id": question_id,
        "record_path": record_path,
    }


def load_record(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        line = f.readline().strip()
        if line.startswith("\ufeff"):
            line = line[1:]
        return json.loads(line)


def find_record_files(output_dir: str) -> List[str]:
    pattern = os.path.join(output_dir, "**", "record_0.jsonl")
    return sorted(glob.glob(pattern, recursive=True))
