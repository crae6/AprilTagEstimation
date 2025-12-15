#!/usr/bin/env python3
"""Summarize ground-truth vs. pose depth for each algorithm in table form."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

ALGORITHMS = ["baseline", "wiener", "inverse"]
DISPLAY = {"baseline": "Original", "wiener": "Wiener", "inverse": "Inverse"}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    default_metrics = root / "motion_blur" / "results_motion15" / "metrics"
    parser = argparse.ArgumentParser(description="Create depth comparison table")
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=default_metrics,
        help="Directory containing per-frame JSON metrics.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "motion_blur" / "depth_table.csv",
        help="Path to CSV/Markdown output (extension determines format).",
    )
    return parser.parse_args()


def best_detection(detections: List[Dict[str, float]]) -> Dict[str, float] | None:
    if not detections:
        return None
    return max(detections, key=lambda d: d.get("decision_margin", float("-inf")))


def collect_rows(metrics_dir: Path) -> List[Dict[str, float | str]]:
    rows: List[Dict[str, float | str]] = []
    for json_path in sorted(metrics_dir.glob("*.json")):
        payload = json.loads(json_path.read_text())
        depth_mm = payload["depth_mm"]
        entry = {
            "raw_path": payload["raw_path"],
            "depth_mm": depth_mm,
        }
        for algo in ALGORITHMS:
            detections = payload["algorithms"][algo]["detections"]
            best = best_detection(detections)
            if best and best.get("pose_t"):
                entry[f"{algo}_pose_mm"] = best["pose_t"][2] * 1000.0
                entry[f"{algo}_margin"] = best.get("decision_margin")
            else:
                entry[f"{algo}_pose_mm"] = None
                entry[f"{algo}_margin"] = None
        rows.append(entry)
    return rows


def format_table(rows: List[Dict[str, float | str]], output: Path) -> None:
    df = pd.DataFrame(rows)
    cols = ["raw_path", "depth_mm"]
    for algo in ALGORITHMS:
        cols.extend([f"{algo}_pose_mm", f"{algo}_margin"])
    df = df[cols]
    if output.suffix.lower() == ".md":
        output.write_text(df.to_markdown(index=False))
    else:
        df.to_csv(output, index=False)
    print(f"Wrote table to {output}")


def main() -> None:
    args = parse_args()
    rows = collect_rows(args.metrics_dir)
    format_table(rows, args.output)


if __name__ == "__main__":
    main()
