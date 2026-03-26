#!/usr/bin/env python3
"""Aggregate outputs/metrics/rerank_blueprint_pass.json into CSV tables under outputs/figures/."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        type=Path,
        default=ROOT / "outputs/metrics/rerank_blueprint_pass.json",
    )
    ap.add_argument("--figures-dir", type=Path, default=ROOT / "outputs/figures")
    ap.add_argument(
        "--report-dir",
        type=Path,
        default=ROOT / "outputs/figures/report_ready_blueprint_rerank",
    )
    args = ap.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    rows = data.get("rows") or {}

    args.figures_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)

    main_path = args.figures_dir / "rerank_main_comparison.csv"
    with main_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "experiment",
                "acc@1",
                "acc@5",
                "n",
                "coarse_target_in_topk_rate",
            ]
        )
        for name, r in sorted(rows.items()):
            w.writerow(
                [
                    name,
                    r.get("acc@1", ""),
                    r.get("acc@5", ""),
                    r.get("n", ""),
                    r.get("coarse_target_in_topk_rate", ""),
                ]
            )

    slice_keys = [
        "acc@1_slice::geometry_fallback_gt_half",
        "acc@1_slice::geometry_fallback_le_half",
        "acc@1_subset::geometry_high_fallback",
        "acc@1_subset::real_box_heavy",
        "acc@1_subset::weak_feature_source",
    ]
    slice_path = args.figures_dir / "rerank_geometry_slices.csv"
    with slice_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["experiment", "slice", "acc"])
        for name, r in sorted(rows.items()):
            strat = r.get("stratified") or {}
            for sk in slice_keys:
                if sk in strat:
                    w.writerow([name, sk, strat[sk]])

    topk_path = args.figures_dir / "rerank_topk_comparison.csv"
    with topk_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["mode", "k", "acc@1", "coarse_target_in_topk_rate"])
        for name, r in sorted(rows.items()):
            if "coarse_topk" in name or "topk" in name:
                k = ""
                if "topk5" in name or "_k5" in name:
                    k = "5"
                elif "topk10" in name or "_k10" in name:
                    k = "10"
                elif "topk20" in name or "_k20" in name:
                    k = "20"
                w.writerow(
                    [
                        name,
                        k,
                        r.get("acc@1", ""),
                        r.get("coarse_target_in_topk_rate", ""),
                    ]
                )

    hn_path = args.figures_dir / "rerank_hard_negative_comparison.csv"
    with hn_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["setting", "acc@1", "acc@5", "coarse_target_in_topk_rate"])
        a = rows.get("full_scene_topk10_rerank")
        b = rows.get("full_scene_topk10_rerank_hardneg")
        if a:
            w.writerow(["rerank_k10", a.get("acc@1"), a.get("acc@5"), a.get("coarse_target_in_topk_rate")])
        if b:
            w.writerow(["rerank_k10_hardneg", b.get("acc@1"), b.get("acc@5"), b.get("coarse_target_in_topk_rate")])

    for p in (main_path, slice_path, topk_path, hn_path):
        dest = args.report_dir / p.name
        dest.write_text(p.read_text(encoding="utf-8"), encoding="utf-8")

    print("Wrote", main_path, slice_path, topk_path, hn_path, "and copies under", args.report_dir)


if __name__ == "__main__":
    main()
