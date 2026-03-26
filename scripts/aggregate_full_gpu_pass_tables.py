#!/usr/bin/env python3
"""Build report-ready CSV tables from stage1_recall_*.json + rerank_full_gpu_pass.json."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--stage1-json",
        type=Path,
        default=None,
        help="Path to stage1_recall_*.json (default: newest under outputs/metrics/)",
    )
    ap.add_argument(
        "--rerank-json",
        type=Path,
        default=ROOT / "outputs/metrics/rerank_full_gpu_pass.json",
    )
    ap.add_argument("--figures-dir", type=Path, default=ROOT / "outputs/figures")
    ap.add_argument(
        "--report-dir",
        type=Path,
        default=ROOT / "outputs/figures/report_ready_blueprint_stage1",
    )
    args = ap.parse_args()

    metrics_dir = ROOT / "outputs/metrics"
    stage1_path = args.stage1_json
    if stage1_path is None:
        cands = sorted(
            metrics_dir.glob("stage1_recall_*.json"),
            key=lambda p: p.stat().st_mtime,
        )
        stage1_path = cands[-1] if cands else None
    if stage1_path is None or not stage1_path.is_file():
        raise SystemExit("No stage1 JSON found; run eval_stage1_recall_pass.py first.")

    s1 = json.loads(stage1_path.read_text(encoding="utf-8"))
    rnk_path = args.rerank_json
    rnk = json.loads(rnk_path.read_text(encoding="utf-8")) if rnk_path.is_file() else {"rows": {}}
    rows_s1 = s1.get("coarse_evals") or []
    rows_r = rnk.get("rows") or {}

    args.figures_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)

    # Full-scene Acc@1: use rerank eval rows (eval_attribute_full_scene) as canonical to match top-K / two-stage section.
    acc1_canon = {
        "coarse_attr_baseline_full": (rows_r.get("full_gpu_attr_baseline_full_scene") or {}).get("acc@1", ""),
        "coarse_geom_recall_full": (rows_r.get("full_gpu_geom_recall_full_scene") or {}).get("acc@1", ""),
    }

    t1 = args.figures_dir / "table1_stage1_coarse_quality.csv"
    with t1.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "model",
                "recall@1",
                "recall@5",
                "recall@10",
                "recall@20",
                "coarse_acc@1_full_scene_canonical",
                "gold_rank_median",
                "gold_rank_mean",
            ]
        )
        for row in rows_s1:
            name = row.get("name", "")
            w.writerow(
                [
                    name,
                    row.get("recall@1", ""),
                    row.get("recall@5", ""),
                    row.get("recall@10", ""),
                    row.get("recall@20", ""),
                    acc1_canon.get(name, row.get("acc@1", "")),
                    row.get("gold_rank_median", ""),
                    row.get("gold_rank_mean", ""),
                ]
            )

    t2 = args.figures_dir / "table2_two_stage_final.csv"
    with t2.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "setup",
                "K",
                "coarse_target_in_topk_rate",
                "topk_recall_success_rate",
                "final_acc@1",
                "final_acc@5",
                "rerank_rescue_rate",
            ]
        )
        keys = [
            "full_gpu_two_stage_k5_geom",
            "full_gpu_two_stage_k10_geom",
            "full_gpu_two_stage_k20_geom",
        ]
        for k, name in [(5, keys[0]), (10, keys[1]), (20, keys[2])]:
            r = rows_r.get(name) or {}
            w.writerow(
                [
                    name,
                    k,
                    r.get("coarse_target_in_topk_rate", ""),
                    r.get("topk_recall_success_rate", r.get("coarse_target_in_topk_rate", "")),
                    r.get("acc@1", ""),
                    r.get("acc@5", ""),
                    r.get("rerank_rescue_rate", ""),
                ]
            )
        # Coarse-only reference rows for same K (geom)
        for k in (5, 10, 20):
            r = rows_r.get(f"full_gpu_geom_recall_topk{k}") or {}
            w.writerow(
                [
                    f"full_gpu_geom_recall_topk{k}_coarse_only",
                    k,
                    r.get("coarse_target_in_topk_rate", ""),
                    r.get("coarse_target_in_topk_rate", ""),
                    r.get("acc@1", ""),
                    r.get("acc@5", ""),
                    "",
                ]
            )

    t3 = args.figures_dir / "table3_slice_analysis.csv"
    with t3.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["source", "model_or_setup", "slice_key", "value"])
        for row in rows_s1:
            name = row.get("name", "")
            slices = row.get("stratified_recall_slices") or {}
            for sk, val in sorted(slices.items()):
                w.writerow(["stage1_recall", name, sk, val])
        for name, r in sorted(rows_r.items()):
            strat = r.get("stratified") or {}
            for sk, val in sorted(strat.items()):
                w.writerow(["rerank_full_gpu", name, sk, val])

    gain = args.figures_dir / "rerank_recall_gain_table.csv"
    with gain.open("w", newline="", encoding="utf-8") as f:
        wg = csv.writer(f)
        wg.writerow(
            [
                "K",
                "coarse_target_in_topk_baseline_attr",
                "coarse_target_in_topk_geom_recall",
                "delta_in_topk",
                "two_stage_acc@1_geom_rerank",
                "coarse_only_topk_acc@1_geom",
            ]
        )
        for k, akey, gkey, tkey in [
            (5, "full_gpu_attr_baseline_topk5", "full_gpu_geom_recall_topk5", "full_gpu_two_stage_k5_geom"),
            (10, "full_gpu_attr_baseline_topk10", "full_gpu_geom_recall_topk10", "full_gpu_two_stage_k10_geom"),
            (20, "full_gpu_attr_baseline_topk20", "full_gpu_geom_recall_topk20", "full_gpu_two_stage_k20_geom"),
        ]:
            ra = rows_r.get(akey) or {}
            rg = rows_r.get(gkey) or {}
            rt = rows_r.get(tkey) or {}
            ca = ra.get("coarse_target_in_topk_rate", "")
            cg = rg.get("coarse_target_in_topk_rate", "")
            dlt = ""
            try:
                dlt = float(cg) - float(ca)
            except (TypeError, ValueError):
                pass
            wg.writerow(
                [
                    k,
                    ca,
                    cg,
                    dlt,
                    rt.get("acc@1", ""),
                    rg.get("acc@1", ""),
                ]
            )

    for p in (t1, t2, t3, gain):
        (args.report_dir / p.name).write_text(p.read_text(encoding="utf-8"), encoding="utf-8")

    print("Wrote", t1, t2, t3, gain, "and copies under", args.report_dir)


if __name__ == "__main__":
    main()
