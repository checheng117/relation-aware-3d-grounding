#!/usr/bin/env python3
"""Train/eval K=10 rerank from shortlist-aligned coarse promotion; write alignment CSV tables."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag3d.utils.logging import setup_logging

log = logging.getLogger(__name__)

CKPT_RERANK_SHORTLIST = ROOT / "outputs/checkpoints_stage1_rerank_shortlist"
SPATIAL_NAME = "coarse_geom_ce_spatial"
REFERENCE_TWO_STAGE = {
    "name": "reference_full_gpu_two_stage_k10_geom",
    "coarse_checkpoint": "outputs/checkpoints_stage1/coarse_geom_recall_last.pt",
    "coarse_model": "coarse_geom",
    "fine_checkpoint": "outputs/checkpoints_stage1_rerank/rerank_k10_stage1_last.pt",
}


def _sanitize(s: str) -> str:
    return "".join(c if c.isalnum() or c in "_-" else "_" for c in s).strip("_")[:80]


def _rel_repo(path_str: str) -> str:
    p = Path(path_str).resolve()
    r = ROOT.resolve()
    try:
        return str(p.relative_to(r))
    except ValueError:
        return path_str


def _pick_promoted(rows: list[dict[str, Any]], top_n: int, force_spatial: bool) -> list[dict[str, Any]]:
    by_score = sorted(rows, key=lambda r: float(r["promote_score_shortlist"]), reverse=True)
    picked: list[dict[str, Any]] = []
    seen: set[str] = set()
    for r in by_score[: max(1, top_n)]:
        n = str(r["name"])
        if n not in seen:
            picked.append(r)
            seen.add(n)
    if force_spatial:
        sp = next((r for r in rows if str(r["name"]) == SPATIAL_NAME), None)
        if sp is not None and SPATIAL_NAME not in seen:
            picked.append(sp)
            seen.add(SPATIAL_NAME)
    return picked


def _write_table1_audit(
    audit: dict[str, Any],
    promoted_names: set[str],
    path: Path,
) -> None:
    rows = list(audit.get("rows") or [])
    old_r = audit.get("ranks_old_promote") or {}
    new_r = audit.get("ranks_shortlist_aligned") or {}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "coarse_recipe",
                "recall@10",
                "recall@20",
                "topk_hit_rate@10",
                "coarse_only_topK10_acc@1",
                "median_gold_rank",
                "old_promote_rank",
                "new_shortlist_rank",
                "promoted_for_rerank_train",
            ]
        )
        for r in sorted(rows, key=lambda x: str(x["name"])):
            name = str(r["name"])
            w.writerow(
                [
                    name,
                    r.get("recall@10", ""),
                    r.get("recall@20", ""),
                    r.get("k10_coarse_target_in_topk_rate", ""),
                    r.get("k10_coarse_only_acc@1", ""),
                    r.get("gold_rank_median", ""),
                    old_r.get(name, ""),
                    new_r.get(name, ""),
                    "yes" if name in promoted_names else "no",
                ]
            )


def _write_table2_rerank(
    audit_rows: list[dict[str, Any]],
    rerank_rows: dict[str, Any],
    promoted: list[dict[str, Any]],
    path: Path,
) -> None:
    by_name = {str(r["name"]): r for r in audit_rows}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "coarse_recipe",
                "promote_score_shortlist",
                "coarse_recall@10",
                "coarse_target_in_topk@10",
                "final_rerank_acc@1",
                "final_rerank_acc@5",
                "rerank_rescue_rate",
                "experiment_two_stage_key",
                "notes",
            ]
        )
        for i, pr in enumerate(promoted):
            name = str(pr["name"])
            san = _sanitize(name)
            ek = f"sa_{i}_{san}_two_stage_k10"
            ck = f"sa_{i}_{san}_coarse_topk10"
            rr = rerank_rows.get(ek) or {}
            cr = rerank_rows.get(ck) or {}
            hit = cr.get("coarse_target_in_topk_rate", pr.get("k10_coarse_target_in_topk_rate", ""))
            notes = ""
            if name == SPATIAL_NAME:
                notes = "included_via_spatial_policy"
            w.writerow(
                [
                    name,
                    pr.get("promote_score_shortlist", ""),
                    pr.get("recall@10", ""),
                    hit,
                    rr.get("acc@1", ""),
                    rr.get("acc@5", ""),
                    rr.get("rerank_rescue_rate", ""),
                    ek,
                    notes,
                ]
            )
        ref = rerank_rows.get(REFERENCE_TWO_STAGE["name"]) or {}
        ref_audit = by_name.get("coarse_geom_recall_full")
        w.writerow(
            [
                "reference_full_gpu_geom_k10",
                ref_audit.get("promote_score_shortlist", "") if ref_audit else "",
                ref_audit.get("recall@10", "") if ref_audit else "",
                ref.get("coarse_target_in_topk_rate", ""),
                ref.get("acc@1", ""),
                ref.get("acc@5", ""),
                ref.get("rerank_rescue_rate", ""),
                REFERENCE_TWO_STAGE["name"],
                "frozen_checkpoint_baseline_no_retrain",
            ]
        )


def _write_table3_failure(
    rerank_rows: dict[str, Any],
    col_keys: list[tuple[str, str]],
    path: Path,
) -> None:
    axes = [
        ("not_in_topk_rate", lambda r: 1.0 - float(r.get("coarse_target_in_topk_rate", 0.0))),
        ("acc@1_subset::same_class_clutter", lambda r: r.get("stratified", {}).get("acc@1_subset::same_class_clutter", "")),
        ("acc@1_slice::geometry_fallback_le_half", lambda r: r.get("stratified", {}).get("acc@1_slice::geometry_fallback_le_half", "")),
        ("acc@1_subset::weak_feature_source", lambda r: r.get("stratified", {}).get("acc@1_subset::weak_feature_source", "")),
        ("acc@1_subset::low_model_margin", lambda r: r.get("stratified", {}).get("acc@1_subset::low_model_margin", "")),
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["failure_or_stress_axis"] + [label for label, _ in col_keys]
        w.writerow(header)
        for axis_name, fn in axes:
            row = [axis_name]
            for _, ek in col_keys:
                rr = rerank_rows.get(ek) or {}
                v = fn(rr)
                row.append(v if v != "" else "")
            w.writerow(row)


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit-json", type=Path, default=ROOT / "outputs/metrics/shortlist_alignment_audit_latest.json")
    ap.add_argument("--top-n", type=int, default=2)
    ap.add_argument("--no-force-spatial", action="store_true", help="Do not append coarse_geom_ce_spatial if outside top-N.")
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()

    audit = json.loads(args.audit_json.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = list(audit.get("rows") or [])
    if not rows:
        log.error("No rows in audit JSON")
        sys.exit(1)

    promoted = _pick_promoted(rows, args.top_n, force_spatial=not args.no_force_spatial)
    promoted_names = {str(r["name"]) for r in promoted}
    log.info("Promoted for rerank training: %s", promoted_names)

    CKPT_RERANK_SHORTLIST.mkdir(parents=True, exist_ok=True)
    train_dir = ROOT / "configs/train/rerank/shortlist_alignment"
    train_dir.mkdir(parents=True, exist_ok=True)

    fine_ckpts: list[tuple[int, str, str, str, str]] = []
    for i, row in enumerate(promoted):
        name = str(row["name"])
        kind = str(row.get("coarse_model", "coarse_geom"))
        ck = str(row["checkpoint"])
        san = _sanitize(name)
        run_name = f"rerank_k10_sa_{san}"
        train_yaml = train_dir / f"generated_sa_{i}_{san}.yaml"
        tcfg = {
            "model": "relation_aware",
            "coarse_model": kind,
            "dataset_config": "configs/dataset/diagnosis_full_geom.yaml",
            "coarse_checkpoint": _rel_repo(ck),
            "rerank_k": 10,
            "parser_mode": "structured",
            "parser_cache_dir": "data/parser_cache/diagnosis",
            "epochs": 8,
            "batch_size": 16,
            "lr": 0.0001,
            "weight_decay": 0.01,
            "seed": 42,
            "num_workers": 2,
            "checkpoint_dir": "outputs/checkpoints_stage1_rerank_shortlist",
            "metrics_file": f"outputs/metrics/rerank_sa_train_{i}_{san}.jsonl",
            "run_name": run_name,
            "device": "cuda",
            "debug_max_batches": None,
            "loss": {"hard_negative": {"enabled": False}},
        }
        train_yaml.write_text(yaml.safe_dump(tcfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
        if not args.skip_train:
            subprocess.run(
                [sys.executable, str(ROOT / "scripts/train_two_stage_rerank.py"), "--config", str(train_yaml)],
                cwd=str(ROOT),
                check=True,
            )
        fine_last = CKPT_RERANK_SHORTLIST / f"{run_name}_last.pt"
        if not fine_last.is_file():
            log.error("Missing fine checkpoint %s", fine_last)
            sys.exit(1)
        rel_coarse = _rel_repo(ck)
        rel_fine = str(fine_last.resolve().relative_to(ROOT.resolve()))
        fine_ckpts.append((i, name, kind, rel_coarse, rel_fine))

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    eval_experiments: list[dict[str, Any]] = []
    table3_cols: list[tuple[str, str]] = []

    for i, name, kind, rel_coarse, rel_fine in fine_ckpts:
        san = _sanitize(name)
        eval_experiments.append(
            {
                "name": f"sa_{i}_{san}_coarse_topk10",
                "type": "coarse_topk_attribute",
                "coarse_checkpoint": rel_coarse,
                "coarse_model": kind,
                "rerank_k": 10,
            }
        )
        ek = f"sa_{i}_{san}_two_stage_k10"
        eval_experiments.append(
            {
                "name": ek,
                "type": "two_stage_rerank",
                "coarse_checkpoint": rel_coarse,
                "coarse_model": kind,
                "fine_checkpoint": rel_fine,
                "rerank_k": 10,
            }
        )
        table3_cols.append((name, ek))

    eval_experiments.append(
        {
            "name": REFERENCE_TWO_STAGE["name"],
            "type": "two_stage_rerank",
            "coarse_checkpoint": REFERENCE_TWO_STAGE["coarse_checkpoint"],
            "coarse_model": REFERENCE_TWO_STAGE["coarse_model"],
            "fine_checkpoint": REFERENCE_TWO_STAGE["fine_checkpoint"],
            "rerank_k": 10,
        }
    )
    table3_cols.append(("reference_full_gpu_geom_k10", REFERENCE_TWO_STAGE["name"]))

    eval_cfg = {
        "dataset_config": "configs/dataset/diagnosis_full_geom.yaml",
        "model_config": "configs/model/relation_aware.yaml",
        "split": "val",
        "batch_size": 16,
        "device": "cuda",
        "margin_thresh": 0.15,
        "seed": 42,
        "parser_mode": "structured",
        "parser_cache_dir": "data/parser_cache/diagnosis",
        "coarse_checkpoint": "outputs/checkpoints_stage1_opt/coarse_geom_ce_only_last.pt",
        "coarse_model": "coarse_geom",
        "output_json": f"outputs/metrics/shortlist_alignment_rerank_{stamp}.json",
        "reference_metrics": {},
        "experiments": eval_experiments,
    }
    eval_path = ROOT / "configs/eval/shortlist_alignment_rerank_generated.yaml"
    eval_path.write_text(yaml.safe_dump(eval_cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")

    subprocess.run(
        [sys.executable, str(ROOT / "scripts/eval_rerank_blueprint.py"), "--config", str(eval_path)],
        cwd=str(ROOT),
        check=True,
    )

    rjson = ROOT / "outputs/metrics" / f"shortlist_alignment_rerank_{stamp}.json"
    latest_r = ROOT / "outputs/metrics/shortlist_alignment_rerank_latest.json"
    if rjson.is_file():
        latest_r.write_text(rjson.read_text(encoding="utf-8"), encoding="utf-8")

    rdata = json.loads(rjson.read_text(encoding="utf-8"))
    rerank_rows: dict[str, Any] = rdata.get("rows") or {}

    fig = ROOT / "outputs/figures"
    report = ROOT / "outputs/figures/report_ready_blueprint_shortlist_alignment"
    report.mkdir(parents=True, exist_ok=True)

    t1 = fig / "shortlist_alignment_promote_audit_table.csv"
    _write_table1_audit(audit, promoted_names, t1)
    t2 = fig / "shortlist_alignment_rerank_table.csv"
    _write_table2_rerank(rows, rerank_rows, promoted, t2)
    t3 = fig / "shortlist_alignment_failure_axes_table.csv"
    _write_table3_failure(rerank_rows, table3_cols, t3)

    for p in (t1, t2, t3):
        (report / p.name).write_text(p.read_text(encoding="utf-8"), encoding="utf-8")

    combined = {
        "stamp": stamp,
        "audit_stamp": audit.get("stamp"),
        "promoted_recipes": list(promoted_names),
        "rerank_metrics_path": str(rjson),
        "old_promote_rule": audit.get("old_promote_rule"),
        "tables": {
            "promote_audit_csv": str(t1),
            "rerank_compare_csv": str(t2),
            "failure_axes_csv": str(t3),
        },
    }
    comb_path = ROOT / "outputs/metrics" / f"shortlist_alignment_summary_{stamp}.json"
    comb_path.write_text(json.dumps(combined, indent=2), encoding="utf-8")
    (ROOT / "outputs/metrics/shortlist_alignment_summary_latest.json").write_text(
        json.dumps(combined, indent=2), encoding="utf-8"
    )

    log.info("Wrote %s, figures, %s", rjson, comb_path)


if __name__ == "__main__":
    main()
