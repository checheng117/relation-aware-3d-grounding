#!/usr/bin/env python3
"""Evaluate coarse optimization sweep; write JSON + stage1 CSV tables."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from rag3d.datasets.collate import make_grounding_collate_fn
from rag3d.datasets.referit3d import ReferIt3DManifestDataset
from rag3d.evaluation.coarse_recall import eval_coarse_stage1_metrics
from rag3d.evaluation.two_stage_eval import eval_attribute_full_scene, load_coarse_model
from rag3d.utils.config import load_yaml_config
from rag3d.utils.logging import setup_logging
from torch.utils.data import DataLoader

log = logging.getLogger(__name__)


def _resolve(p: Path, base: Path) -> Path:
    return p if p.is_absolute() else (base / p).resolve()


def _manifest_path(dcfg: dict, base: Path, split: str) -> Path:
    proc = Path(dcfg.get("processed_dir", "data/processed"))
    if not proc.is_absolute():
        proc = base / proc
    return proc / f"{split}_manifest.jsonl"


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "configs/eval/coarse_optimization.yaml")
    args = ap.parse_args()
    ecfg = load_yaml_config(args.config, base_dir=ROOT)
    dcfg = load_yaml_config(ROOT / ecfg["dataset_config"], base_dir=ROOT)
    mcfg = load_yaml_config(ROOT / ecfg["model_config"], base_dir=ROOT)
    device = torch.device(str(ecfg.get("device", "cuda")))
    margin = float(ecfg.get("margin_thresh", 0.15))
    split = str(ecfg.get("split", "val"))
    manifest = _manifest_path(dcfg, ROOT, split)
    if not manifest.is_file():
        log.error("Manifest not found: %s", manifest)
        sys.exit(1)

    feat_dim = int(mcfg["object_dim"])
    ds = ReferIt3DManifestDataset(manifest)
    loader = DataLoader(
        ds,
        batch_size=int(ecfg.get("batch_size", 16)),
        shuffle=False,
        collate_fn=make_grounding_collate_fn(feat_dim, attach_features=True),
    )

    rows_out: list[dict[str, Any]] = []
    for exp in ecfg.get("experiments", []):
        name = str(exp["name"])
        ckpt = _resolve(Path(exp["checkpoint"]), ROOT)
        kind = str(exp.get("coarse_model", "attribute_only")).lower()
        if not ckpt.is_file():
            log.warning("Skip %s: missing %s", name, ckpt)
            continue
        coarse = load_coarse_model(mcfg, ckpt, device, kind)
        metrics = eval_coarse_stage1_metrics(coarse, loader, device, margin)
        canon = eval_attribute_full_scene(coarse, loader, device, margin)
        row = {
            "name": name,
            "coarse_model": kind,
            "checkpoint": str(ckpt),
            "recipe_tag": exp.get("recipe_tag", ""),
            "geom_encoder": bool(exp.get("geom_encoder", False)),
            "loss_ce_only": bool(exp.get("ce_only", False)),
            "loss_load": bool(exp.get("load_w", False)),
            "loss_sameclass": bool(exp.get("sameclass", False)),
            "loss_hardneg": bool(exp.get("hardneg", False)),
            "loss_spatial": bool(exp.get("spatial", False)),
            **metrics,
            "acc@1_canonical": canon.get("acc@1"),
            "acc@5_canonical": canon.get("acc@5"),
        }
        rows_out.append(row)
        log.info("%s recall@20=%.4f acc@1_canon=%.4f", name, float(metrics.get("recall@20", 0)), float(canon.get("acc@1", 0)))

    stamp = _stamp()
    metrics_dir = ROOT / "outputs/metrics"
    fig_dir = ROOT / "outputs/figures"
    report_dir = ROOT / "outputs/figures/report_ready_blueprint_coarse_opt"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    doc = {
        "stamp": stamp,
        "manifest": str(manifest),
        "split": split,
        "experiments": rows_out,
    }
    out_json = metrics_dir / f"coarse_optimization_{stamp}.json"
    out_json.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    latest = metrics_dir / "coarse_optimization_latest.json"
    latest.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    log.info("Wrote %s and %s", out_json, latest)

    opt_main = fig_dir / "stage1_optimization_table.csv"
    with opt_main.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "name",
                "recipe_tag",
                "coarse_model",
                "recall@1",
                "recall@5",
                "recall@10",
                "recall@20",
                "acc@1",
                "acc@1_canonical",
                "gold_rank_median",
                "gold_rank_mean",
                "coarse_margin_mean",
            ]
        )
        for r in rows_out:
            w.writerow(
                [
                    r.get("name"),
                    r.get("recipe_tag"),
                    r.get("coarse_model"),
                    r.get("recall@1"),
                    r.get("recall@5"),
                    r.get("recall@10"),
                    r.get("recall@20"),
                    r.get("acc@1"),
                    r.get("acc@1_canonical"),
                    r.get("gold_rank_median"),
                    r.get("gold_rank_mean"),
                    r.get("coarse_margin_mean"),
                ]
            )

    ablation = fig_dir / "stage1_loss_ablation_table.csv"
    with ablation.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "name",
                "recipe_tag",
                "geom_encoder",
                "load",
                "sameclass",
                "hardneg_margin",
                "spatial",
                "recall@10",
                "recall@20",
                "acc@1_canonical",
            ]
        )
        for r in rows_out:
            w.writerow(
                [
                    r.get("name"),
                    r.get("recipe_tag"),
                    r.get("geom_encoder"),
                    r.get("loss_load"),
                    r.get("loss_sameclass"),
                    r.get("loss_hardneg"),
                    r.get("loss_spatial"),
                    r.get("recall@10"),
                    r.get("recall@20"),
                    r.get("acc@1_canonical"),
                ]
            )

    slices_path = fig_dir / "stage1_slice_table.csv"
    with slices_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["experiment", "slice_key", "value"])
        for r in rows_out:
            name = str(r.get("name", ""))
            for sk, val in sorted((r.get("stratified_recall_slices") or {}).items()):
                w.writerow([name, sk, val])

    for p in (opt_main, ablation, slices_path):
        (report_dir / p.name).write_text(p.read_text(encoding="utf-8"), encoding="utf-8")

    log.info("Wrote CSVs under %s and %s", fig_dir, report_dir)


if __name__ == "__main__":
    main()
