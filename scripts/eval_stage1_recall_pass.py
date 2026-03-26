#!/usr/bin/env python3
"""Stage-1 recall pass: coarse recall@K + optional two-stage rows; timestamped JSON + CSV artifacts."""

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
from rag3d.evaluation.two_stage_eval import eval_two_stage, load_coarse_model, load_two_stage_model
from rag3d.parsers.cached_parser import CachedParser
from rag3d.parsers.heuristic_parser import HeuristicParser
from rag3d.parsers.structured_rule_parser import StructuredRuleParser
from rag3d.utils.config import load_yaml_config
from rag3d.utils.logging import setup_logging
from torch.utils.data import DataLoader

log = logging.getLogger(__name__)


def _resolve(p: Path | None, base: Path) -> Path:
    if p is None:
        return base
    return p if p.is_absolute() else (base / p).resolve()


def _manifest_path(dcfg: dict, base: Path, split: str) -> Path:
    proc = Path(dcfg.get("processed_dir", "data/processed"))
    if not proc.is_absolute():
        proc = base / proc
    return proc / f"{split}_manifest.jsonl"


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")


def _write_stage1_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "name",
        "coarse_model",
        "recall@1",
        "recall@5",
        "recall@10",
        "recall@20",
        "acc@1",
        "acc@5",
        "gold_rank_mean",
        "coarse_margin_mean",
        "coarse_target_in_topk_rate",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _write_rerank_gain_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["name", "rerank_k", "coarse_model", "acc@1", "acc@5", "coarse_target_in_topk_rate"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "configs/eval/stage1_recall_pass.yaml")
    args = ap.parse_args()
    ecfg = load_yaml_config(args.config, base_dir=ROOT)
    dcfg = load_yaml_config(ROOT / ecfg["dataset_config"], base_dir=ROOT)
    mcfg = load_yaml_config(ROOT / ecfg["model_config"], base_dir=ROOT)
    device = torch.device(str(ecfg.get("device", "cpu")))
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

    parser_mode = str(ecfg.get("parser_mode", "structured")).lower()
    pcache = ROOT / Path(ecfg.get("parser_cache_dir", "data/parser_cache/diagnosis"))
    if parser_mode == "structured":
        parser = CachedParser(StructuredRuleParser(), pcache / "structured")
    else:
        parser = CachedParser(HeuristicParser(), pcache / "heuristic")

    stamp = str(ecfg.get("output_stamp", "")) or _stamp()
    metrics_dir = _resolve(Path(ecfg.get("metrics_dir", "outputs/metrics")), ROOT)
    fig_dir = _resolve(Path(ecfg.get("figures_dir", "outputs/figures")), ROOT)
    report_dir = _resolve(Path(ecfg.get("report_ready_dir", "outputs/figures/report_ready_blueprint_stage1")), ROOT)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    stage1_path = metrics_dir / f"stage1_recall_{stamp}.json"
    two_stage_path = metrics_dir / f"two_stage_recall_pass_{stamp}.json"

    coarse_rows_out: list[dict[str, Any]] = []
    two_stage_rows_out: dict[str, Any] = {"stamp": stamp, "manifest": str(manifest), "split": split, "rows": {}}

    csv_coarse: list[dict[str, Any]] = []
    csv_rerank: list[dict[str, Any]] = []

    default_coarse_kind = str(ecfg.get("default_coarse_model", "attribute_only")).lower()

    for block in ecfg.get("coarse_evals", []):
        name = str(block["name"])
        ckpt = _resolve(Path(block["checkpoint"]), ROOT)
        kind = str(block.get("coarse_model", default_coarse_kind)).lower()
        if not ckpt.is_file():
            log.warning("Skip coarse eval %s: missing checkpoint %s", name, ckpt)
            continue
        coarse = load_coarse_model(mcfg, ckpt, device, kind)
        metrics = eval_coarse_stage1_metrics(coarse, loader, device, margin)
        row = {"name": name, "coarse_model": kind, "checkpoint": str(ckpt), **metrics}
        coarse_rows_out.append(row)
        csv_coarse.append(
            {
                "name": name,
                "coarse_model": kind,
                "recall@1": metrics.get("recall@1", ""),
                "recall@5": metrics.get("recall@5", ""),
                "recall@10": metrics.get("recall@10", ""),
                "recall@20": metrics.get("recall@20", ""),
                "acc@1": metrics.get("acc@1", ""),
                "acc@5": metrics.get("acc@5", ""),
                "gold_rank_mean": metrics.get("gold_rank_mean", ""),
                "coarse_margin_mean": metrics.get("coarse_margin_mean", ""),
                "coarse_target_in_topk_rate": "",
            }
        )
        log.info("Coarse %s recall@20=%.4f", name, float(metrics.get("recall@20", 0.0)))

    for block in ecfg.get("two_stage_evals", []):
        name = str(block["name"])
        c_ckpt = _resolve(Path(block["coarse_checkpoint"]), ROOT)
        f_ckpt = _resolve(Path(block["fine_checkpoint"]), ROOT) if block.get("fine_checkpoint") else None
        rk = int(block.get("rerank_k", 10))
        kind = str(block.get("coarse_model", default_coarse_kind)).lower()
        if not c_ckpt.is_file():
            log.warning("Skip two-stage %s: missing coarse %s", name, c_ckpt)
            continue
        if f_ckpt is None or not f_ckpt.is_file():
            log.warning("Skip two-stage %s: missing fine %s", name, f_ckpt)
            continue
        ts = load_two_stage_model(mcfg, c_ckpt, f_ckpt, rk, device, kind)
        r = eval_two_stage(ts, loader, device, parser, margin)
        two_stage_rows_out["rows"][name] = {
            "coarse_model": kind,
            "rerank_k": rk,
            "coarse_checkpoint": str(c_ckpt),
            "fine_checkpoint": str(f_ckpt),
            **r,
        }
        csv_rerank.append(
            {
                "name": name,
                "rerank_k": rk,
                "coarse_model": kind,
                "acc@1": r.get("acc@1", ""),
                "acc@5": r.get("acc@5", ""),
                "coarse_target_in_topk_rate": r.get("coarse_target_in_topk_rate", ""),
            }
        )

    stage1_doc = {
        "stamp": stamp,
        "manifest": str(manifest),
        "split": split,
        "coarse_evals": coarse_rows_out,
    }
    stage1_path.write_text(json.dumps(stage1_doc, indent=2), encoding="utf-8")
    two_stage_path.write_text(json.dumps(two_stage_rows_out, indent=2), encoding="utf-8")
    log.info("Wrote %s and %s", stage1_path, two_stage_path)

    tcsv = fig_dir / "stage1_recall_table.csv"
    rcsv = fig_dir / "rerank_recall_gain_table.csv"
    _write_stage1_csv(csv_coarse, tcsv)
    _write_rerank_gain_csv(csv_rerank, rcsv)
    # Non-destructive copy into report-ready folder with stamp
    _write_stage1_csv(csv_coarse, report_dir / f"stage1_recall_table_{stamp}.csv")
    _write_rerank_gain_csv(csv_rerank, report_dir / f"rerank_recall_gain_table_{stamp}.csv")
    log.info("Wrote CSV tables under %s and %s", fig_dir, report_dir)

    note_dir = _resolve(Path(ecfg.get("experiment_notes_dir", "outputs/experiment_notes")), ROOT)
    note_dir.mkdir(parents=True, exist_ok=True)
    date_s = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    note_path = note_dir / f"blueprint_stage1_recall_pass_{date_s}.md"
    note_path.write_text(
        f"# Blueprint stage-1 recall pass ({date_s})\n\n"
        f"- Run stamp (UTC): `{stamp}`\n"
        f"- Metrics: `{stage1_path.relative_to(ROOT)}`\n"
        f"- Two-stage: `{two_stage_path.relative_to(ROOT)}`\n"
        f"- Interpret stage-1 / two-stage metrics using README.md (Main findings) and JSON paths above.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
