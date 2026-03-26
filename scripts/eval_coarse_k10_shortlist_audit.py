#!/usr/bin/env python3
"""Run K=10 coarse-topK metrics (inference-aligned) for each coarse checkpoint; merge with sweep JSON."""

from __future__ import annotations

import argparse
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
from rag3d.evaluation.shortlist_promote import assign_ranks, old_promote_key, shortlist_aligned_score
from rag3d.evaluation.two_stage_eval import eval_coarse_topk_attribute, load_coarse_model
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


LEGACY_GEOM_RECALL = {
    "name": "coarse_geom_recall_full",
    "coarse_model": "coarse_geom",
    "checkpoint": "outputs/checkpoints_stage1/coarse_geom_recall_last.pt",
    "recipe_tag": "legacy_geom_recall_full_gpu",
}


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--coarse-json",
        type=Path,
        default=ROOT / "outputs/metrics/coarse_optimization_latest.json",
    )
    ap.add_argument("--dataset-config", type=Path, default=ROOT / "configs/dataset/diagnosis_full_geom.yaml")
    ap.add_argument("--model-config", type=Path, default=ROOT / "configs/model/relation_aware.yaml")
    ap.add_argument("--split", type=str, default="val")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--margin", type=float, default=0.15)
    ap.add_argument("--rerank-k", type=int, default=10)
    ap.add_argument("--include-legacy-geom-recall", action="store_true", default=True)
    ap.add_argument("--no-legacy-geom-recall", action="store_true")
    args = ap.parse_args()
    include_legacy = args.include_legacy_geom_recall and not args.no_legacy_geom_recall

    coarse_doc = json.loads(args.coarse_json.read_text(encoding="utf-8"))
    base_rows: list[dict[str, Any]] = list(coarse_doc.get("experiments") or [])

    if include_legacy:
        ck = ROOT / LEGACY_GEOM_RECALL["checkpoint"]
        if ck.is_file():
            base_rows.append(dict(LEGACY_GEOM_RECALL))
        else:
            log.warning("Legacy checkpoint missing, skip %s", ck)

    dcfg = load_yaml_config(args.dataset_config, base_dir=ROOT)
    mcfg = load_yaml_config(args.model_config, base_dir=ROOT)
    device = torch.device(args.device)
    manifest = _manifest_path(dcfg, ROOT, args.split)
    if not manifest.is_file():
        log.error("Manifest not found: %s", manifest)
        sys.exit(1)

    feat_dim = int(mcfg["object_dim"])
    ds = ReferIt3DManifestDataset(manifest)
    loader = DataLoader(
        ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        collate_fn=make_grounding_collate_fn(feat_dim, attach_features=True),
    )

    merged: list[dict[str, Any]] = []
    for row in base_rows:
        name = str(row["name"])
        ckpt = _resolve(Path(row["checkpoint"]), ROOT)
        kind = str(row.get("coarse_model", "attribute_only")).lower()
        if not ckpt.is_file():
            log.warning("Skip %s: missing %s", name, ckpt)
            continue
        coarse = load_coarse_model(mcfg, ckpt, device, kind)
        m = dict(row)
        if "recall@10" not in m:
            st = eval_coarse_stage1_metrics(coarse, loader, device, float(args.margin))
            for k in (
                "recall@1",
                "recall@5",
                "recall@10",
                "recall@20",
                "gold_rank_median",
                "gold_rank_mean",
                "acc@1",
                "acc@5",
                "n",
            ):
                if k in st:
                    m[k] = st[k]
        topk = eval_coarse_topk_attribute(coarse, loader, device, int(args.rerank_k), float(args.margin))
        m["k10_coarse_target_in_topk_rate"] = float(topk["coarse_target_in_topk_rate"])
        m["k10_coarse_only_acc@1"] = float(topk["acc@1"])
        m["k10_coarse_only_acc@5"] = float(topk["acc@5"])
        m["promote_score_shortlist"] = shortlist_aligned_score(
            float(m.get("recall@10", 0.0)),
            m["k10_coarse_target_in_topk_rate"],
            m["k10_coarse_only_acc@1"],
            float(m.get("gold_rank_median", 0.0)),
            float(m.get("recall@20", 0.0)),
        )
        merged.append(m)
        log.info(
            "%s k10_hit=%.4f k10_acc@1=%.4f shortlist_score=%.4f",
            name,
            m["k10_coarse_target_in_topk_rate"],
            m["k10_coarse_only_acc@1"],
            m["promote_score_shortlist"],
        )

    old_ranks = assign_ranks(merged, key_fn=lambda r: old_promote_key(r))
    new_ranks = assign_ranks(merged, key_fn=lambda r: float(r["promote_score_shortlist"]))

    out_doc = {
        "stamp": _stamp(),
        "manifest": str(manifest),
        "split": args.split,
        "rerank_k": int(args.rerank_k),
        "source_coarse_json": str(args.coarse_json.resolve()),
        "promote_formula": "shortlist_aligned_score (see src/rag3d/evaluation/shortlist_promote.py)",
        "old_promote_rule": "sort by (recall@20, recall@10) descending",
        "rows": merged,
        "ranks_old_promote": old_ranks,
        "ranks_shortlist_aligned": new_ranks,
    }

    metrics_dir = ROOT / "outputs/metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    stamp = out_doc["stamp"]
    out_path = metrics_dir / f"shortlist_alignment_audit_{stamp}.json"
    out_path.write_text(json.dumps(out_doc, indent=2), encoding="utf-8")
    latest = metrics_dir / "shortlist_alignment_audit_latest.json"
    latest.write_text(json.dumps(out_doc, indent=2), encoding="utf-8")
    log.info("Wrote %s and %s", out_path, latest)


if __name__ == "__main__":
    main()
