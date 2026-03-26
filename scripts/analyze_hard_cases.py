#!/usr/bin/env python3
"""Hard-case diagnostics: synthetic, debug manifest, or real manifest + optional checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from rag3d.datasets.collate import collate_grounding_samples, make_grounding_collate_fn
from rag3d.datasets.referit3d import ReferIt3DManifestDataset
from rag3d.datasets.synthetic import make_synthetic_batch
from rag3d.diagnostics.case_analysis import summarize_batch_predictions
from rag3d.parsers.heuristic_parser import HeuristicParser
from rag3d.relation_reasoner.model import RelationAwareModel
from rag3d.utils.config import load_yaml_config
from rag3d.utils.logging import setup_logging
from rag3d.visualization.anchor_vis import save_anchor_bar_data
from rag3d.visualization.qualitative_panels import write_side_by_side_case

import logging
from torch.utils.data import DataLoader

log = logging.getLogger(__name__)


def _resolve(p: Path, base: Path) -> Path:
    return p if p.is_absolute() else (base / p).resolve()


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--config", type=Path, default=ROOT / "configs/dataset/referit3d.yaml")
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--use-debug-subdir", action="store_true")
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "outputs/case_studies")
    ap.add_argument("--fig-dir", type=Path, default=ROOT / "outputs/figures")
    ap.add_argument("--max-batches", type=int, default=50)
    args = ap.parse_args()

    mcfg = load_yaml_config(ROOT / "configs/model/relation_aware.yaml", base_dir=ROOT)
    device = torch.device("cpu")
    parser = HeuristicParser()
    model = RelationAwareModel(
        int(mcfg["object_dim"]),
        int(mcfg["language_dim"]),
        int(mcfg["hidden_dim"]),
        int(mcfg["relation_dim"]),
    )
    ckpt = args.checkpoint
    if ckpt is not None and ckpt.is_file():
        try:
            data = torch.load(ckpt, map_location=device, weights_only=False)
        except TypeError:
            data = torch.load(ckpt, map_location=device)
        model.load_state_dict(data["model"], strict=True)
        log.info("Loaded %s", ckpt)
    model.eval()

    rows: list[dict] = []
    tag_counts: Counter[str] = Counter()
    manifest_path: Path | None = None

    if args.synthetic:
        batch = collate_grounding_samples(make_synthetic_batch().samples)
        tensors = batch.to_tensors(int(mcfg["object_dim"]), device=device)
        parsed_list = [parser.parse(s.utterance) for s in batch.samples]
        pconf = [p.parser_confidence for p in parsed_list]
        with torch.no_grad():
            logits, ad = model(tensors, parsed_list=parsed_list)
        rows = summarize_batch_predictions(
            logits,
            tensors["object_mask"],
            tensors["target_index"],
            batch.samples,
            anchor_dist=ad,
            parser_confidences=pconf,
            parsed_list=parsed_list,
        )
    else:
        dcfg = load_yaml_config(args.config, base_dir=ROOT)
        proc = Path(dcfg.get("processed_dir", "data/processed"))
        if not proc.is_absolute():
            proc = ROOT / proc
        if args.use_debug_subdir:
            proc = proc / "debug"
        manifest = args.manifest or (proc / "val_manifest.jsonl")
        manifest = _resolve(manifest, ROOT)
        manifest_path = manifest
        if not manifest.is_file():
            log.error("Manifest not found: %s", manifest)
            sys.exit(1)
        ds = ReferIt3DManifestDataset(manifest)
        loader = DataLoader(
            ds,
            batch_size=8,
            shuffle=False,
            collate_fn=make_grounding_collate_fn(int(mcfg["object_dim"]), attach_features=True),
        )
        nb = 0
        for batch in loader:
            if nb >= args.max_batches:
                break
            nb += 1
            samples = batch["samples_ref"]
            parsed_list = [parser.parse(s.utterance) for s in samples]
            pconf = [p.parser_confidence for p in parsed_list]
            bt = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            sub = {k: bt[k] for k in ("object_features", "object_mask", "raw_texts")}
            with torch.no_grad():
                logits, ad = model(sub, parsed_list=parsed_list)
            part = summarize_batch_predictions(
                logits,
                batch["object_mask"],
                batch["target_index"],
                samples,
                anchor_dist=ad,
                parser_confidences=pconf,
                parsed_list=parsed_list,
            )
            rows.extend(part)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    qual = args.fig_dir / "qualitative_cases"
    anch = args.fig_dir / "anchor_panels"
    qual.mkdir(parents=True, exist_ok=True)
    anch.mkdir(parents=True, exist_ok=True)

    for r in rows:
        for t in r.get("failure_tags", []):
            tag_counts[str(t)] += 1

    fail_path = args.fig_dir / "failure_summary.json"
    fail_path.parent.mkdir(parents=True, exist_ok=True)
    fail_path.write_text(
        json.dumps(
            {
                "failure_tag_counts": dict(tag_counts),
                "n_rows": len(rows),
                "n_incorrect": sum(1 for r in rows if not r.get("correct", True)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    json_path = args.out_dir / "hard_case_summary.json"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    if rows:
        csv_path = args.out_dir / "hard_case_summary.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(
                    {
                        k: json.dumps(r[k]) if k in ("model_prediction", "bridge_module_output") else r[k]
                        for k in r
                    }
                )

    if args.synthetic:
        sbatch = collate_grounding_samples(make_synthetic_batch().samples)
        for i, s in enumerate(sbatch.samples):
            if i >= len(rows):
                break
            write_side_by_side_case(s, int(rows[i]["pred"]), qual / f"{s.scene_id}.md")
        if rows:
            mp = rows[0].get("model_prediction") or {}
            probs = mp.get("anchor_distribution") or []
            if probs:
                oids = [f"obj_{i}" for i in range(len(probs))]
                save_anchor_bar_data(oids, probs, anch / "anchor_example.csv")
    elif manifest_path is not None and rows:
        dsq = ReferIt3DManifestDataset(manifest_path)
        for i in range(min(5, len(dsq), len(rows))):
            write_side_by_side_case(dsq[i], int(rows[i]["pred"]), qual / f"{dsq[i].scene_id}_{i}.md")
        mp = rows[0].get("model_prediction") or {}
        probs = mp.get("anchor_distribution") or []
        if probs:
            oids = [f"obj_{i}" for i in range(len(probs))]
            save_anchor_bar_data(oids, probs, anch / "anchor_example.csv")

    log.info("Wrote %s, %s, qualitative under %s, anchors under %s", json_path, fail_path, qual, anch)


if __name__ == "__main__":
    main()
