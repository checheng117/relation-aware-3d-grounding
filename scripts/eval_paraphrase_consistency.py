#!/usr/bin/env python3
"""Paraphrase consistency eval: deterministic rephrasings + agreement / anchor drift (non-destructive outputs)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag3d.datasets.collate import make_grounding_collate_fn
from rag3d.datasets.referit3d import ReferIt3DManifestDataset
from rag3d.evaluation.paraphrase_eval import anchor_distribution_drift, paraphrase_consistency_score
from rag3d.evaluation.paraphrase_templates import relation_preserving_paraphrases
from rag3d.parsers.cached_parser import CachedParser
from rag3d.parsers.heuristic_parser import HeuristicParser
from rag3d.relation_reasoner.model import AttributeOnlyModel, RawTextRelationModel, RelationAwareModel
from rag3d.utils.config import load_yaml_config
from rag3d.utils.logging import setup_logging

import logging

log = logging.getLogger(__name__)


def _resolve(p: Path, base: Path) -> Path:
    return p if p.is_absolute() else (base / p).resolve()


def _load_model(kind: str, mcfg: dict, ckpt: Path | None, device: torch.device):
    if kind == "attribute_only":
        m = AttributeOnlyModel(
            int(mcfg["object_dim"]),
            int(mcfg["language_dim"]),
            int(mcfg["hidden_dim"]),
            dropout=float(mcfg.get("dropout", 0.1)),
        )
    elif kind == "raw_text_relation":
        m = RawTextRelationModel(
            int(mcfg["object_dim"]),
            int(mcfg["language_dim"]),
            int(mcfg["hidden_dim"]),
            int(mcfg["relation_dim"]),
            dropout=float(mcfg.get("dropout", 0.1)),
        )
    else:
        m = RelationAwareModel(
            int(mcfg["object_dim"]),
            int(mcfg["language_dim"]),
            int(mcfg["hidden_dim"]),
            int(mcfg["relation_dim"]),
            anchor_temperature=float(mcfg.get("anchor_temperature", 1.0)),
            dropout=float(mcfg.get("dropout", 0.1)),
        )
    m = m.to(device)
    if ckpt is not None and ckpt.is_file():
        try:
            data = torch.load(ckpt, map_location=device, weights_only=False)
        except TypeError:
            data = torch.load(ckpt, map_location=device)
        m.load_state_dict(data["model"], strict=True)
        log.info("Loaded %s", ckpt)
    m.eval()
    return m


def main() -> int:
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--model", choices=("relation_aware", "attribute_only", "raw_text_relation"), default="relation_aware")
    ap.add_argument("--dataset-config", type=Path, default=ROOT / "configs/dataset/referit3d.yaml")
    ap.add_argument("--max-variants", type=int, default=4)
    ap.add_argument("--max-batches", type=int, default=20)
    ap.add_argument(
        "--out-json",
        type=Path,
        default=ROOT / "outputs/metrics/paraphrase_consistency_blueprint.json",
    )
    args = ap.parse_args()

    dcfg = load_yaml_config(args.dataset_config, base_dir=ROOT)
    mcfg = load_yaml_config(ROOT / "configs/model" / f"{args.model}.yaml", base_dir=ROOT)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feat_dim = int(mcfg["object_dim"])
    manifest = _resolve(args.manifest, ROOT)
    if not manifest.is_file():
        log.error("Manifest not found: %s", manifest)
        return 1

    ckpt = _resolve(args.checkpoint, ROOT) if args.checkpoint else None
    model = _load_model(args.model, mcfg, ckpt, device)
    parser = CachedParser(HeuristicParser(), ROOT / "data/parser_cache" / "eval_paraphrase")

    ds = ReferIt3DManifestDataset(manifest)
    loader = DataLoader(
        ds,
        batch_size=4,
        shuffle=False,
        collate_fn=make_grounding_collate_fn(feat_dim, attach_features=True),
    )

    agree_sum = 0.0
    acc_sum = 0.0
    drift_sum = 0.0
    n_batches = 0

    for bi, batch in enumerate(loader):
        if bi >= args.max_batches:
            break
        samples = batch["samples_ref"]
        texts_per_sample = [relation_preserving_paraphrases(s.utterance, args.max_variants) for s in samples]
        max_v = max(len(t) for t in texts_per_sample)
        logits_list: list[torch.Tensor] = []
        panchor_list: list[torch.Tensor] = []
        for vi in range(max_v):
            sub_texts = []
            for tlist in texts_per_sample:
                sub_texts.append(tlist[vi] if vi < len(tlist) else tlist[-1])
            bt = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            bt["raw_texts"] = sub_texts
            feat = {k: bt[k] for k in ("object_features", "object_mask", "raw_texts")}
            with torch.no_grad():
                if args.model == "relation_aware":
                    parsed_list = [parser.parse(t) for t in sub_texts]
                    logits, pa = model(feat, parsed_list=parsed_list)
                    logits_list.append(logits)
                    panchor_list.append(pa)
                else:
                    logits_list.append(model(feat))
        mask = batch["object_mask"].to(device)
        tgt = batch["target_index"].to(device)
        m = paraphrase_consistency_score(logits_list, tgt, mask)
        agree_sum += m["paraphrase_target_agreement"]
        acc_sum += m["paraphrase_mean_acc@1"]
        if panchor_list:
            drift_sum += anchor_distribution_drift(panchor_list, mask)
        n_batches += 1

    out = {
        "manifest": str(manifest),
        "checkpoint": str(ckpt) if ckpt else None,
        "model": args.model,
        "max_variants": args.max_variants,
        "batches_used": n_batches,
        "mean_paraphrase_target_agreement": agree_sum / max(n_batches, 1),
        "mean_paraphrase_mean_acc@1": acc_sum / max(n_batches, 1),
        "mean_anchor_distribution_drift": drift_sum / max(n_batches, 1) if args.model == "relation_aware" else None,
        "note": "Template paraphrases only; replace with curated paraphrases for publication-grade robustness.",
    }
    outp = _resolve(args.out_json, ROOT)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2), encoding="utf-8")
    log.info("Wrote %s", outp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
