#!/usr/bin/env python3
"""End-to-end smoke: synthetic batch → parse → model → eval → artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from rag3d.datasets.collate import collate_grounding_samples
from rag3d.datasets.synthetic import make_synthetic_batch
from rag3d.evaluation.evaluator import Evaluator
from rag3d.parsers.cached_parser import CachedParser
from rag3d.parsers.heuristic_parser import HeuristicParser
from rag3d.relation_reasoner.losses import grounding_cross_entropy
from rag3d.relation_reasoner.model import AttributeOnlyModel, RelationAwareModel
from rag3d.utils.env import ensure_env_loaded
from rag3d.utils.logging import setup_logging

import logging

log = logging.getLogger(__name__)


def main() -> None:
    setup_logging()
    ensure_env_loaded()
    device = torch.device("cpu")
    feat_dim = 256
    batch = collate_grounding_samples(make_synthetic_batch(feat_dim=feat_dim).samples)
    cache = ROOT / "data/parser_cache" / "smoke"
    parser = CachedParser(HeuristicParser(), cache)
    parsed = [parser.parse(s.utterance) for s in batch.samples]

    tensors = batch.to_tensors(feat_dim, device=device)
    target = tensors["target_index"]
    mask = tensors["object_mask"]

    attr = AttributeOnlyModel(feat_dim, 256, 256).to(device)
    relm = RelationAwareModel(feat_dim, 256, 256, 128).to(device)
    attr.train()
    opt = torch.optim.AdamW(list(attr.parameters()) + list(relm.parameters()), lr=1e-3)
    for _ in range(3):
        opt.zero_grad()
        la = attr(tensors)
        loss_a = grounding_cross_entropy(la, target, mask)
        lr, _ = relm(tensors, parsed_list=parsed)
        loss_r = grounding_cross_entropy(lr, target, mask)
        (loss_a + loss_r).backward()
        opt.step()

    attr.eval()
    relm.eval()
    with torch.no_grad():
        la = attr(tensors)
        lr, ad = relm(tensors, parsed_list=parsed)
    meta = [{"relation_type_gold": s.relation_type_gold, "tags": s.tags} for s in batch.samples]
    ev = Evaluator(device=device)
    summary = {
        "attribute_only": ev.evaluate_batch(la, target, mask, meta),
        "relation_aware": ev.evaluate_batch(lr, target, mask, meta),
    }
    out_metrics = ROOT / "outputs/metrics/smoke_summary.json"
    out_metrics.parent.mkdir(parents=True, exist_ok=True)
    out_metrics.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("Smoke OK. Wrote %s", out_metrics)
    print("SMOKE_OK", out_metrics)


if __name__ == "__main__":
    main()
