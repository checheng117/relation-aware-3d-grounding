#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from rag3d.datasets.collate import collate_grounding_samples
from rag3d.datasets.synthetic import make_synthetic_batch
from rag3d.evaluation.evaluator import Evaluator
from rag3d.parsers.heuristic_parser import HeuristicParser
from rag3d.relation_reasoner.model import RelationAwareModel
from rag3d.utils.config import load_yaml_config
from rag3d.utils.logging import setup_logging

import logging

log = logging.getLogger(__name__)


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/metrics/paraphrase_eval.json")
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()
    if not args.synthetic:
        log.error("Use --synthetic for demo paraphrase views.")
        sys.exit(1)

    mcfg = load_yaml_config(ROOT / "configs/model/relation_aware.yaml", base_dir=ROOT)
    device = torch.device("cpu")
    batch = collate_grounding_samples(make_synthetic_batch().samples)
    tensors = batch.to_tensors(int(mcfg["object_dim"]), device=device)
    model = RelationAwareModel(
        int(mcfg["object_dim"]),
        int(mcfg["language_dim"]),
        int(mcfg["hidden_dim"]),
        int(mcfg["relation_dim"]),
    )
    model.eval()
    texts_a = [s.utterance for s in batch.samples]
    texts_b = [t.replace("Pick", "Choose") for t in texts_a]
    hp = HeuristicParser()
    p1 = [hp.parse(t) for t in texts_a]
    p2 = [hp.parse(t) for t in texts_b]
    with torch.no_grad():
        l1, _ = model(tensors, parsed_list=p1)
        l2, _ = model(tensors, parsed_list=p2)
    ev = Evaluator(device=device)
    metrics = ev.paraphrase_eval([l1, l2], tensors["target_index"], tensors["object_mask"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    log.info("Wrote %s (demo uses identical logits unless parser differs)", args.out)


if __name__ == "__main__":
    main()
