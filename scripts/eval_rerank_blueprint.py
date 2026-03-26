#!/usr/bin/env python3
"""Evaluate two-stage rerank + coarse-topK baselines; write blueprint rerank metrics JSON."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from rag3d.datasets.collate import make_grounding_collate_fn
from rag3d.datasets.referit3d import ReferIt3DManifestDataset
from rag3d.evaluation.two_stage_eval import (
    eval_attribute_full_scene,
    eval_coarse_topk_attribute,
    eval_full_scene_relation_aware,
    eval_two_stage,
    load_coarse_model,
    load_two_stage_model,
)
from rag3d.parsers.cached_parser import CachedParser
from rag3d.parsers.heuristic_parser import HeuristicParser
from rag3d.parsers.structured_rule_parser import StructuredRuleParser
from rag3d.relation_reasoner.model import RelationAwareModel
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


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "configs/eval/rerank_blueprint.yaml")
    args = ap.parse_args()
    ecfg = load_yaml_config(args.config, base_dir=ROOT)
    dcfg = load_yaml_config(ROOT / ecfg["dataset_config"], base_dir=ROOT)
    device = torch.device(str(ecfg.get("device", "cpu")))
    seed = int(ecfg.get("seed", 42))
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    margin = float(ecfg.get("margin_thresh", 0.15))
    split = str(ecfg.get("split", "val"))
    manifest = _manifest_path(dcfg, ROOT, split)
    if not manifest.is_file():
        log.error("Manifest not found: %s", manifest)
        sys.exit(1)

    mcfg = load_yaml_config(ROOT / ecfg["model_config"], base_dir=ROOT)
    feat_dim = int(mcfg["object_dim"])
    ds = ReferIt3DManifestDataset(manifest)
    gen = torch.Generator()
    gen.manual_seed(seed)
    loader = DataLoader(
        ds,
        batch_size=int(ecfg.get("batch_size", 16)),
        shuffle=False,
        collate_fn=make_grounding_collate_fn(feat_dim, attach_features=True),
        generator=gen,
    )

    parser_mode = str(ecfg.get("parser_mode", "structured")).lower()
    pcache = ROOT / Path(ecfg.get("parser_cache_dir", "data/parser_cache/diagnosis"))
    if parser_mode == "structured":
        parser = CachedParser(StructuredRuleParser(), pcache / "structured")
    else:
        parser = CachedParser(HeuristicParser(), pcache / "heuristic")

    coarse_ckpt = _resolve(Path(ecfg["coarse_checkpoint"]), ROOT)
    coarse_kind = str(ecfg.get("coarse_model", "attribute_only")).lower()
    out: dict[str, Any] = {
        "manifest": str(manifest),
        "split": split,
        "coarse_model": coarse_kind,
        "rows": {},
        "reference_metrics": {},
    }

    for ref_name, ref_path in (ecfg.get("reference_metrics") or {}).items():
        rp = _resolve(Path(ref_path), ROOT)
        if rp.is_file():
            out["reference_metrics"][ref_name] = json.loads(rp.read_text(encoding="utf-8"))

    for exp in ecfg.get("experiments", []):
        name = str(exp["name"])
        typ = str(exp["type"])
        log.info("Evaluating %s (%s)", name, typ)
        exp_coarse_ckpt = (
            _resolve(Path(exp["coarse_checkpoint"]), ROOT) if exp.get("coarse_checkpoint") else coarse_ckpt
        )
        exp_coarse_kind = str(exp.get("coarse_model", coarse_kind)).lower()
        if typ == "attribute_full_scene":
            coarse = load_coarse_model(mcfg, exp_coarse_ckpt, device, exp_coarse_kind)
            out["rows"][name] = eval_attribute_full_scene(coarse, loader, device, margin)
        elif typ == "coarse_topk_attribute":
            coarse = load_coarse_model(mcfg, exp_coarse_ckpt, device, exp_coarse_kind)
            rk = int(exp.get("rerank_k", 10))
            out["rows"][name] = eval_coarse_topk_attribute(coarse, loader, device, rk, margin)
        elif typ == "two_stage_rerank":
            fine_ckpt = _resolve(Path(exp["fine_checkpoint"]), ROOT) if exp.get("fine_checkpoint") else None
            rk = int(exp.get("rerank_k", 10))
            ts = load_two_stage_model(mcfg, exp_coarse_ckpt, fine_ckpt, rk, device, exp_coarse_kind)
            out["rows"][name] = eval_two_stage(ts, loader, device, parser, margin)
        elif typ == "relation_aware_full_scene":
            ckpt = _resolve(Path(exp["checkpoint"]), ROOT)
            m = RelationAwareModel(
                int(mcfg["object_dim"]),
                int(mcfg["language_dim"]),
                int(mcfg["hidden_dim"]),
                int(mcfg["relation_dim"]),
                anchor_temperature=float(mcfg.get("anchor_temperature", 1.0)),
                dropout=float(mcfg.get("dropout", 0.1)),
            ).to(device)
            try:
                data = torch.load(ckpt, map_location=device, weights_only=False)
            except TypeError:
                data = torch.load(ckpt, map_location=device)
            m.load_state_dict(data["model"], strict=True)
            m.eval()
            out["rows"][name] = eval_full_scene_relation_aware(m, loader, device, parser, margin)
        else:
            log.warning("Skip unknown experiment type %s", typ)

    out_path = _resolve(Path(ecfg["output_json"]), ROOT)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    log.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
