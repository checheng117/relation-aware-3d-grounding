#!/usr/bin/env python3
"""Evaluate one or more models on a manifest; write main + stratified metrics JSON."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from rag3d.datasets.collate import make_grounding_collate_fn
from rag3d.datasets.referit3d import ReferIt3DManifestDataset
from rag3d.datasets.synthetic import make_synthetic_batch
from rag3d.evaluation.metrics import per_sample_correct_at1, per_sample_correct_at5
from rag3d.evaluation.stratified_eval import (
    augment_meta_with_model_margins,
    stratified_accuracy_from_lists,
)
from rag3d.parsers.cached_parser import CachedParser
from rag3d.parsers.heuristic_parser import HeuristicParser
from rag3d.parsers.structured_rule_parser import StructuredRuleParser
from rag3d.relation_reasoner.model import AttributeOnlyModel, RawTextRelationModel, RelationAwareModel
from rag3d.utils.config import load_yaml_config
from rag3d.utils.logging import setup_logging

import logging
from torch.utils.data import DataLoader

log = logging.getLogger(__name__)


def _resolve(p: Path | None, base: Path) -> Path:
    if p is None:
        return base
    return p if p.is_absolute() else (base / p).resolve()


def _manifest_path(ecfg: dict, dcfg: dict, base: Path, use_debug_subdir: bool) -> Path:
    m = ecfg.get("manifest")
    if m:
        return _resolve(Path(m), base)
    proc = Path(dcfg.get("processed_dir", "data/processed"))
    if not proc.is_absolute():
        proc = base / proc
    split = str(ecfg.get("split", "val"))
    if use_debug_subdir:
        proc = proc / str(ecfg.get("debug_processed_subdir", "debug"))
    return proc / f"{split}_manifest.jsonl"


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
        log.info("Loaded checkpoint %s", ckpt)
    else:
        log.warning("Checkpoint missing — evaluating randomly initialized weights: %s", ckpt)
    m.eval()
    return m


def _forward(kind: str, model, batch: dict, parser: HeuristicParser | None, device: torch.device):
    b = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    sub = {k: b[k] for k in ("object_features", "object_mask", "raw_texts")}
    if kind in ("attribute_only", "raw_text_relation"):
        return model(sub)
    assert parser is not None
    samples = batch["samples_ref"]
    parsed_list = [parser.parse(s.utterance) for s in samples]
    logits, _ = model(sub, parsed_list=parsed_list)
    return logits


def run_eval_on_loader(
    kind: str,
    model,
    loader: DataLoader,
    device: torch.device,
    parser: HeuristicParser | None,
    margin_thresh: float,
) -> tuple[dict[str, float], dict[str, float]]:
    c1: list[bool] = []
    c5: list[bool] = []
    meta_flat: list[dict] = []
    for batch in loader:
        logits = _forward(kind, model, batch, parser, device)
        mask = batch["object_mask"].to(device)
        target = batch["target_index"].to(device)
        meta = copy.deepcopy(batch["meta"])
        augment_meta_with_model_margins(logits.detach().cpu(), mask.cpu(), meta, margin_thresh=margin_thresh)
        c1.extend(per_sample_correct_at1(logits, target, mask))
        c5.extend(per_sample_correct_at5(logits, target, mask))
        meta_flat.extend(meta)
    main = {
        "acc@1": sum(c1) / max(len(c1), 1),
        "acc@5": sum(c5) / max(len(c5), 1),
        "n": len(c1),
    }
    strat = stratified_accuracy_from_lists(c1, meta_flat)
    return main, strat


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "configs/eval/default.yaml")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--use-debug-subdir", action="store_true", help="Load manifests from processed_dir/debug/")
    args = ap.parse_args()
    ecfg = load_yaml_config(args.config, base_dir=ROOT)
    use_debug = bool(args.use_debug_subdir or ecfg.get("use_debug_subdir"))
    dcfg = load_yaml_config(ROOT / ecfg["dataset_config"], base_dir=ROOT)
    device = torch.device(str(ecfg.get("device", "cpu")))
    margin = float(ecfg.get("margin_thresh", 0.15))
    ckpt_dir = _resolve(Path(ecfg.get("checkpoint_dir", "outputs/checkpoints")), ROOT)
    out_main = _resolve(Path(ecfg.get("main_results_path", "outputs/metrics/main_results.json")), ROOT)
    out_strat = _resolve(Path(ecfg.get("stratified_results_path", "outputs/metrics/stratified_results.json")), ROOT)
    out_debug = _resolve(Path(ecfg.get("debug_results_path", "outputs/metrics/debug_results.json")), ROOT)

    parser_mode = str(ecfg.get("parser_mode", "heuristic")).lower()
    pcache = ROOT / Path(ecfg.get("parser_cache_dir", "data/parser_cache"))
    if parser_mode == "structured":
        parser = CachedParser(StructuredRuleParser(), pcache / "eval" / "structured")
    else:
        parser = CachedParser(HeuristicParser(), pcache / "eval" / "heuristic")

    if args.synthetic:
        from rag3d.datasets.collate import collate_grounding_samples

        batch = collate_grounding_samples(make_synthetic_batch().samples)
        mcfg_a = load_yaml_config(ROOT / "configs/model/attribute_only.yaml", base_dir=ROOT)
        mcfg_r = load_yaml_config(ROOT / "configs/model/raw_text_relation.yaml", base_dir=ROOT)
        mcfg_o = load_yaml_config(ROOT / "configs/model/relation_aware.yaml", base_dir=ROOT)
        tensors = batch.to_tensors(int(mcfg_a["object_dim"]), device=device)
        tensors["meta"] = [
            {"relation_type_gold": s.relation_type_gold, "tags": dict(s.tags)} for s in batch.samples
        ]
        tensors["samples_ref"] = batch.samples
        main_out: dict[str, dict] = {}
        strat_out: dict[str, dict] = {}
        for kind, mcfg, rn in [
            ("attribute_only", mcfg_a, "baseline"),
            ("raw_text_relation", mcfg_r, "raw_relation"),
            ("relation_aware", mcfg_o, "relation_aware"),
        ]:
            model = _load_model(kind, mcfg, ckpt_dir / f"{rn}_last.pt", device)
            logits = _forward(kind, model, tensors, parser if kind == "relation_aware" else None, device)
            mask = tensors["object_mask"]
            target = tensors["target_index"]
            meta = copy.deepcopy(tensors["meta"])
            augment_meta_with_model_margins(logits.detach().cpu(), mask.cpu(), meta, margin_thresh=margin)
            c1 = per_sample_correct_at1(logits, target, mask)
            c5 = per_sample_correct_at5(logits, target, mask)
            main_out[kind] = {"acc@1": sum(c1) / len(c1), "acc@5": sum(c5) / len(c5), "n": len(c1)}
            strat_out[kind] = stratified_accuracy_from_lists(c1, meta)
        out_main.parent.mkdir(parents=True, exist_ok=True)
        out_main.write_text(json.dumps(main_out, indent=2), encoding="utf-8")
        out_strat.write_text(json.dumps(strat_out, indent=2), encoding="utf-8")
        log.info("Wrote %s and %s", out_main, out_strat)
        return

    manifest = _manifest_path(ecfg, dcfg, ROOT, use_debug_subdir=use_debug)
    if not manifest.is_file():
        log.error("Manifest not found: %s", manifest)
        sys.exit(1)

    mcfg_paths = {
        "attribute_only": ROOT / "configs/model/attribute_only.yaml",
        "raw_text_relation": ROOT / "configs/model/raw_text_relation.yaml",
        "relation_aware": ROOT / "configs/model/relation_aware.yaml",
    }
    default_run_names = {
        "attribute_only": "baseline",
        "raw_text_relation": "raw_relation",
        "relation_aware": "relation_aware",
    }
    ckpt_names = ecfg.get("checkpoint_run_names") or {}
    feat_dim = int(load_yaml_config(mcfg_paths["attribute_only"], base_dir=ROOT)["object_dim"])
    ds = ReferIt3DManifestDataset(manifest)
    loader = DataLoader(
        ds,
        batch_size=int(ecfg.get("batch_size", 16)),
        shuffle=False,
        collate_fn=make_grounding_collate_fn(feat_dim, attach_features=True),
    )

    main_out = {}
    strat_out = {}
    for kind in ecfg.get("models", ["attribute_only", "raw_text_relation", "relation_aware"]):
        mcfg = load_yaml_config(mcfg_paths[kind], base_dir=ROOT)
        rn = str(ckpt_names.get(kind) or ecfg.get("run_name") or default_run_names.get(kind, kind))
        ckpt = ckpt_dir / f"{rn}_last.pt"
        model = _load_model(kind, mcfg, ckpt, device)
        main, strat = run_eval_on_loader(kind, model, loader, device, parser if kind == "relation_aware" else None, margin)
        main_out[kind] = main
        strat_out[kind] = strat

    out_main.parent.mkdir(parents=True, exist_ok=True)
    out_main.write_text(json.dumps(main_out, indent=2), encoding="utf-8")
    out_strat.write_text(json.dumps(strat_out, indent=2), encoding="utf-8")
    if use_debug:
        out_debug.write_text(json.dumps({"main": main_out, "stratified": strat_out}, indent=2), encoding="utf-8")
        log.info("Wrote debug results %s", out_debug)
    log.info("Wrote %s and %s", out_main, out_strat)


if __name__ == "__main__":
    main()
