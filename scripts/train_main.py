#!/usr/bin/env python3
"""Train relation-aware structured model (synthetic, debug, or real manifests)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from rag3d.datasets.collate import collate_grounding_samples
from rag3d.datasets.synthetic import make_synthetic_batch
from rag3d.parsers.cached_parser import CachedParser
from rag3d.parsers.heuristic_parser import HeuristicParser
from rag3d.parsers.structured_rule_parser import StructuredRuleParser
from rag3d.relation_reasoner.losses import grounding_cross_entropy
from rag3d.relation_reasoner.model import RelationAwareModel
from rag3d.training.runner import TrainingConfig, build_loaders, forward_relation_aware, run_training_loop
from rag3d.utils.config import load_yaml_config
from rag3d.utils.logging import setup_logging
from rag3d.utils.seed import set_seed

import logging

log = logging.getLogger(__name__)


def _resolve(p: Path | None, base: Path) -> Path | None:
    if p is None:
        return None
    return p if p.is_absolute() else (base / p).resolve()


def _manifest_paths(tcfg: dict, dcfg: dict, base: Path) -> tuple[Path, Path | None]:
    mode = str(tcfg.get("mode", "real"))
    proc = Path(dcfg.get("processed_dir", "data/processed"))
    if not proc.is_absolute():
        proc = base / proc
    if mode == "debug":
        proc = proc / str(tcfg.get("debug_processed_subdir", "debug"))
    tr = tcfg.get("train_manifest")
    va = tcfg.get("val_manifest")
    train_p = _resolve(Path(tr), base) if tr else proc / "train_manifest.jsonl"
    val_p = _resolve(Path(va), base) if va else proc / "val_manifest.jsonl"
    return train_p, val_p if val_p.is_file() else None


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "configs/train/main.yaml")
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()
    tcfg = load_yaml_config(args.config, base_dir=ROOT)
    dcfg = load_yaml_config(ROOT / tcfg["dataset_config"], base_dir=ROOT)
    mcfg = load_yaml_config(ROOT / "configs/model" / f"{tcfg['model']}.yaml", base_dir=ROOT)
    set_seed(int(tcfg.get("seed", 42)))
    device_s = str(tcfg.get("device", "cpu"))
    device = torch.device(device_s if torch.cuda.is_available() and device_s == "cuda" else "cpu")
    feat_dim = int(mcfg["object_dim"])
    run_name = str(tcfg.get("run_name", "relation_aware"))

    cache_dir = Path(tcfg.get("parser_cache_dir", "data/parser_cache"))
    if not cache_dir.is_absolute():
        cache_dir = ROOT / cache_dir
    parser_mode = str(tcfg.get("parser_mode", "heuristic")).lower()
    if parser_mode == "structured":
        inner = StructuredRuleParser()
        cache_dir = cache_dir / "structured"
    elif parser_mode == "heuristic":
        inner = HeuristicParser()
        cache_dir = cache_dir / "heuristic"
    else:
        log.error("Unknown parser_mode %r; use heuristic or structured.", parser_mode)
        sys.exit(1)
    parser = CachedParser(inner, cache_dir)

    model = RelationAwareModel(
        int(mcfg["object_dim"]),
        int(mcfg["language_dim"]),
        int(mcfg["hidden_dim"]),
        int(mcfg["relation_dim"]),
        anchor_temperature=float(mcfg.get("anchor_temperature", 1.0)),
        dropout=float(mcfg.get("dropout", 0.1)),
    )

    def _forward(m, batch):
        return forward_relation_aware(m, batch, parser)

    tconf = TrainingConfig(
        epochs=int(tcfg.get("epochs", 10)),
        batch_size=int(tcfg.get("batch_size", 8)),
        lr=float(tcfg.get("lr", 1e-4)),
        weight_decay=float(tcfg.get("weight_decay", 0.01)),
        seed=int(tcfg.get("seed", 42)),
        feat_dim=feat_dim,
        checkpoint_dir=_resolve(Path(tcfg["checkpoint_dir"]), ROOT) or ROOT / "outputs/checkpoints",
        metrics_path=_resolve(Path(tcfg.get("metrics_file", "outputs/metrics/train_main_metrics.jsonl")), ROOT)
        or ROOT / "outputs/metrics/train_main_metrics.jsonl",
        device=device_s if device.type == "cuda" else "cpu",
        debug_max_batches=tcfg.get("debug_max_batches"),
        loss=dict(tcfg.get("loss") or {}),
    )
    tconf.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    if tconf.metrics_path.is_file():
        tconf.metrics_path.unlink()

    if args.synthetic or str(tcfg.get("mode")) == "synthetic":
        model = model.to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=tconf.lr, weight_decay=tconf.weight_decay)
        batch = collate_grounding_samples(make_synthetic_batch().samples)
        parsed_list = [parser.parse(s.utterance) for s in batch.samples]
        tensors = batch.to_tensors(feat_dim, device=device)
        target, mask = tensors["target_index"], tensors["object_mask"]
        for epoch in range(tconf.epochs):
            opt.zero_grad()
            logits, _ = model(tensors, parsed_list=parsed_list)
            loss = grounding_cross_entropy(logits, target, mask)
            loss.backward()
            opt.step()
            log.info("epoch=%s loss=%.4f", epoch, float(loss.item()))
        tconf.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict()}, tconf.checkpoint_dir / f"{run_name}_last.pt")
        log.info("Saved %s", tconf.checkpoint_dir / f"{run_name}_last.pt")
        return

    train_p, val_p = _manifest_paths(tcfg, dcfg, ROOT)
    if not train_p.is_file():
        log.error("Train manifest missing: %s — run prepare_data (mock-debug or build).", train_p)
        sys.exit(1)

    train_loader, val_loader = build_loaders(
        train_p,
        val_p,
        tconf,
        num_workers=int(tcfg.get("num_workers", 0)),
    )
    run_training_loop(model, train_loader, val_loader, tconf, _forward, parser=parser, model_name=run_name)


if __name__ == "__main__":
    main()
