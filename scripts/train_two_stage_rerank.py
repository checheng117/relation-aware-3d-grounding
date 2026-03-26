#!/usr/bin/env python3
"""Train fine relation-aware reranker on top-K coarse candidates (coarse checkpoint frozen)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from rag3d.datasets.collate import make_grounding_collate_fn
from rag3d.datasets.referit3d import ReferIt3DManifestDataset
from rag3d.parsers.cached_parser import CachedParser
from rag3d.parsers.heuristic_parser import HeuristicParser
from rag3d.parsers.structured_rule_parser import StructuredRuleParser
from rag3d.relation_reasoner.losses import compute_batch_training_loss
from rag3d.relation_reasoner.model import AttributeOnlyModel, CoarseGeomAttributeModel
from rag3d.relation_reasoner.two_stage_rerank import (
    RelationAwareGeomModel,
    TwoStageCoarseRerankModel,
    forward_two_stage_rerank,
)
from rag3d.training.runner import TrainingConfig, _to_device, build_loaders
from rag3d.utils.config import load_yaml_config
from rag3d.utils.logging import setup_logging
from rag3d.utils.seed import set_seed
from torch.utils.data import DataLoader

log = logging.getLogger(__name__)


def _resolve(p: Path | None, base: Path) -> Path | None:
    if p is None:
        return None
    return p if p.is_absolute() else (base / p).resolve()


def _manifest_paths(tcfg: dict, dcfg: dict, base: Path) -> tuple[Path, Path | None]:
    proc = Path(dcfg.get("processed_dir", "data/processed"))
    if not proc.is_absolute():
        proc = base / proc
    tr = tcfg.get("train_manifest")
    va = tcfg.get("val_manifest")
    train_p = _resolve(Path(tr), base) if tr else proc / "train_manifest.jsonl"
    val_p = _resolve(Path(va), base) if va else proc / "val_manifest.jsonl"
    return train_p, val_p if val_p.is_file() else None


def run_two_stage_training(
    model: TwoStageCoarseRerankModel,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    cfg: TrainingConfig,
    parser: CachedParser,
    model_name: str,
) -> None:
    set_seed(cfg.seed)
    dev = torch.device(cfg.device if torch.cuda.is_available() and cfg.device == "cuda" else "cpu")
    model = model.to(dev)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    cfg.metrics_path.parent.mkdir(parents=True, exist_ok=True)

    history: list[dict] = []
    for epoch in range(cfg.epochs):
        model.train()
        losses: list[float] = []
        n_batches = 0
        for batch in train_loader:
            if cfg.debug_max_batches is not None and n_batches >= cfg.debug_max_batches:
                break
            n_batches += 1
            batch_d = _to_device(batch, dev)
            opt.zero_grad()
            logits = forward_two_stage_rerank(model, batch_d, parser)
            samples = batch_d.get("samples_ref")
            loss = compute_batch_training_loss(
                logits,
                batch_d["target_index"],
                batch_d["object_mask"],
                cfg.loss,
                samples if isinstance(samples, list) else None,
                meta=batch_d.get("meta"),
            )
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))

        row: dict = {"epoch": epoch, "train_loss_mean": sum(losses) / max(len(losses), 1)}
        if val_loader is not None:
            model.eval()
            correct1 = 0
            total = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch_d = _to_device(batch, dev)
                    logits = forward_two_stage_rerank(model, batch_d, parser)
                    pred = logits.argmax(dim=-1)
                    correct1 += (pred == batch_d["target_index"]).sum().item()
                    total += pred.numel()
            row["val_acc@1"] = correct1 / max(total, 1)
        log.info("epoch %s metrics %s", epoch, row)
        history.append(row)
        with cfg.metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        ckpt_path = cfg.checkpoint_dir / f"{model_name}_epoch{epoch}.pt"
        torch.save({"model": model.state_dict(), "epoch": epoch, "metrics": row}, ckpt_path)

    torch.save({"model": model.state_dict(), "history": history}, cfg.checkpoint_dir / f"{model_name}_last.pt")
    log.info("Saved %s", cfg.checkpoint_dir / f"{model_name}_last.pt")


def main() -> None:
    setup_logging()
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=ROOT / "configs/train/rerank/rerank_full_k10.yaml")
    args = ap.parse_args()
    tcfg = load_yaml_config(args.config, base_dir=ROOT)
    dcfg = load_yaml_config(ROOT / tcfg["dataset_config"], base_dir=ROOT)
    mcfg = load_yaml_config(ROOT / "configs/model" / f"{tcfg['model']}.yaml", base_dir=ROOT)
    set_seed(int(tcfg.get("seed", 42)))
    device_s = str(tcfg.get("device", "cpu"))
    device = torch.device(device_s if torch.cuda.is_available() and device_s == "cuda" else "cpu")
    feat_dim = int(mcfg["object_dim"])
    run_name = str(tcfg.get("run_name", "rerank_full_k10"))

    cache_dir = Path(tcfg.get("parser_cache_dir", "data/parser_cache"))
    if not cache_dir.is_absolute():
        cache_dir = ROOT / cache_dir
    parser_mode = str(tcfg.get("parser_mode", "structured")).lower()
    if parser_mode == "structured":
        inner = StructuredRuleParser()
        cache_dir = cache_dir / "structured"
    elif parser_mode == "heuristic":
        inner = HeuristicParser()
        cache_dir = cache_dir / "heuristic"
    else:
        log.error("Unknown parser_mode %r", parser_mode)
        sys.exit(1)
    parser = CachedParser(inner, cache_dir)

    coarse_ckpt = _resolve(Path(tcfg["coarse_checkpoint"]), ROOT)
    if coarse_ckpt is None or not coarse_ckpt.is_file():
        log.error("coarse_checkpoint missing or not a file: %s", tcfg.get("coarse_checkpoint"))
        sys.exit(1)

    coarse_kind = str(tcfg.get("coarse_model", "attribute_only")).lower()
    if coarse_kind == "coarse_geom":
        coarse = CoarseGeomAttributeModel(
            int(mcfg["object_dim"]),
            int(mcfg["language_dim"]),
            int(mcfg["hidden_dim"]),
            dropout=float(mcfg.get("dropout", 0.1)),
        )
    else:
        coarse = AttributeOnlyModel(
            int(mcfg["object_dim"]),
            int(mcfg["language_dim"]),
            int(mcfg["hidden_dim"]),
            dropout=float(mcfg.get("dropout", 0.1)),
        )
    try:
        coarse_sd = torch.load(coarse_ckpt, map_location="cpu", weights_only=False)
    except TypeError:
        coarse_sd = torch.load(coarse_ckpt, map_location="cpu")
    if isinstance(coarse_sd, dict) and "model" in coarse_sd:
        coarse_sd = coarse_sd["model"]
    coarse.load_state_dict(coarse_sd, strict=True)
    log.info("Loaded coarse weights from %s", coarse_ckpt)

    fine = RelationAwareGeomModel(
        int(mcfg["object_dim"]),
        int(mcfg["language_dim"]),
        int(mcfg["hidden_dim"]),
        int(mcfg["relation_dim"]),
        anchor_temperature=float(mcfg.get("anchor_temperature", 1.0)),
        dropout=float(mcfg.get("dropout", 0.1)),
    )
    rerank_k = int(tcfg.get("rerank_k", 10))
    model = TwoStageCoarseRerankModel(coarse, fine, rerank_k=rerank_k)

    train_p, val_p = _manifest_paths(tcfg, dcfg, ROOT)
    if not train_p.is_file():
        log.error("Train manifest missing: %s", train_p)
        sys.exit(1)

    tconf = TrainingConfig(
        epochs=int(tcfg.get("epochs", 8)),
        batch_size=int(tcfg.get("batch_size", 16)),
        lr=float(tcfg.get("lr", 1e-4)),
        weight_decay=float(tcfg.get("weight_decay", 0.01)),
        seed=int(tcfg.get("seed", 42)),
        feat_dim=feat_dim,
        checkpoint_dir=_resolve(Path(tcfg["checkpoint_dir"]), ROOT) or ROOT / "outputs/checkpoints_rerank",
        metrics_path=_resolve(Path(tcfg.get("metrics_file", "outputs/metrics/rerank_default.jsonl")), ROOT)
        or ROOT / "outputs/metrics/rerank_default.jsonl",
        device=device_s if device.type == "cuda" else "cpu",
        debug_max_batches=tcfg.get("debug_max_batches"),
        loss=dict(tcfg.get("loss") or {}),
    )
    tconf.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    if tconf.metrics_path.is_file():
        tconf.metrics_path.unlink()

    train_loader, val_loader = build_loaders(
        train_p,
        val_p,
        tconf,
        num_workers=int(tcfg.get("num_workers", 0)),
    )
    run_two_stage_training(model, train_loader, val_loader, tconf, parser, run_name)


if __name__ == "__main__":
    main()
