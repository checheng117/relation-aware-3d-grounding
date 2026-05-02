"""Config-driven training loop (single GPU, lightweight)."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from rag3d.training.checkpoint_selection import (
    CoarsePipelineSelectionConfig,
    evaluate_coarse_with_fixed_rerank,
)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from rag3d.datasets.collate import make_grounding_collate_fn
from rag3d.datasets.referit3d import ReferIt3DManifestDataset
from rag3d.evaluation.evaluator import Evaluator
from rag3d.parsers.base import BaseParser
from rag3d.relation_reasoner.losses import compute_batch_training_loss
from rag3d.utils.seed import set_seed

log = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    epochs: int = 5
    batch_size: int = 8
    lr: float = 1e-4
    weight_decay: float = 0.01
    seed: int = 42
    feat_dim: int = 256
    checkpoint_dir: Path = Path("outputs/checkpoints")
    metrics_path: Path = Path("outputs/metrics/train_metrics.jsonl")
    device: str = "cuda"
    debug_max_batches: int | None = None
    loss: dict[str, Any] = field(default_factory=dict)
    # If set, each epoch logs coarse val recall@K + acc (stage-1 shortlist utility).
    val_coarse_recall_ks: tuple[int, ...] | None = None
    # Repo root for resolving selection paths.
    repo_root: Path | None = None
    # If set, logs natural two-stage val Acc@1 (frozen reference reranker) for checkpoint selection.
    coarse_pipeline_selection: CoarsePipelineSelectionConfig | None = None


def _snapshot_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Stable checkpoint snapshot detached from live GPU parameter storage."""
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out = {}
    for k, v in batch.items():
        if k in ("meta", "samples_ref"):
            out[k] = v
        elif isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def run_training_loop(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    cfg: TrainingConfig,
    forward_train: Callable[[nn.Module, dict[str, Any]], torch.Tensor],
    parser: BaseParser | None = None,
    model_name: str = "model",
) -> None:
    set_seed(cfg.seed)
    dev = torch.device(cfg.device if torch.cuda.is_available() and cfg.device == "cuda" else "cpu")
    model = model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    cfg.metrics_path.parent.mkdir(parents=True, exist_ok=True)

    evaluator = Evaluator(device=dev)
    history: list[dict[str, Any]] = []
    base = cfg.repo_root or Path(".")
    best_pipe = -1.0
    best_pipe_epoch = -1

    for epoch in range(cfg.epochs):
        model.train()
        losses: list[float] = []
        n_batches = 0
        for batch in train_loader:
            if cfg.debug_max_batches is not None and n_batches >= cfg.debug_max_batches:
                break
            n_batches += 1
            batch_d = _to_device(batch, dev)
            target = batch_d["target_index"]
            mask = batch_d["object_mask"]
            opt.zero_grad()
            logits = forward_train(model, batch_d)
            samples = batch_d.get("samples_ref")
            loss = compute_batch_training_loss(
                logits,
                target,
                mask,
                cfg.loss,
                samples if isinstance(samples, list) else None,
                meta=batch_d.get("meta"),
            )
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))

        row: dict[str, Any] = {"epoch": epoch, "train_loss_mean": sum(losses) / max(len(losses), 1)}
        if val_loader is not None:
            model.eval()
            correct1 = 0
            total = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch_d = _to_device(batch, dev)
                    logits = forward_train(model, batch_d)
                    pred = logits.argmax(dim=-1)
                    correct1 += (pred == batch_d["target_index"]).sum().item()
                    total += pred.numel()
            row["val_acc@1"] = correct1 / max(total, 1)
            # Also evaluator-style acc (masked)
            vaccs = []
            with torch.no_grad():
                for batch in val_loader:
                    batch_d = _to_device(batch, dev)
                    logits = forward_train(model, batch_d)
                    m = evaluator.evaluate_batch(
                        logits, batch_d["target_index"], batch_d["object_mask"], batch_d.get("meta")
                    )
                    vaccs.append(m.get("acc@1", 0.0))
            row["val_acc@1_masked"] = sum(vaccs) / max(len(vaccs), 1)
            if cfg.val_coarse_recall_ks:
                from rag3d.evaluation.coarse_recall import eval_coarse_stage1_metrics

                cr = eval_coarse_stage1_metrics(
                    model,
                    val_loader,
                    dev,
                    margin_thresh=0.15,
                    ks=cfg.val_coarse_recall_ks,
                )
                for key in ("acc@1", "acc@5"):
                    if key in cr:
                        row[f"val_coarse_{key}"] = cr[key]
                for k in cfg.val_coarse_recall_ks:
                    rk = f"recall@{k}"
                    if rk in cr:
                        row[f"val_coarse_{rk}"] = cr[rk]
                if cr.get("stratified_recall_slices"):
                    for sk, sv in cr["stratified_recall_slices"].items():
                        if "same_class_clutter" in sk or "candidate_load::high" in sk:
                            row[f"val_coarse_{sk}"] = sv
        if (
            val_loader is not None
            and cfg.coarse_pipeline_selection is not None
            and cfg.coarse_pipeline_selection.enabled
        ):
            model.eval()
            pipe = evaluate_coarse_with_fixed_rerank(model, val_loader, dev, base, cfg.coarse_pipeline_selection)
            row.update(pipe)
            acc = float(pipe.get("val_pipeline_natural_acc@1", -1.0))
            if acc > best_pipe:
                best_pipe = acc
                best_pipe_epoch = epoch
                best_path = cfg.checkpoint_dir / f"{model_name}_best_pipeline_natural.pt"
                torch.save({"model": _snapshot_state_dict(model), "epoch": epoch, "metrics": row}, best_path)
                log.info("New best pipeline natural acc@1=%.6f -> %s", acc, best_path)
        log.info("epoch %s metrics %s", epoch, row)
        history.append(row)
        with cfg.metrics_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

        ckpt_path = cfg.checkpoint_dir / f"{model_name}_epoch{epoch}.pt"
        torch.save({"model": _snapshot_state_dict(model), "epoch": epoch, "metrics": row}, ckpt_path)

    torch.save({"model": _snapshot_state_dict(model), "history": history}, cfg.checkpoint_dir / f"{model_name}_last.pt")


def build_loaders(
    train_manifest: Path,
    val_manifest: Path | None,
    cfg: TrainingConfig,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader | None]:
    train_ds = ReferIt3DManifestDataset(train_manifest)
    collate = make_grounding_collate_fn(cfg.feat_dim, attach_features=True)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate,
    )
    val_loader = None
    if val_manifest is not None and val_manifest.is_file():
        val_ds = ReferIt3DManifestDataset(val_manifest)
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate,
        )
    return train_loader, val_loader


def forward_attribute(model: nn.Module, batch: dict[str, Any]) -> torch.Tensor:
    return model({k: batch[k] for k in ("object_features", "object_mask", "raw_texts")})


def forward_raw_text_relation(model: nn.Module, batch: dict[str, Any]) -> torch.Tensor:
    return model({k: batch[k] for k in ("object_features", "object_mask", "raw_texts")})


def forward_relation_aware(model: nn.Module, batch: dict[str, Any], parser: BaseParser) -> torch.Tensor:
    samples = batch["samples_ref"]
    parsed_list = [parser.parse(s.utterance) for s in samples]
    logits, _ = model(
        {k: batch[k] for k in ("object_features", "object_mask", "raw_texts")},
        parsed_list=parsed_list,
    )
    return logits
