#!/usr/bin/env python3
"""Train Implicit Relation Modeling v3 model (Chunked Dense).

Memory-safe version using chunked dense pairwise computation.
Preserves full N² coverage semantics without memory spikes.

Usage:
    python scripts/train_implicit_relation_v3.py \
        --config configs/implicit_relation_v3.yaml \
        --device cuda
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "repro" / "referit3d_baseline" / "src"))

from rag3d.datasets.referit3d import ReferIt3DManifestDataset
from rag3d.utils.logging import setup_logging
from rag3d.utils.seed import set_seed
from rag3d.utils.config import load_yaml_config
from rag3d.models.relation_aware_implicit_v3 import RelationAwareImplicitV3, build_relation_aware_implicit_v3

log = logging.getLogger(__name__)


def get_gpu_stats() -> Dict[str, float]:
    """Get GPU memory, temperature, and power stats."""
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(",")
            return {
                "gpu_memory_mib": float(parts[0].strip()),
                "gpu_temp_c": float(parts[1].strip()),
                "gpu_power_w": float(parts[2].strip().split(".")[0]),
            }
    except Exception:
        pass
    return {"gpu_memory_mib": 0.0, "gpu_temp_c": 0.0, "gpu_power_w": 0.0}


class IndexedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return (idx, self.dataset[idx])


def build_class_vocabulary(manifest_paths: List[Path]) -> tuple:
    class_names = set()
    for manifest_path in manifest_paths:
        if not manifest_path.exists():
            continue
        with open(manifest_path) as f:
            for line in f:
                sample = json.loads(line)
                for obj in sample.get("objects", []):
                    class_names.add(obj["class_name"])

    sorted_classes = sorted(list(class_names))
    class_to_idx = {name: idx for idx, name in enumerate(sorted_classes)}
    return class_to_idx, sorted_classes


def collate_fn(
    batch: List[tuple],
    feat_dim: int = 256,
    text_features: Optional[np.ndarray] = None,
    class_to_idx: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """Collate samples for SimplePointEncoder mode."""
    B = len(batch)
    max_n = max(len(s.objects) for idx, s in batch)

    object_features = torch.zeros(B, max_n, feat_dim)
    object_mask = torch.zeros(B, max_n, dtype=torch.bool)
    target_index = torch.zeros(B, dtype=torch.long)
    texts = []
    sample_indices = []
    centers = torch.zeros(B, max_n, 3)
    sizes = torch.zeros(B, max_n, 3)

    class_indices = None
    if class_to_idx is not None:
        class_indices = torch.zeros(B, max_n, dtype=torch.long)

    for i, (idx, sample) in enumerate(batch):
        texts.append(sample.utterance)
        sample_indices.append(idx)

        for j, obj in enumerate(sample.objects):
            if obj.center:
                center_tensor = torch.tensor(obj.center).float()
                object_features[i, j, 0:3] = center_tensor / 5.0
                centers[i, j] = center_tensor
            if obj.size:
                size_tensor = torch.tensor(obj.size).float()
                object_features[i, j, 3:6] = size_tensor / 2.0
                sizes[i, j] = size_tensor

            import hashlib
            class_hash = int(hashlib.md5(obj.class_name.encode()).hexdigest()[:8], 16)
            feat_hash = class_hash % (feat_dim - 6)
            object_features[i, j, 6 + feat_hash] = 1.0

            if class_to_idx is not None:
                class_indices[i, j] = class_to_idx.get(obj.class_name, 0)

            object_mask[i, j] = True

        target_index[i] = sample.target_index

    result = {
        "object_features": object_features,
        "object_mask": object_mask,
        "target_index": target_index,
        "texts": texts,
        "samples_ref": [s for idx, s in batch],
        "centers": centers,
        "sizes": sizes,
    }

    if class_indices is not None:
        result["class_indices"] = class_indices

    if text_features is not None and len(sample_indices) == B:
        result["bert_features"] = torch.tensor(text_features[sample_indices], dtype=torch.float32)

    return result


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float = 1.0,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    n_batches = 0

    criterion = nn.CrossEntropyLoss()
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        object_features = batch["object_features"].to(device)
        object_mask = batch["object_mask"].to(device)
        target_index = batch["target_index"].to(device)
        centers = batch["centers"].to(device)
        sizes = batch["sizes"].to(device)

        if "bert_features" in batch:
            text_features = batch["bert_features"].to(device)
        else:
            text_features = torch.randn(object_features.shape[0], 768, device=device)

        class_indices = None
        if "class_indices" in batch:
            class_indices = batch["class_indices"].to(device)

        outputs = model(
            points=object_features,
            object_mask=object_mask,
            text_features=text_features,
            class_indices=class_indices,
            centers=centers,
            sizes=sizes,
        )
        logits = outputs["logits"]

        # Sync after forward to prevent async timeout
        torch.cuda.synchronize()

        loss = criterion(logits, target_index)
        loss.backward()

        # Sync after backward to ensure gradient complete
        torch.cuda.synchronize()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        pred = logits.argmax(dim=-1)
        correct += (pred == target_index).sum().item()
        total += target_index.numel()
        n_batches += 1

    return {
        "loss": total_loss / max(n_batches, 1),
        "acc": correct / max(total, 1),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    model.eval()
    correct_at_1 = 0
    correct_at_5 = 0
    total = 0

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        object_features = batch["object_features"].to(device)
        object_mask = batch["object_mask"].to(device)
        target_index = batch["target_index"].to(device)
        centers = batch["centers"].to(device)
        sizes = batch["sizes"].to(device)

        if "bert_features" in batch:
            text_features = batch["bert_features"].to(device)
        else:
            text_features = torch.randn(object_features.shape[0], 768, device=device)

        class_indices = None
        if "class_indices" in batch:
            class_indices = batch["class_indices"].to(device)

        outputs = model(
            points=object_features,
            object_mask=object_mask,
            text_features=text_features,
            class_indices=class_indices,
            centers=centers,
            sizes=sizes,
        )
        logits = outputs["logits"]

        pred = logits.argmax(dim=-1)
        correct_at_1 += (pred == target_index).sum().item()

        for i in range(len(target_index)):
            valid_mask = object_mask[i]
            num_valid = valid_mask.sum().item()
            k = min(5, num_valid)
            if k > 0:
                topk = logits[i][valid_mask].topk(k).indices
                valid_indices = torch.where(valid_mask)[0]
                topk_original = valid_indices[topk]
                if target_index[i] in topk_original:
                    correct_at_5 += 1

        total += target_index.numel()

    return {
        "acc_at_1": correct_at_1 / max(total, 1),
        "acc_at_5": correct_at_5 / max(total, 1),
    }


def train(config_path: Path, device_str: str = "cuda", resume_path: Optional[Path] = None):
    setup_logging()
    config = load_yaml_config(config_path, base_dir=ROOT)

    set_seed(config.get("training", {}).get("seed", 42))

    device = torch.device(device_str if torch.cuda.is_available() and device_str == "cuda" else "cpu")
    log.info(f"Using device: {device}")

    model = build_relation_aware_implicit_v3(config)
    model = model.to(device)
    log.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    dataset_config = config.get("dataset", {})
    manifest_dir = ROOT / dataset_config.get("manifest_dir", "data/processed")

    train_manifest = manifest_dir / dataset_config.get("train_manifest", "train_manifest.jsonl")
    val_manifest = manifest_dir / dataset_config.get("val_manifest", "val_manifest.jsonl")
    test_manifest = manifest_dir / dataset_config.get("test_manifest", "test_manifest.jsonl")

    train_dataset = IndexedDataset(ReferIt3DManifestDataset(train_manifest))
    val_dataset = IndexedDataset(ReferIt3DManifestDataset(val_manifest))

    log.info(f"Train samples: {len(train_dataset)}")
    log.info(f"Val samples: {len(val_dataset)}")

    training_config = config.get("training", {})
    batch_size = training_config.get("batch_size", 16)
    feat_dim = config.get("model", {}).get("point_input_dim", 256)

    bert_dir = ROOT / dataset_config.get("bert_dir", "data/text_features")
    train_bert_path = bert_dir / "train_bert_embeddings.npy"
    val_bert_path = bert_dir / "val_bert_embeddings.npy"

    train_bert_features = None
    val_bert_features = None

    if train_bert_path.exists():
        train_bert_features = np.load(train_bert_path)
        log.info(f"Loaded train BERT features: {train_bert_features.shape}")

    if val_bert_path.exists():
        val_bert_features = np.load(val_bert_path)
        log.info(f"Loaded val BERT features: {val_bert_features.shape}")

    model_config = config.get("model", {})
    use_learned_class_embedding = model_config.get("use_learned_class_embedding", False)
    class_to_idx = None
    if use_learned_class_embedding:
        class_to_idx, class_vocab = build_class_vocabulary([train_manifest, val_manifest, test_manifest])
        log.info(f"Class vocabulary size: {len(class_vocab)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, feat_dim, train_bert_features, class_to_idx),
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get("evaluation", {}).get("batch_size", batch_size * 2),
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, feat_dim, val_bert_features, class_to_idx),
        num_workers=0,
    )

    lr = float(training_config.get("lr", 1e-4))
    weight_decay = float(training_config.get("weight_decay", 1e-4))
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    epochs = training_config.get("epochs", 30)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    checkpoint_dir = ROOT / config.get("checkpoint", {}).get("dir", "outputs/implicit_relation_v3")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0
    history = []
    start_epoch = 0

    # Resume from checkpoint if provided
    if resume_path is not None and resume_path.exists():
        log.info(f"Resuming from checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1  # Continue from next epoch
        best_val_acc = checkpoint.get("val_acc", 0.0)
        log.info(f"Resumed from epoch {start_epoch}, best val acc: {best_val_acc:.4f}")

    debug_max_batches = config.get("debug", {}).get("max_batches")

    log.info(f"Starting training for {epochs} epochs (from epoch {start_epoch + 1})...")
    log.info(f"Chunk size: {model_config.get('chunk_size', 8)}")
    log.info(f"Goal: Match v1's 31.26% with stable training")

    training_start_time = datetime.now()

    for epoch in range(start_epoch, epochs):
        epoch_start_time = datetime.now()
        log.info(f"Epoch {epoch + 1}/{epochs}")

        train_metrics = train_epoch(
            model, train_loader, optimizer, device,
            grad_clip=training_config.get("grad_clip", 1.0),
            max_batches=debug_max_batches,
        )

        val_metrics = evaluate(
            model, val_loader, device,
            max_batches=debug_max_batches,
        )

        scheduler.step()

        # Sync before clearing cache to ensure all operations complete
        torch.cuda.synchronize()

        # Clear CUDA cache to prevent memory accumulation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Get epoch timing and GPU stats
        epoch_end_time = datetime.now()
        epoch_runtime_sec = (epoch_end_time - epoch_start_time).total_seconds()
        gpu_stats = get_gpu_stats()

        log.info(
            f"Epoch {epoch + 1} complete: "
            f"Train loss: {train_metrics['loss']:.4f}, "
            f"Train acc: {train_metrics['acc']:.4f}, "
            f"Val acc@1: {val_metrics['acc_at_1']:.4f}, "
            f"Val acc@5: {val_metrics['acc_at_5']:.4f}, "
            f"Runtime: {epoch_runtime_sec:.1f}s, "
            f"GPU mem: {gpu_stats['gpu_memory_mib']:.0f}MiB, "
            f"Temp: {gpu_stats['gpu_temp_c']:.0f}C, "
            f"Power: {gpu_stats['gpu_power_w']:.0f}W"
        )

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "val_acc_at_1": val_metrics["acc_at_1"],
            "val_acc_at_5": val_metrics["acc_at_5"],
            "lr": optimizer.param_groups[0]['lr'],
            "epoch_runtime_sec": epoch_runtime_sec,
            "gpu_memory_mib": gpu_stats["gpu_memory_mib"],
            "gpu_temp_c": gpu_stats["gpu_temp_c"],
            "gpu_power_w": gpu_stats["gpu_power_w"],
        })

        if val_metrics["acc_at_1"] > best_val_acc:
            best_val_acc = val_metrics["acc_at_1"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": best_val_acc,
                "config": config,
            }, checkpoint_dir / "best_model.pt")
            log.info(f"Saved best model with val acc: {best_val_acc:.4f}")

        # Save checkpoint every epoch for crash recovery
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_acc": val_metrics["acc_at_1"],
            "config": config,
            "history": history,
        }, checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pt")
        log.info(f"Saved checkpoint at epoch {epoch + 1}")

    # Save final model
    torch.save(model.state_dict(), checkpoint_dir / "final_model.pt")

    total_runtime_sec = (datetime.now() - training_start_time).total_seconds()
    log.info(f"Total training runtime: {total_runtime_sec:.1f}s ({total_runtime_sec/60:.1f} min)")

    history_path = checkpoint_dir / "training_history.json"
    with history_path.open("w") as f:
        json.dump(history, f, indent=2)

    log.info(f"Training complete. Best val acc: {best_val_acc:.4f}")
    log.info(f"Baseline: 30.79% Test Acc@1")
    log.info(f"v1 result: 31.26% (crashed at epoch 17)")
    log.info(f"v2 result: 28.55% (stable but degraded)")

    return history, best_val_acc


def main():
    parser = argparse.ArgumentParser(description="Train Implicit Relation v3")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/implicit_relation_v3.yaml")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--resume", type=Path, default=None, help="Resume from checkpoint")
    args = parser.parse_args()

    train(args.config, args.device, args.resume)


if __name__ == "__main__":
    main()