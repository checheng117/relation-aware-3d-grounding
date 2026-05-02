#!/usr/bin/env python3
"""Train SAT baseline model.

This script trains the SAT-style baseline on NR3D data.

Usage:
    python repro/sat_baseline/scripts/train_sat.py \
        --config configs/sat_baseline.yaml \
        --device cuda
"""

from __future__ import annotations

# Disable PyTorch nested tensor feature to avoid SIGILL crash on certain GPU/CUDA configurations
# This is a prototype feature in PyTorch that can cause hardware incompatibility issues
import os
os.environ['TORCH_DISABLE_NESTED_TENSOR'] = '1'

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

# Add paths
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "repro" / "sat_baseline" / "src"))

from rag3d.datasets.referit3d import ReferIt3DManifestDataset
from rag3d.datasets.schemas import GroundingSample
from rag3d.utils.logging import setup_logging
from rag3d.utils.seed import set_seed
from rag3d.utils.config import load_yaml_config

from rag3d.models.sat_model import SATModel, build_sat_model

log = logging.getLogger(__name__)


class IndexedDataset(torch.utils.data.Dataset):
    """Wrapper dataset that returns (index, sample) tuples for proper BERT alignment."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        return (idx, sample)


def collate_fn(
    batch: List[tuple],
    feat_dim: int = 256,
    text_features: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Collate samples into batch tensors (SAT-compatible).

    Uses real BERT features and geometry features (center/size + class hash).

    Args:
        batch: List of (index, sample) tuples from IndexedDataset
        feat_dim: Feature dimension (default 256)
        text_features: Pre-computed BERT features array

    Returns:
        Dict with object_features, object_mask, target_index, bert_features
    """
    B = len(batch)
    max_n = max(len(s.objects) for idx, s in batch)

    # Initialize tensors
    object_features = torch.zeros(B, max_n, feat_dim)
    object_mask = torch.zeros(B, max_n, dtype=torch.bool)
    target_index = torch.zeros(B, dtype=torch.long)
    texts = []
    sample_indices = []

    for i, (idx, sample) in enumerate(batch):
        texts.append(sample.utterance)
        sample_indices.append(idx)

        for j, obj in enumerate(sample.objects):
            # Use geometry features: center + size + class hash
            # Channels 0-2: center (normalized)
            # Channels 3-5: size (normalized)
            # Channels 6+: class name hash (one-hot semantic signal)
            if obj.center:
                object_features[i, j, 0:3] = torch.tensor(obj.center).float() / 5.0
            if obj.size:
                object_features[i, j, 3:6] = torch.tensor(obj.size).float() / 2.0
            # Class name hash as semantic feature
            import hashlib
            class_hash = int(hashlib.md5(obj.class_name.encode()).hexdigest()[:8], 16)
            feat_hash = class_hash % (feat_dim - 6)
            object_features[i, j, 6 + feat_hash] = 1.0
            object_mask[i, j] = True

        target_index[i] = sample.target_index

    result = {
        "object_features": object_features,
        "object_mask": object_mask,
        "target_index": target_index,
        "texts": texts,
        "samples_ref": [s for idx, s in batch],
    }

    # Add real BERT features
    if text_features is not None:
        if len(sample_indices) == B:
            bert_features = torch.tensor(
                text_features[sample_indices],
                dtype=torch.float32,
            )
            result["bert_features"] = bert_features

    return result


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float = 1.0,
    max_batches: Optional[int] = None,
    use_bert: bool = True,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    n_batches = 0

    criterion = nn.CrossEntropyLoss()

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        # Move to device
        object_features = batch["object_features"].to(device)
        object_mask = batch["object_mask"].to(device)
        target_index = batch["target_index"].to(device)

        # Get BERT features
        if use_bert and "bert_features" in batch:
            text_features = batch["bert_features"].to(device)
        else:
            batch_size = object_features.shape[0]
            text_features = torch.randn(batch_size, 768, device=device)

        # Forward pass
        outputs = model(
            points=object_features,
            object_mask=object_mask,
            text_features=text_features,
        )
        logits = outputs["logits"]

        # Compute loss
        loss = criterion(logits, target_index)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        # Track metrics
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
    use_bert: bool = True,
) -> Dict[str, float]:
    """Evaluate model."""
    model.eval()
    correct_at_1 = 0
    correct_at_5 = 0
    correct_at_0_5 = 0
    total = 0

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        object_features = batch["object_features"].to(device)
        object_mask = batch["object_mask"].to(device)
        target_index = batch["target_index"].to(device)

        # Get BERT features
        if use_bert and "bert_features" in batch:
            text_features = batch["bert_features"].to(device)
        else:
            batch_size = object_features.shape[0]
            text_features = torch.randn(batch_size, 768, device=device)

        outputs = model(
            points=object_features,
            object_mask=object_mask,
            text_features=text_features,
        )
        logits = outputs["logits"]

        # Acc@1
        pred = logits.argmax(dim=-1)
        correct_at_1 += (pred == target_index).sum().item()

        # Acc@5 and Acc@0.5
        for i in range(len(target_index)):
            valid_mask = object_mask[i]
            num_valid = valid_mask.sum().item()

            # Acc@5
            k5 = min(5, num_valid)
            if k5 > 0:
                topk5 = logits[i][valid_mask].topk(k5).indices
                valid_indices = torch.where(valid_mask)[0]
                topk5_original = valid_indices[topk5]
                if target_index[i] in topk5_original:
                    correct_at_5 += 1

            # Acc@0.5 (top 50%)
            k0_5 = max(1, int(num_valid * 0.5))
            if k0_5 > 0:
                topk0_5 = logits[i][valid_mask].topk(k0_5).indices
                topk0_5_original = valid_indices[topk0_5]
                if target_index[i] in topk0_5_original:
                    correct_at_0_5 += 1

        total += target_index.numel()

    return {
        "acc_at_1": correct_at_1 / max(total, 1),
        "acc_at_5": correct_at_5 / max(total, 1),
        "acc_at_0_5": correct_at_0_5 / max(total, 1),
    }


def train(config_path: Path, device_str: str = "cuda", smoke: bool = False):
    """Main training loop."""
    setup_logging()
    config = load_yaml_config(config_path, base_dir=ROOT)

    # Set seed
    set_seed(config.get("training", {}).get("seed", 42))

    # Device
    device = torch.device(device_str if torch.cuda.is_available() and device_str == "cuda" else "cpu")
    log.info(f"Using device: {device}")

    # Build SAT model
    model = build_sat_model(config)
    model = model.to(device)
    log.info(f"SAT Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Load data
    dataset_config = config.get("dataset", {})
    manifest_dir = ROOT / dataset_config.get("manifest_dir", "data/processed/scene_disjoint/expanded_nr3d")

    train_manifest = manifest_dir / dataset_config.get("train_manifest", "train_manifest.jsonl")
    val_manifest = manifest_dir / dataset_config.get("val_manifest", "val_manifest.jsonl")
    test_manifest = manifest_dir / dataset_config.get("test_manifest", "test_manifest.jsonl")

    train_dataset = IndexedDataset(ReferIt3DManifestDataset(train_manifest))
    val_dataset = IndexedDataset(ReferIt3DManifestDataset(val_manifest))

    log.info(f"Train samples: {len(train_dataset)}")
    log.info(f"Val samples: {len(val_dataset)}")

    training_config = config.get("training", {})
    batch_size = training_config.get("batch_size", 16)
    shuffle = training_config.get("shuffle", True)  # Allow config to control shuffle
    feat_dim = config.get("model", {}).get("point_output_dim", 256)

    # Load BERT features
    bert_dir = ROOT / dataset_config.get("bert_dir", "data/text_features/full_official_nr3d")
    train_bert_path = bert_dir / "train_bert_embeddings.npy"
    val_bert_path = bert_dir / "val_bert_embeddings.npy"

    use_bert = True
    train_bert_features = None
    val_bert_features = None

    if train_bert_path.exists():
        train_bert_features = np.load(train_bert_path)
        log.info(f"Loaded train BERT features: {train_bert_features.shape}")
    else:
        log.warning("Train BERT features not found, using random features")
        use_bert = False

    if val_bert_path.exists():
        val_bert_features = np.load(val_bert_path)
        log.info(f"Loaded val BERT features: {val_bert_features.shape}")
    else:
        log.warning("Val BERT features not found, using random features")
        use_bert = False

    # Create loaders with conservative settings for stability
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,  # Use config setting
        collate_fn=lambda b: collate_fn(b, feat_dim, train_bert_features),
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get("evaluation", {}).get("batch_size", batch_size * 2),
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, feat_dim, val_bert_features),
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
    )

    # Optimizer
    lr = float(training_config.get("lr", 1e-4))
    weight_decay = float(training_config.get("weight_decay", 1e-4))
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Scheduler
    epochs = training_config.get("epochs", 30)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    # Checkpoint directory - timestamped
    timestamp = datetime.now().strftime("%Y%m%d")
    checkpoint_dir = ROOT / config.get("checkpoint", {}).get("dir", "outputs/sat_baseline") / f"{timestamp}_run"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save config snapshot
    config_snapshot_path = checkpoint_dir / "config.yaml"
    import yaml
    with open(config_snapshot_path, "w") as f:
        yaml.dump(config, f)
    log.info(f"Config snapshot saved to {config_snapshot_path}")

    # Training loop
    best_val_acc = 0.0
    history = []

    # Smoke run uses fewer batches and epochs
    debug_max_batches = config.get("debug", {}).get("max_batches")
    if smoke:
        epochs = 2
        debug_max_batches = 10
        log.info("SMOKE RUN: 2 epochs, 10 batches per epoch")

    for epoch in range(epochs):
        log.info(f"Epoch {epoch + 1}/{epochs}")

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, device,
            grad_clip=training_config.get("grad_clip", 1.0),
            max_batches=debug_max_batches,
            use_bert=use_bert,
        )

        # Evaluate
        val_metrics = evaluate(
            model, val_loader, device,
            max_batches=debug_max_batches,
            use_bert=use_bert,
        )

        # Update scheduler
        scheduler.step()

        # Clear GPU cache to prevent memory accumulation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Log
        log.info(
            f"Train loss: {train_metrics['loss']:.4f}, "
            f"Train acc: {train_metrics['acc']:.4f}, "
            f"Val acc@1: {val_metrics['acc_at_1']:.4f}, "
            f"Val acc@5: {val_metrics['acc_at_5']:.4f}, "
            f"Val acc@0.5: {val_metrics['acc_at_0_5']:.4f}"
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "val_acc_at_1": val_metrics["acc_at_1"],
            "val_acc_at_5": val_metrics["acc_at_5"],
            "val_acc_at_0_5": val_metrics["acc_at_0_5"],
            "lr": optimizer.param_groups[0]['lr'],
        })

        # Save best model
        if val_metrics["acc_at_1"] > best_val_acc:
            best_val_acc = val_metrics["acc_at_1"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": best_val_acc,
                "config": config,
            }, checkpoint_dir / "best_model.pt")
            log.info(f"Saved best model with val acc@1: {best_val_acc:.4f}")

    # Save final model and history
    torch.save(model.state_dict(), checkpoint_dir / "final_model.pt")

    history_path = checkpoint_dir / "training_history.json"
    with history_path.open("w") as f:
        json.dump(history, f, indent=2)

    log.info(f"Training complete. Best val acc@1: {best_val_acc:.4f}")
    log.info(f"Target comparison: ReferIt3DNet baseline = 30.79% Test Acc@1")

    return history, checkpoint_dir


def main():
    parser = argparse.ArgumentParser(description="Train SAT baseline")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/sat_baseline.yaml",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--smoke", action="store_true", help="Run smoke test (2 epochs, 10 batches)")
    args = parser.parse_args()

    train(args.config, args.device, smoke=args.smoke)


if __name__ == "__main__":
    main()