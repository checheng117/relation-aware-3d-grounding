#!/usr/bin/env python3
"""Train ReferIt3DNet baseline model.

This script trains the official ReferIt3DNet baseline on NR3D data.

Usage:
    python repro/referit3d_baseline/scripts/train.py \\
        --config repro/referit3d_baseline/configs/official_baseline.yaml \\
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

# Add paths
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "repro" / "referit3d_baseline" / "src"))

from rag3d.datasets.referit3d import ReferIt3DManifestDataset
from rag3d.datasets.schemas import GroundingSample
from rag3d.datasets.collate import make_grounding_collate_fn
from rag3d.utils.logging import setup_logging
from rag3d.utils.seed import set_seed
from rag3d.utils.config import load_yaml_config

from referit3d_net import ReferIt3DNet

log = logging.getLogger(__name__)


class GradualWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Gradually warm-up learning rate over several epochs.

    After warmup, switches to the main scheduler.
    """

    def __init__(self, optimizer, multiplier: float, total_epoch: int,
                 after_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None):
        self.multiplier = multiplier
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        self.last_epoch = -1
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch >= self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]

        return [base_lr * ((self.multiplier - 1) * self.last_epoch / self.total_epoch + 1)
                for base_lr in self.base_lrs]

    def step(self, epoch=None, metrics=None):
        if self.last_epoch >= self.total_epoch:
            if self.after_scheduler:
                if metrics is not None:
                    self.after_scheduler.step(metrics)
                else:
                    self.after_scheduler.step(epoch)
        else:
            super().step(epoch)


def build_model(config: Dict[str, Any]) -> ReferIt3DNet:
    """Build ReferIt3DNet model from config."""
    model_config = config.get("model", {})
    return ReferIt3DNet(
        point_input_dim=model_config.get("point_input_dim", 256),
        point_hidden_dim=model_config.get("point_hidden_dim", 128),
        point_output_dim=model_config.get("point_output_dim", 256),
        lang_input_dim=model_config.get("lang_input_dim", 768),
        lang_hidden_dim=model_config.get("lang_hidden_dim", 256),
        lang_output_dim=model_config.get("lang_output_dim", 256),
        fusion_dim=model_config.get("fusion_dim", 512),
        dropout=model_config.get("dropout", 0.1),
        encoder_type=model_config.get("encoder_type", "simple_point"),
        pointnetpp_num_points=model_config.get("pointnetpp_num_points", 1024),
        # Learned class embedding parameters
        use_learned_class_embedding=model_config.get("use_learned_class_embedding", False),
        num_object_classes=model_config.get("num_classes", 516),
        class_embed_dim=model_config.get("class_embed_dim", 64),
    )

# Geometry directory for raw points
GEOMETRY_DIR = ROOT / "data/geometry"


def load_scene_points(scene_id: str, geometry_dir: Path = GEOMETRY_DIR) -> dict:
    """Load raw points from geometry file for a scene.

    Returns dict mapping object_id -> points tensor.
    """
    geom_path = geometry_dir / f"{scene_id}_geometry.npz"
    if not geom_path.exists():
        return {}

    data = np.load(geom_path, allow_pickle=True)
    object_ids = data["object_ids"]
    centers = data["centers"]
    sizes = data["sizes"]

    result = {}
    for i, oid in enumerate(object_ids):
        # Get points for this object
        points_key = f"points_{i}"
        if points_key in data:
            points = data[points_key]  # [P, 3] XYZ coordinates
            center = centers[i]  # [3]
            size = sizes[i]  # [3]
            # Normalize: points relative to center, scaled by size
            # Avoid division by zero
            safe_size = np.maximum(size, 0.1)
            normalized_points = (points - center) / safe_size
            result[int(oid)] = normalized_points

    return result


# No caching - load fresh each time to avoid memory issues
def load_scene_geometry(scene_id: str, geometry_dir: Path = GEOMETRY_DIR) -> Optional[dict]:
    """Load center/size geometry from file for a scene (no cache).

    Returns dict mapping object_id -> {center, size}, or None if not available.
    """
    geom_path = geometry_dir / f"{scene_id}_geometry.npz"
    if not geom_path.exists():
        return None

    try:
        data = np.load(geom_path, allow_pickle=True)
        object_ids = data["object_ids"]
        centers = data["centers"]
        sizes = data["sizes"]

        result = {}
        for i, oid in enumerate(object_ids):
            result[int(oid)] = {
                "center": centers[i].tolist(),  # Convert to list for stability
                "size": sizes[i].tolist(),
            }
        data.close()  # Explicitly close
        return result
    except Exception as e:
        return None


def sample_points(points: np.ndarray, num_points: int = 1024) -> np.ndarray:
    """Sample fixed number of points from point cloud.

    Args:
        points: [P, 3] raw points
        num_points: target number of points

    Returns:
        sampled: [num_points, 3] sampled points
    """
    P = points.shape[0]
    if P >= num_points:
        # Random sampling
        indices = np.random.choice(P, num_points, replace=False)
        return points[indices]
    else:
        # Pad by repeating
        indices = np.random.choice(P, num_points, replace=True)
        return points[indices]


def collate_fn(
    batch: List[tuple],  # List of (index, sample) tuples from IndexedDataset
    feat_dim: int = 256,
    text_features: Optional[np.ndarray] = None,
    class_to_idx: Optional[Dict[str, int]] = None,  # For learned class embedding
) -> Dict[str, Any]:
    """Collate samples into batch tensors (SimplePointEncoder mode).

    Uses real BERT features and geometry features (center/size + class hash).
    The class hash provides semantic signal for distinguishing object types.

    If class_to_idx is provided, also outputs class_indices for learned embedding.

    NOTE: Real geometry integration attempted but caused memory issues.
    Current: uses manifest center/size (which are fallback values).
    """
    B = len(batch)
    max_n = max(len(s.objects) for idx, s in batch)

    # Initialize tensors
    object_features = torch.zeros(B, max_n, feat_dim)
    object_mask = torch.zeros(B, max_n, dtype=torch.bool)
    target_index = torch.zeros(B, dtype=torch.long)
    texts = []
    sample_indices = []

    # For learned class embedding
    class_indices = None
    if class_to_idx is not None:
        class_indices = torch.zeros(B, max_n, dtype=torch.long)

    for i, (idx, sample) in enumerate(batch):
        texts.append(sample.utterance)
        sample_indices.append(idx)  # Direct index for BERT lookup

        for j, obj in enumerate(sample.objects):
            # Use geometry features: center + size + class hash
            # Channels 0-2: center (normalized)
            # Channels 3-5: size (normalized)
            # Channels 6+: class name hash (one-hot semantic signal)
            if obj.center:
                object_features[i, j, 0:3] = torch.tensor(obj.center).float() / 5.0  # Normalize
            if obj.size:
                object_features[i, j, 3:6] = torch.tensor(obj.size).float() / 2.0  # Normalize
            # Class name hash as semantic feature - use deterministic hash
            import hashlib
            class_hash = int(hashlib.md5(obj.class_name.encode()).hexdigest()[:8], 16)
            feat_hash = class_hash % (feat_dim - 6)
            object_features[i, j, 6 + feat_hash] = 1.0

            # Learned class embedding index
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
    }

    # Add class indices for learned embedding
    if class_indices is not None:
        result["class_indices"] = class_indices

    # Add real BERT features - always use index alignment since manifest/features aligned
    if text_features is not None:
        if len(sample_indices) == B:
            bert_features = torch.tensor(
                text_features[sample_indices],
                dtype=torch.float32,
            )
            result["bert_features"] = bert_features

    return result


def collate_fn_pointnetpp(
    batch: List[tuple],
    num_points: int = 1024,
    text_features: Optional[np.ndarray] = None,
    include_class_semantics: bool = True,
) -> Dict[str, Any]:
    """Collate samples into batch tensors with RAW POINTS (PointNet++ mode).

    Loads raw XYZ points from geometry files, normalizes to object bbox,
    and samples fixed number of points per object.

    Now includes class semantics (class hash) to match SimplePointEncoder.

    Returns:
        object_points: [B, N, P, 3] raw normalized point coordinates
        class_features: [B, N, 250] class hash features (channels 6-255 style)
    """
    B = len(batch)
    max_n = max(len(s.objects) for idx, s in batch)

    # Initialize tensors
    object_points = torch.zeros(B, max_n, num_points, 3)
    class_features = torch.zeros(B, max_n, 250)  # Match SimplePointEncoder class hash dim
    object_mask = torch.zeros(B, max_n, dtype=torch.bool)
    target_index = torch.zeros(B, dtype=torch.long)
    texts = []
    sample_indices = []
    scene_ids = []

    # Cache for loading points per scene
    points_cache: dict = {}

    for i, (idx, sample) in enumerate(batch):
        texts.append(sample.utterance)
        sample_indices.append(idx)
        scene_ids.append(sample.scene_id)

        # Load points for this scene (cached)
        scene_id = sample.scene_id
        if scene_id not in points_cache:
            points_cache[scene_id] = load_scene_points(scene_id)
        scene_points = points_cache[scene_id]

        for j, obj in enumerate(sample.objects):
            oid = int(obj.object_id)
            if oid in scene_points:
                raw_points = scene_points[oid]  # [P, 3] normalized points
                sampled = sample_points(raw_points, num_points)  # [num_points, 3]
                object_points[i, j] = torch.tensor(sampled, dtype=torch.float32)
            else:
                # Fallback: use zeros (will be masked)
                object_points[i, j] = torch.zeros(num_points, 3)

            # Add class semantics (class hash) - same as SimplePointEncoder
            if include_class_semantics:
                import hashlib
                class_hash = int(hashlib.md5(obj.class_name.encode()).hexdigest()[:8], 16)
                feat_hash = class_hash % 250  # Match 250 channels (was 256-6)
                class_features[i, j, feat_hash] = 1.0

            object_mask[i, j] = True

        target_index[i] = sample.target_index

    result = {
        "object_points": object_points,  # [B, N, P, 3] for PointNet++
        "class_features": class_features,  # [B, N, 250] class semantics
        "object_mask": object_mask,
        "target_index": target_index,
        "texts": texts,
        "samples_ref": [s for idx, s in batch],
        "scene_ids": scene_ids,
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
    object_mask = torch.zeros(B, max_n, dtype=torch.bool)
    target_index = torch.zeros(B, dtype=torch.long)
    texts = []
    sample_indices = []

    for i, (idx, sample) in enumerate(batch):
        texts.append(sample.utterance)
        sample_indices.append(idx)  # Direct index for BERT lookup

        for j, obj in enumerate(sample.objects):
            # Use geometry features: center + size + class hash
            # Channels 0-2: center (tanh-normalized)
            # Channels 3-5: size (tanh-normalized)
            # Channels 6+: class name hash (one-hot semantic signal)
            if obj.center:
                object_features[i, j, 0:3] = torch.tensor(obj.center) / 5.0  # Normalize
            if obj.size:
                object_features[i, j, 3:6] = torch.tensor(obj.size) / 2.0  # Normalize
            # Class name hash as semantic feature - use deterministic hash
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

    # Add real BERT features - always use index alignment since manifest/features aligned
    if text_features is not None:
        if len(sample_indices) == B:
            bert_features = torch.tensor(
                text_features[sample_indices],
                dtype=torch.float32,
            )
            result["bert_features"] = bert_features

    return result


class IndexedDataset(torch.utils.data.Dataset):
    """Wrapper dataset that returns (index, sample) tuples for proper BERT alignment.

    Returns tuple instead of setting attribute on Pydantic model.
    """

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        # Return tuple (index, sample) for collate
        return (idx, sample)


def build_class_vocabulary(manifest_paths: List[Path]) -> tuple:
    """Build class vocabulary from manifest files.

    Returns:
        class_to_idx: Dict mapping class name to index
        idx_to_class: List mapping index to class name
    """
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


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float = 1.0,
    max_batches: Optional[int] = None,
    use_bert: bool = True,
    encoder_type: str = "simple_point",
    gradient_accumulation_steps: int = 1,
) -> Dict[str, float]:
    """Train for one epoch."""
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

        # Move to device - handle both input formats
        object_mask = batch["object_mask"].to(device)
        target_index = batch["target_index"].to(device)

        # PointNet++ uses raw points, SimplePointEncoder uses features
        class_features = None
        if encoder_type == "pointnetpp" and "object_points" in batch:
            points_input = batch["object_points"].to(device)  # [B, N, P, 3]
            # Get class features if available
            if "class_features" in batch:
                class_features = batch["class_features"].to(device)  # [B, N, 250]
        else:
            points_input = batch["object_features"].to(device)  # [B, N, feat_dim]

        # Forward pass
        # Use BERT features if available, otherwise random
        if use_bert and "bert_features" in batch:
            text_features = batch["bert_features"].to(device)
        else:
            batch_size = points_input.shape[0]
            text_features = torch.randn(batch_size, 768, device=device)

        # Get class indices for learned embedding
        class_indices = None
        if "class_indices" in batch:
            class_indices = batch["class_indices"].to(device)

        outputs = model(
            points=points_input,
            object_mask=object_mask,
            text_features=text_features,
            class_features=class_features,
            class_indices=class_indices,
        )
        logits = outputs["logits"]

        # Compute loss
        loss = criterion(logits, target_index)

        # Scale loss for gradient accumulation
        loss = loss / gradient_accumulation_steps

        # Backward pass
        loss.backward()

        # Only update weights after gradient accumulation steps
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()
            optimizer.zero_grad()

        # Track metrics (use unscaled loss for logging)
        total_loss += loss.item() * gradient_accumulation_steps
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
    encoder_type: str = "simple_point",
) -> Dict[str, float]:
    """Evaluate model."""
    model.eval()
    correct_at_1 = 0
    correct_at_5 = 0
    total = 0

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        object_mask = batch["object_mask"].to(device)
        target_index = batch["target_index"].to(device)

        # PointNet++ uses raw points, SimplePointEncoder uses features
        class_features = None
        if encoder_type == "pointnetpp" and "object_points" in batch:
            points_input = batch["object_points"].to(device)  # [B, N, P, 3]
            # Get class features if available
            if "class_features" in batch:
                class_features = batch["class_features"].to(device)  # [B, N, 250]
        else:
            points_input = batch["object_features"].to(device)  # [B, N, feat_dim]

        # Generate language features (placeholder)
        if use_bert and "bert_features" in batch:
            text_features = batch["bert_features"].to(device)
        else:
            batch_size = points_input.shape[0]
            text_features = torch.randn(batch_size, 768, device=device)

        # Get class indices for learned embedding
        class_indices = None
        if "class_indices" in batch:
            class_indices = batch["class_indices"].to(device)

        outputs = model(
            points=points_input,
            object_mask=object_mask,
            text_features=text_features,
            class_features=class_features,
            class_indices=class_indices,
        )
        logits = outputs["logits"]

        # Acc@1
        pred = logits.argmax(dim=-1)
        correct_at_1 += (pred == target_index).sum().item()

        # Acc@5
        for i in range(len(target_index)):
            valid_mask = object_mask[i]
            num_valid = valid_mask.sum().item()
            k = min(5, num_valid)
            if k > 0:
                topk = logits[i][valid_mask].topk(k).indices
                # Map back to original indices
                valid_indices = torch.where(valid_mask)[0]
                topk_original = valid_indices[topk]
                if target_index[i] in topk_original:
                    correct_at_5 += 1

        total += target_index.numel()

    return {
        "acc_at_1": correct_at_1 / max(total, 1),
        "acc_at_5": correct_at_5 / max(total, 1),
    }


def train(
    config_path: Path,
    device_str: str = "cuda",
    resume_checkpoint: Optional[Path] = None,
):
    """Main training loop."""
    setup_logging()
    config = load_yaml_config(config_path, base_dir=ROOT)

    # Set seed
    set_seed(config.get("training", {}).get("seed", 42))

    # Device
    device = torch.device(device_str if torch.cuda.is_available() and device_str == "cuda" else "cpu")
    log.info(f"Using device: {device}")

    # Build model
    model = build_model(config)
    model = model.to(device)
    log.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Determine encoder type and collate mode
    model_config = config.get("model", {})
    encoder_type = model_config.get("encoder_type", "simple_point")
    log.info(f"Encoder type: {encoder_type}")

    # Load data
    dataset_config = config.get("dataset", {})
    manifest_dir = ROOT / dataset_config.get("manifest_dir", "data/processed")

    train_manifest = manifest_dir / dataset_config.get("train_manifest", "train_manifest.jsonl")
    val_manifest = manifest_dir / dataset_config.get("val_manifest", "val_manifest.jsonl")
    test_manifest = manifest_dir / dataset_config.get("test_manifest", "test_manifest.jsonl")  # For full vocabulary

    train_dataset = IndexedDataset(ReferIt3DManifestDataset(train_manifest))
    val_dataset = IndexedDataset(ReferIt3DManifestDataset(val_manifest))

    log.info(f"Train samples: {len(train_dataset)}")
    log.info(f"Val samples: {len(val_dataset)}")

    training_config = config.get("training", {})
    batch_size = training_config.get("batch_size", 16)
    feat_dim = model_config.get("point_output_dim", 256)
    num_points = model_config.get("pointnetpp_num_points", 1024)

    # Load BERT features if available
    use_bert = True
    bert_config = config.get("dataset", {}).get("bert_dir", None)
    if bert_config:
        bert_dir = ROOT / bert_config
    else:
        bert_dir = ROOT / "data/text_features"
    train_bert_path = bert_dir / "train_bert_embeddings.npy"
    val_bert_path = bert_dir / "val_bert_embeddings.npy"

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

    # Build class vocabulary for learned class embedding
    use_learned_class_embedding = model_config.get("use_learned_class_embedding", False)
    class_to_idx = None
    class_vocab = None
    if use_learned_class_embedding:
        log.info("Building class vocabulary for learned class embedding...")
        # Use all splits for full vocabulary coverage (516 classes)
        class_to_idx, class_vocab = build_class_vocabulary([train_manifest, val_manifest, test_manifest])
        log.info(f"Class vocabulary size: {len(class_vocab)}")
        configured_num_classes = int(model_config.get("num_classes", len(class_vocab)))
        if len(class_vocab) > configured_num_classes:
            raise ValueError(
                "Learned class vocabulary is larger than model embedding table: "
                f"vocab={len(class_vocab)} configured_num_classes={configured_num_classes}. "
                "Use a sorted-vocabulary config with num_classes >= vocab size."
            )

    # Select collate function based on encoder type
    if encoder_type == "pointnetpp":
        log.info(f"Using PointNet++ collate with {num_points} points per object")
        # Check if class semantics should be used
        use_class_semantics = model_config.get("use_class_semantics", True)
        log.info(f"Class semantics enabled: {use_class_semantics}")
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda b: collate_fn_pointnetpp(b, num_points, train_bert_features, use_class_semantics),
            num_workers=0,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.get("evaluation", {}).get("batch_size", batch_size * 2),
            shuffle=False,
            collate_fn=lambda b: collate_fn_pointnetpp(b, num_points, val_bert_features, use_class_semantics),
            num_workers=0,
        )
    else:
        log.info(f"Using SimplePointEncoder collate with {feat_dim} feature dim")
        if use_learned_class_embedding:
            log.info("Using learned class embedding in collate")
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

    # Optimizer with optional separate encoder learning rate
    use_separate_encoder_lr = training_config.get("use_separate_encoder_lr", False)
    encoder_lr_factor = training_config.get("encoder_lr_factor", 0.1)
    base_lr = training_config.get("lr", 1e-4)

    if use_separate_encoder_lr:
        # Separate parameter groups for encoder and rest
        encoder_params = []
        other_params = []
        for name, param in model.named_parameters():
            if 'point_encoder' in name:
                encoder_params.append(param)
            else:
                other_params.append(param)

        param_groups = [
            {'params': encoder_params, 'lr': base_lr * encoder_lr_factor},
            {'params': other_params, 'lr': base_lr}
        ]
        log.info(f"Using separate encoder LR: encoder={base_lr * encoder_lr_factor:.6f}, rest={base_lr:.6f}")
    else:
        param_groups = model.parameters()

    optimizer = AdamW(
        param_groups,
        lr=base_lr,
        weight_decay=training_config.get("weight_decay", 1e-4),
    )

    # Scheduler configuration
    epochs = training_config.get("epochs", 30)
    scheduler_type = training_config.get("scheduler_type", "cosine")

    if scheduler_type == "multistep":
        milestones = training_config.get("scheduler_milestones", [25, 40, 50, 60, 70, 80, 90])
        gamma = training_config.get("scheduler_gamma", 0.65)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
        log.info(f"Using MultiStepLR scheduler: milestones={milestones}, gamma={gamma}")
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
        log.info(f"Using CosineAnnealingLR scheduler: T_max={epochs}")

    # Warmup scheduler (wraps main scheduler)
    warmup_epochs = training_config.get("warmup_epochs", 0)
    if warmup_epochs > 0:
        warmup_multiplier = training_config.get("warmup_multiplier", 1.0)
        scheduler = GradualWarmupScheduler(
            optimizer,
            multiplier=warmup_multiplier,
            total_epoch=warmup_epochs,
            after_scheduler=scheduler
        )
        log.info(f"Using warmup: {warmup_epochs} epochs")

    # Training configuration
    gradient_accumulation_steps = training_config.get("gradient_accumulation_steps", 1)
    early_stopping_patience = training_config.get("early_stopping_patience", None)

    # Training loop
    checkpoint_dir = ROOT / config.get("checkpoint", {}).get("dir", "outputs/repro/referit3d_baseline")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_val_acc = -1.0
    history = []
    epochs_without_improvement = 0
    start_epoch = 0

    debug_max_batches = config.get("debug", {}).get("max_batches")

    if resume_checkpoint is not None:
        if not resume_checkpoint.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_checkpoint}")
        resume = torch.load(resume_checkpoint, map_location=device)
        if "model_state_dict" not in resume:
            raise ValueError(f"Resume checkpoint lacks model_state_dict: {resume_checkpoint}")
        model.load_state_dict(resume["model_state_dict"])
        if "optimizer_state_dict" in resume:
            optimizer.load_state_dict(resume["optimizer_state_dict"])
        start_epoch = int(resume.get("epoch", -1)) + 1
        best_val_acc = float(resume.get("val_acc", 0.0))
        history_path = checkpoint_dir / "training_history.json"
        if history_path.exists():
            try:
                history = json.loads(history_path.read_text())
            except json.JSONDecodeError:
                log.warning("Could not parse existing training history; starting a new history list")
        log.info(
            "Resumed from %s at next epoch %d with best val acc %.4f",
            resume_checkpoint,
            start_epoch + 1,
            best_val_acc,
        )

    if class_vocab is not None:
        vocab_path = checkpoint_dir / "class_vocabulary.json"
        with vocab_path.open("w") as f:
            json.dump(
                {
                    "class_vocabulary": class_vocab,
                    "class_to_idx": class_to_idx,
                    "source_manifests": [str(train_manifest), str(val_manifest), str(test_manifest)],
                    "ordering": "sorted(class_name)",
                },
                f,
                indent=2,
            )
        log.info(f"Saved class vocabulary to {vocab_path}")

    for epoch in range(start_epoch, epochs):
        log.info(f"Epoch {epoch + 1}/{epochs}")

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, device,
            grad_clip=training_config.get("grad_clip", 1.0),
            max_batches=debug_max_batches,
            use_bert=use_bert,
            encoder_type=encoder_type,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )

        # Evaluate
        val_metrics = evaluate(
            model, val_loader, device,
            max_batches=debug_max_batches,
            use_bert=use_bert,
            encoder_type=encoder_type,
        )

        # Update scheduler
        scheduler.step()

        # Log
        log.info(
            f"Train loss: {train_metrics['loss']:.4f}, "
            f"Train acc: {train_metrics['acc']:.4f}, "
            f"Val acc@1: {val_metrics['acc_at_1']:.4f}, "
            f"Val acc@5: {val_metrics['acc_at_5']:.4f}"
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["acc"],
            "val_acc_at_1": val_metrics["acc_at_1"],
            "val_acc_at_5": val_metrics["acc_at_5"],
            "lr": optimizer.param_groups[0]['lr'],
        })

        # Save best model
        if val_metrics["acc_at_1"] > best_val_acc:
            best_val_acc = val_metrics["acc_at_1"]
            epochs_without_improvement = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": best_val_acc,
                "config": config,
                "class_vocabulary": class_vocab,
                "class_to_idx": class_to_idx,
                "class_vocabulary_ordering": "sorted(class_name)" if class_vocab is not None else None,
            }, checkpoint_dir / "best_model.pt")
            log.info(f"Saved best model with val acc: {best_val_acc:.4f}")
        else:
            epochs_without_improvement += 1
            if early_stopping_patience and epochs_without_improvement >= early_stopping_patience:
                log.info(f"Early stopping triggered after {epochs_without_improvement} epochs without improvement")
                break

    # Save final model and history
    torch.save(model.state_dict(), checkpoint_dir / "final_model.pt")

    history_path = checkpoint_dir / "training_history.json"
    with history_path.open("w") as f:
        json.dump(history, f, indent=2)

    log.info(f"Training complete. Best val acc: {best_val_acc:.4f}")
    log.info(f"Target: 35.6% official baseline")

    return history


def main():
    parser = argparse.ArgumentParser(description="Train ReferIt3DNet baseline")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "repro/referit3d_baseline/configs/official_baseline.yaml",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    args = parser.parse_args()

    train(args.config, args.device, args.resume_checkpoint)


if __name__ == "__main__":
    main()
