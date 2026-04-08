#!/usr/bin/env python3
"""Evaluate ReferIt3DNet baseline model.

Usage:
    python repro/referit3d_baseline/scripts/evaluate.py \\
        --checkpoint outputs/repro/referit3d_baseline/best_model.pt \\
        --split test
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
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "repro" / "referit3d_baseline" / "src"))

from rag3d.datasets.referit3d import ReferIt3DManifestDataset
from rag3d.datasets.schemas import GroundingSample
from rag3d.utils.logging import setup_logging
from rag3d.utils.config import load_yaml_config

from referit3d_net import ReferIt3DNet

log = logging.getLogger(__name__)

# Geometry directory for raw points
GEOMETRY_DIR = ROOT / "data/geometry"


def load_scene_points(scene_id: str, geometry_dir: Path = GEOMETRY_DIR) -> dict:
    """Load raw points from geometry file for a scene."""
    geom_path = geometry_dir / f"{scene_id}_geometry.npz"
    if not geom_path.exists():
        return {}

    data = np.load(geom_path, allow_pickle=True)
    object_ids = data["object_ids"]
    centers = data["centers"]
    sizes = data["sizes"]

    result = {}
    for i, oid in enumerate(object_ids):
        points_key = f"points_{i}"
        if points_key in data:
            points = data[points_key]
            center = centers[i]
            size = sizes[i]
            safe_size = np.maximum(size, 0.1)
            normalized_points = (points - center) / safe_size
            result[int(oid)] = normalized_points

    return result


def sample_points(points: np.ndarray, num_points: int = 1024) -> np.ndarray:
    """Sample fixed number of points."""
    P = points.shape[0]
    if P >= num_points:
        indices = np.random.choice(P, num_points, replace=False)
        return points[indices]
    else:
        indices = np.random.choice(P, num_points, replace=True)
        return points[indices]


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
    """Collate samples into batch tensors with BERT features and geometry features."""
    B = len(batch)
    max_n = max(len(s.objects) for idx, s in batch)

    object_features = torch.zeros(B, max_n, feat_dim)
    object_mask = torch.zeros(B, max_n, dtype=torch.bool)
    target_index = torch.zeros(B, dtype=torch.long)
    texts = []
    scene_ids = []
    target_ids = []
    sample_indices = []

    for i, (idx, sample) in enumerate(batch):
        texts.append(sample.utterance)
        scene_ids.append(sample.scene_id)
        target_ids.append(sample.target_object_id)
        sample_indices.append(idx)

        for j, obj in enumerate(sample.objects):
            # Use geometry features: center + size + class hash
            if obj.center:
                object_features[i, j, 0:3] = torch.tensor(obj.center) / 5.0
            if obj.size:
                object_features[i, j, 3:6] = torch.tensor(obj.size) / 2.0
            # Use deterministic hash for class name features
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
        "scene_ids": scene_ids,
        "target_ids": target_ids,
    }

    # Add real BERT features
    if text_features is not None:
        bert_features = torch.tensor(text_features[sample_indices], dtype=torch.float32)
        result["bert_features"] = bert_features

    return result


def collate_fn_pointnetpp(
    batch: List[tuple],
    num_points: int = 1024,
    text_features: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Collate samples into batch tensors with RAW POINTS (PointNet++ mode)."""
    B = len(batch)
    max_n = max(len(s.objects) for idx, s in batch)

    object_points = torch.zeros(B, max_n, num_points, 3)
    object_mask = torch.zeros(B, max_n, dtype=torch.bool)
    target_index = torch.zeros(B, dtype=torch.long)
    texts = []
    scene_ids = []
    target_ids = []
    sample_indices = []

    points_cache: dict = {}

    for i, (idx, sample) in enumerate(batch):
        texts.append(sample.utterance)
        scene_ids.append(sample.scene_id)
        target_ids.append(sample.target_object_id)
        sample_indices.append(idx)

        scene_id = sample.scene_id
        if scene_id not in points_cache:
            points_cache[scene_id] = load_scene_points(scene_id)
        scene_points = points_cache[scene_id]

        for j, obj in enumerate(sample.objects):
            oid = int(obj.object_id)
            if oid in scene_points:
                raw_points = scene_points[oid]
                sampled = sample_points(raw_points, num_points)
                object_points[i, j] = torch.tensor(sampled, dtype=torch.float32)
            else:
                object_points[i, j] = torch.zeros(num_points, 3)
            object_mask[i, j] = True

        target_index[i] = sample.target_index

    result = {
        "object_points": object_points,
        "object_mask": object_mask,
        "target_index": target_index,
        "texts": texts,
        "samples_ref": [s for idx, s in batch],
        "scene_ids": scene_ids,
        "target_ids": target_ids,
    }

    if text_features is not None:
        bert_features = torch.tensor(text_features[sample_indices], dtype=torch.float32)
        result["bert_features"] = bert_features

    return result


def build_model_from_checkpoint(checkpoint_path: Path, device: torch.device) -> tuple:
    """Load model from checkpoint. Returns (model, encoder_type)."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "config" in checkpoint:
        config = checkpoint["config"]
        model_config = config.get("model", {})
    else:
        model_config = {}

    model = ReferIt3DNet(
        point_input_dim=model_config.get("point_input_dim", 256),
        point_hidden_dim=model_config.get("point_hidden_dim", 128),
        point_output_dim=model_config.get("point_output_dim", 256),
        lang_output_dim=model_config.get("lang_output_dim", 256),
        fusion_dim=model_config.get("fusion_dim", 512),
        dropout=model_config.get("dropout", 0.1),
        encoder_type=model_config.get("encoder_type", "simple_point"),
        pointnetpp_num_points=model_config.get("pointnetpp_num_points", 1024),
    )

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()
    encoder_type = model_config.get("encoder_type", "simple_point")
    return model, encoder_type


@torch.no_grad()
def evaluate(
    model: ReferIt3DNet,
    loader: DataLoader,
    device: torch.device,
    use_bert: bool = True,
    encoder_type: str = "simple_point",
) -> Dict[str, Any]:
    """Evaluate model and return detailed metrics."""
    model.eval()

    all_predictions = []
    correct_at_1 = 0
    correct_at_5 = 0
    total = 0
    bert_used_count = 0

    for batch in loader:
        object_mask = batch["object_mask"].to(device)
        target_index = batch["target_index"].to(device)

        # PointNet++ uses raw points, SimplePointEncoder uses features
        if encoder_type == "pointnetpp" and "object_points" in batch:
            points_input = batch["object_points"].to(device)  # [B, N, P, 3]
        else:
            points_input = batch["object_features"].to(device)  # [B, N, feat_dim]

        # Use real BERT features if available
        if use_bert and "bert_features" in batch:
            text_features = batch["bert_features"].to(device)
            bert_used_count += len(batch["bert_features"])
        else:
            batch_size = points_input.shape[0]
            text_features = torch.randn(batch_size, 768, device=device)

        outputs = model(
            points=points_input,
            object_mask=object_mask,
            text_features=text_features,
        )
        logits = outputs["logits"]

        for i in range(len(batch["samples_ref"])):
            sample = batch["samples_ref"][i]
            sample_logits = logits[i]
            sample_mask = object_mask[i]
            target = target_index[i].item()

            # Mask invalid objects
            masked_logits = sample_logits.clone()
            masked_logits[~sample_mask] = float("-inf")

            # Top-1
            pred_top1 = masked_logits.argmax().item()
            is_correct_at_1 = (pred_top1 == target)

            # Top-5
            num_valid = sample_mask.sum().item()
            k = min(5, num_valid)
            topk_indices = masked_logits.topk(k).indices.tolist()
            is_correct_at_5 = target in topk_indices

            all_predictions.append({
                "scene_id": sample.scene_id,
                "utterance": sample.utterance,
                "target_id": sample.target_object_id,
                "target_index": target,
                "pred_top1": pred_top1,
                "pred_top5": topk_indices,
                "correct_at_1": is_correct_at_1,
                "correct_at_5": is_correct_at_5,
            })

            if is_correct_at_1:
                correct_at_1 += 1
            if is_correct_at_5:
                correct_at_5 += 1
            total += 1

    return {
        "acc_at_1": correct_at_1 / max(total, 1),
        "acc_at_5": correct_at_5 / max(total, 1),
        "total_samples": total,
        "bert_used_count": bert_used_count,
        "predictions": all_predictions,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate ReferIt3DNet baseline")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    parser.add_argument("--manifest-dir", type=Path, default=None)
    parser.add_argument("--bert-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    setup_logging()

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    log.info(f"Using device: {device}")

    # Load model
    log.info(f"Loading model from {args.checkpoint}")
    model, encoder_type = build_model_from_checkpoint(args.checkpoint, device)
    log.info(f"Encoder type: {encoder_type}")

    # Load data - use configurable manifest directory
    if args.manifest_dir:
        manifest_dir = args.manifest_dir
    else:
        # Try to get from checkpoint config
        checkpoint = torch.load(args.checkpoint, map_location=device)
        if "config" in checkpoint:
            config = checkpoint["config"]
            dataset_config = config.get("dataset", {})
            manifest_dir_str = dataset_config.get("manifest_dir", "data/processed")
            manifest_dir = ROOT / manifest_dir_str
        else:
            manifest_dir = ROOT / "data/processed"
    manifest_path = manifest_dir / f"{args.split}_manifest.jsonl"

    base_dataset = ReferIt3DManifestDataset(manifest_path)
    dataset = IndexedDataset(base_dataset)
    log.info(f"Loaded {len(dataset)} samples from {args.split} split")

    # Load BERT features - use configurable bert directory
    if args.bert_dir:
        bert_dir = args.bert_dir
    else:
        # Try to get from checkpoint config
        checkpoint = torch.load(args.checkpoint, map_location=device)
        if "config" in checkpoint:
            config = checkpoint["config"]
            dataset_config = config.get("dataset", {})
            bert_dir_str = dataset_config.get("bert_dir", None)
            if bert_dir_str:
                bert_dir = ROOT / bert_dir_str
            else:
                bert_dir = ROOT / "data/text_features"
        else:
            bert_dir = ROOT / "data/text_features"
    bert_path = bert_dir / f"{args.split}_bert_embeddings.npy"
    use_bert = True
    text_features = None

    if bert_path.exists():
        text_features = np.load(bert_path)
        log.info(f"Loaded BERT features: {text_features.shape}")
        if text_features.shape[0] != len(dataset):
            log.warning(f"BERT feature count mismatch: {text_features.shape[0]} vs {len(dataset)}")
            use_bert = False
    else:
        log.warning(f"BERT features not found: {bert_path}")
        use_bert = False

    # Select collate based on encoder type
    if encoder_type == "pointnetpp":
        log.info("Using PointNet++ collate with raw points")
        loader = DataLoader(
            dataset,
            batch_size=32,
            shuffle=False,
            collate_fn=lambda b: collate_fn_pointnetpp(b, 1024, text_features),
            num_workers=0,
        )
    else:
        log.info("Using SimplePointEncoder collate with hand-crafted features")
        loader = DataLoader(
            dataset,
            batch_size=32,
            shuffle=False,
            collate_fn=lambda b: collate_fn(b, 256, text_features),
            num_workers=0,
        )

    # Evaluate
    log.info("Evaluating...")
    results = evaluate(model, loader, device, use_bert=use_bert, encoder_type=encoder_type)

    log.info(f"Results:")
    log.info(f"  Acc@1: {results['acc_at_1']:.4f} ({results['acc_at_1']*100:.2f}%)")
    log.info(f"  Acc@5: {results['acc_at_5']:.4f} ({results['acc_at_5']*100:.2f}%)")
    log.info(f"  Total samples: {results['total_samples']}")
    log.info(f"  BERT features used: {results['bert_used_count']}/{results['total_samples']}")
    log.info(f"  Encoder type: {encoder_type}")
    log.info(f"  Target: 35.6% official baseline")

    # Gap analysis
    gap = results['acc_at_1'] - 0.356
    if abs(gap) <= 0.02:
        log.info(f"  Gap: {gap*100:+.2f}% (EXACT REPRODUCTION)")
    elif abs(gap) <= 0.05:
        log.info(f"  Gap: {gap*100:+.2f}% (ACCEPTABLE REPRODUCTION)")
    else:
        log.info(f"  Gap: {gap*100:+.2f}% (PARTIAL REPRODUCTION)")

    # Save results
    output_dir = args.output_dir or args.checkpoint.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / f"eval_{args.split}_results.json"
    with results_path.open("w") as f:
        json.dump({
            "split": args.split,
            "checkpoint": str(args.checkpoint),
            "timestamp": datetime.now().isoformat(),
            "encoder_type": encoder_type,
            "acc_at_1": results["acc_at_1"],
            "acc_at_5": results["acc_at_5"],
            "total_samples": results["total_samples"],
            "bert_used_count": results["bert_used_count"],
            "bert_coverage": results["bert_used_count"] / results["total_samples"],
            "target": 0.356,
            "gap": gap,
        }, f, indent=2)

    predictions_path = output_dir / f"eval_{args.split}_predictions.json"
    with predictions_path.open("w") as f:
        json.dump(results["predictions"], f, indent=2)

    log.info(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()