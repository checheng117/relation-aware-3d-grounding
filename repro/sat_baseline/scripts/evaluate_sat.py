#!/usr/bin/env python3
"""Evaluate SAT baseline model.

Usage:
    python repro/sat_baseline/scripts/evaluate_sat.py \
        --checkpoint outputs/sat_baseline/20260414_run/best_model.pt \
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
sys.path.insert(0, str(ROOT / "repro" / "sat_baseline" / "src"))

from rag3d.datasets.referit3d import ReferIt3DManifestDataset
from rag3d.utils.logging import setup_logging
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
    """Collate samples into batch tensors with BERT features."""
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
            if obj.center:
                object_features[i, j, 0:3] = torch.tensor(obj.center).float() / 5.0
            if obj.size:
                object_features[i, j, 3:6] = torch.tensor(obj.size).float() / 2.0
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

    if text_features is not None:
        bert_features = torch.tensor(text_features[sample_indices], dtype=torch.float32)
        result["bert_features"] = bert_features

    return result


def build_model_from_checkpoint(checkpoint_path: Path, device: torch.device) -> SATModel:
    """Load SAT model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "config" in checkpoint:
        config = checkpoint["config"]
    else:
        # Fallback config
        config = {"model": {"point_output_dim": 256, "lang_input_dim": 768}}

    model = build_sat_model(config)
    model = model.to(device)

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model


@torch.no_grad()
def evaluate(
    model: SATModel,
    loader: DataLoader,
    device: torch.device,
    use_bert: bool = True,
) -> Dict[str, Any]:
    """Evaluate model and return detailed metrics."""
    model.eval()

    all_predictions = []
    correct_at_1 = 0
    correct_at_5 = 0
    correct_at_0_5 = 0
    total = 0
    bert_used_count = 0

    for batch in loader:
        object_features = batch["object_features"].to(device)
        object_mask = batch["object_mask"].to(device)
        target_index = batch["target_index"].to(device)

        # Use real BERT features if available
        if use_bert and "bert_features" in batch:
            text_features = batch["bert_features"].to(device)
            bert_used_count += len(batch["bert_features"])
        else:
            batch_size = object_features.shape[0]
            text_features = torch.randn(batch_size, 768, device=device)

        outputs = model(
            points=object_features,
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
            k5 = min(5, num_valid)
            topk5_indices = masked_logits.topk(k5).indices.tolist()
            is_correct_at_5 = target in topk5_indices

            # Top-50% (Acc@0.5)
            k0_5 = max(1, int(num_valid * 0.5))
            topk0_5_indices = masked_logits.topk(k0_5).indices.tolist()
            is_correct_at_0_5 = target in topk0_5_indices

            all_predictions.append({
                "scene_id": sample.scene_id,
                "utterance": sample.utterance,
                "target_id": sample.target_object_id,
                "target_index": target,
                "pred_top1": pred_top1,
                "pred_top5": topk5_indices,
                "pred_top0_5": topk0_5_indices,
                "correct_at_1": is_correct_at_1,
                "correct_at_5": is_correct_at_5,
                "correct_at_0_5": is_correct_at_0_5,
                "num_objects": num_valid,
            })

            if is_correct_at_1:
                correct_at_1 += 1
            if is_correct_at_5:
                correct_at_5 += 1
            if is_correct_at_0_5:
                correct_at_0_5 += 1
            total += 1

    return {
        "acc_at_1": correct_at_1 / max(total, 1),
        "acc_at_5": correct_at_5 / max(total, 1),
        "acc_at_0_5": correct_at_0_5 / max(total, 1),
        "total_samples": total,
        "bert_used_count": bert_used_count,
        "predictions": all_predictions,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate SAT baseline")
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
    log.info(f"Loading SAT model from {args.checkpoint}")
    model = build_model_from_checkpoint(args.checkpoint, device)
    log.info(f"SAT Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Load data
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if "config" in checkpoint:
        config = checkpoint["config"]
        dataset_config = config.get("dataset", {})
        manifest_dir_str = dataset_config.get("manifest_dir", "data/processed/scene_disjoint/expanded_nr3d")
        bert_dir_str = dataset_config.get("bert_dir", "data/text_features/full_official_nr3d")
    else:
        manifest_dir_str = "data/processed/scene_disjoint/expanded_nr3d"
        bert_dir_str = "data/text_features/full_official_nr3d"

    manifest_dir = args.manifest_dir if args.manifest_dir else ROOT / manifest_dir_str
    bert_dir = args.bert_dir if args.bert_dir else ROOT / bert_dir_str

    manifest_path = manifest_dir / f"{args.split}_manifest.jsonl"
    base_dataset = ReferIt3DManifestDataset(manifest_path)
    dataset = IndexedDataset(base_dataset)
    log.info(f"Loaded {len(dataset)} samples from {args.split} split")

    # Load BERT features
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

    # Create loader
    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, 256, text_features),
        num_workers=0,
    )

    # Evaluate
    log.info("Evaluating...")
    results = evaluate(model, loader, device, use_bert=use_bert)

    log.info(f"Results:")
    log.info(f"  Acc@1: {results['acc_at_1']:.4f} ({results['acc_at_1']*100:.2f}%)")
    log.info(f"  Acc@5: {results['acc_at_5']:.4f} ({results['acc_at_5']*100:.2f}%)")
    log.info(f"  Acc@0.5: {results['acc_at_0_5']:.4f} ({results['acc_at_0_5']*100:.2f}%)")
    log.info(f"  Total samples: {results['total_samples']}")
    log.info(f"  BERT features used: {results['bert_used_count']}/{results['total_samples']}")

    # Comparison targets
    log.info(f"  ReferIt3DNet baseline:")
    log.info(f"    Test Acc@1: 30.79%")
    log.info(f"    Test Acc@5: 91.75%")
    log.info(f"  SAT paper (Nr3D):")
    log.info(f"    Acc@0.5: 49.2% (reported)")

    # Gap analysis
    baseline_acc1 = 0.3079
    gap_acc1 = results['acc_at_1'] - baseline_acc1
    sat_paper_acc0_5 = 0.492
    gap_acc0_5 = results['acc_at_0_5'] - sat_paper_acc0_5

    log.info(f"  Gap vs baseline (Acc@1): {gap_acc1*100:+.2f}%")
    log.info(f"  Gap vs SAT paper (Acc@0.5): {gap_acc0_5*100:+.2f}%")

    # Save results
    output_dir = args.output_dir or args.checkpoint.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / f"eval_{args.split}_results.json"
    with results_path.open("w") as f:
        json.dump({
            "split": args.split,
            "checkpoint": str(args.checkpoint),
            "timestamp": datetime.now().isoformat(),
            "model_type": "SAT",
            "acc_at_1": results["acc_at_1"],
            "acc_at_5": results["acc_at_5"],
            "acc_at_0_5": results["acc_at_0_5"],
            "total_samples": results["total_samples"],
            "bert_used_count": results["bert_used_count"],
            "bert_coverage": results["bert_used_count"] / results["total_samples"],
            "baseline_comparison": {
                "referit3dnet_acc_at_1": 0.3079,
                "referit3dnet_acc_at_5": 0.9175,
                "gap_acc_at_1": gap_acc1,
            },
            "sat_paper_comparison": {
                "sat_paper_acc_at_0_5": 0.492,
                "gap_acc_at_0_5": gap_acc0_5,
            },
        }, f, indent=2)

    predictions_path = output_dir / f"eval_{args.split}_predictions.json"
    with predictions_path.open("w") as f:
        json.dump(results["predictions"], f, indent=2)

    log.info(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()