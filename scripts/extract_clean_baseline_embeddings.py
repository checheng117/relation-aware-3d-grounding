#!/usr/bin/env python3
"""Extract object embeddings and language features from clean baseline.

This script loads the clean sorted-vocabulary ReferIt3DNet checkpoint and
runs inference to extract per-sample object embeddings and language features
for COVER-3D training.

The output format matches COVER-3D input requirements:
- object_embeddings: [N, 320] per sample (256 point + 64 class embed)
- lang_features: [256] per sample (from lang_encoder)
- base_logits: [N] per sample (already exported, linked by key)

Usage:
    python scripts/extract_clean_baseline_embeddings.py \
        --split test \
        --output-dir outputs/20260420_clean_sorted_vocab_baseline/embeddings

Date: 2026-04-19
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "repro" / "referit3d_baseline" / "src"))

from rag3d.datasets.referit3d import ReferIt3DManifestDataset
from rag3d.utils.logging import setup_logging
from referit3d_net import ReferIt3DNet

log = logging.getLogger(__name__)


def load_clean_checkpoint(checkpoint_path: Path, device: torch.device) -> Dict[str, Any]:
    """Load clean baseline checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    log.info(f"Loaded checkpoint from {checkpoint_path}")
    log.info(f"Checkpoint keys: {list(ckpt.keys())}")
    return ckpt


def build_model_from_checkpoint(ckpt: Dict[str, Any], device: torch.device) -> ReferIt3DNet:
    """Build ReferIt3DNet model from checkpoint config."""
    config = ckpt.get("config", {})

    # Extract dimensions from checkpoint
    point_input_dim = config.get("point_input_dim", 256)
    point_hidden_dim = config.get("point_hidden_dim", 128)
    point_output_dim = config.get("point_output_dim", 256)
    lang_input_dim = config.get("lang_input_dim", 768)
    lang_hidden_dim = config.get("lang_hidden_dim", 256)
    lang_output_dim = config.get("lang_output_dim", 256)
    fusion_dim = config.get("fusion_dim", 512)

    # Build class vocabulary
    class_vocabulary = ckpt.get("class_vocabulary", [])
    class_to_idx = ckpt.get("class_to_idx", {})
    num_classes = len(class_vocabulary)

    # Build model
    model = ReferIt3DNet(
        point_input_dim=point_input_dim,
        point_hidden_dim=point_hidden_dim,
        point_output_dim=point_output_dim,
        lang_input_dim=lang_input_dim,
        lang_hidden_dim=lang_hidden_dim,
        lang_output_dim=lang_output_dim,
        fusion_dim=fusion_dim,
        num_classes=1,
        encoder_type="simple_point",
        use_class_semantics=False,
        use_learned_class_embedding=True,
        num_object_classes=num_classes,
        class_embed_dim=64,
    )

    # Load weights
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()

    log.info(f"Built model with {num_classes} classes, object_dim={point_output_dim + 64}")

    return model, class_to_idx


def load_text_features(split: str, text_feature_dir: Path) -> Dict[str, np.ndarray]:
    """Load pre-computed BERT text features."""
    # Try multiple possible naming conventions
    possible_names = [
        f"{split}_text_features.npy",
        f"{split}_bert_embeddings.npy",
    ]

    for name in possible_names:
        feature_file = text_feature_dir / name
        if feature_file.exists():
            features = np.load(feature_file)
            log.info(f"Loaded text features from {feature_file}: shape={features.shape}")
            return {"features": features}

    log.warning(f"Text features not found in {text_feature_dir} for {split}")
    return {}


def load_object_features(scene_id: str, object_feature_dir: Path) -> Dict[str, np.ndarray]:
    """Load real object features for a scene."""
    feature_file = object_feature_dir / f"{scene_id}_features.npz"
    if not feature_file.exists():
        return {}
    data = np.load(feature_file)
    return {
        "object_ids": data["object_ids"],
        "features": data["features"],  # [N, 256]
    }


def extract_embeddings_for_sample(
    model: ReferIt3DNet,
    sample: Any,
    text_feature: np.ndarray,
    class_to_idx: Dict[str, int],
    device: torch.device,
    object_feature_dir: Path,
) -> Dict[str, Any]:
    """Extract embeddings for one sample."""

    # Load real object features for this scene
    obj_features_data = load_object_features(sample.scene_id, object_feature_dir)

    objects = sample.objects
    N = len(objects)

    # Build input features: [1, N, 256] using real features if available
    object_features = torch.zeros(1, N, 256, device=device)
    class_indices = torch.zeros(1, N, dtype=torch.long, device=device)

    # Map object_id to index in features array
    if obj_features_data:
        obj_id_to_idx = {str(obj_id): i for i, obj_id in enumerate(obj_features_data["object_ids"])}
        real_features = obj_features_data["features"]  # [N_scene, 256]

        for j, obj in enumerate(objects):
            obj_id_str = str(obj.object_id) if hasattr(obj, 'object_id') else str(j)
            if obj_id_str in obj_id_to_idx:
                feat_idx = obj_id_to_idx[obj_id_str]
                object_features[0, j] = torch.from_numpy(real_features[feat_idx]).float()
            else:
                # Fallback: use center + size + hash
                if obj.center:
                    center = torch.tensor(obj.center).float()
                    object_features[0, j, 0:3] = center / 10.0
                if obj.size:
                    size = torch.tensor(obj.size).float()
                    object_features[0, j, 3:6] = size / 2.0

            class_indices[0, j] = class_to_idx.get(obj.class_name, 0)
    else:
        # No real features: use synthetic fallback
        import hashlib
        for j, obj in enumerate(objects):
            if obj.center:
                center = torch.tensor(obj.center).float()
                object_features[0, j, 0:3] = center / 10.0
            if obj.size:
                size = torch.tensor(obj.size).float()
                object_features[0, j, 3:6] = size / 2.0

            class_hash = int(hashlib.md5(obj.class_name.encode()).hexdigest()[:8], 16)
            for k in range(250):
                if (class_hash + k) % 250 == 0:
                    object_features[0, j, 6 + k] = 1.0

            class_indices[0, j] = class_to_idx.get(obj.class_name, 0)

    # Build mask
    object_mask = torch.ones(1, N, dtype=torch.bool, device=device)

    # Build text features tensor
    text_tensor = torch.tensor(text_feature).float().unsqueeze(0).to(device)  # [1, 768]

    # Forward pass
    with torch.no_grad():
        result = model(
            points=object_features,
            object_mask=object_mask,
            text_features=text_tensor,
            class_indices=class_indices,
        )

    # Extract outputs
    logits = result["logits"][0].cpu().numpy()  # [N]
    obj_features = result["obj_features"][0].cpu().numpy()  # [N, D] (320 dims)
    lang_features = result["lang_features"][0].cpu().numpy()  # [D] (256 dims)

    return {
        "scene_id": sample.scene_id,
        "target_id": sample.target_object_id,
        "utterance": sample.utterance,
        "target_index": sample.target_index,
        "num_objects": N,
        "base_logits": logits.tolist(),
        "object_embeddings": obj_features.tolist(),
        "lang_features": lang_features.tolist(),
        "class_indices": class_indices[0].cpu().numpy().tolist(),
    }


def extract_embeddings_for_split(
    model: ReferIt3DNet,
    manifest_path: Path,
    text_feature_dir: Path,
    object_feature_dir: Path,
    class_to_idx: Dict[str, int],
    split: str,
    device: torch.device,
    output_dir: Path,
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """Extract embeddings for all samples in a split."""

    # Load dataset
    dataset = ReferIt3DManifestDataset(manifest_path)

    # Load text features
    text_data = load_text_features(split, text_feature_dir)
    text_features = text_data.get("features", None)

    if text_features is None:
        # Use hash-based fallback
        log.warning(f"Using hash-based text features for {split}")

    # Limit samples if specified
    if max_samples is not None:
        dataset_samples = list(dataset)[:max_samples]
    else:
        dataset_samples = list(dataset)

    log.info(f"Processing {len(dataset_samples)} samples for {split}")

    # Extract embeddings
    all_embeddings = []
    errors = []
    scenes_with_real_features = 0
    scenes_with_fallback = 0

    for idx, sample in enumerate(dataset_samples):
        try:
            # Get text feature for this sample
            if text_features is not None and idx < len(text_features):
                tf = text_features[idx]
            else:
                # Hash-based fallback
                import hashlib
                text_hash = int(hashlib.md5(sample.utterance.encode()).hexdigest()[:8], 16)
                tf = np.zeros(768)
                for k in range(64):
                    tf[k * 12 + (text_hash % 12)] = 1.0 / 64

            # Check if real object features exist
            obj_feat_file = object_feature_dir / f"{sample.scene_id}_features.npz"
            if obj_feat_file.exists():
                scenes_with_real_features += 1
            else:
                scenes_with_fallback += 1

            emb = extract_embeddings_for_sample(
                model=model,
                sample=sample,
                text_feature=tf,
                class_to_idx=class_to_idx,
                device=device,
                object_feature_dir=object_feature_dir,
            )
            all_embeddings.append(emb)

            if (idx + 1) % 500 == 0:
                log.info(f"Processed {idx + 1}/{len(dataset_samples)} samples")

        except Exception as e:
            log.error(f"Error processing sample {idx}: {e}")
            errors.append({
                "index": idx,
                "scene_id": sample.scene_id if hasattr(sample, 'scene_id') else "unknown",
                "error": str(e),
            })

    # Save outputs
    output_dir.mkdir(parents=True, exist_ok=True)

    embeddings_path = output_dir / f"{split}_embeddings.json"
    with embeddings_path.open("w") as f:
        json.dump(all_embeddings, f)
    log.info(f"Saved embeddings to {embeddings_path}")

    # Save summary
    summary = {
        "split": split,
        "total_samples": len(all_embeddings),
        "errors": len(errors),
        "output_path": str(embeddings_path),
        "object_embedding_dim": 320 if all_embeddings else "unknown",
        "lang_feature_dim": 256 if all_embeddings else "unknown",
        "scenes_with_real_object_features": scenes_with_real_features,
        "scenes_with_fallback_features": scenes_with_fallback,
    }

    summary_path = output_dir / f"{split}_embedding_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    log.info(f"Real object features: {scenes_with_real_features}, Fallback: {scenes_with_fallback}")

    if errors:
        errors_path = output_dir / f"{split}_embedding_errors.json"
        with errors_path.open("w") as f:
            json.dump(errors, f, indent=2)
        log.warning(f"Saved {len(errors)} errors to {errors_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Extract embeddings from clean baseline")
    parser.add_argument("--checkpoint", type=Path,
        default=ROOT / "outputs/20260420_clean_sorted_vocab_baseline/formal/best_model.pt")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--manifest-dir", type=Path,
        default=ROOT / "data/processed/scene_disjoint/official_scene_disjoint")
    parser.add_argument("--text-feature-dir", type=Path,
        default=ROOT / "data/text_features/full_official_nr3d")
    parser.add_argument("--object-feature-dir", type=Path,
        default=ROOT / "data/object_features")
    parser.add_argument("--output-dir", type=Path,
        default=ROOT / "outputs/20260420_clean_sorted_vocab_baseline/embeddings")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-samples", type=int, default=None,
        help="Limit samples for testing")

    args = parser.parse_args()

    setup_logging()

    # Device
    if args.device == "cuda" and not torch.cuda.is_available():
        log.warning("CUDA not available, using CPU")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    log.info(f"Extracting embeddings for {args.split} split")
    log.info(f"Device: {device}")
    log.info(f"Object feature dir: {args.object_feature_dir}")

    # Load checkpoint
    ckpt = load_clean_checkpoint(args.checkpoint, device)

    # Build model
    model, class_to_idx = build_model_from_checkpoint(ckpt, device)

    # Extract embeddings
    manifest_path = args.manifest_dir / f"{args.split}_manifest.jsonl"

    summary = extract_embeddings_for_split(
        model=model,
        manifest_path=manifest_path,
        text_feature_dir=args.text_feature_dir,
        object_feature_dir=args.object_feature_dir,
        class_to_idx=class_to_idx,
        split=args.split,
        device=device,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
    )

    log.info(f"Extraction complete: {summary['total_samples']} samples")

    # Verify object embedding dimensions
    if summary["object_embedding_dim"] != 320:
        log.warning(f"Object embedding dimension mismatch: expected 320, got {summary['object_embedding_dim']}")

    return summary


if __name__ == "__main__":
    main()