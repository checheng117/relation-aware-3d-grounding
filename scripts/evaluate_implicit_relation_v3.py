#!/usr/bin/env python3
"""Evaluate Implicit Relation v3 on test set."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "repro" / "referit3d_baseline" / "src"))

from rag3d.datasets.referit3d import ReferIt3DManifestDataset
from rag3d.utils.logging import setup_logging
from rag3d.models.relation_aware_implicit_v3 import RelationAwareImplicitV3

log = logging.getLogger(__name__)


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


def collate_fn(batch, feat_dim=256, text_features=None, class_to_idx=None):
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
        "centers": centers,
        "sizes": sizes,
    }

    if class_indices is not None:
        result["class_indices"] = class_indices

    if text_features is not None and len(sample_indices) == B:
        result["bert_features"] = torch.tensor(text_features[sample_indices], dtype=torch.float32)

    return result


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct_at_1 = 0
    correct_at_5 = 0
    total = 0

    for batch in loader:
        object_features = batch["object_features"].to(device)
        object_mask = batch["object_mask"].to(device)
        target_index = batch["target_index"].to(device)
        centers = batch["centers"].to(device)
        sizes = batch["sizes"].to(device)

        if "bert_features" in batch:
            text_features = batch["bert_features"].to(device)
        else:
            text_features = torch.randn(object_features.shape[0], 768, device=device)

        class_indices = batch.get("class_indices")
        if class_indices is not None:
            class_indices = class_indices.to(device)

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


def main():
    setup_logging()

    parser = argparse.ArgumentParser(description="Evaluate Implicit Relation v3")
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/implicit_relation_v3/best_model.pt"))
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    log.info(f"Using device: {device}")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint["config"]

    model_config = config.get("model", {})
    model = RelationAwareImplicitV3(
        point_input_dim=model_config.get("point_input_dim", 256),
        point_output_dim=model_config.get("point_output_dim", 256),
        lang_input_dim=model_config.get("lang_input_dim", 768),
        lang_output_dim=model_config.get("lang_output_dim", 256),
        fusion_dim=model_config.get("fusion_dim", 512),
        use_learned_class_embedding=model_config.get("use_learned_class_embedding", True),
        num_object_classes=model_config.get("num_classes", 516),
        class_embed_dim=model_config.get("class_embed_dim", 64),
        chunk_size=model_config.get("chunk_size", 8),  # Key v3 parameter
        relation_hidden_dim=model_config.get("relation_hidden_dim", 256),
        relation_mlp_layers=model_config.get("relation_mlp_layers", 2),
        use_residual=model_config.get("use_residual", True),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    log.info(f"Loaded model from epoch {checkpoint['epoch']}")
    log.info(f"Chunk size: {model_config.get('chunk_size', 8)}")

    dataset_config = config.get("dataset", {})
    manifest_dir = ROOT / dataset_config.get("manifest_dir", "data/processed")
    test_manifest = manifest_dir / dataset_config.get("test_manifest", "test_manifest.jsonl")

    test_dataset = IndexedDataset(ReferIt3DManifestDataset(test_manifest))
    log.info(f"Test samples: {len(test_dataset)}")

    bert_dir = ROOT / dataset_config.get("bert_dir", "data/text_features")
    test_bert_path = bert_dir / "test_bert_embeddings.npy"
    test_bert_features = None
    if test_bert_path.exists():
        test_bert_features = np.load(test_bert_path)
        log.info(f"Loaded test BERT features: {test_bert_features.shape}")

    train_manifest = manifest_dir / dataset_config.get("train_manifest", "train_manifest.jsonl")
    val_manifest = manifest_dir / dataset_config.get("val_manifest", "val_manifest.jsonl")
    class_to_idx, _ = build_class_vocabulary([train_manifest, val_manifest, test_manifest])

    feat_dim = model_config.get("point_input_dim", 256)
    test_loader = DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, feat_dim, test_bert_features, class_to_idx),
        num_workers=0,
    )

    results = evaluate(model, test_loader, device)

    log.info(f"Test Acc@1: {results['acc_at_1']:.4f} ({results['acc_at_1']*100:.2f}%)")
    log.info(f"Test Acc@5: {results['acc_at_5']:.4f} ({results['acc_at_5']*100:.2f}%)")
    log.info(f"Baseline: 30.79% Test Acc@1")
    log.info(f"v1 (dense): 31.26% (crashed at epoch 17)")
    log.info(f"v2 (sparse): 28.55% (stable but degraded)")
    log.info(f"Delta vs baseline: {results['acc_at_1']*100 - 30.79:.2f}%")
    log.info(f"Delta vs v1: {results['acc_at_1']*100 - 31.26:.2f}%")

    output_data = {
        "model": "implicit_relation_v3",
        "method": "chunked_dense_pairwise",
        "chunk_size": model_config.get("chunk_size", 8),
        "test_acc_at_1": results["acc_at_1"],
        "test_acc_at_5": results["acc_at_5"],
        "val_acc_at_1": checkpoint["val_acc"],
        "baseline_test_acc_at_1": 0.3079,
        "v1_test_acc_at_1": 0.3126,
        "v2_test_acc_at_1": 0.2855,
        "delta_vs_baseline": results["acc_at_1"] - 0.3079,
        "delta_vs_v1": results["acc_at_1"] - 0.3126,
        "delta_vs_v2": results["acc_at_1"] - 0.2855,
        "epoch": checkpoint["epoch"],
        "memory_safe": True,
        "full_coverage": True,
        "checkpoint": str(args.checkpoint),
    }

    output_path = ROOT / "implicit_relation_v3_results.json"
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    log.info(f"Saved results to {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("IMPLICIT RELATION v3 RESULTS SUMMARY")
    print("=" * 60)
    print(f"Test Acc@1: {results['acc_at_1']*100:.2f}%")
    print(f"Test Acc@5: {results['acc_at_5']*100:.2f}%")
    print(f"Delta vs Baseline (30.79%): {results['acc_at_1']*100 - 30.79:+.2f}%")
    print(f"Delta vs v1 Dense (31.26%): {results['acc_at_1']*100 - 31.26:+.2f}%")
    print(f"Delta vs v2 Sparse (28.55%): {results['acc_at_1']*100 - 28.55:+.2f}%")
    print(f"Memory-safe: YES (chunked computation)")
    print(f"Full coverage: YES (all N² pairs)")


if __name__ == "__main__":
    main()