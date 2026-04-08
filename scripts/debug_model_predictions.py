#!/usr/bin/env python3
"""Debug script to check the actual prediction contracts and model outputs."""

import argparse
import sys
from pathlib import Path
import json
import torch
import numpy as np
from typing import Dict, Any, List

# Add src to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag3d.datasets.referit3d import ReferIt3DManifestDataset
from rag3d.datasets.collate import collate_grounding_samples
from rag3d.relation_reasoner.model import AttributeOnlyModel, RawTextRelationModel
from rag3d.relation_reasoner.structured_relation_model import StructuredRelationModel
from rag3d.training.runner import forward_attribute, forward_raw_text_relation


def debug_model_predictions(max_samples=10):
    """Debug the actual predictions and contracts of the models."""

    print(f"Debugging model predictions for {max_samples} samples...")

    # Load validation dataset
    eval_split = "val"
    manifest_path = Path("data/processed") / f"{eval_split}_manifest.jsonl"

    if not manifest_path.exists():
        print(f"ERROR: Manifest not found at {manifest_path}")
        return

    dataset = ReferIt3DManifestDataset(manifest_path=manifest_path)
    print(f"Loaded {len(dataset)} samples from {eval_split} split")

    # Initialize models
    model_config = {
        "object_dim": 256,
        "language_dim": 256,
        "hidden_dim": 256,
        "relation_dim": 128,
        "dropout": 0.1,
        "anchor_temperature": 1.0
    }

    attr_model = AttributeOnlyModel(
        object_dim=model_config["object_dim"],
        language_dim=model_config["language_dim"],
        hidden_dim=model_config["hidden_dim"],
        dropout=model_config["dropout"]
    )

    raw_text_model = RawTextRelationModel(
        object_dim=model_config["object_dim"],
        language_dim=model_config["language_dim"],
        hidden_dim=model_config["hidden_dim"],
        relation_dim=model_config["relation_dim"],
        dropout=model_config["dropout"]
    )

    struct_model = StructuredRelationModel(
        object_dim=model_config["object_dim"],
        language_dim=model_config["language_dim"],
        hidden_dim=model_config["hidden_dim"],
        relation_dim=model_config["relation_dim"],
        anchor_temperature=model_config["anchor_temperature"],
        dropout=model_config["dropout"]
    )

    # Create subset for debugging
    from torch.utils.data import DataLoader, Subset
    subset_indices = list(range(min(max_samples, len(dataset))))
    subset_dataset = Subset(dataset, subset_indices)

    eval_loader = DataLoader(
        subset_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_grounding_samples
    )

    device = torch.device("cpu")

    print("\\n*** Debugging Model Prediction Contracts ***\\n")

    for batch_idx, batch in enumerate(eval_loader):
        if batch_idx >= 5:  # Only debug first 5 samples
            break

        sample = batch.samples[0]
        tensors = batch.to_tensors(256, device=device)

        print(f"Sample {batch_idx}: Scene {sample.scene_id}")
        print(f"  Utterance: '{sample.utterance[:50]}...'")
        print(f"  GT Target: {sample.target_object_id} (index {sample.target_index})")
        print(f"  Candidates: {len(sample.objects)} objects")

        # Get candidate IDs for mapping
        candidate_ids = [obj.object_id for obj in sample.objects]
        print(f"  Candidate IDs: {candidate_ids[:10]}...")  # Show first 10

        # Test Attribute-only model
        print(f"  Attribute model:")
        attr_logits = forward_attribute(attr_model, tensors)
        attr_probs = torch.softmax(attr_logits[0], dim=-1)
        attr_mask = tensors["object_mask"][0]

        # Check mask alignment
        print(f"    Logits shape: {attr_logits.shape}")
        print(f"    Probs shape: {attr_probs.shape}")
        print(f"    Mask shape: {attr_mask.shape}")
        print(f"    Num valid objects (mask sum): {attr_mask.sum().item()}")
        print(f"    Actual objects in sample: {len(sample.objects)}")

        # Apply mask to probabilities
        attr_masked_probs = attr_probs[0].clone() if len(attr_probs.shape) > 1 else attr_probs.clone()
        attr_masked_probs[~attr_mask] = float('-inf')

        # Get predictions
        attr_top1_idx = torch.argmax(attr_masked_probs)
        _, attr_top5_indices = torch.topk(attr_masked_probs, k=min(5, len(attr_masked_probs)), largest=True)

        # Map indices to object IDs
        attr_pred_top1 = "OUT_OF_BOUNDS"
        if attr_top1_idx < len(candidate_ids):
            attr_pred_top1 = candidate_ids[attr_top1_idx.item()]
        else:
            print(f"    WARNING: Top-1 index {attr_top1_idx} >= candidate count {len(candidate_ids)}")

        attr_pred_top5 = []
        for idx in attr_top5_indices:
            if idx < len(candidate_ids):
                attr_pred_top5.append(candidate_ids[idx.item()])
            else:
                attr_pred_top5.append("OUT_OF_BOUNDS")

        print(f"    Pred Top-1 index: {attr_top1_idx} -> ID: {attr_pred_top1} (GT: {sample.target_object_id}, Correct: {attr_pred_top1 == sample.target_object_id})")
        print(f"    Pred Top-5 indices: {attr_top5_indices.tolist()} -> IDs: {attr_pred_top5}")

        # Test Raw-text relation model
        print(f"  Raw-text model:")
        raw_text_logits = forward_raw_text_relation(raw_text_model, tensors)
        raw_text_probs = torch.softmax(raw_text_logits[0], dim=-1)
        raw_text_mask = tensors["object_mask"][0]

        print(f"    Logits shape: {raw_text_logits.shape}")
        print(f"    Probs shape: {raw_text_probs.shape}")
        print(f"    Mask shape: {raw_text_mask.shape}")
        print(f"    Num valid objects: {raw_text_mask.sum().item()}")

        raw_text_masked_probs = raw_text_probs[0].clone() if len(raw_text_probs.shape) > 1 else raw_text_probs.clone()
        raw_text_masked_probs[~raw_text_mask] = float('-inf')

        raw_text_top1_idx = torch.argmax(raw_text_masked_probs)
        _, raw_text_top5_indices = torch.topk(raw_text_masked_probs, k=min(5, len(raw_text_masked_probs)), largest=True)

        raw_text_pred_top1 = "OUT_OF_BOUNDS"
        if raw_text_top1_idx < len(candidate_ids):
            raw_text_pred_top1 = candidate_ids[raw_text_top1_idx.item()]
        else:
            print(f"    WARNING: Raw-text Top-1 index {raw_text_top1_idx} >= candidate count {len(candidate_ids)}")

        raw_text_pred_top5 = []
        for idx in raw_text_top5_indices:
            if idx < len(candidate_ids):
                raw_text_pred_top5.append(candidate_ids[idx.item()])
            else:
                raw_text_pred_top5.append("OUT_OF_BOUNDS")

        print(f"    Pred Top-1 index: {raw_text_top1_idx} -> ID: {raw_text_pred_top1} (GT: {sample.target_object_id}, Correct: {raw_text_pred_top1 == sample.target_object_id})")
        print(f"    Pred Top-5 indices: {raw_text_top5_indices.tolist()} -> IDs: {raw_text_pred_top5}")

        # Test Structured relation model
        print(f"  Structured model:")
        struct_results = struct_model(tensors, parsed_list=None)
        struct_logits = struct_results['logits'][0] if isinstance(struct_results, dict) and 'logits' in struct_results else struct_results
        struct_probs = torch.softmax(struct_logits, dim=-1)
        struct_mask = tensors["object_mask"][0]

        print(f"    Logits shape: {struct_logits.shape}")
        print(f"    Probs shape: {struct_probs.shape}")
        print(f"    Mask shape: {struct_mask.shape}")
        print(f"    Num valid objects: {struct_mask.sum().item()}")

        struct_masked_probs = struct_probs.clone()
        struct_masked_probs[~struct_mask] = float('-inf')

        struct_top1_idx = torch.argmax(struct_masked_probs)
        _, struct_top5_indices = torch.topk(struct_masked_probs, k=min(5, len(struct_masked_probs)), largest=True)

        struct_pred_top1 = "OUT_OF_BOUNDS"
        if struct_top1_idx < len(candidate_ids):
            struct_pred_top1 = candidate_ids[struct_top1_idx.item()]
        else:
            print(f"    WARNING: Struct Top-1 index {struct_top1_idx} >= candidate count {len(candidate_ids)}")

        struct_pred_top5 = []
        for idx in struct_top5_indices:
            if idx < len(candidate_ids):
                struct_pred_top5.append(candidate_ids[idx.item()])
            else:
                struct_pred_top5.append("OUT_OF_BOUNDS")

        print(f"    Pred Top-1 index: {struct_top1_idx} -> ID: {struct_pred_top1} (GT: {sample.target_object_id}, Correct: {struct_pred_top1 == sample.target_object_id})")
        print(f"    Pred Top-5 indices: {struct_top5_indices.tolist()} -> IDs: {struct_pred_top5}")
        if 'anchor_dist' in struct_results:
            print(f"    Anchor entropy: {struct_results['anchor_entropy'][0].item():.4f}")
            print(f"    Top anchor ID: {struct_results['top_anchor_id'][0].item()}")

        print()  # Empty line between samples

    # Now test the metric calculation directly
    print("*** Testing Metric Calculation Directly ***\\n")

    # Create prediction records as they would be in actual evaluation
    sample_predictions = []
    sample_targets = []

    for batch_idx, batch in enumerate(eval_loader):
        if batch_idx >= max_samples:
            break

        sample = batch.samples[0]
        tensors = batch.to_tensors(256, device=device)
        candidate_ids = [obj.object_id for obj in sample.objects]

        # Get attribute-only prediction for this sample
        attr_logits = forward_attribute(attr_model, tensors)
        attr_probs = torch.softmax(attr_logits[0], dim=-1)
        attr_mask = tensors["object_mask"][0]

        attr_masked_probs = attr_probs.clone()
        attr_masked_probs[~attr_mask] = float('-inf')

        attr_top1_idx = torch.argmax(attr_masked_probs)
        _, attr_top5_indices = torch.topk(attr_masked_probs, k=min(5, len(attr_masked_probs)), largest=True)

        attr_pred_top1 = candidate_ids[attr_top1_idx.item()] if attr_top1_idx < len(candidate_ids) else "OUT_OF_BOUNDS"
        attr_pred_top5 = [candidate_ids[idx.item()] if idx < len(candidate_ids) else "OUT_OF_BOUNDS" for idx in attr_top5_indices]

        pred_record = {
            'scene_id': sample.scene_id,
            'target_id': sample.target_object_id,
            'pred_top1': attr_pred_top1,
            'pred_top5': attr_pred_top5,
            'candidate_object_ids': candidate_ids,
            'confidence_scores': attr_probs[attr_mask].tolist(),
            'model_type': 'attribute_only'
        }

        target_record = {
            'scene_id': sample.scene_id,
            'target_id': sample.target_object_id,
            'candidate_object_ids': candidate_ids,
            'utterance': sample.utterance
        }

        sample_predictions.append(pred_record)
        sample_targets.append(target_record)

    # Compute metrics
    from rag3d.evaluation.metrics import compute_overall_metrics
    metrics = compute_overall_metrics(sample_predictions, sample_targets)
    print(f"Direct metric calculation:")
    print(f"  Acc@1: {metrics['acc_at_1']}")
    print(f"  Acc@5: {metrics['acc_at_5']}")
    print(f"  Total samples: {metrics['total_samples']}")


def main():
    parser = argparse.ArgumentParser(description='Debug model predictions and contracts')
    parser.add_argument('--max-samples', type=int, default=10, help='Max samples to debug')

    args = parser.parse_args()

    debug_model_predictions(args.max_samples)


if __name__ == "__main__":
    main()