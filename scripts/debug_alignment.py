#!/usr/bin/env python3
"""Debug prediction-target alignment issue."""

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


def debug_prediction_target_alignment(max_samples=10):
    """Debug the alignment between predictions and targets."""

    print(f"Debugging prediction-target alignment for {max_samples} samples...")

    # Load validation dataset
    eval_split = "val"
    manifest_path = Path("data/processed") / f"{eval_split}_manifest.jsonl"

    if not manifest_path.exists():
        print(f"ERROR: Manifest not found at {manifest_path}")
        return

    dataset = ReferIt3DManifestDataset(manifest_path=manifest_path)
    print(f"Loaded {len(dataset)} samples from {eval_split} split")

    # Initialize a model to run a small test
    model_config = {
        "object_dim": 256,
        "language_dim": 256,
        "hidden_dim": 256,
        "dropout": 0.1
    }

    attr_model = AttributeOnlyModel(**model_config)

    # Create small dataloader
    from torch.utils.data import DataLoader, Subset
    subset_indices = list(range(min(max_samples, len(dataset))))
    subset_dataset = Subset(dataset, subset_indices)

    eval_loader = DataLoader(
        subset_dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=collate_grounding_samples
    )

    device = torch.device("cpu")

    # Collect real predictions and targets to inspect
    predictions = []
    targets = []

    for batch_idx, batch in enumerate(eval_loader):
        if len(predictions) >= max_samples:
            break

        tensors = batch.to_tensors(256, device=device)

        # Get model predictions
        attr_logits = forward_attribute(attr_model, tensors)
        attr_probs = torch.softmax(attr_logits, dim=-1)

        # Convert to prediction records (following the same pattern as actual evaluation)
        for i in range(len(batch.samples)):
            if len(predictions) >= max_samples:
                break

            sample = batch.samples[i]

            # Apply mask to probabilities
            sample_logits = attr_logits[i]
            sample_probs = attr_probs[i]
            sample_mask = tensors["object_mask"][i]

            masked_probs = sample_probs.clone()
            masked_probs[~sample_mask] = float('-inf')

            # Get top predictions
            top1_idx = torch.argmax(masked_probs)
            top5_values, top5_indices = torch.topk(masked_probs, k=min(5, len(sample_mask)), largest=True)

            # Get candidate object IDs from the sample
            candidate_ids = [obj.object_id for obj in sample.objects]

            # Map prediction indices to object IDs
            pred_top1_id = candidate_ids[top1_idx.item()] if top1_idx < len(candidate_ids) else "OUT_OF_BOUNDS"
            pred_top5_ids = []
            for idx in top5_indices:
                if idx < len(candidate_ids):
                    pred_top5_ids.append(candidate_ids[idx.item()])
                else:
                    pred_top5_ids.append("OUT_OF_BOUNDS")

            # Create prediction record in the same format as evaluation
            pred_record = {
                'scene_id': sample.scene_id,
                'target_id': sample.target_object_id,  # This is what the model predicts should match
                'pred_top1': pred_top1_id,
                'pred_top5': pred_top5_ids,
                'candidate_object_ids': candidate_ids,
                'confidence_scores': sample_probs[sample_mask].tolist(),
                'model_type': 'attribute_only'
            }

            target_record = {
                'scene_id': sample.scene_id,
                'target_id': sample.target_object_id,  # This is the ground truth
                'candidate_object_ids': candidate_ids,
                'utterance': sample.utterance
            }

            predictions.append(pred_record)
            targets.append(target_record)

            # Debug info for this specific sample
            print(f"\nSample {len(predictions)}:")
            print(f"  Scene: {sample.scene_id}")
            print(f"  Utterance: {sample.utterance}")
            print(f"  GT Target: {sample.target_object_id} (at index {sample.target_index})")
            print(f"  Pred Top-1: {pred_top1_id}")
            print(f"  Pred Top-5: {pred_top5_ids}")
            print(f"  Correct Top-1: {pred_top1_id == sample.target_object_id}")
            print(f"  GT in Top-5: {sample.target_object_id in pred_top5_ids}")
            print(f"  Candidate count: {len(candidate_ids)}")

    # Now test the actual metrics calculation
    print(f"\nTesting metric calculation...")

    # Calculate metrics manually to confirm
    acc_at_1 = 0
    acc_at_5 = 0
    for pred, target in zip(predictions, targets):
        print(f"\nTesting: pred['target_id'] = {pred['target_id']}, target['target_id'] = {target['target_id']}")
        print(f"Pred top-1: {pred['pred_top1']}")

        # Acc@1
        if 'pred_top1' in pred and pred['pred_top1'] == target['target_id']:
            acc_at_1 += 1
            print(f"  -> Acc@1 CORRECT")
        else:
            print(f"  -> Acc@1 INCORRECT")

        # Acc@5
        if 'pred_top5' in pred and target['target_id'] in pred['pred_top5']:
            acc_at_5 += 1
            print(f"  -> Acc@5 CORRECT")
        else:
            print(f"  -> Acc@5 INCORRECT")

    total_samples = len(predictions)
    print(f"\nCalculated Metrics:")
    print(f"  Acc@1: {acc_at_1}/{total_samples} = {acc_at_1/total_samples if total_samples > 0 else 0:.4f}")
    print(f"  Acc@5: {acc_at_5}/{total_samples} = {acc_at_5/total_samples if total_samples > 0 else 0:.4f}")

    # Check if the issue might be elsewhere - let's look at the evaluation code again
    print(f"\nInvestigating the evaluation code...")

    # Import the actual metric function to check it
    from rag3d.evaluation.metrics import compute_overall_metrics

    eval_metrics = compute_overall_metrics(predictions, targets)
    print(f"From actual evaluation function:")
    print(f"  Acc@1: {eval_metrics['acc_at_1']}")
    print(f"  Acc@5: {eval_metrics['acc_at_5']}")

    print(f"\nThe two calculations match: {abs(eval_metrics['acc_at_1'] - acc_at_1/total_samples if total_samples > 0 else 0) < 1e-6}")


def main():
    parser = argparse.ArgumentParser(description='Debug prediction-target alignment')
    parser.add_argument('--max-samples', type=int, default=10, help='Max samples to debug')

    args = parser.parse_args()

    debug_prediction_target_alignment(args.max_samples)


if __name__ == "__main__":
    main()