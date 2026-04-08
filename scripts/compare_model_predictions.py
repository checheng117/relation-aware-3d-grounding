#!/usr/bin/env python3
"""Compare all three models on the same samples to see if they're actually identical."""

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


def compare_all_three_models(max_samples=20):
    """Compare all three models on the same samples."""

    print(f"Comparing all three models on {max_samples} samples...")

    # Load validation dataset
    eval_split = "val"
    manifest_path = Path("data/processed") / f"{eval_split}_manifest.jsonl"

    if not manifest_path.exists():
        print(f"ERROR: Manifest not found at {manifest_path}")
        return

    dataset = ReferIt3DManifestDataset(manifest_path=manifest_path)
    print(f"Loaded {len(dataset)} samples from {eval_split} split")

    # Initialize all three models
    model_config = {
        "object_dim": 256,
        "language_dim": 256,
        "hidden_dim": 256,
        "relation_dim": 128,
        "dropout": 0.1,
        "anchor_temperature": 1.0
    }

    # Attribute-only model
    attr_model = AttributeOnlyModel(
        object_dim=model_config["object_dim"],
        language_dim=model_config["language_dim"],
        hidden_dim=model_config["hidden_dim"],
        dropout=model_config["dropout"]
    )

    # Raw-text relation model
    raw_text_model = RawTextRelationModel(
        object_dim=model_config["object_dim"],
        language_dim=model_config["language_dim"],
        hidden_dim=model_config["hidden_dim"],
        relation_dim=model_config["relation_dim"],
        dropout=model_config["dropout"]
    )

    # Structured relation model
    struct_model = StructuredRelationModel(
        object_dim=model_config["object_dim"],
        language_dim=model_config["language_dim"],
        hidden_dim=model_config["hidden_dim"],
        relation_dim=model_config["relation_dim"],
        anchor_temperature=model_config["anchor_temperature"],
        dropout=model_config["dropout"]
    )

    # Create small dataset subset
    from torch.utils.data import DataLoader, Subset
    subset_indices = list(range(min(max_samples, len(dataset))))
    subset_dataset = Subset(dataset, subset_indices)

    eval_loader = DataLoader(
        subset_dataset,
        batch_size=1,  # Use batch size 1 for easier comparison
        shuffle=False,
        collate_fn=collate_grounding_samples
    )

    device = torch.device("cpu")

    # Run each model on the same samples and collect results
    all_results = {
        'attribute_only': {'predictions': [], 'targets': [], 'accuracies': []},
        'raw_text_relation': {'predictions': [], 'targets': [], 'accuracies': []},
        'structured_relation': {'predictions': [], 'targets': [], 'accuracies': []}
    }

    for batch_idx, batch in enumerate(eval_loader):
        if batch_idx >= max_samples:
            break

        sample = batch.samples[0]  # Only one sample per batch
        tensors = batch.to_tensors(256, device=device)

        print(f"\nSample {batch_idx + 1}: Scene {sample.scene_id}")
        print(f"  Utterance: '{sample.utterance}'")
        print(f"  GT Target: {sample.target_object_id} (index {sample.target_index})")

        # Get candidate IDs
        candidate_ids = [obj.object_id for obj in sample.objects]
        print(f"  Candidates: {len(candidate_ids)} objects")

        # Run Attribute-only model
        attr_logits = forward_attribute(attr_model, tensors)
        attr_probs = torch.softmax(attr_logits[0], dim=-1)  # Get first (only) sample
        attr_mask = tensors["object_mask"][0]

        attr_masked_probs = attr_probs.clone()
        attr_masked_probs[~attr_mask] = float('-inf')

        attr_top1_idx = torch.argmax(attr_masked_probs)
        attr_top5_values, attr_top5_indices = torch.topk(attr_masked_probs, k=min(5, len(attr_masked_probs)), largest=True)

        attr_pred_top1 = candidate_ids[attr_top1_idx.item()] if attr_top1_idx < len(candidate_ids) else "OUT_OF_BOUNDS"
        attr_pred_top5 = []
        for idx in attr_top5_indices:
            if idx < len(candidate_ids):
                attr_pred_top5.append(candidate_ids[idx.item()])
            else:
                attr_pred_top5.append("OUT_OF_BOUNDS")

        attr_correct_top1 = attr_pred_top1 == sample.target_object_id
        attr_correct_top5 = sample.target_object_id in attr_pred_top5

        all_results['attribute_only']['predictions'].append(attr_pred_top1)
        all_results['attribute_only']['targets'].append(sample.target_object_id)
        all_results['attribute_only']['accuracies'].append((attr_correct_top1, attr_correct_top5))

        # Run Raw-text relation model
        raw_text_logits = forward_raw_text_relation(raw_text_model, tensors)
        raw_text_probs = torch.softmax(raw_text_logits[0], dim=-1)  # Get first sample
        raw_text_mask = tensors["object_mask"][0]

        raw_text_masked_probs = raw_text_probs.clone()
        raw_text_masked_probs[~raw_text_mask] = float('-inf')

        raw_text_top1_idx = torch.argmax(raw_text_masked_probs)
        raw_text_top5_values, raw_text_top5_indices = torch.topk(raw_text_masked_probs, k=min(5, len(raw_text_masked_probs)), largest=True)

        raw_text_pred_top1 = candidate_ids[raw_text_top1_idx.item()] if raw_text_top1_idx < len(candidate_ids) else "OUT_OF_BOUNDS"
        raw_text_pred_top5 = []
        for idx in raw_text_top5_indices:
            if idx < len(candidate_ids):
                raw_text_pred_top5.append(candidate_ids[idx.item()])
            else:
                raw_text_pred_top5.append("OUT_OF_BOUNDS")

        raw_text_correct_top1 = raw_text_pred_top1 == sample.target_object_id
        raw_text_correct_top5 = sample.target_object_id in raw_text_pred_top5

        all_results['raw_text_relation']['predictions'].append(raw_text_pred_top1)
        all_results['raw_text_relation']['targets'].append(sample.target_object_id)
        all_results['raw_text_relation']['accuracies'].append((raw_text_correct_top1, raw_text_correct_top5))

        # Run Structured relation model
        struct_results = struct_model(tensors, parsed_list=None)
        struct_logits = struct_results['logits'][0]  # Get first sample
        struct_probs = torch.softmax(struct_logits, dim=-1)
        struct_mask = tensors["object_mask"][0]

        struct_masked_probs = struct_probs.clone()
        struct_masked_probs[~struct_mask] = float('-inf')

        struct_top1_idx = torch.argmax(struct_masked_probs)
        struct_top5_values, struct_top5_indices = torch.topk(struct_masked_probs, k=min(5, len(struct_masked_probs)), largest=True)

        struct_pred_top1 = candidate_ids[struct_top1_idx.item()] if struct_top1_idx < len(candidate_ids) else "OUT_OF_BOUNDS"
        struct_pred_top5 = []
        for idx in struct_top5_indices:
            if idx < len(candidate_ids):
                struct_pred_top5.append(candidate_ids[idx.item()])
            else:
                struct_pred_top5.append("OUT_OF_BOUNDS")

        struct_correct_top1 = struct_pred_top1 == sample.target_object_id
        struct_correct_top5 = sample.target_object_id in struct_pred_top5

        all_results['structured_relation']['predictions'].append(struct_pred_top1)
        all_results['structured_relation']['targets'].append(sample.target_object_id)
        all_results['structured_relation']['accuracies'].append((struct_correct_top1, struct_correct_top5))

        # Print comparison for this sample
        print(f"  Attribute-only:     pred={attr_pred_top1:8} top5={attr_pred_top5[:3]} [{'✓' if attr_correct_top1 else '✗'}@1, {'✓' if attr_correct_top5 else '✗'}@5]")
        print(f"  Raw-text relation:  pred={raw_text_pred_top1:8} top5={raw_text_pred_top5[:3]} [{'✓' if raw_text_correct_top1 else '✗'}@1, {'✓' if raw_text_correct_top5 else '✗'}@5]")
        print(f"  Structured relation: pred={struct_pred_top1:8} top5={struct_pred_top5[:3]} [{'✓' if struct_correct_top1 else '✗'}@1, {'✓' if struct_correct_top5 else '✗'}@5]")

    # Calculate overall accuracies
    print(f"\n{'='*80}")
    print(f"SUMMARY COMPARISON OVER {max_samples} SAMPLES")
    print(f"{'='*80}")

    for model_name, results in all_results.items():
        acc_at_1 = sum(1 for acc_tuple in results['accuracies'] if acc_tuple[0]) / len(results['accuracies']) if results['accuracies'] else 0
        acc_at_5 = sum(1 for acc_tuple in results['accuracies'] if acc_tuple[1]) / len(results['accuracies']) if results['accuracies'] else 0

        print(f"{model_name:20} Acc@1: {acc_at_1:.4f} ({sum(1 for acc_tuple in results['accuracies'] if acc_tuple[0])}/{len(results['accuracies'])}) | Acc@5: {acc_at_5:.4f} ({sum(1 for acc_tuple in results['accuracies'] if acc_tuple[1])}/{len(results['accuracies'])})")

    # Check if predictions are identical
    attr_preds = all_results['attribute_only']['predictions']
    raw_text_preds = all_results['raw_text_relation']['predictions']
    struct_preds = all_results['structured_relation']['predictions']

    attr_equals_raw = attr_preds == raw_text_preds
    raw_equals_struct = raw_text_preds == struct_preds
    attr_equals_struct = attr_preds == struct_preds

    print(f"\nPrediction identity check:")
    print(f"Attribute == Raw-text: {attr_equals_raw}")
    print(f"Raw-text == Structured: {raw_equals_struct}")
    print(f"Attribute == Structured: {attr_equals_struct}")

    if all([attr_equals_raw, raw_equals_struct, attr_equals_struct]):
        print("\n⚠️  WARNING: All models are producing identical predictions!")
        print("This suggests an infrastructure issue where models aren't being differentiated properly.")
    else:
        print("\n✓ Models are producing different predictions, which is expected.")


def main():
    parser = argparse.ArgumentParser(description='Compare all three models on same samples')
    parser.add_argument('--max-samples', type=int, default=20, help='Max samples to compare')

    args = parser.parse_args()

    compare_all_three_models(args.max_samples)


if __name__ == "__main__":
    main()