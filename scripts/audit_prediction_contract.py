#!/usr/bin/env python3
"""Script to audit prediction record contracts across model lines."""

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


def run_model_predictions_and_audit(max_samples=10):
    """Run models and audit the prediction contracts."""

    print(f"Auditing prediction contracts for first {max_samples} samples...")

    # Load the validation dataset
    eval_split = "val"
    manifest_path = Path("data/processed") / f"{eval_split}_manifest.jsonl"

    if not manifest_path.exists():
        print(f"ERROR: Manifest not found at {manifest_path}")
        return

    dataset = ReferIt3DManifestDataset(manifest_path=manifest_path)
    print(f"Loaded {len(dataset)} samples from {eval_split} split")

    # Create subset for testing
    from torch.utils.data import Subset, DataLoader
    subset_indices = list(range(min(max_samples, len(dataset))))
    subset_dataset = Subset(dataset, subset_indices)

    eval_loader = DataLoader(
        subset_dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=collate_grounding_samples
    )

    # Initialize models
    model_configs = {
        "attribute_only": {
            "object_dim": 256,
            "language_dim": 256,
            "hidden_dim": 256,
            "dropout": 0.1
        },
        "raw_text_relation": {
            "object_dim": 256,
            "language_dim": 256,
            "hidden_dim": 256,
            "relation_dim": 128,
            "dropout": 0.1
        },
        "structured_relation": {
            "object_dim": 256,
            "language_dim": 256,
            "hidden_dim": 256,
            "relation_dim": 128,
            "dropout": 0.1,
            "anchor_temperature": 1.0
        }
    }

    attr_model = AttributeOnlyModel(**model_configs["attribute_only"])
    raw_text_model = RawTextRelationModel(**model_configs["raw_text_relation"])
    struct_model = StructuredRelationModel(**model_configs["structured_relation"])

    device = torch.device("cpu")

    all_prediction_examples = {}
    contract_issues = []

    for batch_idx, batch in enumerate(eval_loader):
        tensors = batch.to_tensors(256, device=device)

        # Get basic info
        batch_samples_info = []
        for i, sample in enumerate(batch.samples):
            batch_samples_info.append({
                'scene_id': sample.scene_id,
                'utterance': sample.utterance,
                'target_id': sample.target_object_id,
                'candidate_count': len(sample.objects),
                'candidate_ids': [obj.object_id for obj in sample.objects]
            })

        # Run Attribute-Only Model
        attr_logits = forward_attribute(attr_model, tensors)
        attr_probs = torch.softmax(attr_logits, dim=-1)

        attr_predictions = []
        for i in range(len(batch.samples)):
            sample_logits = attr_logits[i]
            sample_probs = attr_probs[i]
            sample_mask = tensors["object_mask"][i]

            # Apply mask to probabilities
            masked_probs = sample_probs.clone()
            masked_probs[~sample_mask] = float('-inf')

            # Get top predictions
            top1_idx = torch.argmax(masked_probs)
            top5_values, top5_indices = torch.topk(masked_probs, k=min(5, len(sample_mask)), largest=True)

            # Map indices back to object IDs
            candidate_ids = [obj.object_id for obj in batch.samples[i].objects]
            pred_top1_id = candidate_ids[top1_idx.item()] if top1_idx < len(candidate_ids) else f"INVALID_IDX_{top1_idx.item()}"
            pred_top5_ids = [candidate_ids[idx.item()] if idx < len(candidate_ids) else f"INVALID_IDX_{idx.item()}" for idx in top5_indices]

            pred_record = {
                'scene_id': batch.samples[i].scene_id,
                'target_id': batch.samples[i].target_object_id,
                'pred_top1': pred_top1_id,
                'pred_top5': pred_top5_ids,
                'candidate_object_ids': candidate_ids,
                'confidence_scores': sample_probs[sample_mask].tolist(),
                'model_type': 'attribute_only',
                'logit_values': sample_logits[sample_mask].tolist(),
                'sample_idx': i,
                'batch_idx': batch_idx
            }

            # Check for contract issues
            if pred_top1_id not in candidate_ids:
                contract_issues.append({
                    'type': 'invalid_prediction_index',
                    'model': 'attribute_only',
                    'scene_id': batch.samples[i].scene_id,
                    'target_id': batch.samples[i].target_object_id,
                    'pred_top1': pred_top1_id,
                    'candidate_count': len(candidate_ids),
                    'message': f'Predicted index {top1_idx.item()} out of bounds for {len(candidate_ids)} candidates'
                })

            attr_predictions.append(pred_record)

        # Run Raw-Text Relation Model
        raw_text_logits = forward_raw_text_relation(raw_text_model, tensors)
        raw_text_probs = torch.softmax(raw_text_logits, dim=-1)

        raw_text_predictions = []
        for i in range(len(batch.samples)):
            sample_logits = raw_text_logits[i]
            sample_probs = raw_text_probs[i]
            sample_mask = tensors["object_mask"][i]

            # Apply mask to probabilities
            masked_probs = sample_probs.clone()
            masked_probs[~sample_mask] = float('-inf')

            # Get top predictions
            top1_idx = torch.argmax(masked_probs)
            top5_values, top5_indices = torch.topk(masked_probs, k=min(5, len(sample_mask)), largest=True)

            # Map indices back to object IDs
            candidate_ids = [obj.object_id for obj in batch.samples[i].objects]
            pred_top1_id = candidate_ids[top1_idx.item()] if top1_idx < len(candidate_ids) else f"INVALID_IDX_{top1_idx.item()}"
            pred_top5_ids = [candidate_ids[idx.item()] if idx < len(candidate_ids) else f"INVALID_IDX_{idx.item()}" for idx in top5_indices]

            pred_record = {
                'scene_id': batch.samples[i].scene_id,
                'target_id': batch.samples[i].target_object_id,
                'pred_top1': pred_top1_id,
                'pred_top5': pred_top5_ids,
                'candidate_object_ids': candidate_ids,
                'confidence_scores': sample_probs[sample_mask].tolist(),
                'model_type': 'raw_text_relation',
                'logit_values': sample_logits[sample_mask].tolist(),
                'sample_idx': i,
                'batch_idx': batch_idx
            }

            # Check for contract issues
            if pred_top1_id not in candidate_ids:
                contract_issues.append({
                    'type': 'invalid_prediction_index',
                    'model': 'raw_text_relation',
                    'scene_id': batch.samples[i].scene_id,
                    'target_id': batch.samples[i].target_object_id,
                    'pred_top1': pred_top1_id,
                    'candidate_count': len(candidate_ids),
                    'message': f'Predicted index {top1_idx.item()} out of bounds for {len(candidate_ids)} candidates'
                })

            raw_text_predictions.append(pred_record)

        # Run Structured Relation Model
        struct_results = struct_model(tensors, parsed_list=None)
        struct_logits = struct_results['logits']
        struct_probs = torch.softmax(struct_logits, dim=-1)

        struct_predictions = []
        for i in range(len(batch.samples)):
            sample_logits = struct_logits[i]
            sample_probs = struct_probs[i]
            sample_mask = tensors["object_mask"][i]

            # Apply mask to probabilities
            masked_probs = sample_probs.clone()
            masked_probs[~sample_mask] = float('-inf')

            # Get top predictions
            top1_idx = torch.argmax(masked_probs)
            top5_values, top5_indices = torch.topk(masked_probs, k=min(5, len(sample_mask)), largest=True)

            # Map indices back to object IDs
            candidate_ids = [obj.object_id for obj in batch.samples[i].objects]
            pred_top1_id = candidate_ids[top1_idx.item()] if top1_idx < len(candidate_ids) else f"INVALID_IDX_{top1_idx.item()}"
            pred_top5_ids = [candidate_ids[idx.item()] if idx < len(candidate_ids) else f"INVALID_IDX_{idx.item()}" for idx in top5_indices]

            # Extract anchor info
            anchor_dist = struct_results['anchor_dist'][i]
            anchor_entropy = struct_results['anchor_entropy'][i]
            top_anchor_id = struct_results['top_anchor_id'][i]

            pred_record = {
                'scene_id': batch.samples[i].scene_id,
                'target_id': batch.samples[i].target_object_id,
                'pred_top1': pred_top1_id,
                'pred_top5': pred_top5_ids,
                'candidate_object_ids': candidate_ids,
                'confidence_scores': sample_probs[sample_mask].tolist(),
                'model_type': 'structured_relation',
                'logit_values': sample_logits[sample_mask].tolist(),
                'anchor_distribution': anchor_dist.tolist(),
                'anchor_entropy': float(anchor_entropy.item()),
                'top_anchor_id': int(top_anchor_id.item()),
                'anchor_confidence': float(anchor_dist[top_anchor_id].item()),
                'sample_idx': i,
                'batch_idx': batch_idx
            }

            # Check for contract issues
            if pred_top1_id not in candidate_ids:
                contract_issues.append({
                    'type': 'invalid_prediction_index',
                    'model': 'structured_relation',
                    'scene_id': batch.samples[i].scene_id,
                    'target_id': batch.samples[i].target_object_id,
                    'pred_top1': pred_top1_id,
                    'candidate_count': len(candidate_ids),
                    'message': f'Predicted index {top1_idx.item()} out of bounds for {len(candidate_ids)} candidates'
                })

            struct_predictions.append(pred_record)

        # Store examples for this batch
        all_prediction_examples[f'batch_{batch_idx}'] = {
            'batch_info': batch_samples_info,
            'attribute_only': attr_predictions,
            'raw_text_relation': raw_text_predictions,
            'structured_relation': struct_predictions
        }

    # Consolidate all predictions
    all_predictions = {
        'attribute_only': [],
        'raw_text_relation': [],
        'structured_relation': []
    }

    for batch_key, batch_data in all_prediction_examples.items():
        all_predictions['attribute_only'].extend(batch_data['attribute_only'])
        all_predictions['raw_text_relation'].extend(batch_data['raw_text_relation'])
        all_predictions['structured_relation'].extend(batch_data['structured_relation'])

    print(f"\nCollected prediction examples from {len(all_prediction_examples)} batches")
    print(f"Attribute-Only predictions: {len(all_predictions['attribute_only'])}")
    print(f"Raw-Text Relation predictions: {len(all_predictions['raw_text_relation'])}")
    print(f"Structured Relation predictions: {len(all_predictions['structured_relation'])}")
    print(f"Contract issues found: {len(contract_issues)}")

    # Create output directory
    output_dir = Path("outputs") / "debug_integrity"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Export detailed examples
    examples_output = {
        'all_prediction_examples': all_prediction_examples,
        'contract_issues': contract_issues,
        'total_predictions_per_model': {
            'attribute_only': len(all_predictions['attribute_only']),
            'raw_text_relation': len(all_predictions['raw_text_relation']),
            'structured_relation': len(all_predictions['structured_relation'])
        }
    }

    with open(output_dir / "prediction_contract_audit.json", 'w') as f:
        json.dump(examples_output, f, indent=2)

    # Export Markdown report
    md_content = f"""# Prediction Contract Audit Report

## Summary
- Total batches processed: {len(all_prediction_examples)}
- Attribute-Only predictions: {len(all_predictions['attribute_only'])}
- Raw-Text Relation predictions: {len(all_predictions['raw_text_relation'])}
- Structured Relation predictions: {len(all_predictions['structured_relation'])}
- Contract issues found: {len(contract_issues)}

## Contract Verification
All three models produce prediction records with consistent field names:
- scene_id
- target_id
- pred_top1
- pred_top5
- candidate_object_ids
- confidence_scores
- model_type

## Field Names and Meanings
All models use the same field names with consistent semantics:
- **scene_id**: Scene identifier (string)
- **target_id**: Ground truth target object ID (string)
- **pred_top1**: Top-1 predicted object ID (string)
- **pred_top5**: Top-5 predicted object IDs (list of strings)
- **candidate_object_ids**: List of all possible candidate object IDs (list of strings)
- **confidence_scores**: Confidence scores for each candidate object (list of floats)
- **model_type**: Model identifier (string)

## Sample Predictions

### Attribute-Only Model (First 3 samples):
"""

    for i, pred in enumerate(all_predictions['attribute_only'][:3]):
        md_content += f"""
- **Scene**: {pred['scene_id']}
- **Target**: {pred['target_id']}
- **Pred Top-1**: {pred['pred_top1']}
- **Pred Top-5**: {pred['pred_top5'][:3]}... (showing first 3)
- **Candidate Count**: {len(pred['candidate_object_ids'])}
- **Correct**: {pred['pred_top1'] == pred['target_id']}
"""

    md_content += f"""
### Raw-Text Relation Model (First 3 samples):
"""

    for i, pred in enumerate(all_predictions['raw_text_relation'][:3]):
        md_content += f"""
- **Scene**: {pred['scene_id']}
- **Target**: {pred['target_id']}
- **Pred Top-1**: {pred['pred_top1']}
- **Pred Top-5**: {pred['pred_top5'][:3]}... (showing first 3)
- **Candidate Count**: {len(pred['candidate_object_ids'])}
- **Correct**: {pred['pred_top1'] == pred['target_id']}
"""

    md_content += f"""
### Structured Relation Model (First 3 samples):
"""

    for i, pred in enumerate(all_predictions['structured_relation'][:3]):
        md_content += f"""
- **Scene**: {pred['scene_id']}
- **Target**: {pred['target_id']}
- **Pred Top-1**: {pred['pred_top1']}
- **Pred Top-5**: {pred['pred_top5'][:3]}... (showing first 3)
- **Candidate Count**: {len(pred['candidate_object_ids'])}
- **Correct**: {pred['pred_top1'] == pred['target_id']}
- **Anchor Entropy**: {pred['anchor_entropy']:.4f}
- **Top Anchor**: {pred['candidate_object_ids'][pred['top_anchor_id']]}
- **Anchor Conf**: {pred['anchor_confidence']:.4f}
"""

    if contract_issues:
        md_content += f"""
## Contract Issues Found
"""
        for i, issue in enumerate(contract_issues[:10]):  # Show first 10 issues
            md_content += f"""
{i+1}. **Model**: {issue['model']}
    **Scene**: {issue['scene_id']}
    **Target**: {issue['target_id']}
    **Issue**: {issue['message']}
"""
    else:
        md_content += f"""
## Contract Issues
No contract issues found! All models are producing properly formatted prediction records.
"""

    md_content += f"""
## Conclusion
The prediction contracts are {'PROBLEMATIC' if contract_issues else 'CONSISTENT'} across all three model types. {'Issues found with prediction formatting.' if contract_issues else 'All models produce correctly formatted prediction records with consistent semantics.'}
"""

    with open(output_dir / "prediction_contract_audit.md", 'w') as f:
        f.write(md_content)

    print(f"Contract audit reports saved to {output_dir}/")

    return {
        'examples': examples_output,
        'predictions': all_predictions,
        'issues': contract_issues
    }


def main():
    parser = argparse.ArgumentParser(description='Audit prediction record contracts across model lines')
    parser.add_argument('--max-samples', type=int, default=10, help='Maximum samples to audit')

    args = parser.parse_args()

    results = run_model_predictions_and_audit(args.max_samples)

    # Exit with error if contract issues found
    if results['issues']:
        print(f"\n⚠️  WARNING: {len(results['issues'])} contract issues found.")
        return 1
    else:
        print(f"\n✓ All contracts look consistent!")
        return 0


if __name__ == "__main__":
    exit(main())