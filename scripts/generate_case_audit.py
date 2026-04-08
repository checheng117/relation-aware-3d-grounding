#!/usr/bin/env python3
"""Script to generate human-readable case audit for debugging."""

import argparse
import sys
from pathlib import Path
import json
from typing import Dict, Any, List

# Add src to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag3d.datasets.referit3d import ReferIt3DManifestDataset
from rag3d.datasets.collate import collate_grounding_samples
from rag3d.relation_reasoner.model import AttributeOnlyModel, RawTextRelationModel
from rag3d.relation_reasoner.structured_relation_model import StructuredRelationModel
from rag3d.training.runner import forward_attribute, forward_raw_text_relation


def run_case_audit(max_samples=20):
    """Run case audit to manually inspect predictions."""

    print(f"Generating human-readable case audit for {max_samples} samples...")

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
        batch_size=1,
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

    cases = []

    for batch_idx, batch in enumerate(eval_loader):
        if len(cases) >= max_samples:
            break

        tensors = batch.to_tensors(256, device=device)
        sample = batch.samples[0]  # Only one sample per batch for this audit

        # Get utterance and target info
        utterance = sample.utterance
        gt_target = sample.target_object_id
        scene_id = sample.scene_id

        # Get candidates
        candidate_objects = [(obj.object_id, obj.class_name, obj.center) for obj in sample.objects]

        # Run each model
        # Attribute-only
        attr_logits = forward_attribute(attr_model, tensors)
        attr_probs = attr_logits.softmax(dim=-1)[0]  # First (and only) sample
        attr_mask = tensors["object_mask"][0]
        attr_masked_probs = attr_probs.masked_fill(~attr_mask, float('-inf'))
        attr_topk = torch.topk(attr_masked_probs, k=min(5, len(attr_masked_probs)), dim=-1)
        attr_pred_ids = [candidate_objects[i][0] for i in attr_topk.indices if i < len(candidate_objects)]

        # Raw-text relation
        raw_text_logits = forward_raw_text_relation(raw_text_model, tensors)
        raw_text_probs = raw_text_logits.softmax(dim=-1)[0]
        raw_text_masked_probs = raw_text_probs.masked_fill(~attr_mask, float('-inf'))  # Use same mask
        raw_text_topk = torch.topk(raw_text_masked_probs, k=min(5, len(raw_text_masked_probs)), dim=-1)
        raw_text_pred_ids = [candidate_objects[i][0] for i in raw_text_topk.indices if i < len(candidate_objects)]

        # Structured relation
        struct_results = struct_model(tensors, parsed_list=None)
        struct_logits = struct_results['logits']
        struct_probs = struct_logits.softmax(dim=-1)[0]
        struct_masked_probs = struct_probs.masked_fill(~attr_mask, float('-inf'))
        struct_topk = torch.topk(struct_masked_probs, k=min(5, len(struct_masked_probs)), dim=-1)
        struct_pred_ids = [candidate_objects[i][0] for i in struct_topk.indices if i < len(candidate_objects)]

        # Check if GT is in top-k
        attr_gt_in_top5 = gt_target in attr_pred_ids[:5]
        raw_text_gt_in_top5 = gt_target in raw_text_pred_ids[:5]
        struct_gt_in_top5 = gt_target in struct_pred_ids[:5]

        case = {
            'case_id': len(cases),
            'scene_id': scene_id,
            'utterance': utterance,
            'gt_target': gt_target,
            'gt_target_class': next((obj.class_name for obj in sample.objects if obj.object_id == gt_target), "Unknown"),
            'candidates': [
                {
                    'object_id': obj.object_id,
                    'class_name': obj.class_name,
                    'center': obj.center
                }
                for obj in sample.objects
            ],
            'attribute_only': {
                'top1_pred': attr_pred_ids[0] if len(attr_pred_ids) > 0 else 'None',
                'top5_preds': attr_pred_ids[:5],
                'gt_in_top5': attr_gt_in_top5,
                'correct_top1': attr_pred_ids[0] == gt_target if len(attr_pred_ids) > 0 else False
            },
            'raw_text_relation': {
                'top1_pred': raw_text_pred_ids[0] if len(raw_text_pred_ids) > 0 else 'None',
                'top5_preds': raw_text_pred_ids[:5],
                'gt_in_top5': raw_text_gt_in_top5,
                'correct_top1': raw_text_pred_ids[0] == gt_target if len(raw_text_pred_ids) > 0 else False
            },
            'structured_relation': {
                'top1_pred': struct_pred_ids[0] if len(struct_pred_ids) > 0 else 'None',
                'top5_preds': struct_pred_ids[:5],
                'gt_in_top5': struct_gt_in_top5,
                'correct_top1': struct_pred_ids[0] == gt_target if len(struct_pred_ids) > 0 else False,
                'anchor_info': {
                    'top_anchor_id': int(struct_results['top_anchor_id'][0].item()),
                    'anchor_entropy': float(struct_results['anchor_entropy'][0].item()),
                    'anchor_confidence': float(struct_results['anchor_dist'][0][struct_results['top_anchor_id'][0]].item())
                }
            }
        }

        cases.append(case)

    # Create output directory
    output_dir = Path("outputs") / "debug_integrity"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Export JSON
    with open(output_dir / "case_audit_20.json", 'w') as f:
        json.dump(cases, f, indent=2)

    # Create human-readable report
    md_content = f"""# Case Audit Report - {max_samples} Samples

## Overview
Manual audit of 20 sample cases to verify model predictions and identify potential issues.

## Summary Statistics
- Total cases audited: {len(cases)}
- Attribute-Only correct predictions (top-1): {sum(1 for c in cases if c['attribute_only']['correct_top1'])}/{len(cases)} ({sum(1 for c in cases if c['attribute_only']['correct_top1'])/len(cases)*100:.1f}%)
- Raw-Text Relation correct predictions (top-1): {sum(1 for c in cases if c['raw_text_relation']['correct_top1'])}/{len(cases)} ({sum(1 for c in cases if c['raw_text_relation']['correct_top1'])/len(cases)*100:.1f}%)
- Structured Relation correct predictions (top-1): {sum(1 for c in cases if c['structured_relation']['correct_top1'])}/{len(cases)} ({sum(1 for c in cases if c['structured_relation']['correct_top1'])/len(cases)*100:.1f}%)
- Average candidates per scene: {sum(len(c['candidates']) for c in cases) / len(cases):.1f}

## Individual Case Analysis

"""

    for i, case in enumerate(cases):
        md_content += f"""
### Case {i+1}: Scene {case['scene_id']}

**Utterance**: "{case['utterance']}"
**GT Target**: {case['gt_target']} ({case['gt_target_class']})
**Number of Candidates**: {len(case['candidates'])}

#### Attribute-Only
- Top-1 Prediction: {case['attribute_only']['top1_pred']} {'✓' if case['attribute_only']['correct_top1'] else '✗'}
- Top-5 Predictions: {case['attribute_only']['top5_preds'][:5]}
- GT in Top-5: {'Yes' if case['attribute_only']['gt_in_top5'] else 'No'}

#### Raw-Text Relation
- Top-1 Prediction: {case['raw_text_relation']['top1_pred']} {'✓' if case['raw_text_relation']['correct_top1'] else '✗'}
- Top-5 Predictions: {case['raw_text_relation']['top5_preds'][:5]}
- GT in Top-5: {'Yes' if case['raw_text_relation']['gt_in_top5'] else 'No'}

#### Structured Relation
- Top-1 Prediction: {case['structured_relation']['top1_pred']} {'✓' if case['structured_relation']['correct_top1'] else '✗'}
- Top-5 Predictions: {case['structured_relation']['top5_preds'][:5]}
- GT in Top-5: {'Yes' if case['structured_relation']['gt_in_top5'] else 'No'}
- **Anchor Info**: Top anchor={case['candidates'][case['structured_relation']['anchor_info']['top_anchor_id']]['object_id']} (class: {case['candidates'][case['structured_relation']['anchor_info']['top_anchor_id']]['class_name']}), entropy={case['structured_relation']['anchor_info']['anchor_entropy']:.3f}, conf={case['structured_relation']['anchor_info']['anchor_confidence']:.3f}

"""

    md_content += f"""
## Analysis and Conclusions

Based on this manual audit of {max_samples} cases:

1. **Attribute-Only Performance**: {sum(1 for c in cases if c['attribute_only']['correct_top1'])/len(cases)*100:.1f}% top-1 accuracy
2. **Raw-Text Relation Performance**: {sum(1 for c in cases if c['raw_text_relation']['correct_top1'])/len(cases)*100:.1f}% top-1 accuracy
3. **Structured Relation Performance**: {sum(1 for c in cases if c['structured_relation']['correct_top1'])/len(cases)*100:.1f}% top-1 accuracy

"""
    model_perf_text = "performing poorly, suggesting possible evaluation contract issues" if all([acc < 0.1 for acc in [
        sum(1 for c in cases if c['attribute_only']['correct_top1'])/len(cases),
        sum(1 for c in cases if c['raw_text_relation']['correct_top1'])/len(cases),
        sum(1 for c in cases if c['structured_relation']['correct_top1'])/len(cases)
    ]]) else "showing varying performance, suggesting the evaluation framework works correctly"

    md_content += f"The models are {model_perf_text}.\n"

    with open(output_dir / "case_audit_20.md", 'w') as f:
        f.write(md_content)

    print(f"Case audit reports saved to {output_dir}/")
    print(f"Attribute-Only top-1 accuracy: {sum(1 for c in cases if c['attribute_only']['correct_top1'])/len(cases)*100:.1f}%")
    print(f"Raw-Text Relation top-1 accuracy: {sum(1 for c in cases if c['raw_text_relation']['correct_top1'])/len(cases)*100:.1f}%")
    print(f"Structured Relation top-1 accuracy: {sum(1 for c in cases if c['structured_relation']['correct_top1'])/len(cases)*100:.1f}%")


def main():
    parser = argparse.ArgumentParser(description='Generate human-readable case audit for debugging')
    parser.add_argument('--max-samples', type=int, default=20, help='Maximum samples to audit')

    args = parser.parse_args()

    run_case_audit(args.max_samples)


if __name__ == "__main__":
    import torch
    main()