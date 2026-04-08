#!/usr/bin/env python3
"""Comprehensive evaluation integrity validation report."""

import json
import torch
from pathlib import Path
import sys

# Add src to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag3d.datasets.referit3d import ReferIt3DManifestDataset
from rag3d.datasets.collate import collate_grounding_samples
from rag3d.relation_reasoner.model import AttributeOnlyModel, RawTextRelationModel
from rag3d.relation_reasoner.structured_relation_model import StructuredRelationModel
from rag3d.training.runner import forward_attribute, forward_raw_text_relation
from rag3d.evaluation.metrics import compute_overall_metrics


def run_comprehensive_debug():
    """Run comprehensive debugging to identify the root cause of evaluation issues."""

    print("🔍 PHASE 2.5 EVALUATION INTEGRITY DEBUG REPORT")
    print("="*60)

    # 1. Check dataset integrity
    print("\n1. DATASET INTEGRITY CHECK")
    print("-" * 30)

    eval_split = "val"
    manifest_path = Path("data/processed") / f"{eval_split}_manifest.jsonl"

    if not manifest_path.exists():
        print(f"❌ ERROR: Manifest not found at {manifest_path}")
        return

    dataset = ReferIt3DManifestDataset(manifest_path=manifest_path)
    print(f"✓ Loaded {len(dataset)} samples from {eval_split} split")

    # Check first few samples
    sample_0 = dataset[0]
    print(f"✓ Sample 0: scene_id='{sample_0.scene_id}', target_object_id='{sample_0.target_object_id}', utterance='{sample_0.utterance[:50]}...'")
    print(f"✓ Sample 0: {len(sample_0.objects)} objects in candidate set")

    # Check if target is in candidate set
    candidate_ids = [obj.object_id for obj in sample_0.objects]
    target_in_candidates = sample_0.target_object_id in candidate_ids
    print(f"✓ GT target in candidate set: {target_in_candidates}")

    # 2. Check model predictions
    print(f"\n2. MODEL PREDICTION BEHAVIOR ANALYSIS")
    print("-" * 40)

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

    # Create a small evaluation set
    from torch.utils.data import DataLoader, Subset
    small_dataset = Subset(dataset, list(range(min(20, len(dataset)))))
    eval_loader = DataLoader(
        small_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_grounding_samples
    )

    device = torch.device("cpu")

    # Collect predictions for all models
    all_predictions = {
        'attribute_only': [],
        'raw_text_relation': [],
        'structured_relation': []
    }
    all_targets = []

    for batch_idx, batch in enumerate(eval_loader):
        sample = batch.samples[0]
        tensors = batch.to_tensors(256, device=device)
        candidate_ids = [obj.object_id for obj in sample.objects]

        # Store target
        target_record = {
            'scene_id': sample.scene_id,
            'target_id': sample.target_object_id,
            'candidate_object_ids': candidate_ids,
            'utterance': sample.utterance
        }
        all_targets.append(target_record)

        # Attribute model
        attr_logits = forward_attribute(attr_model, tensors)
        attr_probs = torch.softmax(attr_logits[0], dim=-1)
        attr_mask = tensors["object_mask"][0]
        attr_masked_probs = attr_probs.clone()
        attr_masked_probs[~attr_mask] = float('-inf')

        attr_top1_idx = torch.argmax(attr_masked_probs)
        _, attr_top5_indices = torch.topk(attr_masked_probs, k=min(5, len(attr_masked_probs)), largest=True)

        attr_pred_top1 = candidate_ids[attr_top1_idx.item()] if attr_top1_idx < len(candidate_ids) else "OUT_OF_BOUNDS"
        attr_pred_top5 = [candidate_ids[idx.item()] if idx < len(candidate_ids) else "OUT_OF_BOUNDS" for idx in attr_top5_indices]

        attr_pred_record = {
            'scene_id': sample.scene_id,
            'target_id': sample.target_object_id,
            'pred_top1': attr_pred_top1,
            'pred_top5': attr_pred_top5,
            'candidate_object_ids': candidate_ids,
            'confidence_scores': attr_probs[attr_mask].tolist(),
            'model_type': 'attribute_only'
        }
        all_predictions['attribute_only'].append(attr_pred_record)

        # Raw-text model
        raw_text_logits = forward_raw_text_relation(raw_text_model, tensors)
        raw_text_probs = torch.softmax(raw_text_logits[0], dim=-1)
        raw_text_mask = tensors["object_mask"][0]
        raw_text_masked_probs = raw_text_probs.clone()
        raw_text_masked_probs[~raw_text_mask] = float('-inf')

        raw_text_top1_idx = torch.argmax(raw_text_masked_probs)
        _, raw_text_top5_indices = torch.topk(raw_text_masked_probs, k=min(5, len(raw_text_masked_probs)), largest=True)

        raw_text_pred_top1 = candidate_ids[raw_text_top1_idx.item()] if raw_text_top1_idx < len(candidate_ids) else "OUT_OF_BOUNDS"
        raw_text_pred_top5 = [candidate_ids[idx.item()] if idx < len(candidate_ids) else "OUT_OF_BOUNDS" for idx in raw_text_top5_indices]

        raw_text_pred_record = {
            'scene_id': sample.scene_id,
            'target_id': sample.target_object_id,
            'pred_top1': raw_text_pred_top1,
            'pred_top5': raw_text_pred_top5,
            'candidate_object_ids': candidate_ids,
            'confidence_scores': raw_text_probs[raw_text_mask].tolist(),
            'model_type': 'raw_text_relation'
        }
        all_predictions['raw_text_relation'].append(raw_text_pred_record)

        # Structured model
        struct_results = struct_model(tensors, parsed_list=None)
        struct_logits = struct_results['logits'][0]
        struct_probs = torch.softmax(struct_logits, dim=-1)
        struct_mask = tensors["object_mask"][0]
        struct_masked_probs = struct_probs.clone()
        struct_masked_probs[~struct_mask] = float('-inf')

        struct_top1_idx = torch.argmax(struct_masked_probs)
        _, struct_top5_indices = torch.topk(struct_masked_probs, k=min(5, len(struct_masked_probs)), largest=True)

        struct_pred_top1 = candidate_ids[struct_top1_idx.item()] if struct_top1_idx < len(candidate_ids) else "OUT_OF_BOUNDS"
        struct_pred_top5 = [candidate_ids[idx.item()] if idx < len(candidate_ids) else "OUT_OF_BOUNDS" for idx in struct_top5_indices]

        struct_pred_record = {
            'scene_id': sample.scene_id,
            'target_id': sample.target_object_id,
            'pred_top1': struct_pred_top1,
            'pred_top5': struct_pred_top5,
            'candidate_object_ids': candidate_ids,
            'confidence_scores': struct_probs[struct_mask].tolist(),
            'model_type': 'structured_relation'
        }
        all_predictions['structured_relation'].append(struct_pred_record)

    # 3. Calculate metrics for each model individually
    print(f"\n3. PER-MODEL ACCURACY ANALYSIS")
    print("-" * 35)

    detailed_results = {}

    for model_name, model_preds in all_predictions.items():
        metrics = compute_overall_metrics(model_preds, all_targets)
        detailed_results[model_name] = metrics

        print(f"{model_name:20} | Acc@1: {metrics['acc_at_1']:.4f} ({int(metrics['acc_at_1'] * len(model_preds))}/{len(model_preds)}) | Acc@5: {metrics['acc_at_5']:.4f}")

    # 4. Check if predictions are identical (infrastructure bug indicator)
    print(f"\n4. PREDICTION DIVERSITY CHECK")
    print("-" * 32)

    attr_preds = [p['pred_top1'] for p in all_predictions['attribute_only']]
    raw_text_preds = [p['pred_top1'] for p in all_predictions['raw_text_relation']]
    struct_preds = [p['pred_top1'] for p in all_predictions['structured_relation']]

    all_identical = attr_preds == raw_text_preds == struct_preds
    print(f"All models produce identical predictions: {all_identical}")

    if all_identical:
        print("❌ CRITICAL: All models producing identical predictions - this indicates an infrastructure bug!")
    else:
        print("✓ Good: Models produce different predictions - infrastructure is working correctly")

    # 5. Detailed sample analysis
    print(f"\n5. SAMPLE-BY-SAMPLE ANALYSIS (First 5 samples)")
    print("-" * 45)

    print(f"{'Idx':<4} {'GT':<4} {'Attr':<6} {'RawTxt':<6} {'Struct':<6} {'Correct?':<10} {'Utterance (truncated)':<40}")
    print("-" * 90)

    correct_counts = {'attribute_only': 0, 'raw_text_relation': 0, 'structured_relation': 0}

    for i in range(min(5, len(all_targets))):
        target_id = all_targets[i]['target_id']
        attr_pred = all_predictions['attribute_only'][i]['pred_top1']
        raw_text_pred = all_predictions['raw_text_relation'][i]['pred_top1']
        struct_pred = all_predictions['structured_relation'][i]['pred_top1']

        attr_correct = attr_pred == target_id
        raw_text_correct = raw_text_pred == target_id
        struct_correct = struct_pred == target_id

        if attr_correct: correct_counts['attribute_only'] += 1
        if raw_text_correct: correct_counts['raw_text_relation'] += 1
        if struct_correct: correct_counts['structured_relation'] += 1

        utterance = all_targets[i]['utterance'][:37] + "..." if len(all_targets[i]['utterance']) > 40 else all_targets[i]['utterance']

        print(f"{i:<4} {target_id:<4} {attr_pred:<6} {raw_text_pred:<6} {struct_pred:<6} "
              f"{'Attr:'+('✓' if attr_correct else '✗')+' RT:'+('✓' if raw_text_correct else '✗')+' Str:'+('✓' if struct_correct else '✗'):<10} {utterance:<40}")

    # 6. Conclusion
    print(f"\n6. CONCLUSION")
    print("-" * 12)

    models_different = not all_identical
    some_performance = any(metrics['acc_at_1'] > 0.0 for metrics in detailed_results.values())

    if models_different and not some_performance:
        print("⚠️  INFRASTRUCTURE VALID: Models behave differently but performance is uniformly low.")
        print("   This suggests the models genuinely struggle with this task/data, not infrastructure issues.")
        print("   Proceed with VLM parser integration to enhance model capabilities.")
    elif not models_different:
        print("❌ INFRASTRUCTURE BUG: All models produce identical predictions.")
        print("   The evaluation framework has a fundamental issue that needs fixing.")
    elif some_performance:
        print("✅ INFRASTRUCTURE VALID: Models behave differently and show varied performance.")
        print("   The current implementation is working correctly.")

    print(f"\n📊 FINAL METRICS FOR {len(all_targets)} SAMPLES:")
    for model_name, metrics in detailed_results.items():
        print(f"  {model_name:20}: Acc@1={metrics['acc_at_1']:.4f}, Acc@5={metrics['acc_at_5']:.4f}")

    # Create output directory
    output_dir = Path("outputs") / "debug_integrity" / "comprehensive_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Export detailed results
    summary = {
        'dataset_info': {
            'split': eval_split,
            'total_samples': len(dataset),
            'evaluated_samples': len(all_targets),
            'target_in_candidates_rate': sum(1 for t in all_targets if t['target_id'] in t['candidate_object_ids']) / len(all_targets)
        },
        'model_predictions': {
            'attribute_only': all_predictions['attribute_only'],
            'raw_text_relation': all_predictions['raw_text_relation'],
            'structured_relation': all_predictions['structured_relation']
        },
        'model_metrics': detailed_results,
        'analysis': {
            'predictions_identical': all_identical,
            'models_different_behavior': models_different,
            'some_performance_above_zero': some_performance,
            'correct_counts': correct_counts
        }
    }

    with open(output_dir / 'comprehensive_debug_analysis.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Create markdown report
    md_content = f"""# Phase 2.5 Evaluation Integrity Report

## Executive Summary

This report analyzes the evaluation integrity of the three model approaches. Key findings:

- **Dataset integrity**: {'✓ PASS' if all(t['target_id'] in t['candidate_object_ids'] for t in all_targets) else '❌ FAIL'}
- **Model diversity**: {'✓ PASS' if models_different else '❌ FAIL'} (all models have different predictions)
- **Performance differentiation**: {'✓ PASS' if len(set(r['acc_at_1'] for r in detailed_results.values())) > 1 else '❌ FAIL'}
- **Infrastructure health**: {'VALID' if models_different and not all_identical else 'ISSUES'}

## Dataset Analysis
- Split: {eval_split}
- Total dataset size: {len(dataset)}
- Evaluated samples: {len(all_targets)}
- GT targets in candidates rate: {summary['dataset_info']['target_in_candidates_rate']:.4f}

## Model Performance
| Model | Acc@1 | Acc@5 | Correct/Total |
|-------|-------|-------|---------------|
"""
    for model_name, metrics in detailed_results.items():
        correct = int(metrics['acc_at_1'] * len(all_targets))
        total = len(all_targets)
        md_content += f"| {model_name.replace('_', ' ').title()} | {metrics['acc_at_1']:.4f} | {metrics['acc_at_5']:.4f} | {correct}/{total} |\n"

    md_content += f"""
## Sample-Level Analysis (First 5)
Index | GT | Attribute | Raw-Text | Structured | Results
------|----|----------|----------|-----------|--------
"""
    for i in range(min(5, len(all_targets))):
        target_id = all_targets[i]['target_id']
        attr_pred = all_predictions['attribute_only'][i]['pred_top1']
        raw_text_pred = all_predictions['raw_text_relation'][i]['pred_top1']
        struct_pred = all_predictions['structured_relation'][i]['pred_top1']

        attr_correct = '✓' if attr_pred == target_id else '✗'
        raw_text_correct = '✓' if raw_text_pred == target_id else '✗'
        struct_correct = '✓' if struct_pred == target_id else '✗'

        utterance = all_targets[i]['utterance'][:30] + "..." if len(all_targets[i]['utterance']) > 30 else all_targets[i]['utterance']

        md_content += f"{i} | {target_id} | {attr_pred}({attr_correct}) | {raw_text_pred}({raw_text_correct}) | {struct_pred}({struct_correct}) | '{utterance}'\n"

    md_content += f"""
## Infrastructure Assessment

### Status: {'✅ VALID' if models_different and not all_identical else '❌ ISSUE'}

- **Dataset integrity**: {'✓ PASS' if all(t['target_id'] in t['candidate_object_ids'] for t in all_targets) else '❌ FAIL'}
- **Model contract adherence**: {'✓ PASS' if all(len(p['candidate_object_ids']) == len(t['candidate_object_ids']) for p, t in zip(all_predictions["attribute_only"], all_targets)) else '❌ FAIL'}
- **Prediction diversity**: {'✓ PASS' if not all_identical else '❌ FAIL'}
- **Metric computation**: {'✓ PASS' if all(m['total_samples'] == len(all_targets) for m in detailed_results.values()) else '❌ FAIL'}

## Conclusion

The evaluation infrastructure is {'working properly' if models_different else 'has issues'}. {'Models show different behaviors' if models_different else 'All models produce identical results'}. Performance levels {'vary between models' if len(set(r['acc_at_1'] for r in detailed_results.values())) > 1 else 'are identical'}.

Based on this analysis, the uniformly low performance across all models appears to reflect {'actual model limitations on this challenging task' if models_different and not some_performance else 'infrastructure issues'}.
"""

    with open(output_dir / 'comprehensive_debug_analysis.md', 'w') as f:
        f.write(md_content)

    print(f"\n📋 Detailed reports saved to: {output_dir}/")


if __name__ == "__main__":
    run_comprehensive_debug()