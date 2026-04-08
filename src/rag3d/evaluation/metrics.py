"""Evaluator-first metrics for 3D grounding."""
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
import json
from pathlib import Path
import csv
from collections import defaultdict
import torch


def accuracy_at_k(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, k: int) -> float:
    """Masked top-k accuracy, batch mean."""
    logits = logits.masked_fill(~mask, float("-inf"))
    topk = logits.topk(k, dim=-1).indices
    correct = (topk == target.unsqueeze(1)).any(dim=1)
    return float(correct.float().mean().item())


def logit_top12_margin(logits_row: torch.Tensor, mask_row: torch.Tensor) -> float:
    vals = logits_row[mask_row]
    if vals.numel() < 2:
        return 1e6
    top2 = torch.topk(vals.float(), k=min(2, vals.numel())).values
    if top2.numel() < 2:
        return 1e6
    return float(top2[0] - top2[1])


def per_sample_correct_at1(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> list[bool]:
    row = logits.masked_fill(~mask, float("-inf"))
    pred = row.argmax(dim=-1)
    return [bool(pred[i].item() == target[i].item()) for i in range(logits.size(0))]


def per_sample_correct_at5(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> list[bool]:
    row = logits.masked_fill(~mask, float("-inf"))
    out: list[bool] = []
    for i in range(logits.size(0)):
        k = min(5, int(mask[i].sum().item()))
        if k < 1:
            out.append(False)
            continue
        topk = row[i].topk(k).indices
        out.append(bool((topk == target[i]).any().item()))
    return out


def compute_overall_metrics(
    predictions: List[Dict[str, Any]],
    targets: List[Dict[str, Any]]
) -> Dict[str, float]:
    """
    Compute overall metrics for 3D grounding.

    Args:
        predictions: List of prediction dictionaries with 'pred_top1', 'pred_top5', etc.
        targets: List of ground truth dictionaries with 'target_id', etc.

    Returns:
        Dictionary of overall metrics
    """
    # Calculate accuracy metrics
    acc_at_1 = 0
    acc_at_5 = 0
    total_samples = len(predictions)

    for pred, target in zip(predictions, targets):
        target_id = target.get('target_id', target.get('target_obj_id', ''))

        # Acc@1
        if 'pred_top1' in pred and pred['pred_top1'] == target_id:
            acc_at_1 += 1

        # Acc@5
        if 'pred_top5' in pred and target_id in pred['pred_top5']:
            acc_at_5 += 1

    results = {
        'acc_at_1': acc_at_1 / total_samples if total_samples > 0 else 0.0,
        'acc_at_5': acc_at_5 / total_samples if total_samples > 0 else 0.0,
        'total_samples': total_samples,
        'candidate_count_avg': np.mean([
            len(pred.get('candidate_object_ids', []))
            for pred in predictions
        ]) if predictions else 0.0
    }

    return results


def compute_stratified_metrics(
    predictions: List[Dict[str, Any]],
    targets: List[Dict[str, Any]],
    tags: List[Dict[str, Any]]
) -> Dict[str, Dict[str, float]]:
    """
    Compute stratified metrics based on difficulty and relation tags.

    Args:
        predictions: List of prediction dictionaries
        targets: List of ground truth dictionaries
        tags: List of tag dictionaries with relation/difficulty tags

    Returns:
        Nested dictionary of stratified metrics
    """
    # Group samples by different criteria
    groups = {
        'by_relation_type': defaultdict(list),
        'by_same_class_clutter': defaultdict(list),
        'by_occlusion_heaviness': defaultdict(list),
        'by_attribute_dominance': defaultdict(list),
        'by_relation_heaviness': defaultdict(list)
    }

    # Assign each sample to groups
    for i, (pred, target, tag) in enumerate(zip(predictions, targets, tags)):
        target_id = target.get('target_id', target.get('target_obj_id', ''))

        # Group by relation type if available
        relation_tags = tag.get('relation_tags', [])
        if relation_tags:
            for rel_tag in relation_tags:
                groups['by_relation_type'][rel_tag].append((pred, target))
        else:
            groups['by_relation_type']['unknown'].append((pred, target))

        # Group by same-class clutter
        same_class_count = tag.get('same_class_clutter_count', 0)
        clutter_level = 'none' if same_class_count == 0 else (
            'low' if same_class_count < 3 else 'high'
        )
        groups['by_same_class_clutter'][clutter_level].append((pred, target))

        # Group by other heuristics
        is_relation_heavy = tag.get('relation_heavy', False)
        groups['by_relation_heaviness']['relation_heavy' if is_relation_heavy else 'relation_light'].append((pred, target))

        is_attr_dominant = tag.get('attribute_dominant', False)
        groups['by_attribute_dominance']['attr_dominant' if is_attr_dominant else 'attr_light'].append((pred, target))

        # For occlusion heaviness, use heuristic if available
        occl_heavy = tag.get('occlusion_heavy', False)
        groups['by_occlusion_heaviness']['occl_heavy' if occl_heavy else 'occl_light'].append((pred, target))

    # Compute metrics for each group
    stratified_results = {}
    for group_type, group_dict in groups.items():
        stratified_results[group_type] = {}
        for subgroup_name, subgroup_data in group_dict.items():
            subgroup_preds = [item[0] for item in subgroup_data]
            subgroup_targets = [item[1] for item in subgroup_data]

            if subgroup_preds:
                subgroup_metrics = compute_overall_metrics(subgroup_preds, subgroup_targets)
                stratified_results[group_type][subgroup_name] = subgroup_metrics
            else:
                stratified_results[group_type][subgroup_name] = {'acc_at_1': 0.0, 'acc_at_5': 0.0, 'total_samples': 0}

    return stratified_results


def compute_diagnostic_metrics(
    predictions: List[Dict[str, Any]],
    targets: List[Dict[str, Any]],
    scores: Optional[List[List[float]]] = None
) -> Dict[str, Any]:
    """
    Compute diagnostic metrics for 3D grounding.

    Args:
        predictions: List of prediction dictionaries
        targets: List of ground truth dictionaries
        scores: Optional list of confidence scores for each candidate

    Returns:
        Dictionary of diagnostic metrics
    """
    target_margins = []
    candidate_counts = []
    failed_predictions = []

    for i, (pred, target) in enumerate(zip(predictions, targets)):
        target_id = target.get('target_id', target.get('target_obj_id', ''))
        pred_top1 = pred.get('pred_top1', '')

        # Count candidates
        candidate_ids = pred.get('candidate_object_ids', [])
        candidate_counts.append(len(candidate_ids))

        # Compute target margin if scores are available
        if scores and i < len(scores):
            pred_scores = scores[i]
            if target_id in candidate_ids:
                target_idx = candidate_ids.index(target_id)

                # Get top score
                top_score = max(pred_scores)
                target_score = pred_scores[target_idx]

                # Compute margin (top score - target score)
                margin = top_score - target_score
                target_margins.append(margin)

        # Track failed predictions
        if pred_top1 != target_id:
            failed_predictions.append({
                'index': i,
                'scene_id': target.get('scene_id', ''),
                'target_id': target_id,
                'predicted_id': pred_top1,
                'utterance': target.get('utterance', target.get('sentence', ''))
            })

    # Calculate statistics
    results = {
        'avg_target_margin': np.mean(target_margins) if target_margins else 0.0,
        'std_target_margin': np.std(target_margins) if target_margins else 0.0,
        'median_target_margin': np.median(target_margins) if target_margins else 0.0,
        'avg_candidate_count': np.mean(candidate_counts) if candidate_counts else 0.0,
        'failed_predictions': failed_predictions,
        'failure_rate': len(failed_predictions) / len(predictions) if predictions else 0.0
    }

    return results


def export_results_to_json(results: Dict[str, Any], output_path: Path) -> None:
    """Export results to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Handle numpy types for JSON serialization
    def serialize_for_json(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (list, tuple)):
            return [serialize_for_json(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: serialize_for_json(value) for key, value in obj.items()}
        else:
            return obj

    serialized_results = serialize_for_json(results)

    with open(output_path, 'w') as f:
        json.dump(serialized_results, f, indent=2)


def export_results_to_csv(results: Dict[str, Any], output_path: Path) -> None:
    """Export results to CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Flatten the results structure for CSV
    flattened_data = []

    # Process overall metrics
    if 'overall' in results:
        for key, value in results['overall'].items():
            if isinstance(value, (int, float)):
                flattened_data.append(['overall', key, value])

    # Process stratified metrics
    if 'stratified' in results:
        for group_type, groups in results['stratified'].items():
            for group_name, metrics in groups.items():
                for metric_name, metric_value in metrics.items():
                    if isinstance(metric_value, (int, float)):
                        flattened_data.append([f'{group_type}_{group_name}', metric_name, metric_value])

    # Write to CSV
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['group', 'metric', 'value'])
        writer.writerows(flattened_data)


def export_results_to_markdown(results: Dict[str, Any], output_path: Path) -> None:
    """Export results to Markdown summary."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    md_content = "# Evaluation Results Summary\n\n"

    # Add overall metrics
    if 'overall' in results:
        md_content += "## Overall Metrics\n\n"
        md_content += "| Metric | Value |\n|--------|-------|\n"
        for key, value in results['overall'].items():
            if isinstance(value, float):
                md_content += f"| {key} | {value:.4f} |\n"
            else:
                md_content += f"| {key} | {value} |\n"
        md_content += "\n"

    # Add stratified metrics
    if 'stratified' in results:
        md_content += "## Stratified Metrics\n\n"
        for group_type, groups in results['stratified'].items():
            md_content += f"### {group_type.replace('_', ' ').title()}\n\n"
            md_content += "| Subgroup | Acc@1 | Acc@5 | Samples |\n|----------|-------|-------|---------|\n"
            for group_name, metrics in groups.items():
                acc_at_1 = metrics.get('acc_at_1', 0.0)
                acc_at_5 = metrics.get('acc_at_5', 0.0)
                total_samples = metrics.get('total_samples', 0)
                md_content += f"| {group_name} | {acc_at_1:.4f} | {acc_at_5:.4f} | {total_samples} |\n"
            md_content += "\n"

    # Add diagnostic info
    if 'diagnostic' in results:
        md_content += "## Diagnostic Information\n\n"
        diagnostic = results['diagnostic']
        md_content += f"- Average Target Margin: {diagnostic.get('avg_target_margin', 0.0):.4f}\n"
        md_content += f"- Average Candidate Count: {diagnostic.get('avg_candidate_count', 0.0):.2f}\n"
        md_content += f"- Failure Rate: {diagnostic.get('failure_rate', 0.0):.4f}\n"
        md_content += f"- Failed Predictions: {len(diagnostic.get('failed_predictions', []))}\n"

    with open(output_path, 'w') as f:
        f.write(md_content)
