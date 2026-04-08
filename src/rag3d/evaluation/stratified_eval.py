"""Enhanced stratified evaluation functions for 3D grounding."""
from __future__ import annotations
from typing import Any, Dict, List, Optional
import torch
from rag3d.evaluation.metrics import accuracy_at_k
from rag3d.datasets.schema import GroundingSample
from rag3d.datasets.adapters import adapt_referit3d_sample_to_schema
from pathlib import Path
import json
import numpy as np
from collections import defaultdict


def tag_samples_heuristically(samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Apply heuristic tagging to samples for stratified evaluation.

    Args:
        samples: List of samples to tag

    Returns:
        List of tag dictionaries for each sample
    """
    tags_list = []

    for sample in samples:
        # Initialize tag dictionary
        tags = {
            'relation_tags': [],
            'difficulty_tags': [],
            'same_class_clutter_count': 0,
            'relation_heavy': False,
            'attribute_dominant': False,
            'occlusion_heavy': False
        }

        # Extract utterance for analysis
        utterance = sample.get('utterance', sample.get('sentence', '')).lower()

        # Tag based on relation keywords
        relation_keywords = [
            'left', 'right', 'behind', 'front', 'in front of', 'next to',
            'beside', 'between', 'among', 'near', 'close to', 'far from',
            'above', 'below', 'on top of', 'under', 'beneath', 'adjacent to',
            'closest', 'furthest', 'biggest', 'smallest', 'largest'
        ]

        found_relations = []
        for keyword in relation_keywords:
            if keyword in utterance:
                found_relations.append(keyword)

        tags['relation_tags'] = found_relations
        tags['relation_heavy'] = len(found_relations) > 0

        # Tag based on attribute keywords (color, size, shape)
        attribute_keywords = [
            'red', 'blue', 'green', 'yellow', 'black', 'white', 'brown', 'gray', 'grey',
            'small', 'large', 'big', 'tall', 'short', 'wide', 'narrow', 'thin', 'thick',
            'round', 'square', 'rectangular', 'circular', 'triangular'
        ]

        found_attributes = []
        for keyword in attribute_keywords:
            if keyword in utterance:
                found_attributes.append(keyword)

        tags['attribute_dominant'] = len(found_attributes) > len(found_relations)

        # Tag same-class clutter
        # This requires checking candidate objects against target object class
        target_id = sample.get('target_id', '')
        candidate_ids = sample.get('candidate_object_ids', [])

        # Placeholder - this would be computed based on actual object classes
        tags['same_class_clutter_count'] = 0

        tags_list.append(tags)

    return tags_list


def compute_and_export_stratified_evaluation(
    predictions: List[Dict[str, Any]],
    targets: List[Dict[str, Any]],
    output_dir: Path,
    export_formats: List[str] = ['json', 'csv', 'markdown']
) -> Dict[str, Any]:
    """
    Compute stratified evaluation and export results.

    Args:
        predictions: List of prediction dictionaries
        targets: List of ground truth dictionaries
        output_dir: Directory to save results
        export_formats: List of formats to export ('json', 'csv', 'markdown')

    Returns:
        Complete evaluation results
    """
    from .metrics import compute_stratified_metrics, export_results_to_json, export_results_to_csv, export_results_to_markdown

    # Generate heuristic tags for stratification
    tags = tag_samples_heuristically(targets)

    # Compute stratified metrics
    stratified_results = compute_stratified_metrics(predictions, targets, tags)

    # Prepare complete results dictionary
    results = {
        'stratified': stratified_results,
        'tags_used': [tag for tag in tags]  # Include tags for reference
    }

    # Export results in specified formats
    output_dir.mkdir(parents=True, exist_ok=True)

    for fmt in export_formats:
        if fmt == 'json':
            export_results_to_json(results, output_dir / 'stratified_results.json')
        elif fmt == 'csv':
            export_results_to_csv(results, output_dir / 'stratified_results.csv')
        elif fmt == 'markdown':
            export_results_to_markdown(results, output_dir / 'stratified_results.md')

    return results


def evaluate_by_difficulty_levels(
    predictions: List[Dict[str, Any]],
    targets: List[Dict[str, Any]]
) -> Dict[str, Dict[str, float]]:
    """
    Evaluate separately on easy, medium, and hard samples based on heuristics.

    Args:
        predictions: List of prediction dictionaries
        targets: List of ground truth dictionaries

    Returns:
        Dictionary of metrics by difficulty level
    """
    # Tag samples by difficulty heuristics
    tags = tag_samples_heuristically(targets)

    # Categorize samples by difficulty
    easy_samples = []
    medium_samples = []
    hard_samples = []

    for i, (pred, target, tag) in enumerate(zip(predictions, targets, tags)):
        # Determine difficulty based on heuristics
        relation_complexity = len(tag['relation_tags'])
        attribute_complexity = len(tag['attribute_dominant'])
        same_class_clutter = tag['same_class_clutter_count']

        # Very basic heuristic for categorization
        if same_class_clutter == 0 and relation_complexity <= 1:
            easy_samples.append((pred, target))
        elif same_class_clutter <= 2 and relation_complexity <= 2:
            medium_samples.append((pred, target))
        else:
            hard_samples.append((pred, target))

    # Split predictions and targets by difficulty
    easy_preds, easy_targets = zip(*easy_samples) if easy_samples else ([], [])
    medium_preds, medium_targets = zip(*medium_samples) if medium_samples else ([], [])
    hard_preds, hard_targets = zip(*hard_samples) if hard_samples else ([], [])

    # Calculate metrics for each difficulty level
    results = {}

    if easy_preds:
        easy_metrics = compute_overall_metrics_new(list(easy_preds), list(easy_targets))
        results['easy'] = easy_metrics
    else:
        results['easy'] = {'acc_at_1': 0.0, 'acc_at_5': 0.0, 'total_samples': 0}

    if medium_preds:
        medium_metrics = compute_overall_metrics_new(list(medium_preds), list(medium_targets))
        results['medium'] = medium_metrics
    else:
        results['medium'] = {'acc_at_1': 0.0, 'acc_at_5': 0.0, 'total_samples': 0}

    if hard_preds:
        hard_metrics = compute_overall_metrics_new(list(hard_preds), list(hard_targets))
        results['hard'] = hard_metrics
    else:
        results['hard'] = {'acc_at_1': 0.0, 'acc_at_5': 0.0, 'total_samples': 0}

    return results


def compute_overall_metrics_new(
    predictions: List[Dict[str, Any]],
    targets: List[Dict[str, Any]]
) -> Dict[str, float]:
    """
    Compute overall metrics compatible with existing system.

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
                # Use the compute_overall_metrics from the metrics module
                from .metrics import compute_overall_metrics
                subgroup_metrics = compute_overall_metrics(subgroup_preds, subgroup_targets)
                stratified_results[group_type][subgroup_name] = subgroup_metrics
            else:
                stratified_results[group_type][subgroup_name] = {'acc_at_1': 0.0, 'acc_at_5': 0.0, 'total_samples': 0}

    return stratified_results


def stratified_accuracy(
    logits: torch.Tensor,
    target_index: torch.Tensor,
    mask: torch.Tensor,
    meta: list[dict[str, Any]],
) -> dict[str, float]:
    """meta[i] may include relation_type, tags: {same_class_clutter, ...}."""
    out: dict[str, float] = {}
    if not meta or len(meta) != logits.size(0):
        return out

    def _subset(pred: callable) -> torch.Tensor:
        return torch.tensor(
            [bool(pred(meta[i])) for i in range(len(meta))],
            device=logits.device,
            dtype=torch.bool,
        )

    rel_keys = {m.get("relation_type_gold") or m.get("relation_type") for m in meta}
    for rk in rel_keys:
        if rk is None:
            continue
        sel = _subset(lambda m, r=rk: (m.get("relation_type_gold") or m.get("relation_type")) == r)
        if sel.any():
            key = f"acc@1_rel::{rk}"
            out[key] = accuracy_at_k(logits[sel], target_index[sel], mask[sel], 1)

    sel_clutter = _subset(lambda m: (m.get("tags") or {}).get("same_class_clutter", False))
    if sel_clutter.any():
        out["acc@1_subset::same_class_clutter"] = accuracy_at_k(
            logits[sel_clutter], target_index[sel_clutter], mask[sel_clutter], 1
        )

    sel_occ = _subset(lambda m: (m.get("tags") or {}).get("occlusion_heavy", False))
    if sel_occ.any():
        out["acc@1_subset::occlusion_heavy"] = accuracy_at_k(
            logits[sel_occ], target_index[sel_occ], mask[sel_occ], 1
        )

    sel_anchor = _subset(lambda m: (m.get("tags") or {}).get("anchor_confusion", False))
    if sel_anchor.any():
        out["acc@1_subset::anchor_confusion"] = accuracy_at_k(
            logits[sel_anchor], target_index[sel_anchor], mask[sel_anchor], 1
        )

    sel_pf = _subset(lambda m: (m.get("tags") or {}).get("parser_failure", False))
    if sel_pf.any():
        out["acc@1_subset::parser_failure"] = accuracy_at_k(logits[sel_pf], target_index[sel_pf], mask[sel_pf], 1)

    sel_lm = _subset(lambda m: (m.get("tags") or {}).get("low_model_margin", False))
    if sel_lm.any():
        out["acc@1_subset::low_model_margin"] = accuracy_at_k(
            logits[sel_lm], target_index[sel_lm], mask[sel_lm], 1
        )

    return out


def augment_meta_with_model_margins(
    logits: torch.Tensor,
    mask: torch.Tensor,
    meta: list[dict[str, Any]],
    margin_thresh: float = 0.15,
) -> None:
    """In-place: add tags['low_model_margin'] from predicted logit margin."""
    from rag3d.evaluation.metrics import logit_top12_margin

    for i in range(logits.size(0)):
        marg = logit_top12_margin(logits[i], mask[i])
        m = dict(meta[i])
        tags = dict(m.get("tags") or {})
        tags["low_model_margin"] = marg < margin_thresh
        m["tags"] = tags
        meta[i] = m


def stratified_accuracy_from_lists(correct: list[bool], meta: list[dict[str, Any]]) -> dict[str, float]:
    """Dataset-wide accuracy on named strata (per-sample lists)."""
    n = len(correct)
    out: dict[str, float] = {}
    if n == 0 or len(meta) != n:
        return out

    def acc_where(pred: Any) -> float | None:
        idx = [i for i in range(n) if pred(meta[i])]
        if not idx:
            return None
        return sum(correct[i] for i in idx) / len(idx)

    rel_keys = {m.get("relation_type_gold") or m.get("relation_type") for m in meta}
    for rk in rel_keys:
        if rk is None:
            continue
        v = acc_where(lambda m, r=rk: (m.get("relation_type_gold") or m.get("relation_type")) == r)
        if v is not None:
            out[f"acc@1_rel::{rk}"] = float(v)

    for key, tag in [
        ("acc@1_subset::same_class_clutter", "same_class_clutter"),
        ("acc@1_subset::occlusion_heavy", "occlusion_heavy"),
        ("acc@1_subset::anchor_confusion", "anchor_confusion"),
        ("acc@1_subset::parser_failure", "parser_failure"),
        ("acc@1_subset::low_model_margin", "low_model_margin"),
        ("acc@1_subset::geometry_high_fallback", "geometry_high_fallback"),
        ("acc@1_subset::real_box_heavy", "real_box_heavy"),
        ("acc@1_subset::weak_feature_source", "weak_feature_source"),
    ]:
        v = acc_where(lambda m, t=tag: bool((m.get("tags") or {}).get(t, False)))
        if v is not None:
            out[key] = float(v)

    gfb = acc_where(
        lambda m: float(m.get("geometry_fallback_fraction") or (m.get("tags") or {}).get("geometry_fallback_fraction") or 0.0)
        > 0.5
    )
    if gfb is not None:
        out["acc@1_slice::geometry_fallback_gt_half"] = float(gfb)
    gfb_lo = acc_where(
        lambda m: float(m.get("geometry_fallback_fraction") or (m.get("tags") or {}).get("geometry_fallback_fraction") or 0.0)
        <= 0.5
    )
    if gfb_lo is not None:
        out["acc@1_slice::geometry_fallback_le_half"] = float(gfb_lo)

    return out


def augment_meta_geometry_fallback_tags(meta: list[dict[str, Any]], samples: list[Any]) -> None:
    """In-place: tags for blueprint geometry slices (uses ``SceneObject.geometry_quality`` when present)."""
    for i, m in enumerate(meta):
        if i >= len(samples):
            continue
        s = samples[i]
        objs = getattr(s, "objects", None) or []
        if not objs:
            continue
        n = len(objs)
        fb = sum(1 for o in objs if getattr(o, "geometry_quality", None) == "fallback_centroid") / max(n, 1)
        real = sum(1 for o in objs if getattr(o, "geometry_quality", None) == "obb_aabb") / max(n, 1)
        syn = sum(1 for o in objs if getattr(o, "feature_source", None) == "synthetic_collate") / max(n, 1)
        mm = dict(m)
        tags = dict(mm.get("tags") or {})
        tags["geometry_high_fallback"] = fb > 0.5
        tags["real_box_heavy"] = real >= 0.5
        tags["weak_feature_source"] = syn > 0.8
        mm["geometry_fallback_fraction"] = fb
        mm["tags"] = tags
        meta[i] = mm
