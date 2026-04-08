"""Failure taxonomy for 3D grounding diagnostics."""
from typing import Dict, List, Any, Optional
from enum import Enum


class FailureType(Enum):
    SAME_CLASS_CONFUSION = "same_class_confusion"
    ANCHOR_CONFUSION_RISK = "anchor_confusion_risk"
    RELATION_MISMATCH_POSSIBLE = "relation_mismatch_possible"
    LOW_MARGIN_AMBIGUITY = "low_margin_ambiguity"
    PARSER_NOT_AVAILABLE = "parser_not_available"
    OTHER = "other"


def classify_prediction_failure(
    prediction: Dict[str, Any],
    target: Dict[str, Any],
    scores: Optional[List[float]] = None,
    threshold_low_margin: float = 0.1
) -> List[FailureType]:
    """
    Classify the type of failure for a single prediction.

    Args:
        prediction: Prediction dictionary
        target: Target/Ground truth dictionary
        scores: Confidence scores for all candidates
        threshold_low_margin: Threshold for determining low-margin ambiguity

    Returns:
        List of failure types
    """
    failures = []

    target_id = target.get('target_id', target.get('target_obj_id', ''))
    predicted_id = prediction.get('pred_top1', '')

    # If prediction is correct, return empty list
    if predicted_id == target_id:
        return []

    # Basic confusion classification
    # This would normally require access to object class information
    candidate_ids = prediction.get('candidate_object_ids', [])
    candidate_classes = prediction.get('candidate_object_classes', [])  # Would need to be provided

    # Check for same-class confusion (simplified heuristic)
    if target_id in candidate_ids:
        target_idx = candidate_ids.index(target_id)
        pred_idx = candidate_ids.index(predicted_id) if predicted_id in candidate_ids else -1

        # If scores are provided, check for low margin
        if scores and len(scores) > max(target_idx, pred_idx) and pred_idx != -1:
            margin = scores[pred_idx] - scores[target_idx]
            if abs(margin) < threshold_low_margin:
                failures.append(FailureType.LOW_MARGIN_AMBIGUITY)

    # General catch-all for incorrect predictions
    if predicted_id != target_id:
        failures.append(FailureType.OTHER)

    return failures


def apply_heuristic_hard_case_tags(
    samples: List[Dict[str, Any]],
    predictions: List[Dict[str, Any]],
    targets: List[Dict[str, Any]],
    scores: Optional[List[List[float]]] = None
) -> List[Dict[str, Any]]:
    """
    Apply heuristic tags for hard cases based on the data available.

    Args:
        samples: List of original samples
        predictions: List of prediction dictionaries
        targets: List of target dictionaries
        scores: Optional list of confidence scores for each sample

    Returns:
        List of tag dictionaries for each sample
    """
    tags_list = []

    for i, (sample, pred, target) in enumerate(zip(samples, predictions, targets)):
        tags = {
            'same_class_clutter': False,
            'relation_heavy': False,
            'attribute_dominant': False,
            'occlusion_heavy': False,
            'low_margin_ambiguity': False,
            'failure_classification': []
        }

        # Determine if it's a same-class clutter case
        # This would normally require access to object classes
        target_id = target.get('target_id', target.get('target_obj_id', ''))

        # Check for same class clutter
        candidate_ids = target.get('candidate_object_ids', [])
        # Placeholder heuristic: if there are many candidates, it might be cluttered
        tags['same_class_clutter'] = len(candidate_ids) > 10  # Simplified heuristic

        # Determine if relation-heavy
        utterance = target.get('utterance', target.get('sentence', '')).lower()
        relation_keywords = ['left', 'right', 'behind', 'front', 'next to', 'between', 'in front of']
        tags['relation_heavy'] = any(keyword in utterance for keyword in relation_keywords)

        # Determine if attribute dominant
        attribute_keywords = ['red', 'blue', 'green', 'large', 'small', 'big', 'tall', 'round', 'square']
        tags['attribute_dominant'] = any(keyword in utterance for keyword in attribute_keywords)

        # Determine if it's a failure case and classify failure
        predicted_id = pred.get('pred_top1', '')
        if predicted_id != target_id:
            sample_scores = scores[i] if scores and i < len(scores) else None
            failures = classify_prediction_failure(pred, target, sample_scores)
            tags['failure_classification'] = [f.value for f in failures]

            # Mark as low margin if applicable
            if FailureType.LOW_MARGIN_AMBIGUITY in failures:
                tags['low_margin_ambiguity'] = True

        tags_list.append(tags)

    return tags_list


def generate_failure_summary(
    all_tags: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Generate a summary of failure types and hard cases.

    Args:
        all_tags: List of tag dictionaries for all samples

    Returns:
        Summary dictionary with failure statistics
    """
    total_samples = len(all_tags)
    if total_samples == 0:
        return {}

    # Count occurrences of each tag
    same_class_clutter_count = sum(1 for tag in all_tags if tag.get('same_class_clutter', False))
    relation_heavy_count = sum(1 for tag in all_tags if tag.get('relation_heavy', False))
    attribute_dominant_count = sum(1 for tag in all_tags if tag.get('attribute_dominant', False))
    low_margin_count = sum(1 for tag in all_tags if tag.get('low_margin_ambiguity', False))

    # Collect all failure classifications
    all_failures = []
    for tag in all_tags:
        all_failures.extend(tag.get('failure_classification', []))

    failure_counts = {}
    for failure in all_failures:
        failure_counts[failure] = failure_counts.get(failure, 0) + 1

    summary = {
        'total_samples': total_samples,
        'same_class_clutter_count': same_class_clutter_count,
        'same_class_clutter_ratio': same_class_clutter_count / total_samples if total_samples > 0 else 0,
        'relation_heavy_count': relation_heavy_count,
        'relation_heavy_ratio': relation_heavy_count / total_samples if total_samples > 0 else 0,
        'attribute_dominant_count': attribute_dominant_count,
        'attribute_dominant_ratio': attribute_dominant_count / total_samples if total_samples > 0 else 0,
        'low_margin_ambiguity_count': low_margin_count,
        'low_margin_ambiguity_ratio': low_margin_count / total_samples if total_samples > 0 else 0,
        'failure_counts': failure_counts,
        'failure_ratios': {
            k: v / total_samples if total_samples > 0 else 0 for k, v in failure_counts.items()
        }
    }

    return summary