"""Heuristic tagging for hard cases in 3D grounding."""
from typing import Dict, List, Any
import re


def tag_same_class_clutter(
    sample: Dict[str, Any],
    objects_info: List[Dict[str, Any]]
) -> bool:
    """
    Tag samples with same-class clutter based on object classes in the scene.

    Args:
        sample: Sample dictionary with target_id and candidate_object_ids
        objects_info: List of object dictionaries with class information

    Returns:
        True if same-class clutter detected, False otherwise
    """
    target_id = sample.get('target_id', sample.get('target_obj_id', ''))
    candidate_ids = sample.get('candidate_object_ids', [])

    # Build object id to class mapping
    obj_class_map = {}
    for obj in objects_info:
        obj_id = obj.get('object_id', obj.get('id', ''))
        obj_class = obj.get('class_name', obj.get('label', obj.get('category', 'unknown')))
        obj_class_map[obj_id] = obj_class

    if target_id not in obj_class_map:
        return False

    target_class = obj_class_map[target_id]
    same_class_candidates = [
        obj_id for obj_id in candidate_ids
        if obj_id in obj_class_map and obj_class_map[obj_id] == target_class
    ]

    # At least 2 other objects of same class (excluding the target)
    return len(same_class_candidates) >= 2


def tag_relation_heaviness(
    sample: Dict[str, Any]
) -> bool:
    """
    Tag samples that are heavily dependent on spatial relations.

    Args:
        sample: Sample dictionary with utterance/sentence

    Returns:
        True if relation-heavy, False otherwise
    """
    utterance = sample.get('utterance', sample.get('sentence', '')).lower()

    # Spatial relation keywords
    relation_keywords = [
        'left of', 'right of', 'behind', 'in front of', 'front of', 'next to',
        'beside', 'between', 'among', 'near', 'close to', 'far from',
        'above', 'below', 'on top of', 'under', 'beneath', 'adjacent to',
        'closest to', 'furthest from', 'nearest', 'farthest',
        'left side', 'right side', 'back side', 'front side'
    ]

    # Count occurrences of relation keywords
    count = 0
    for keyword in relation_keywords:
        count += len(re.findall(r'\b' + re.escape(keyword) + r'\b', utterance))

    # If at least one relation keyword is present, consider it relation-heavy
    return count > 0


def tag_attribute_dominance(
    sample: Dict[str, Any]
) -> bool:
    """
    Tag samples that are dominated by attribute descriptors (color, size, etc.).

    Args:
        sample: Sample dictionary with utterance/sentence

    Returns:
        True if attribute-dominant, False otherwise
    """
    utterance = sample.get('utterance', sample.get('sentence', '')).lower()

    # Attribute keywords (color, size, shape, etc.)
    color_keywords = [
        'red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'brown',
        'black', 'white', 'gray', 'grey', 'silver', 'gold', 'color'
    ]

    size_keywords = [
        'small', 'large', 'big', 'tiny', 'huge', 'massive', 'little',
        'mini', 'micro', 'giant', 'enormous', 'massive', 'sizable'
    ]

    shape_keywords = [
        'round', 'square', 'rectangular', 'circular', 'triangular', 'oval',
        'sphere', 'cube', 'cylinder', 'cone', 'pyramid'
    ]

    # Count occurrences
    attr_count = 0
    for keywords in [color_keywords, size_keywords, shape_keywords]:
        for keyword in keywords:
            attr_count += len(re.findall(r'\b' + re.escape(keyword) + r'\b', utterance))

    # If at least one attribute keyword is present
    return attr_count > 0


def tag_occlusion_heaviness(
    sample: Dict[str, Any],
    objects_info: List[Dict[str, Any]]
) -> bool:
    """
    Tag samples with heavy occlusion based on object visibility data.

    Args:
        sample: Sample dictionary
        objects_info: List of object dictionaries with visibility information

    Returns:
        True if occlusion-heavy, False otherwise
    """
    candidate_ids = sample.get('candidate_object_ids', [])

    # Extract visibility info for candidate objects
    visibilities = []
    for obj in objects_info:
        obj_id = obj.get('object_id', obj.get('id', ''))
        if obj_id in candidate_ids:
            visibility = obj.get('visibility_proxy', obj.get('visible_ratio', 1.0))
            if visibility is not None:
                visibilities.append(visibility)

    if not visibilities:
        # If no visibility data available, default to False
        return False

    # Calculate average visibility
    avg_visibility = sum(visibilities) / len(visibilities) if visibilities else 1.0

    # If average visibility is low, consider it occlusion-heavy
    return avg_visibility < 0.3  # Threshold: less than 30% visible on average


def generate_hard_case_tags(
    samples: List[Dict[str, Any]],
    objects_list: List[List[Dict[str, Any]]]
) -> List[Dict[str, Any]]:
    """
    Generate comprehensive hard case tags for a batch of samples.

    Args:
        samples: List of sample dictionaries
        objects_list: List of object lists corresponding to each sample

    Returns:
        List of tag dictionaries for each sample
    """
    all_tags = []

    for sample, objects in zip(samples, objects_list):
        tags = {}

        # Apply different tagging functions
        tags['same_class_clutter'] = tag_same_class_clutter(sample, objects)
        tags['relation_heavy'] = tag_relation_heaviness(sample)
        tags['attribute_dominant'] = tag_attribute_dominance(sample)
        tags['occlusion_heavy'] = tag_occlusion_heaviness(sample, objects)

        # Additional heuristics
        utterance = sample.get('utterance', sample.get('sentence', ''))
        tags['utterance_length'] = len(utterance.split())

        candidate_count = len(sample.get('candidate_object_ids', []))
        tags['candidate_count'] = candidate_count

        # Long utterances with many candidates might be harder
        tags['complex_query'] = tags['utterance_length'] > 10 and candidate_count > 15

        all_tags.append(tags)

    return all_tags


def summarize_hard_cases(
    tags_list: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Summarize statistics about hard cases.

    Args:
        tags_list: List of tag dictionaries

    Returns:
        Summary dictionary with hard case statistics
    """
    if not tags_list:
        return {}

    n_samples = len(tags_list)

    # Count different types of hard cases
    stats = {
        'total_samples': n_samples,
        'same_class_clutter_count': sum(1 for tags in tags_list if tags.get('same_class_clutter', False)),
        'relation_heavy_count': sum(1 for tags in tags_list if tags.get('relation_heavy', False)),
        'attribute_dominant_count': sum(1 for tags in tags_list if tags.get('attribute_dominant', False)),
        'occlusion_heavy_count': sum(1 for tags in tags_list if tags.get('occlusion_heavy', False)),
        'complex_query_count': sum(1 for tags in tags_list if tags.get('complex_query', False)),
    }

    # Calculate ratios
    for key in list(stats.keys()):  # Create a copy of keys to iterate
        if key.endswith('_count'):
            ratio_key = key.replace('_count', '_ratio')
            stats[ratio_key] = stats[key] / n_samples if n_samples > 0 else 0

    # Additional statistics
    utterance_lengths = [tags.get('utterance_length', 0) for tags in tags_list]
    stats['avg_utterance_length'] = sum(utterance_lengths) / len(utterance_lengths) if utterance_lengths else 0

    candidate_counts = [tags.get('candidate_count', 0) for tags in tags_list]
    stats['avg_candidate_count'] = sum(candidate_counts) / len(candidate_counts) if candidate_counts else 0

    return stats