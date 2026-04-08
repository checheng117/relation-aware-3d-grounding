"""Adapters for converting between current data format and unified schema."""
from typing import List, Dict, Any, Optional
import torch
import numpy as np
from .schema import GroundingSample, ObjectRecord, GroundingBatch


def adapt_referit3d_sample_to_schema(
    sample_data: Dict[str, Any],
    candidate_objects: List[Dict[str, Any]]
) -> GroundingSample:
    """
    Convert ReferIt3D sample to unified schema.

    Args:
        sample_data: Original sample from referit3d dataset
        candidate_objects: List of candidate objects for this sample

    Returns:
        GroundingSample in unified schema
    """
    # Extract object IDs from candidate objects
    candidate_ids = [obj.get('object_id') or obj.get('id') for obj in candidate_objects]
    candidate_ids = [cid for cid in candidate_ids if cid is not None]

    # Attempt to extract relation tags (if available)
    relation_tags = sample_data.get('relation_tags', [])
    if not relation_tags and 'sentence' in sample_data:
        # Basic relation extraction heuristic
        sentence = sample_data['sentence'].lower()
        basic_relations = ['left', 'right', 'behind', 'front', 'near', 'next to', 'between']
        relation_tags = [rel for rel in basic_relations if rel in sentence]

    # Attempt to extract difficulty tags (if available)
    difficulty_tags = sample_data.get('difficulty_tags', [])

    return GroundingSample(
        scene_id=sample_data.get('scan_id', sample_data.get('scene_id', '')),
        utterance=sample_data.get('sentence', sample_data.get('utterance', '')),
        target_id=str(sample_data.get('target_id', sample_data.get('instance_id', ''))),
        candidate_object_ids=candidate_ids,
        relation_tags=relation_tags if relation_tags else None,
        difficulty_tags=difficulty_tags if difficulty_tags else None,
        split=sample_data.get('split'),
        metadata={
            k: v for k, v in sample_data.items()
            if k not in ['scan_id', 'scene_id', 'sentence', 'utterance', 'target_id',
                         'instance_id', 'relation_tags', 'difficulty_tags', 'split']
        }
    )


def adapt_object_record_to_schema(obj_data: Dict[str, Any]) -> ObjectRecord:
    """
    Convert object record to unified schema.

    Args:
        obj_data: Object data from referit3d dataset

    Returns:
        ObjectRecord in unified schema
    """
    # Handle center and size (convert to numpy arrays if needed)
    center = obj_data.get('center', [0.0, 0.0, 0.0])
    if not isinstance(center, np.ndarray):
        center = np.array(center)

    size = obj_data.get('size', [0.0, 0.0, 0.0])
    if not isinstance(size, np.ndarray):
        size = np.array(size)

    bbox = obj_data.get('bbox')
    if bbox is not None and not isinstance(bbox, np.ndarray):
        bbox = np.array(bbox)

    sampled_points = obj_data.get('sampled_points')
    if sampled_points is not None and not isinstance(sampled_points, np.ndarray):
        sampled_points = np.array(sampled_points)

    return ObjectRecord(
        object_id=str(obj_data.get('object_id', obj_data.get('id', ''))),
        class_name=obj_data.get('class_name', obj_data.get('label', obj_data.get('category', 'unknown'))),
        center=center,
        size=size,
        bbox=bbox,
        sampled_points=sampled_points,
        point_indices=obj_data.get('point_indices'),
        visibility_proxy=obj_data.get('visibility_proxy', obj_data.get('visible_ratio')),
        embedding_cache_key=obj_data.get('embedding_cache_key'),
        metadata={
            k: v for k, v in obj_data.items()
            if k not in ['object_id', 'id', 'class_name', 'label', 'category', 'center', 'size',
                         'bbox', 'sampled_points', 'point_indices', 'visibility_proxy',
                         'visible_ratio', 'embedding_cache_key']
        }
    )


def adapt_grounding_batch_from_raw(
    raw_samples: List[Dict[str, Any]],
    raw_objects_list: List[List[Dict[str, Any]]]
) -> GroundingBatch:
    """
    Convert raw dataset format to unified GroundingBatch.

    Args:
        raw_samples: List of raw samples from dataset
        raw_objects_list: List of object lists corresponding to each sample

    Returns:
        GroundingBatch in unified schema
    """
    samples = []
    all_objects = []

    for sample_data, objects_data in zip(raw_samples, raw_objects_list):
        # Convert sample
        sample = adapt_referit3d_sample_to_schema(sample_data, objects_data)
        samples.append(sample)

        # Convert objects and add to all_objects
        for obj_data in objects_data:
            obj_record = adapt_object_record_to_schema(obj_data)
            all_objects.append(obj_record)

    # Remove duplicate objects by ID
    unique_objects = {}
    for obj in all_objects:
        if obj.object_id not in unique_objects:
            unique_objects[obj.object_id] = obj

    return GroundingBatch(
        samples=samples,
        objects=list(unique_objects.values())
    )


def validate_schema_compliance(data: Any) -> bool:
    """
    Validate that data conforms to schema requirements.

    Args:
        data: Data to validate (GroundingSample, ObjectRecord, or GroundingBatch)

    Returns:
        True if compliant, raises exception if not
    """
    if hasattr(data, 'validate'):
        return data.validate()
    else:
        # Basic checks for other types of data
        if isinstance(data, dict):
            return True  # Basic dict check passed
        elif isinstance(data, list):
            for item in data:
                validate_schema_compliance(item)
            return True
        else:
            raise ValueError(f"Unknown data type for validation: {type(data)}")