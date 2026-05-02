"""Unified data schema for 3D grounding datasets."""
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import numpy.typing as npt


@dataclass
class ObjectRecord:
    """Object-level schema for 3D grounding."""
    object_id: str
    class_name: str
    center: npt.NDArray  # Shape: [3], (x, y, z) coordinates
    size: npt.NDArray    # Shape: [3], (width, height, depth) dimensions
    bbox: Optional[npt.NDArray] = None  # Shape: [2, 3], min/max corners
    sampled_points: Optional[npt.NDArray] = None  # Shape: [N, 3], point cloud sampling
    point_indices: Optional[List[int]] = None  # Indices into scene point cloud
    visibility_proxy: Optional[float] = None  # Visibility score or fraction visible
    embedding_cache_key: Optional[str] = None  # Key for cached embeddings
    metadata: Optional[Dict[str, Any]] = None  # Additional object properties


@dataclass
class GroundingSample:
    """Sample-level schema for 3D grounding."""
    scene_id: str
    utterance: str  # Natural language description
    target_id: str  # ID of target object
    candidate_object_ids: List[str]  # IDs of all candidate objects in scene
    relation_tags: Optional[List[str]] = None  # Relation types mentioned
    difficulty_tags: Optional[List[str]] = None  # Difficulty indicators
    split: Optional[str] = None  # train/val/test split
    metadata: Optional[Dict[str, Any]] = None  # Additional sample properties

    def validate(self) -> bool:
        """Basic validation of the sample."""
        if not self.scene_id:
            raise ValueError("scene_id is required")
        if not self.utterance:
            raise ValueError("utterance is required")
        if not self.target_id:
            raise ValueError("target_id is required")
        if not self.candidate_object_ids:
            raise ValueError("candidate_object_ids is required")
        if self.target_id not in self.candidate_object_ids:
            raise ValueError("target_id must be in candidate_object_ids")
        return True


@dataclass
class GroundingBatch:
    """Batch-level schema for 3D grounding."""
    samples: List[GroundingSample]
    objects: List[ObjectRecord]  # All objects referenced in samples

    def validate(self) -> bool:
        """Validate the entire batch."""
        for sample in self.samples:
            sample.validate()
        return True