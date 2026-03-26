"""Coarse geometry-aware stage-1 model."""

from __future__ import annotations

import torch

from rag3d.datasets.schemas import GroundingSample, SceneObject
from rag3d.relation_reasoner.model import CoarseGeomAttributeModel


def test_coarse_geom_forward_shape() -> None:
    objs = [
        SceneObject(
            object_id="a",
            class_name="chair",
            center=(0.0, 0.0, 0.0),
            size=(1.0, 1.0, 1.0),
            geometry_quality="obb_aabb",
            feature_source="synthetic_collate",
            feature_vector=[0.1] * 8,
        ),
        SceneObject(
            object_id="b",
            class_name="table",
            center=(2.0, 0.0, 0.0),
            size=(1.0, 1.0, 1.0),
            feature_vector=[0.2] * 8,
        ),
    ]
    s = GroundingSample(
        scene_id="s",
        utterance="the chair",
        target_object_id="a",
        target_index=0,
        objects=objs,
    )
    m = CoarseGeomAttributeModel(8, 32, 32, dropout=0.0)
    batch = {
        "object_features": torch.randn(1, 2, 8),
        "object_mask": torch.ones(1, 2, dtype=torch.bool),
        "raw_texts": [s.utterance],
        "samples_ref": [s],
    }
    out = m(batch)
    assert out.shape == (1, 2)
    assert torch.isfinite(out).all()
