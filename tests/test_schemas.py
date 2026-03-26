from __future__ import annotations

import torch

from rag3d.datasets.collate import collate_grounding_samples
from rag3d.datasets.schemas import SceneObject
from rag3d.datasets.synthetic import make_synthetic_batch


def test_grounding_batch_tensors():
    batch = make_synthetic_batch(batch_size=1, n_objects=3, feat_dim=32)
    g = collate_grounding_samples(batch.samples)
    t = g.to_tensors(32, torch.device("cpu"))
    assert t["object_features"].shape == (1, 3, 32)
    assert t["object_mask"].sum().item() == 3


def test_scene_object_roundtrip():
    o = SceneObject(
        object_id="a",
        class_name="chair",
        center=(0.0, 0.0, 0.0),
        size=(1.0, 1.0, 1.0),
        feature_vector=[0.0] * 4,
    )
    d = o.model_dump()
    assert d["object_id"] == "a"
