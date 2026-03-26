"""Tiny synthetic scenes for smoke tests (no external data)."""

from __future__ import annotations

import numpy as np

from rag3d.datasets.schemas import GroundingBatch, GroundingSample, SceneObject
from rag3d.datasets.transforms import attach_synthetic_features


def make_synthetic_batch(batch_size: int = 2, n_objects: int = 5, feat_dim: int = 256) -> GroundingBatch:
    samples: list[GroundingSample] = []
    rng = np.random.default_rng(0)
    for b in range(batch_size):
        objs: list[SceneObject] = []
        for j in range(n_objects):
            v = rng.standard_normal(feat_dim).astype(float)
            v = (v / (np.linalg.norm(v) + 1e-8)).tolist()
            objs.append(
                SceneObject(
                    object_id=f"obj_{b}_{j}",
                    class_name="chair" if j % 2 == 0 else "table",
                    center=(float(j), 0.0, 0.0),
                    size=(0.5, 0.5, 0.5),
                    visibility_occlusion_proxy=0.9 - 0.05 * j,
                    feature_vector=v,
                )
            )
        target_index = b % n_objects
        sample = GroundingSample(
            scene_id=f"synth_{b}",
            utterance="Pick the chair left of the table near the wall.",
            target_object_id=objs[target_index].object_id,
            target_index=target_index,
            objects=objs,
            relation_type_gold="left-of",
            tags={
                "same_class_clutter": True,
                "occlusion_heavy": False,
                "anchor_confusion": True,
            },
        )
        samples.append(attach_synthetic_features(sample, feat_dim))
    return GroundingBatch(samples=samples)
