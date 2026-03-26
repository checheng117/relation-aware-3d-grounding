from __future__ import annotations

from typing import Any

import torch

from rag3d.datasets.schemas import GroundingBatch, GroundingSample
from rag3d.datasets.transforms import attach_synthetic_features


def collate_grounding_samples(samples: list[GroundingSample]) -> GroundingBatch:
    return GroundingBatch(samples=samples)


def make_grounding_collate_fn(
    feat_dim: int,
    attach_features: bool = True,
) -> Any:
    """DataLoader collate: list[GroundingSample] -> dict of CPU tensors + meta + samples_ref."""

    def _collate(batch: list[GroundingSample]) -> dict[str, Any]:
        if attach_features:
            batch = [attach_synthetic_features(s, feat_dim) for s in batch]
        gb = collate_grounding_samples(batch)
        tensors = gb.to_tensors(feat_dim, torch.device("cpu"))
        meta: list[dict[str, Any]] = []
        for s in gb.samples:
            tg = dict(s.tags)
            meta.append(
                {
                    "relation_type_gold": s.relation_type_gold,
                    "relation_type": s.relation_type_gold,
                    "tags": tg,
                    "scene_id": s.scene_id,
                    "utterance_id": s.utterance_id,
                    "n_objects": tg.get("n_objects", len(s.objects)),
                    "candidate_load": tg.get("candidate_load"),
                    "geometry_fallback_fraction": tg.get("geometry_fallback_fraction"),
                }
            )
        tensors["meta"] = meta
        tensors["samples_ref"] = gb.samples
        return tensors

    return _collate
