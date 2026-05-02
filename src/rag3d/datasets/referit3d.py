"""ReferIt3D / ScanNet-style dataset: manifests, validation, PyTorch Dataset."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterator

import torch
from torch.utils.data import Dataset

from rag3d.datasets.schemas import GroundingSample, SceneObject
from rag3d.utils.io import load_manifest_records

log = logging.getLogger(__name__)


def validate_raw_root(raw_root: Path) -> None:
    if not raw_root.exists():
        raise FileNotFoundError(
            f"ReferIt3D raw root missing: {raw_root}. "
            "See docs/DATASET_SETUP.md for layout; set raw_root in configs/dataset/referit3d.yaml."
        )


def manifest_dict_to_sample(r: dict[str, Any]) -> GroundingSample:
    objs = []
    for o in r.get("objects", []):
        gq = o.get("geometry_quality")
        fs = o.get("feature_source")
        objs.append(
            SceneObject(
                object_id=str(o["object_id"]),
                class_name=str(o["class_name"]),
                center=tuple(float(x) for x in o["center"]),
                size=tuple(float(x) for x in o["size"]),
                bbox=tuple(float(x) for x in o["bbox"]) if o.get("bbox") else None,
                visibility_occlusion_proxy=o.get("visibility_occlusion_proxy"),
                feature_vector=o.get("feature_vector"),
                geometry_quality=gq if gq in ("obb_aabb", "fallback_centroid", "unknown") else "unknown",
                feature_source=fs
                if fs in ("synthetic_collate", "aggregated_file", "user_provided", "unknown")
                else "unknown",
            )
        )
    return GroundingSample(
        scene_id=str(r["scene_id"]),
        utterance=str(r["utterance"]),
        target_object_id=str(r["target_object_id"]),
        target_index=int(r["target_index"]),
        objects=objs,
        utterance_id=r.get("utterance_id"),
        relation_type_gold=r.get("relation_type_gold"),
        tags=r.get("tags") or {},
    )


def load_manifest_samples(path: Path) -> list[GroundingSample]:
    rows = load_manifest_records(path)
    return [manifest_dict_to_sample(r) for r in rows]


class ReferIt3DManifestDataset(Dataset):
    """Torch Dataset from processed JSON or JSONL manifest."""

    def __init__(self, manifest_path: Path, raw_root: Path | None = None) -> None:
        if raw_root is not None:
            validate_raw_root(raw_root)
        self._samples = load_manifest_samples(manifest_path)
        if not self._samples:
            raise ValueError(f"Empty manifest: {manifest_path}")

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> GroundingSample:
        return self._samples[idx]

    def __iter__(self) -> Iterator[GroundingSample]:
        return iter(self._samples)


# Back-compat alias
ReferIt3DFrameworkDataset = ReferIt3DManifestDataset
