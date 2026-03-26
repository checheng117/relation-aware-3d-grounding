"""ScanNet-style scene instance loading (aggregation.json → SceneObject list)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from rag3d.datasets.schemas import SceneObject
from rag3d.utils.io import read_json

log = logging.getLogger(__name__)


def resolve_aggregation_path(scene_dir: Path, scene_id: str) -> Path | None:
    """Locate a ScanNet-style aggregation JSON under ``scene_dir`` (handles naming variants)."""
    if not scene_dir.is_dir():
        return None
    candidates = [
        scene_dir / f"{scene_id}_vh_clean_aggregation.json",
        scene_dir / f"{scene_id}_vh_clean.aggregation.json",
    ]
    for c in candidates:
        if c.is_file():
            return c
    for p in sorted(scene_dir.glob("*.json")):
        low = p.name.lower()
        if "vh_clean" in low and "aggregation" in low:
            return p
        if low.endswith(".aggregation.json") and scene_id in p.name:
            return p
    return None


def _obb_to_center_size(obb: dict[str, Any] | None) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Return (center_xyz), (size_xyz) from ScanNet-style OBB block."""
    if not obb:
        return (0.0, 0.0, 0.0), (0.1, 0.1, 0.1)
    c = obb.get("centroid") or [0.0, 0.0, 0.0]
    ax = obb.get("axesLengths") or [0.1, 0.1, 0.1]
    return (float(c[0]), float(c[1]), float(c[2])), (float(ax[0]), float(ax[1]), float(ax[2]))


def scene_objects_from_aggregation(agg: dict[str, Any]) -> list[SceneObject]:
    """Parse ScanNet `*_vh_clean_aggregation.json` dict into SceneObject list (order = segGroups order)."""
    groups = agg.get("segGroups") or agg.get("seg_groups") or []
    out: list[SceneObject] = []
    for g in groups:
        oid = g.get("objectId")
        if oid is None:
            continue
        label = str(g.get("label") or "object")
        obb = g.get("obb")
        has_real_obb = bool(
            obb and isinstance(obb, dict) and "centroid" in obb and "axesLengths" in obb
        )
        center, size = _obb_to_center_size(obb if isinstance(obb, dict) else None)
        bbox = None
        if has_real_obb:
            c = obb["centroid"]
            a = obb["axesLengths"]
            bbox = (
                float(c[0]) - float(a[0]) / 2,
                float(c[1]) - float(a[1]) / 2,
                float(c[2]) - float(a[2]) / 2,
                float(c[0]) + float(a[0]) / 2,
                float(c[1]) + float(a[1]) / 2,
                float(c[2]) + float(a[2]) / 2,
            )
        geo_q: str = "obb_aabb" if has_real_obb else "fallback_centroid"
        out.append(
            SceneObject(
                object_id=str(oid),
                class_name=label,
                center=center,
                size=size,
                bbox=bbox,
                visibility_occlusion_proxy=g.get("visibility_proxy"),
                geometry_quality=geo_q,  # type: ignore[arg-type]
                feature_source="aggregated_file",
            )
        )
    return out


def load_scene_aggregation_path(path: Path) -> list[SceneObject]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing aggregation file: {path}")
    data = read_json(path)
    return scene_objects_from_aggregation(data)


def load_scene_objects_for_scene_id(scans_root: Path, scene_id: str) -> list[SceneObject]:
    scene_dir = scans_root / scene_id
    agg_path = resolve_aggregation_path(scene_dir, scene_id)
    if agg_path is None:
        raise FileNotFoundError(f"No aggregation JSON found for scene {scene_id} under {scene_dir}")
    return load_scene_aggregation_path(agg_path)


def load_scene_objects_from_processed(scene_json: Path) -> list[SceneObject]:
    """Load a user-prepared scene JSON: { \"objects\": [ {object_id, class_name, center, size}, ... ] }."""
    data = read_json(scene_json)
    out: list[SceneObject] = []
    for o in data.get("objects", []):
        out.append(
            SceneObject(
                object_id=str(o["object_id"]),
                class_name=str(o["class_name"]),
                center=tuple(float(x) for x in o["center"]),
                size=tuple(float(x) for x in o["size"]),
                geometry_quality=o.get("geometry_quality", "unknown"),  # type: ignore[arg-type]
                feature_source=o.get("feature_source", "user_provided"),  # type: ignore[arg-type]
            )
        )
    return out
