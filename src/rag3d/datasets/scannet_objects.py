"""ScanNet-style scene instance loading (aggregation.json → SceneObject list).

Supports two geometry sources:
1. Pre-extracted geometry files (data/geometry/<scene_id>_geometry.npz) - REAL geometry
2. Aggregation JSON files - fallback with placeholder geometry
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from rag3d.datasets.schemas import SceneObject
from rag3d.utils.io import read_json

log = logging.getLogger(__name__)

# Default geometry directory for pre-extracted geometry
DEFAULT_GEOMETRY_DIR = Path("data/geometry")

# Default object feature directory
DEFAULT_FEATURE_DIR = Path("data/object_features")


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


def resolve_geometry_path(scene_id: str, geometry_dir: Path | None = None) -> Path | None:
    """Locate pre-extracted geometry file for a scene."""
    gdir = geometry_dir or DEFAULT_GEOMETRY_DIR
    if not gdir.is_dir():
        return None
    candidates = [
        gdir / f"{scene_id}_geometry.npz",
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


def scene_objects_from_geometry_file(
    geometry_path: Path,
    agg_path: Path | None = None,
    feature_dir: Path | None = None,
) -> list[SceneObject]:
    """Load SceneObject list from pre-extracted geometry file.

    This uses REAL geometry extracted from Pointcept data:
    - Real centers computed from point bboxes
    - Real sizes computed from point bboxes
    - Real point counts
    - Optional: pre-computed point features

    Args:
        geometry_path: Path to *_geometry.npz file
        agg_path: Optional path to aggregation.json for additional metadata (labels)
        feature_dir: Optional directory containing pre-computed object features

    Returns:
        List of SceneObject with real geometry and optional features
    """
    data = np.load(geometry_path, allow_pickle=True)

    object_ids = data["object_ids"]
    centers = data["centers"]
    sizes = data["sizes"]
    bboxes = data["bboxes"]
    point_counts = data.get("point_counts", np.zeros(len(object_ids), dtype=np.int32))
    labels = data.get("labels", np.array([f"object_{i}" for i in object_ids]))

    # Load aggregation for additional labels if available
    agg_labels: dict[int, str] = {}
    if agg_path is not None and agg_path.is_file():
        try:
            agg_data = read_json(agg_path)
            for g in agg_data.get("segGroups", []):
                oid = g.get("objectId")
                label = g.get("label", "object")
                if oid is not None:
                    agg_labels[int(oid)] = label
        except Exception:
            pass

    # Load pre-computed features if available
    features_dict: dict[int, np.ndarray] = {}
    if feature_dir is not None and feature_dir.is_dir():
        scene_id = geometry_path.stem.replace("_geometry", "")
        feat_path = feature_dir / f"{scene_id}_features.npz"
        if feat_path.is_file():
            try:
                feat_data = np.load(feat_path, allow_pickle=True)
                feat_ids = feat_data["object_ids"]
                feat_vectors = feat_data["features"]
                for idx, oid in enumerate(feat_ids):
                    features_dict[int(oid)] = feat_vectors[idx]
                log.debug(f"Loaded {len(features_dict)} object features for {scene_id}")
            except Exception as e:
                log.warning(f"Failed to load features for {scene_id}: {e}")

    out: list[SceneObject] = []
    for i, oid in enumerate(object_ids):
        oid_int = int(oid)
        oid_str = str(oid_int)

        # Use aggregation label if available, otherwise geometry label
        label = agg_labels.get(oid_int, str(labels[i]) if i < len(labels) else f"object_{oid_int}")

        center = tuple(float(x) for x in centers[i])
        size = tuple(float(x) for x in sizes[i])
        bbox = tuple(float(x) for x in bboxes[i])

        # Get pre-computed feature vector if available
        feature_vector = None
        feat_source: str = "pointcept_extracted"
        if oid_int in features_dict:
            feature_vector = features_dict[oid_int].tolist()
            feat_source = "point_features_computed"

        out.append(
            SceneObject(
                object_id=oid_str,
                class_name=label,
                center=center,
                size=size,
                bbox=bbox,
                visibility_occlusion_proxy=None,
                feature_vector=feature_vector,
                geometry_quality="point_bbox",  # type: ignore[arg-type]
                feature_source=feat_source,  # type: ignore[arg-type]
            )
        )

    return out


def load_scene_objects_with_geometry(
    scans_dir: Path,
    scene_id: str,
    geometry_dir: Path | None = None,
    feature_dir: Path | None = None,
    prefer_geometry_file: bool = True,
) -> list[SceneObject]:
    """Load scene objects with preference for real geometry files.

    Args:
        scans_dir: Directory containing scene subdirectories with aggregation JSON
        scene_id: Scene identifier (e.g., "scene0002_00")
        geometry_dir: Directory containing pre-extracted geometry files
        feature_dir: Directory containing pre-computed object features
        prefer_geometry_file: If True, use geometry file if available (recommended)

    Returns:
        List of SceneObject with geometry from best available source
    """
    if prefer_geometry_file:
        geom_path = resolve_geometry_path(scene_id, geometry_dir)
        if geom_path is not None:
            scene_dir = scans_dir / scene_id
            agg_path = resolve_aggregation_path(scene_dir, scene_id)
            try:
                objects = scene_objects_from_geometry_file(geom_path, agg_path, feature_dir)
                log.debug("Loaded real geometry for %s from %s", scene_id, geom_path)
                return objects
            except Exception as e:
                log.warning("Failed to load geometry file %s: %s, falling back to aggregation", geom_path, e)

    # Fallback to aggregation-based loading
    return load_scene_objects_for_scene_id(scans_dir, scene_id)
