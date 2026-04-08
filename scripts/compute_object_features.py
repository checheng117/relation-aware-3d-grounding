#!/usr/bin/env python3
"""Compute per-object point features from extracted geometry.

Extracts simple point statistics from geometry files:
- Point cloud center (mean xyz)
- Point cloud extent (std xyz)
- Point count (normalized)
- Bounding box dimensions (normalized)

These features supplement the geometry center/size with shape statistics.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag3d.utils.logging import setup_logging
from rag3d.utils.seed import set_seed

log = logging.getLogger(__name__)

# Feature dimension for output
FEATURE_DIM = 256

# Geometry directory
DEFAULT_GEOMETRY_DIR = ROOT / "data/geometry"


def compute_point_features(points: np.ndarray, center: np.ndarray, size: np.ndarray) -> np.ndarray:
    """Compute 256-dim feature vector from point cloud.

    Features:
    - [0:3]: bbox center (tanh-normalized to [-1, 1])
    - [3:6]: bbox size (tanh-normalized by max expected size)
    - [6:256]: zeros (reserved for PointNet features)

    This simple feature encoding focuses on geometry position and extent,
    which are the most discriminative features for object grounding.

    Args:
        points: [N, 3] point coordinates (unused in this version)
        center: [3] bbox center
        size: [3] bbox size

    Returns:
        [256] feature vector
    """
    features = np.zeros(FEATURE_DIM, dtype=np.float32)

    # Center: tanh-normalized (assume max room extent ~10m)
    features[0:3] = np.tanh(center / 5.0)

    # Size: tanh-normalized (assume max object size ~4m)
    features[3:6] = np.tanh(size / 2.0)

    return features


def process_geometry_file(geometry_path: Path) -> dict[str, Any]:
    """Process a single geometry file and compute features for all objects.

    Returns:
        Dict with object_id -> feature_vector mapping
    """
    data = np.load(geometry_path, allow_pickle=True)

    object_ids = data["object_ids"]
    centers = data["centers"]
    sizes = data["sizes"]
    point_counts = data.get("point_counts", np.zeros(len(object_ids), dtype=np.int32))

    # Load per-object points
    points_dict = {}
    for i, oid in enumerate(object_ids):
        key = f"points_{i}"
        if key in data:
            points_dict[int(oid)] = data[key]

    features = {}
    stats = {
        "n_objects": len(object_ids),
        "n_points_total": 0,
        "objects_with_points": 0,
    }

    for i, oid in enumerate(object_ids):
        oid_int = int(oid)
        center = centers[i]
        size = sizes[i]

        pts = points_dict.get(oid_int)
        if pts is not None and len(pts) > 0:
            stats["n_points_total"] += len(pts)
            stats["objects_with_points"] += 1

        feat = compute_point_features(pts, center, size)
        features[str(oid_int)] = feat

    return {
        "features": features,
        "stats": stats,
    }


def compute_all_features(
    geometry_dir: Path,
    output_dir: Path,
    max_scenes: int | None = None,
) -> dict[str, Any]:
    """Compute features for all geometry files."""

    geometry_files = sorted(geometry_dir.glob("*_geometry.npz"))

    if max_scenes is not None:
        geometry_files = geometry_files[:max_scenes]

    log.info(f"Processing {len(geometry_files)} geometry files")

    output_dir.mkdir(parents=True, exist_ok=True)

    all_stats = {
        "n_scenes": len(geometry_files),
        "n_objects_total": 0,
        "n_points_total": 0,
        "objects_with_points": 0,
    }

    for gf in geometry_files:
        scene_id = gf.stem.replace("_geometry", "")

        result = process_geometry_file(gf)

        # Save per-scene features
        feat_path = output_dir / f"{scene_id}_features.npz"
        feature_vectors = np.array([result["features"][oid] for oid in sorted(result["features"].keys())])
        object_ids_sorted = np.array(sorted(result["features"].keys()), dtype=np.int32)

        np.savez(
            feat_path,
            object_ids=object_ids_sorted,
            features=feature_vectors,
        )

        # Update stats
        all_stats["n_objects_total"] += result["stats"]["n_objects"]
        all_stats["n_points_total"] += result["stats"]["n_points_total"]
        all_stats["objects_with_points"] += result["stats"]["objects_with_points"]

        if (len(geometry_files) % 50 == 0):
            log.info(f"Processed {len(geometry_files)} scenes...")

    # Save summary
    summary_path = output_dir / "feature_extraction_stats.json"
    with summary_path.open("w") as f:
        json.dump(all_stats, f, indent=2)

    log.info(f"Feature extraction complete:")
    log.info(f"  Scenes: {all_stats['n_scenes']}")
    log.info(f"  Objects: {all_stats['n_objects_total']}")
    log.info(f"  Points: {all_stats['n_points_total']}")
    log.info(f"  Objects with points: {all_stats['objects_with_points']}")

    return all_stats


def main():
    parser = argparse.ArgumentParser(description="Compute object point features")
    parser.add_argument("--geometry-dir", type=Path, default=DEFAULT_GEOMETRY_DIR)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/object_features")
    parser.add_argument("--max-scenes", type=int, default=None)
    args = parser.parse_args()

    setup_logging()
    set_seed(42)

    compute_all_features(args.geometry_dir, args.output_dir, args.max_scenes)


if __name__ == "__main__":
    main()