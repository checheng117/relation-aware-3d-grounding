#!/usr/bin/env python3
"""Prepare ScanNet geometry for ReferIt3D baseline reproduction.

This script extracts per-object point clouds from ScanNet scenes.
Since we don't have actual ScanNet mesh files, this implementation
uses available aggregation data to generate approximate geometry.

For full reproduction, replace with real ScanNet mesh processing.

Output:
- Per-scene geometry files with object point clouds
- Geometry statistics and validation report
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag3d.utils.logging import setup_logging

log = logging.getLogger(__name__)


@dataclass
class ObjectGeometry:
    """Geometry for a single object."""

    object_id: str
    class_name: str
    center: np.ndarray  # [3]
    size: np.ndarray  # [3]
    points: Optional[np.ndarray] = None  # [N, 3] or [N, 6] with RGB
    num_points: int = 0


@dataclass
class SceneGeometry:
    """Geometry for a single scene."""

    scene_id: str
    objects: List[ObjectGeometry]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "objects": [
                {
                    "object_id": obj.object_id,
                    "class_name": obj.class_name,
                    "center": obj.center.tolist(),
                    "size": obj.size.tolist(),
                    "num_points": obj.num_points,
                }
                for obj in self.objects
            ],
        }


def load_aggregation(scene_dir: Path) -> Dict[str, Any]:
    """Load ScanNet aggregation file for a scene."""
    agg_files = list(scene_dir.glob("*.aggregation.json"))
    if not agg_files:
        raise FileNotFoundError(f"No aggregation file in {scene_dir}")

    with agg_files[0].open("r") as f:
        return json.load(f)


def parse_aggregation(aggregation: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    """Parse aggregation into objectId -> info mapping."""
    objects = {}
    for seg_group in aggregation.get("segGroups", []):
        obj_id = seg_group.get("objectId")
        if obj_id is not None:
            objects[obj_id] = {
                "id": obj_id,
                "label": seg_group.get("label", "unknown"),
                "segments": seg_group.get("segments", []),
            }
    return objects


def generate_synthetic_geometry(
    obj_id: int,
    label: str,
    num_points: int = 1024,
    seed: Optional[int] = None,
) -> ObjectGeometry:
    """
    Generate synthetic geometry for an object.

    NOTE: This is a PLACEHOLDER for real ScanNet mesh extraction.
    For actual reproduction, use real point clouds from ScanNet meshes.

    Args:
        obj_id: Object ID
        label: Object class name
        num_points: Number of points to generate
        seed: Random seed for reproducibility

    Returns:
        ObjectGeometry with synthetic points
    """
    if seed is not None:
        np.random.seed(seed + obj_id)

    # Generate random center (centered around object ID for differentiation)
    center = np.array([
        (obj_id % 10) * 0.5 + np.random.uniform(-0.1, 0.1),
        (obj_id // 10) * 0.3 + np.random.uniform(-0.1, 0.1),
        np.random.uniform(0.3, 1.5),
    ])

    # Generate size based on object class
    size_scales = {
        "chair": (0.5, 0.5, 0.8),
        "table": (1.0, 0.8, 0.7),
        "door": (0.8, 0.05, 2.0),
        "wall": (2.0, 0.1, 2.5),
        "floor": (3.0, 3.0, 0.1),
        "ceiling": (3.0, 3.0, 0.1),
        "cabinet": (0.6, 0.4, 1.2),
        "shelf": (0.8, 0.3, 1.5),
        "lamp": (0.2, 0.2, 0.5),
        "sofa": (1.5, 0.8, 0.6),
        "bed": (1.5, 2.0, 0.4),
        "desk": (1.2, 0.6, 0.75),
        "pillow": (0.4, 0.3, 0.15),
        "plant": (0.3, 0.3, 0.6),
        "bookshelf": (0.8, 0.3, 1.8),
        "window": (1.0, 0.1, 1.2),
        "picture": (0.6, 0.02, 0.4),
        "tv": (1.0, 0.1, 0.6),
        "monitor": (0.5, 0.1, 0.4),
        "curtain": (1.0, 0.1, 2.0),
        "book": (0.2, 0.15, 0.03),
        "box": (0.3, 0.2, 0.15),
        "bag": (0.3, 0.2, 0.25),
        "object": (0.3, 0.3, 0.3),  # Default
    }

    default_size = (0.3, 0.3, 0.3)
    base_size = size_scales.get(label.lower(), default_size)
    size = np.array(base_size) * np.random.uniform(0.8, 1.2, 3)

    # Generate synthetic points within bounding box
    points = np.random.uniform(-0.5, 0.5, (num_points, 3)) * size + center

    # Optionally add RGB (random colors for synthetic)
    rgb = np.random.randint(0, 256, (num_points, 3), dtype=np.uint8)
    points_with_rgb = np.concatenate([points, rgb.astype(np.float32) / 255.0], axis=1)

    return ObjectGeometry(
        object_id=str(obj_id),
        class_name=label,
        center=center.astype(np.float32),
        size=size.astype(np.float32),
        points=points_with_rgb,
        num_points=num_points,
    )


def extract_scene_geometry(
    scene_dir: Path,
    num_points_per_object: int = 1024,
    seed: Optional[int] = None,
    use_synthetic: bool = True,
) -> SceneGeometry:
    """
    Extract geometry for all objects in a scene.

    Args:
        scene_dir: Path to scene directory
        num_points_per_object: Number of points per object
        seed: Random seed
        use_synthetic: If True, use synthetic geometry (placeholder)

    Returns:
        SceneGeometry with all objects
    """
    scene_id = scene_dir.name

    # Load aggregation
    try:
        aggregation = load_aggregation(scene_dir)
    except FileNotFoundError as e:
        log.warning(f"Skipping {scene_id}: {e}")
        return SceneGeometry(scene_id=scene_id, objects=[])

    # Parse object info
    objects_info = parse_aggregation(aggregation)

    # Extract geometry for each object
    objects = []
    for obj_id, info in objects_info.items():
        if use_synthetic:
            # Generate synthetic geometry
            obj_geom = generate_synthetic_geometry(
                obj_id=obj_id,
                label=info["label"],
                num_points=num_points_per_object,
                seed=seed,
            )
        else:
            # TODO: Extract from real mesh
            log.warning("Real mesh extraction not implemented, using synthetic")
            obj_geom = generate_synthetic_geometry(
                obj_id=obj_id,
                label=info["label"],
                num_points=num_points_per_object,
                seed=seed,
            )

        objects.append(obj_geom)

    return SceneGeometry(scene_id=scene_id, objects=objects)


def prepare_all_scenes(
    scans_dir: Path,
    output_dir: Path,
    num_points_per_object: int = 1024,
    seed: Optional[int] = None,
    max_scenes: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Prepare geometry for all scenes.

    Args:
        scans_dir: Directory containing scene directories
        output_dir: Output directory for geometry files
        num_points_per_object: Points per object
        seed: Random seed
        max_scenes: Maximum scenes to process (for testing)

    Returns:
        Statistics dictionary
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all scene directories
    scene_dirs = sorted([d for d in scans_dir.iterdir() if d.is_dir()])
    if max_scenes:
        scene_dirs = scene_dirs[:max_scenes]

    log.info(f"Processing {len(scene_dirs)} scenes")

    stats = {
        "total_scenes": len(scene_dirs),
        "total_objects": 0,
        "points_per_object": num_points_per_object,
        "synthetic_mode": True,
        "scenes": [],
    }

    for i, scene_dir in enumerate(scene_dirs):
        if (i + 1) % 50 == 0:
            log.info(f"Processing scene {i + 1}/{len(scene_dirs)}")

        # Extract geometry
        scene_geom = extract_scene_geometry(
            scene_dir,
            num_points_per_object=num_points_per_object,
            seed=seed,
        )

        # Save to file
        output_file = output_dir / f"{scene_geom.scene_id}_geometry.npz"
        if scene_geom.objects:
            # Save point clouds
            object_ids = []
            class_names = []
            centers = []
            sizes = []
            points_list = []

            for obj in scene_geom.objects:
                object_ids.append(obj.object_id)
                class_names.append(obj.class_name)
                centers.append(obj.center)
                sizes.append(obj.size)
                if obj.points is not None:
                    points_list.append(obj.points)

            np.savez(
                output_file,
                object_ids=np.array(object_ids),
                class_names=np.array(class_names),
                centers=np.array(centers),
                sizes=np.array(sizes),
                **{f"points_{i}": p for i, p in enumerate(points_list)},
            )

            stats["total_objects"] += len(scene_geom.objects)
            stats["scenes"].append({
                "scene_id": scene_geom.scene_id,
                "num_objects": len(scene_geom.objects),
                "output_file": str(output_file),
            })

    # Save statistics
    stats_file = output_dir / "geometry_statistics.json"
    with stats_file.open("w") as f:
        json.dump(stats, f, indent=2)

    log.info(f"Processed {stats['total_objects']} objects from {len(scene_dirs)} scenes")
    log.info(f"Saved statistics to {stats_file}")

    return stats


def validate_geometry(output_dir: Path) -> Dict[str, Any]:
    """Validate generated geometry files."""
    geom_files = list(output_dir.glob("*_geometry.npz"))

    if not geom_files:
        return {"error": "No geometry files found"}

    validation = {
        "total_files": len(geom_files),
        "total_objects": 0,
        "point_counts": [],
        "center_ranges": {"min": [], "max": []},
        "size_ranges": {"min": [], "max": []},
        "issues": [],
    }

    for geom_file in geom_files[:10]:  # Sample first 10 for validation
        data = np.load(geom_file, allow_pickle=True)
        object_ids = data["object_ids"]
        validation["total_objects"] += len(object_ids)

        centers = data["centers"]
        sizes = data["sizes"]

        validation["center_ranges"]["min"].append(centers.min(axis=0).tolist())
        validation["center_ranges"]["max"].append(centers.max(axis=0).tolist())
        validation["size_ranges"]["min"].append(sizes.min(axis=0).tolist())
        validation["size_ranges"]["max"].append(sizes.max(axis=0).tolist())

        # Check for degenerate geometry
        if np.any(sizes < 0.01):
            validation["issues"].append(f"{geom_file.name}: very small objects detected")

    return validation


def main():
    parser = argparse.ArgumentParser(description="Prepare ScanNet geometry")
    parser.add_argument(
        "--scans-dir",
        type=Path,
        default=ROOT / "data/raw/referit3d/scans",
        help="Directory containing scene directories",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data/geometry",
        help="Output directory for geometry files",
    )
    parser.add_argument("--num-points", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-scenes", type=int, default=None)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    setup_logging()

    if args.validate_only:
        stats = validate_geometry(args.output_dir)
        print(json.dumps(stats, indent=2))
        return

    # Check if scans directory exists
    if not args.scans_dir.exists():
        log.error(f"Scans directory not found: {args.scans_dir}")
        log.error("Please ensure ScanNet data is available")
        return

    # Prepare geometry
    stats = prepare_all_scenes(
        scans_dir=args.scans_dir,
        output_dir=args.output_dir,
        num_points_per_object=args.num_points,
        seed=args.seed,
        max_scenes=args.max_scenes,
    )

    # Print summary
    print(f"\nGeometry Preparation Summary:")
    print(f"  Total scenes: {stats['total_scenes']}")
    print(f"  Total objects: {stats['total_objects']}")
    print(f"  Synthetic mode: {stats['synthetic_mode']}")
    print(f"  Output directory: {args.output_dir}")

    if stats["synthetic_mode"]:
        print("\n  ⚠️  WARNING: Using SYNTHETIC geometry (placeholder)")
        print("  For actual reproduction, replace with real ScanNet point clouds")


if __name__ == "__main__":
    main()