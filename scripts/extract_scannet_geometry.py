#!/usr/bin/env python3
"""Extract real per-object geometry from Pointcept ScanNet data.

This script extracts per-object point clouds from Pointcept's scannet.tar.gz
using instance.npy labels, computing real center/size/bbox from actual points.

Output:
- Per-scene geometry files: data/geometry/<scene_id>_geometry.npz
  - object_ids: array of object IDs (matching aggregation objectId)
  - centers: [N, 3] real center coordinates
  - sizes: [N, 3] real bounding box sizes
  - bboxes: [N, 6] (min_x, min_y, min_z, max_x, max_y, max_z)
  - point_counts: [N] number of points per object
  - points_{i}: [M, 3] or [M, 6] per-object point cloud (optional, controlled by --save-points)
  - geometry_quality: ["point_bbox", ...] per-object quality flag
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import sys
import tarfile
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag3d.utils.logging import setup_logging

log = logging.getLogger(__name__)


def find_pointcept_tar() -> Path | None:
    """Locate Pointcept scannet.tar.gz in HuggingFace cache."""
    cache_base = Path.home() / ".cache" / "huggingface" / "hub"
    if not cache_base.is_dir():
        return None

    for repo_dir in cache_base.iterdir():
        if not repo_dir.is_dir():
            continue
        if "pointcept" in repo_dir.name.lower() or "scannet-compressed" in repo_dir.name.lower():
            snapshots_dir = repo_dir / "snapshots"
            if snapshots_dir.is_dir():
                for snapshot_dir in snapshots_dir.iterdir():
                    if snapshot_dir.is_dir():
                        tar_path = snapshot_dir / "scannet.tar.gz"
                        if tar_path.is_file():
                            return tar_path
    return None


def load_aggregation_object_map(agg_path: Path) -> dict[int, str]:
    """Load aggregation JSON and return objectId -> label mapping."""
    data = json.loads(agg_path.read_text(encoding="utf-8"))
    obj_map = {}
    for group in data.get("segGroups", []):
        obj_id = group.get("objectId")
        label = group.get("label", "object")
        if obj_id is not None:
            obj_map[int(obj_id)] = label
    return obj_map


def extract_scene_geometry_from_tar(
    tar: tarfile.TarFile,
    scene_id: str,
    split: str,
    object_label_map: dict[int, str],
    save_points: bool = False,
    max_points_per_object: int = 4096,
) -> dict[str, Any] | None:
    """Extract per-object geometry from tar for a single scene.

    Args:
        tar: Open tarfile
        scene_id: Scene ID (e.g., "scene0002_00")
        split: "train", "val", or "test"
        object_label_map: objectId -> label mapping from aggregation
        save_points: Whether to save actual point coordinates
        max_points_per_object: Max points to save per object (for storage efficiency)

    Returns:
        Dict with geometry arrays, or None if extraction failed
    """
    prefix = f"./{split}/{scene_id}/"

    # Find required files in tar
    coord_info = None
    instance_info = None

    for member in tar.getmembers():
        if member.name.startswith(prefix):
            fname = member.name[len(prefix):]
            if fname == "coord.npy":
                coord_info = member
            elif fname == "instance.npy":
                instance_info = member

    if coord_info is None or instance_info is None:
        log.warning("Missing coord.npy or instance.npy for %s", scene_id)
        return None

    # Extract coord and instance arrays
    def _extract_npy(info: tarfile.TarInfo) -> np.ndarray:
        f = tar.extractfile(info)
        if f is None:
            raise IOError(f"Cannot extract {info.name}")
        return np.load(io.BytesIO(f.read()))

    coord = _extract_npy(coord_info)  # [N, 3] point coordinates
    instance = _extract_npy(instance_info)  # [N] instance labels

    if coord.shape[0] != instance.shape[0]:
        log.warning("Coord/instance mismatch for %s: %s vs %s", scene_id, coord.shape, instance.shape)
        return None

    # Get unique instance IDs (excluding 0 which is typically unlabeled)
    unique_instances = np.unique(instance)
    unique_instances = unique_instances[unique_instances > 0]

    if len(unique_instances) == 0:
        log.warning("No valid instances for %s", scene_id)
        return None

    # Extract per-object geometry
    object_ids = []
    centers = []
    sizes = []
    bboxes = []
    point_counts = []
    geometry_qualities = []
    points_list = []
    labels = []

    for inst_id in sorted(unique_instances):
        mask = instance == inst_id
        obj_points = coord[mask]

        if len(obj_points) < 3:
            continue  # Skip objects with too few points

        # Compute real bbox from points
        min_coords = obj_points.min(axis=0)
        max_coords = obj_points.max(axis=0)
        center = (min_coords + max_coords) / 2
        size = max_coords - min_coords

        # Handle degenerate sizes (flat objects)
        size = np.maximum(size, 0.01)

        bbox = np.concatenate([min_coords, max_coords])

        # Get label from aggregation if available
        label = object_label_map.get(int(inst_id), f"object_{int(inst_id)}")

        object_ids.append(int(inst_id))
        centers.append(center)
        sizes.append(size)
        bboxes.append(bbox)
        point_counts.append(len(obj_points))
        geometry_qualities.append("point_bbox")
        labels.append(label)

        if save_points:
            # Subsample if too many points
            if len(obj_points) > max_points_per_object:
                indices = np.random.choice(len(obj_points), max_points_per_object, replace=False)
                obj_points = obj_points[indices]
            points_list.append(obj_points)

    if len(object_ids) == 0:
        log.warning("No valid objects extracted for %s", scene_id)
        return None

    result = {
        "scene_id": scene_id,
        "object_ids": np.array(object_ids, dtype=np.int32),
        "labels": labels,
        "centers": np.array(centers, dtype=np.float32),
        "sizes": np.array(sizes, dtype=np.float32),
        "bboxes": np.array(bboxes, dtype=np.float32),
        "point_counts": np.array(point_counts, dtype=np.int32),
        "geometry_qualities": geometry_qualities,
        "total_points": int(coord.shape[0]),
    }

    if save_points:
        result["points_list"] = points_list

    return result


def save_geometry_npz(geometry: dict[str, Any], output_path: Path) -> None:
    """Save geometry to npz file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_dict = {
        "object_ids": geometry["object_ids"],
        "centers": geometry["centers"],
        "sizes": geometry["sizes"],
        "bboxes": geometry["bboxes"],
        "point_counts": geometry["point_counts"],
    }

    # Save labels as string array (need special handling)
    save_dict["labels"] = np.array(geometry["labels"], dtype=object)

    # Save points if available
    if "points_list" in geometry:
        for i, pts in enumerate(geometry["points_list"]):
            save_dict[f"points_{i}"] = pts

    np.savez(output_path, **save_dict)

    # Save quality flags as separate JSON (npz doesn't handle string lists well)
    meta = {
        "scene_id": geometry["scene_id"],
        "geometry_qualities": geometry["geometry_qualities"],
        "total_points": geometry["total_points"],
        "num_objects": len(geometry["object_ids"]),
    }
    meta_path = output_path.with_suffix(".json")
    with meta_path.open("w") as f:
        json.dump(meta, f, indent=2)


def determine_split(scene_id: str, tar: tarfile.TarFile) -> str | None:
    """Determine which split a scene belongs to by checking tar structure."""
    for split in ["train", "val", "test"]:
        prefix = f"./{split}/{scene_id}/"
        for member in tar.getmembers():
            if member.name.startswith(prefix):
                return split
    return None


def main():
    setup_logging()
    ap = argparse.ArgumentParser(description="Extract real geometry from Pointcept ScanNet data.")
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--nr3d-json", type=Path, default=ROOT / "data/raw/referit3d/annotations/nr3d_annotations.json")
    ap.add_argument("--scans-dir", type=Path, default=ROOT / "data/raw/referit3d/scans")
    ap.add_argument("--output-dir", type=Path, default=ROOT / "data/geometry")
    ap.add_argument("--pointcept-tar", type=Path, default=None)
    ap.add_argument("--save-points", action="store_true", help="Save actual point coordinates (larger files)")
    ap.add_argument("--max-points-per-object", type=int, default=4096)
    ap.add_argument("--max-scenes", type=int, default=None)
    ap.add_argument("--scene-ids", type=str, default="", help="Comma-separated scene IDs to process")
    args = ap.parse_args()

    # Load NR3D scenes
    if not args.nr3d_json.is_file():
        log.error("NR3D JSON not found: %s", args.nr3d_json)
        return 1

    data = json.loads(args.nr3d_json.read_text(encoding="utf-8"))
    nr3d_scenes = set(row["scene_id"] for row in data if isinstance(row, dict) and row.get("scene_id"))

    # Override with specific scenes if provided
    if args.scene_ids.strip():
        nr3d_scenes = set(s.strip() for s in args.scene_ids.split(",") if s.strip())

    if args.max_scenes:
        nr3d_scenes = set(sorted(nr3d_scenes)[:args.max_scenes])

    log.info("Processing %s scenes", len(nr3d_scenes))

    # Find Pointcept tar
    tar_path = args.pointcept_tar
    if tar_path is None:
        tar_path = find_pointcept_tar()

    if tar_path is None:
        log.error("Pointcept tar not found")
        return 1

    log.info("Using Pointcept tar: %s", tar_path)

    # Process scenes
    stats = {
        "total": len(nr3d_scenes),
        "success": 0,
        "failed": 0,
        "total_objects": 0,
        "total_points": 0,
        "scenes": [],
    }

    with tarfile.open(tar_path, "r:gz") as tar:
        for i, scene_id in enumerate(sorted(nr3d_scenes)):
            if (i + 1) % 50 == 0:
                log.info("Processing scene %s/%s: %s", i + 1, len(nr3d_scenes), scene_id)

            # Determine split
            split = determine_split(scene_id, tar)
            if split is None:
                log.warning("Scene %s not found in tar", scene_id)
                stats["failed"] += 1
                continue

            # Load aggregation for object labels
            agg_path = args.scans_dir / scene_id / f"{scene_id}.aggregation.json"
            object_label_map = {}
            if agg_path.is_file():
                object_label_map = load_aggregation_object_map(agg_path)

            # Extract geometry
            geometry = extract_scene_geometry_from_tar(
                tar,
                scene_id,
                split,
                object_label_map,
                save_points=args.save_points,
                max_points_per_object=args.max_points_per_object,
            )

            if geometry is None:
                stats["failed"] += 1
                continue

            # Save geometry
            output_path = args.output_dir / f"{scene_id}_geometry.npz"
            save_geometry_npz(geometry, output_path)

            stats["success"] += 1
            stats["total_objects"] += len(geometry["object_ids"])
            stats["total_points"] += geometry["total_points"]
            stats["scenes"].append({
                "scene_id": scene_id,
                "num_objects": len(geometry["object_ids"]),
                "total_points": geometry["total_points"],
            })

    # Save statistics
    stats_path = args.output_dir / "geometry_extraction_stats.json"
    with stats_path.open("w") as f:
        json.dump(stats, f, indent=2)

    log.info("=== Extraction Summary ===")
    log.info("Total scenes: %s", stats["total"])
    log.info("Successful: %s", stats["success"])
    log.info("Failed: %s", stats["failed"])
    log.info("Total objects: %s", stats["total_objects"])
    log.info("Total points: %s", stats["total_points"])
    log.info("Output directory: %s", args.output_dir)

    return 0 if stats["success"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())