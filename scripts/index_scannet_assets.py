#!/usr/bin/env python3
"""Index ScanNet geometry assets for NR3D baseline reproduction.

This script discovers and validates geometry assets from multiple sources:
1. Pointcept scannet.tar.gz (cached in HuggingFace hub)
2. Local aggregation JSON files
3. Any local PLY/mesh files

Outputs:
- Per-scene asset availability summary
- Missing file report
- Total scenes with usable geometry
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tarfile
from collections import defaultdict
from pathlib import Path

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
        # Match datasets--Pointcept--scannet-compressed or similar patterns
        if "pointcept" in repo_dir.name.lower() or "scannet-compressed" in repo_dir.name.lower():
            # Look in snapshots subdirectory
            snapshots_dir = repo_dir / "snapshots"
            if snapshots_dir.is_dir():
                for snapshot_dir in snapshots_dir.iterdir():
                    if snapshot_dir.is_dir():
                        tar_path = snapshot_dir / "scannet.tar.gz"
                        if tar_path.is_file():
                            return tar_path
    return None


def index_pointcept_tar(tar_path: Path, nr3d_scenes: set[str]) -> dict[str, dict]:
    """Index Pointcept tar for NR3D scenes, checking geometry assets."""
    assets = defaultdict(dict)

    required_files = ["coord.npy", "instance.npy"]
    optional_files = ["color.npy", "normal.npy", "segment20.npy", "segment200.npy"]

    with tarfile.open(tar_path, "r|gz") as tar:
        while True:
            info = tar.next()
            if info is None:
                break

            parts = info.name.strip("./").split("/")
            if len(parts) < 3 or parts[0] not in ("train", "val", "test"):
                continue

            scene_id = parts[1]
            if scene_id not in nr3d_scenes:
                continue

            filename = parts[2] if len(parts) >= 3 else ""

            # Check for required files
            if filename in required_files:
                assets[scene_id][filename] = True
            elif filename in optional_files:
                assets[scene_id][filename] = True

    # Fill missing entries
    for scene_id in nr3d_scenes:
        if scene_id not in assets:
            assets[scene_id] = {}
        for f in required_files + optional_files:
            assets[scene_id].setdefault(f, False)

    return dict(assets)


def index_local_aggregation(scans_dir: Path, nr3d_scenes: set[str]) -> dict[str, dict]:
    """Index local aggregation JSON files."""
    assets = {}

    for scene_id in nr3d_scenes:
        scene_dir = scans_dir / scene_id
        agg_files = list(scene_dir.glob("*.aggregation.json")) if scene_dir.is_dir() else []

        assets[scene_id] = {
            "aggregation_json": len(agg_files) > 0,
            "aggregation_path": str(agg_files[0]) if agg_files else None,
        }

    return assets


def merge_asset_indices(pointcept: dict, aggregation: dict) -> dict[str, dict]:
    """Merge asset indices from multiple sources."""
    merged = defaultdict(dict)

    for scene_id in pointcept:
        merged[scene_id]["pointcept"] = pointcept[scene_id]
        merged[scene_id]["has_coord"] = pointcept[scene_id].get("coord.npy", False)
        merged[scene_id]["has_instance"] = pointcept[scene_id].get("instance.npy", False)
        merged[scene_id]["has_color"] = pointcept[scene_id].get("color.npy", False)

    for scene_id in aggregation:
        merged[scene_id]["aggregation"] = aggregation[scene_id]
        merged[scene_id]["has_aggregation"] = aggregation[scene_id].get("aggregation_json", False)

    # Compute usability status
    for scene_id in merged:
        has_points = merged[scene_id].get("has_coord", False)
        has_instances = merged[scene_id].get("has_instance", False)
        has_agg = merged[scene_id].get("has_aggregation", False)

        # Can extract real geometry if we have coord + instance
        can_extract_geometry = has_points and has_instances

        merged[scene_id]["can_extract_geometry"] = can_extract_geometry
        merged[scene_id]["geometry_source"] = (
            "pointcept" if can_extract_geometry else
            "aggregation_only" if has_agg else
            "missing"
        )

    return dict(merged)


def load_nr3d_scenes(nr3d_json_path: Path) -> set[str]:
    """Load scene IDs from NR3D annotations."""
    data = json.loads(nr3d_json_path.read_text(encoding="utf-8"))
    scenes = set()
    for row in data:
        if isinstance(row, dict):
            sid = str(row.get("scene_id") or "").strip()
            if sid:
                scenes.add(sid)
    return scenes


def generate_summary(assets: dict[str, dict]) -> dict:
    """Generate summary statistics."""
    total = len(assets)
    can_extract = sum(1 for a in assets.values() if a.get("can_extract_geometry", False))
    has_aggregation = sum(1 for a in assets.values() if a.get("has_aggregation", False))
    has_coord = sum(1 for a in assets.values() if a.get("has_coord", False))
    has_instance = sum(1 for a in assets.values() if a.get("has_instance", False))
    has_color = sum(1 for a in assets.values() if a.get("has_color", False))
    missing_all = sum(1 for a in assets.values() if a.get("geometry_source") == "missing")

    return {
        "total_nr3d_scenes": total,
        "scenes_can_extract_geometry": can_extract,
        "scenes_has_aggregation": has_aggregation,
        "scenes_has_coord": has_coord,
        "scenes_has_instance": has_instance,
        "scenes_has_color": has_color,
        "scenes_missing_all_geometry": missing_all,
        "geometry_coverage_pct": round(can_extract / total * 100, 2) if total > 0 else 0,
        "missing_scene_ids": sorted([s for s, a in assets.items() if a.get("geometry_source") == "missing"]),
    }


def main():
    setup_logging()
    ap = argparse.ArgumentParser(description="Index ScanNet geometry assets for NR3D scenes.")
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--nr3d-json", type=Path, default=ROOT / "data/raw/referit3d/annotations/nr3d_annotations.json")
    ap.add_argument("--scans-dir", type=Path, default=ROOT / "data/raw/referit3d/scans")
    ap.add_argument("--output-dir", type=Path, default=ROOT / "reports")
    ap.add_argument("--pointcept-tar", type=Path, default=None, help="Override Pointcept tar path")
    args = ap.parse_args()

    # Load NR3D scenes
    if not args.nr3d_json.is_file():
        log.error("NR3D JSON not found: %s", args.nr3d_json)
        return 1

    nr3d_scenes = load_nr3d_scenes(args.nr3d_json)
    log.info("NR3D scenes needed: %s", len(nr3d_scenes))

    # Find Pointcept tar
    tar_path = args.pointcept_tar
    if tar_path is None:
        tar_path = find_pointcept_tar()

    if tar_path is None:
        log.warning("Pointcept tar not found in HuggingFace cache")
        pointcept_assets = {s: {} for s in nr3d_scenes}
    else:
        log.info("Found Pointcept tar: %s", tar_path)
        pointcept_assets = index_pointcept_tar(tar_path, nr3d_scenes)
        log.info("Indexed Pointcept assets for %s scenes", len(pointcept_assets))

    # Index local aggregation
    agg_assets = index_local_aggregation(args.scans_dir, nr3d_scenes)
    log.info("Indexed local aggregation for %s scenes", len(agg_assets))

    # Merge indices
    merged = merge_asset_indices(pointcept_assets, agg_assets)

    # Generate summary
    summary = generate_summary(merged)

    log.info("=== Geometry Asset Summary ===")
    log.info("Total NR3D scenes: %s", summary["total_nr3d_scenes"])
    log.info("Can extract geometry (coord+instance): %s (%.1f%%)",
             summary["scenes_can_extract_geometry"], summary["geometry_coverage_pct"])
    log.info("Has aggregation JSON: %s", summary["scenes_has_aggregation"])
    log.info("Missing all geometry: %s", summary["scenes_missing_all_geometry"])

    # Save outputs
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Full asset index
    index_path = args.output_dir / "scannet_asset_index.json"
    with index_path.open("w") as f:
        json.dump(merged, f, indent=2)
    log.info("Saved asset index to %s", index_path)

    # Summary
    summary_path = args.output_dir / "scannet_asset_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    log.info("Saved summary to %s", summary_path)

    # Markdown report
    md_path = args.output_dir / "scannet_asset_index.md"
    md_content = f"""# ScanNet Asset Index Report

## Summary

| Metric | Count |
|--------|-------|
| Total NR3D scenes | {summary['total_nr3d_scenes']} |
| Can extract geometry | {summary['scenes_can_extract_geometry']} ({summary['geometry_coverage_pct']}%) |
| Has aggregation JSON | {summary['scenes_has_aggregation']} |
| Has coord.npy | {summary['scenes_has_coord']} |
| Has instance.npy | {summary['scenes_has_instance']} |
| Has color.npy | {summary['scenes_has_color']} |
| Missing all geometry | {summary['scenes_missing_all_geometry']} |

## Geometry Sources

- **Pointcept tar**: Contains `coord.npy`, `instance.npy`, `color.npy`, `normal.npy` per scene
- **Local aggregation**: `{summary['scenes_has_aggregation']}` scenes have `*.aggregation.json`
- **Real geometry extraction**: Possible for `{summary['scenes_can_extract_geometry']}` scenes

## Missing Scenes

{', '.join(summary['missing_scene_ids']) if summary['missing_scene_ids'] else 'None'}

## Next Steps

1. Extract per-object geometry from Pointcept tar using `instance.npy` labels
2. Compute center/size from real point bboxes
3. Wire into `scannet_objects.py` geometry pipeline
"""
    with md_path.open("w") as f:
        f.write(md_content)
    log.info("Saved markdown report to %s", md_path)

    return 0 if summary["scenes_can_extract_geometry"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())