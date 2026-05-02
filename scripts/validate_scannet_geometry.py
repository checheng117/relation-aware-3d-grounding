#!/usr/bin/env python3
"""Validate extracted ScanNet geometry quality.

Checks:
- Non-zero object count per scene
- Non-default centers (not all 0,0,0)
- Non-default sizes (not all 0.1,0.1,0.1)
- Reasonable center/size ranges
- Point count distribution
- Proportion of real geometry vs fallback
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

log = logging.getLogger(__name__)


def validate_scene_geometry(npz_path: Path) -> dict[str, Any]:
    """Validate a single scene's geometry file."""
    result = {
        "scene_id": npz_path.stem.replace("_geometry", ""),
        "valid": True,
        "issues": [],
    }

    try:
        data = np.load(npz_path, allow_pickle=True)

        # Check object count
        object_ids = data["object_ids"]
        num_objects = len(object_ids)
        result["num_objects"] = num_objects

        if num_objects == 0:
            result["valid"] = False
            result["issues"].append("zero_objects")
            return result

        # Check centers
        centers = data["centers"]
        default_center_count = np.sum(np.all(centers == 0.0, axis=1))
        result["default_center_count"] = int(default_center_count)

        if default_center_count > 0:
            result["issues"].append(f"default_centers:{default_center_count}")

        # Check sizes
        sizes = data["sizes"]
        default_size = np.array([0.1, 0.1, 0.1])
        default_size_count = np.sum(np.all(np.isclose(sizes, default_size, atol=0.01), axis=1))
        result["default_size_count"] = int(default_size_count)

        if default_size_count > 0:
            result["issues"].append(f"default_sizes:{default_size_count}")

        # Check ranges
        center_ranges = {
            "min": centers.min(axis=0).tolist(),
            "max": centers.max(axis=0).tolist(),
        }
        size_ranges = {
            "min": sizes.min(axis=0).tolist(),
            "max": sizes.max(axis=0).tolist(),
        }
        result["center_ranges"] = center_ranges
        result["size_ranges"] = size_ranges

        # Check for degenerate sizes
        min_size = sizes.min(axis=1)
        if np.any(min_size < 0.01):
            degenerate_count = np.sum(min_size < 0.01)
            result["issues"].append(f"degenerate_sizes:{degenerate_count}")

        # Check point counts
        if "point_counts" in data:
            point_counts = data["point_counts"]
            result["total_points"] = int(point_counts.sum())
            result["avg_points_per_object"] = float(point_counts.mean())
            result["min_points"] = int(point_counts.min())
            result["max_points"] = int(point_counts.max())

            if np.any(point_counts < 10):
                low_point_count = np.sum(point_counts < 10)
                result["issues"].append(f"low_point_objects:{low_point_count}")

        # Load metadata for quality check
        meta_path = npz_path.with_suffix(".json")
        if meta_path.is_file():
            with meta_path.open() as f:
                meta = json.load(f)
            result["geometry_qualities"] = meta.get("geometry_qualities", [])

            # Count fallback geometry
            fallback_count = sum(1 for q in meta.get("geometry_qualities", []) if "fallback" in q.lower())
            result["fallback_geometry_count"] = fallback_count

            if fallback_count > 0:
                result["issues"].append(f"fallback_geometry:{fallback_count}")

    except Exception as e:
        result["valid"] = False
        result["issues"].append(f"error:{str(e)}")

    return result


def main():
    setup_logging()
    ap = argparse.ArgumentParser(description="Validate extracted ScanNet geometry.")
    ap.add_argument("--geometry-dir", type=Path, default=ROOT / "data/geometry")
    ap.add_argument("--output-dir", type=Path, default=ROOT / "reports")
    ap.add_argument("--nr3d-json", type=Path, default=ROOT / "data/raw/referit3d/annotations/nr3d_annotations.json")
    args = ap.parse_args()

    if not args.geometry_dir.is_dir():
        log.error("Geometry directory not found: %s", args.geometry_dir)
        return 1

    # Load NR3D scenes for coverage check
    nr3d_scenes = set()
    if args.nr3d_json.is_file():
        data = json.loads(args.nr3d_json.read_text(encoding="utf-8"))
        nr3d_scenes = set(row["scene_id"] for row in data if isinstance(row, dict) and row.get("scene_id"))

    # Find all geometry files
    geom_files = sorted(args.geometry_dir.glob("*_geometry.npz"))
    log.info("Found %s geometry files", len(geom_files))

    # Validate each file
    results = []
    stats = {
        "total_files": len(geom_files),
        "valid_files": 0,
        "invalid_files": 0,
        "total_objects": 0,
        "total_points": 0,
        "default_center_objects": 0,
        "default_size_objects": 0,
        "fallback_geometry_objects": 0,
        "issues_histogram": {},
    }

    for gf in geom_files:
        result = validate_scene_geometry(gf)
        results.append(result)

        if result["valid"]:
            stats["valid_files"] += 1
            stats["total_objects"] += result.get("num_objects", 0)
            stats["total_points"] += result.get("total_points", 0)
            stats["default_center_objects"] += result.get("default_center_count", 0)
            stats["default_size_objects"] += result.get("default_size_count", 0)
            stats["fallback_geometry_objects"] += result.get("fallback_geometry_count", 0)
        else:
            stats["invalid_files"] += 1

        for issue in result.get("issues", []):
            key = issue.split(":")[0] if ":" in issue else issue
            stats["issues_histogram"][key] = stats["issues_histogram"].get(key, 0) + 1

    # Check coverage
    geom_scenes = set(r["scene_id"] for r in results)
    missing_scenes = nr3d_scenes - geom_scenes
    stats["nr3d_coverage"] = len(geom_scenes & nr3d_scenes)
    stats["nr3d_total"] = len(nr3d_scenes)
    stats["missing_scenes"] = sorted(missing_scenes)

    # Calculate geometry quality
    total_objects = stats["total_objects"]
    real_geometry_objects = total_objects - stats["fallback_geometry_objects"]
    stats["real_geometry_pct"] = round(real_geometry_objects / total_objects * 100, 2) if total_objects > 0 else 0

    # Save results
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Detailed results
    results_path = args.output_dir / "geometry_validation_results.json"
    with results_path.open("w") as f:
        json.dump(results, f, indent=2)

    # Summary
    summary_path = args.output_dir / "geometry_validation_summary.json"
    with summary_path.open("w") as f:
        json.dump(stats, f, indent=2)

    # Markdown report
    md_content = f"""# ScanNet Geometry Validation Report

## Summary

| Metric | Value |
|--------|-------|
| Total geometry files | {stats['total_files']} |
| Valid files | {stats['valid_files']} |
| Invalid files | {stats['invalid_files']} |
| NR3D coverage | {stats['nr3d_coverage']}/{stats['nr3d_total']} |
| Total objects | {stats['total_objects']} |
| Total points | {stats['total_points']:,} |

## Geometry Quality

| Metric | Count |
|--------|-------|
| Objects with real geometry | {real_geometry_objects} ({stats['real_geometry_pct']}%) |
| Objects with fallback geometry | {stats['fallback_geometry_objects']} |
| Objects with default centers | {stats['default_center_objects']} |
| Objects with default sizes | {stats['default_size_objects']} |

## Issues Histogram

| Issue | Count |
|-------|-------|
"""
    for issue, count in sorted(stats["issues_histogram"].items()):
        md_content += f"| {issue} | {count} |\n"

    if stats["missing_scenes"]:
        md_content += f"\n## Missing Scenes ({len(stats['missing_scenes'])})\n\n"
        md_content += ", ".join(stats["missing_scenes"][:50])
        if len(stats["missing_scenes"]) > 50:
            md_content += f"\n... and {len(stats['missing_scenes']) - 50} more"

    md_content += """

## Conclusion

"""
    if stats["real_geometry_pct"] >= 99:
        md_content += "✅ **Geometry extraction successful**: 99%+ objects have real geometry.\n"
    elif stats["real_geometry_pct"] >= 90:
        md_content += "⚠️ **Partial success**: 90%+ objects have real geometry, but some fallbacks exist.\n"
    else:
        md_content += "❌ **Issues detected**: Significant fallback geometry present.\n"

    md_path = args.output_dir / "geometry_validation_summary.md"
    with md_path.open("w") as f:
        f.write(md_content)

    # Print summary
    log.info("=== Validation Summary ===")
    log.info("Valid files: %s/%s", stats["valid_files"], stats["total_files"])
    log.info("Total objects: %s", stats["total_objects"])
    log.info("Total points: %s", stats["total_points"])
    log.info("Real geometry: %.1f%%", stats["real_geometry_pct"])
    log.info("Default centers: %s", stats["default_center_objects"])
    log.info("Default sizes: %s", stats["default_size_objects"])

    return 0 if stats["valid_files"] == stats["total_files"] and stats["real_geometry_pct"] >= 95 else 1


if __name__ == "__main__":
    raise SystemExit(main())