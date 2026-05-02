"""Build manifest records from official NR3D CSV format.

Official NR3D format from https://referit3d.github.io/
Contains 41,503 samples across 641 ScanNet scenes.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


def parse_nr3d_official_row(row: dict[str, str]) -> dict[str, Any] | None:
    """Parse official Nr3D CSV format.

    Official columns:
    - assignmentid: unique sample ID
    - scan_id: ScanNet scene ID (e.g., scene0525_00)
    - utterance: natural language reference
    - target_id: target object ID within scene
    - instance_type: object class name
    - tokens: tokenized utterance (JSON list)
    - Additional metadata: correct_guess, speaker_id, listener_id, etc.
    """
    scene_id = row.get("scan_id", "").strip()
    if not scene_id:
        return None

    utterance = row.get("utterance", "").strip()
    if not utterance:
        return None

    target_oid = row.get("target_id", "").strip()
    if not target_oid:
        return None

    sample_id = row.get("assignmentid", "").strip()
    if not sample_id:
        sample_id = f"{scene_id}::{target_oid}"

    instance_type = row.get("instance_type", "object").strip()

    # Parse boolean fields
    def parse_bool(v: str) -> bool:
        return v.strip().lower() in ("true", "1", "yes")

    return {
        "scene_id": scene_id,
        "utterance": utterance,
        "target_object_id": target_oid,
        "utterance_id": sample_id,
        "instance_type": instance_type,
        "uses_spatial_lang": parse_bool(row.get("uses_spatial_lang", "False")),
        "uses_color_lang": parse_bool(row.get("uses_color_lang", "False")),
        "uses_shape_lang": parse_bool(row.get("uses_shape_lang", "False")),
        "uses_object_lang": parse_bool(row.get("uses_object_lang", "False")),
        "mentions_target_class": parse_bool(row.get("mentions_target_class", "False")),
        "correct_guess": parse_bool(row.get("correct_guess", "True")),
    }


def records_from_nr3d_official_csv(
    csv_path: Path,
    geometry_scenes: set[str] | None = None,
    max_rows: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load records from official Nr3D CSV.

    Args:
        csv_path: Path to nr3d_official.csv
        geometry_scenes: Set of scene IDs with available geometry. If None, all scenes allowed.
        max_rows: Maximum rows to process (for testing)

    Returns:
        Tuple of (records, stats)
    """
    if not csv_path.is_file():
        raise FileNotFoundError(f"Nr3D CSV not found: {csv_path}")

    records: list[dict[str, Any]] = []
    skipped_no_geometry = 0
    skipped_no_meta = 0

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if max_rows is not None and i >= max_rows:
                break

            meta = parse_nr3d_official_row(row)
            if not meta:
                skipped_no_meta += 1
                continue

            # Check geometry availability
            if geometry_scenes is not None and meta["scene_id"] not in geometry_scenes:
                skipped_no_geometry += 1
                continue

            records.append(meta)

    stats = {
        "source": "nr3d_official_csv",
        "csv_path": str(csv_path),
        "total_rows_processed": i + 1 if 'i' in dir() else 0,
        "records_kept": len(records),
        "skipped_no_meta": skipped_no_meta,
        "skipped_no_geometry": skipped_no_geometry,
        "geometry_filter_applied": geometry_scenes is not None,
        "unique_scenes": len(set(r["scene_id"] for r in records)) if records else 0,
    }

    return records, stats


def load_nr3d_official_scene_ids(csv_path: Path) -> set[str]:
    """Get set of scene IDs in official Nr3D."""
    scene_ids: set[str] = set()
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            scene_id = row.get("scan_id", "").strip()
            if scene_id:
                scene_ids.add(scene_id)
    return scene_ids