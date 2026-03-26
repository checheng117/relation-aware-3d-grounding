"""NR3D + aggregation with entity_only candidate_set."""

from __future__ import annotations

import json
from pathlib import Path

from rag3d.datasets.builder import build_records_nr3d_hf_with_scans
from rag3d.datasets.layout import ReferIt3DRawLayout


def test_entity_only_uses_entity_subset(tmp_path: Path) -> None:
    sid = "scene_ent_00"
    scans = tmp_path / "scans" / sid
    scans.mkdir(parents=True)
    agg = {
        "segGroups": [
            {"objectId": 9, "label": "plant", "obb": {"centroid": [1.0, 0.0, 0.0], "axesLengths": [0.2, 0.2, 0.3]}},
            {"objectId": 34, "label": "bookshelf", "obb": {"centroid": [2.0, 0.0, 0.0], "axesLengths": [0.5, 0.5, 1.0]}},
            {"objectId": 77, "label": "floor", "obb": {"centroid": [0.0, 0.0, 0.0], "axesLengths": [4.0, 4.0, 0.1]}},
        ]
    }
    (scans / f"{sid}_vh_clean_aggregation.json").write_text(json.dumps(agg), encoding="utf-8")

    ann = tmp_path / "annotations"
    ann.mkdir()
    rows = [
        {
            "scene_id": sid,
            "object_id": 9,
            "object_name": "plant",
            "answer": 0,
            "descriptions": ["the plant"],
            "entities": ["9_plant", "34_bookshelf"],
            "unique_id": "u1",
        }
    ]
    jp = ann / "nr3d_annotations.json"
    jp.write_text(json.dumps(rows), encoding="utf-8")
    layout = ReferIt3DRawLayout(tmp_path, "scans", "annotations")

    recs, stats = build_records_nr3d_hf_with_scans(jp, layout, candidate_set="entity_only")
    assert len(recs) == 1
    assert stats["candidate_set"] == "entity_only"
    assert len(recs[0]["objects"]) == 2
    assert stats["skipped_entity_subset_incomplete"] == 0

    full, stats_f = build_records_nr3d_hf_with_scans(jp, layout, candidate_set="full_scene")
    assert len(full[0]["objects"]) == 3
    assert stats_f["candidate_set"] == "full_scene"


def test_entity_only_skips_missing_entity_in_scene(tmp_path: Path) -> None:
    sid = "scene_ent_01"
    scans = tmp_path / "scans" / sid
    scans.mkdir(parents=True)
    agg = {
        "segGroups": [
            {"objectId": 9, "label": "plant", "obb": {"centroid": [1.0, 0.0, 0.0], "axesLengths": [0.2, 0.2, 0.3]}},
        ]
    }
    (scans / f"{sid}_vh_clean_aggregation.json").write_text(json.dumps(agg), encoding="utf-8")
    ann = tmp_path / "annotations"
    ann.mkdir()
    rows = [
        {
            "scene_id": sid,
            "object_id": 9,
            "object_name": "plant",
            "answer": 0,
            "descriptions": ["the plant"],
            "entities": ["99_missing", "9_plant"],
            "unique_id": "u1",
        }
    ]
    jp = ann / "nr3d_annotations.json"
    jp.write_text(json.dumps(rows), encoding="utf-8")
    layout = ReferIt3DRawLayout(tmp_path, "scans", "annotations")
    recs, stats = build_records_nr3d_hf_with_scans(jp, layout, candidate_set="entity_only")
    assert recs == []
    assert stats["skipped_entity_subset_incomplete"] >= 1
