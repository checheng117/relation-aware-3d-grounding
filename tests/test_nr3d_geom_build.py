"""NR3D JSON + ScanNet aggregation merge."""

from __future__ import annotations

import json
from pathlib import Path

from rag3d.datasets.builder import build_records_nr3d_hf_with_scans
from rag3d.datasets.layout import ReferIt3DRawLayout


def test_build_nr3d_hf_with_scans_merges_obb(tmp_path: Path) -> None:
    sid = "scene_geom_test_00"
    scans = tmp_path / "scans" / sid
    scans.mkdir(parents=True)
    agg = {
        "segGroups": [
            {
                "objectId": 9,
                "label": "plant",
                "obb": {"centroid": [1.0, 0.0, 0.0], "axesLengths": [0.2, 0.2, 0.3]},
            },
            {
                "objectId": 34,
                "label": "bookshelf",
                "obb": {"centroid": [2.0, 0.0, 0.0], "axesLengths": [0.5, 0.5, 1.0]},
            },
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
    recs, stats = build_records_nr3d_hf_with_scans(jp, layout)
    assert len(recs) == 1
    assert stats["records_kept"] == 1
    assert stats["geometry_backed"] is True
    assert len(recs[0]["objects"]) == 2
    assert tuple(recs[0]["objects"][0]["center"]) == (1.0, 0.0, 0.0)
