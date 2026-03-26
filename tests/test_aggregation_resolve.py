"""Aggregation path resolution (ScanNet filename variants)."""

from __future__ import annotations

import json
from pathlib import Path

from rag3d.datasets.scannet_objects import load_scene_objects_for_scene_id, resolve_aggregation_path


def test_resolve_canonical(tmp_path: Path) -> None:
    sid = "scene0000_00"
    d = tmp_path / sid
    d.mkdir()
    p = d / f"{sid}_vh_clean_aggregation.json"
    p.write_text("{}", encoding="utf-8")
    assert resolve_aggregation_path(d, sid) == p


def test_resolve_dot_form(tmp_path: Path) -> None:
    sid = "scene0525_00"
    d = tmp_path / sid
    d.mkdir()
    p = d / f"{sid}_vh_clean.aggregation.json"
    p.write_text("{}", encoding="utf-8")
    assert resolve_aggregation_path(d, sid) == p


def test_load_scene_minimal_seg_groups(tmp_path: Path) -> None:
    sid = "scene_test_00"
    d = tmp_path / sid
    d.mkdir()
    agg = {
        "segGroups": [
            {
                "objectId": 9,
                "label": "plant",
                "obb": {"centroid": [1.0, 2.0, 3.0], "axesLengths": [0.4, 0.4, 0.5]},
            }
        ]
    }
    (d / f"{sid}_vh_clean_aggregation.json").write_text(json.dumps(agg), encoding="utf-8")
    objs = load_scene_objects_for_scene_id(tmp_path, sid)
    assert len(objs) == 1
    assert objs[0].object_id == "9"
