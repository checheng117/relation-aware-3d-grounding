"""NR3D HF JSON -> manifest record conversion."""

from rag3d.datasets.nr3d_hf import record_from_nr3d_row, records_from_nr3d_hf_json


def test_record_from_nr3d_row_basic() -> None:
    row = {
        "unique_id": 0,
        "scene_id": "scene0525_00",
        "object_id": "9",
        "object_name": "plant",
        "descriptions": ["The plant on the left.", "Wrong description."],
        "answer": 0,
        "entities": ["9_plant", "34_bookshelf", "38_window"],
        "image": "860.jpg",
    }
    rec = record_from_nr3d_row(row)
    assert rec is not None
    assert rec["scene_id"] == "scene0525_00"
    assert rec["utterance"] == "The plant on the left."
    assert rec["target_object_id"] == "9"
    assert len(rec["objects"]) == 3
    assert 0 <= rec["target_index"] < len(rec["objects"])


def test_record_adds_target_if_missing_from_entities() -> None:
    row = {
        "unique_id": 34,
        "scene_id": "scene0001_00",
        "object_id": "38",
        "object_name": "pillow",
        "descriptions": ["Pick the pillow."],
        "answer": 0,
        "entities": ["13_bed", "39_pillow"],
    }
    rec = record_from_nr3d_row(row)
    assert rec is not None
    ids = [o["object_id"] for o in rec["objects"]]
    assert "38" in ids
    assert rec["target_object_id"] == "38"
