"""Build manifest records from NR3D-style JSON hosted on Hugging Face (e.g. chouss/nr3d).

These JSON files contain real NR3D referring expressions and object ids. They do **not**
include ScanNet ``*_vh_clean_aggregation.json`` geometry. For this codebase, per-object
``feature_vector`` is optional: training collates attach deterministic synthetic features
when missing (see ``attach_synthetic_features``). Centers/sizes here are placeholders so
``GroundingSample`` validates; replace with real OBBs after you place ScanNet scans.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from rag3d.datasets.schemas import GroundingSample, SceneObject
from rag3d.datasets.transforms import apply_tags_to_sample

log = logging.getLogger(__name__)

_ENTITY_RE = re.compile(r"^(\d+)_(.+)$")


def _parse_entity(token: str) -> tuple[str, str] | None:
    m = _ENTITY_RE.match(str(token).strip())
    if not m:
        return None
    return m.group(1), m.group(2).replace("_", " ")


def _placeholder_center_size(oid: str, idx: int) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    h = (hash((oid, idx)) % 10000) / 10000.0
    return (float(idx) * 0.25 + h * 0.01, h * 0.5, (h * 0.3) % 0.2), (0.2, 0.2, 0.3)


def ordered_entity_object_ids_from_nr3d_row(row: dict[str, Any]) -> list[str]:
    """Object ids appearing in ``entities``, in first-seen order, then target if not listed."""
    order: list[str] = []
    seen: set[str] = set()
    for tok in row.get("entities") or []:
        p = _parse_entity(tok)
        if not p:
            continue
        oid, _ = p
        if oid not in seen:
            seen.add(oid)
            order.append(oid)
    target_oid = str(row.get("object_id", "")).strip()
    if target_oid and target_oid not in seen:
        order.append(target_oid)
    return order


def scene_objects_entity_subset_from_aggregation(
    row: dict[str, Any],
    full_scene_objects: list[SceneObject],
) -> list[SceneObject] | None:
    """NR3D entity-only candidate set with geometry taken from aggregated scene objects.

    Returns ``None`` if any referenced id is missing from ``full_scene_objects``.
    """
    want_order = ordered_entity_object_ids_from_nr3d_row(row)
    if not want_order:
        return None
    by_id: dict[str, SceneObject] = {}
    for o in full_scene_objects:
        by_id[str(o.object_id)] = o
    out: list[SceneObject] = []
    for oid in want_order:
        o = by_id.get(oid)
        if o is None:
            return None
        out.append(o)
    return out


def scene_objects_from_nr3d_row(row: dict[str, Any]) -> list[SceneObject]:
    """Build a minimal object list: entities plus target if missing."""
    target_oid = str(row.get("object_id", "")).strip()
    object_name = str(row.get("object_name") or "object").strip()
    seen: dict[str, SceneObject] = {}
    order: list[str] = []
    for tok in row.get("entities") or []:
        p = _parse_entity(tok)
        if not p:
            continue
        oid, cname = p
        if oid not in seen:
            c, s = _placeholder_center_size(oid, len(order))
            seen[oid] = SceneObject(object_id=oid, class_name=cname, center=c, size=s)
            order.append(oid)
    if target_oid and target_oid not in seen:
        c, s = _placeholder_center_size(target_oid, len(order))
        seen[target_oid] = SceneObject(
            object_id=target_oid,
            class_name=object_name,
            center=c,
            size=s,
        )
        order.append(target_oid)
    return [seen[i] for i in order if i in seen]


def parse_nr3d_row_meta(row: dict[str, Any]) -> dict[str, str] | None:
    """Utterance + ids from an NR3D HF JSON row (no geometry). Used to merge with ScanNet aggregation."""
    scene_id = str(row.get("scene_id") or "").strip()
    if not scene_id:
        return None
    descs = row.get("descriptions")
    if not isinstance(descs, list) or not descs:
        return None
    ans = row.get("answer", 0)
    try:
        ai = int(ans)
    except (TypeError, ValueError):
        ai = 0
    if ai < 0 or ai >= len(descs):
        ai = 0
    utterance = str(descs[ai]).strip()
    if not utterance:
        return None
    target_oid = str(row.get("object_id", "")).strip()
    if not target_oid:
        return None
    uid = row.get("utterance_id") or row.get("unique_id")
    uid_s = str(uid) if uid is not None else f"{scene_id}::{target_oid}"
    return {
        "scene_id": scene_id,
        "utterance": utterance,
        "target_object_id": target_oid,
        "utterance_id": uid_s,
    }


def record_from_nr3d_row(row: dict[str, Any]) -> dict[str, Any] | None:
    meta = parse_nr3d_row_meta(row)
    if not meta:
        return None
    scene_id = meta["scene_id"]
    target_oid = meta["target_object_id"]
    objects = scene_objects_from_nr3d_row(row)
    indices = [i for i, o in enumerate(objects) if o.object_id == target_oid]
    if not indices:
        log.warning("Skipping row unique_id=%s: target %s not in object list.", row.get("unique_id"), target_oid)
        return None
    sample = GroundingSample(
        scene_id=scene_id,
        utterance=meta["utterance"],
        target_object_id=target_oid,
        target_index=indices[0],
        objects=objects,
        utterance_id=meta["utterance_id"],
        relation_type_gold=None,
        tags={},
    )
    sample = apply_tags_to_sample(sample)
    return sample.model_dump()


def records_from_nr3d_hf_json(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array in {path}")
    out: list[dict[str, Any]] = []
    for row in data:
        if not isinstance(row, dict):
            continue
        rec = record_from_nr3d_row(row)
        if rec:
            out.append(rec)
    return out
