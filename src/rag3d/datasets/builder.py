"""Build processed manifests from ScanNet scans + CSV annotations or from imports / mock."""

from __future__ import annotations

import csv
import logging
from collections import Counter
from pathlib import Path
from typing import Any

from rag3d.datasets.layout import ReferIt3DRawLayout
from rag3d.datasets.nr3d_hf import parse_nr3d_row_meta, scene_objects_entity_subset_from_aggregation
from rag3d.datasets.scannet_objects import load_scene_objects_for_scene_id
from rag3d.datasets.schemas import GroundingSample, SceneObject
from rag3d.datasets.transforms import apply_tags_to_sample
from rag3d.utils.io import load_manifest_records, write_json, write_jsonl

log = logging.getLogger(__name__)


def _norm_key(row: dict[str, str], *aliases: str) -> str | None:
    keys = {k.lower().strip(): v for k, v in row.items()}
    for a in aliases:
        if a.lower() in keys and keys[a.lower()].strip():
            return keys[a.lower()].strip()
    return None


def csv_row_to_record(
    row: dict[str, str],
    objects: list[SceneObject],
    scene_id: str,
) -> dict[str, Any] | None:
    utterance = _norm_key(row, "utterance", "utterance_text", "text", "description")
    target_oid = _norm_key(row, "target_object_id", "target_id", "object_id", "target")
    if not utterance or not target_oid:
        return None
    uid = _norm_key(row, "utterance_id", "id", "sample_id")
    rel = _norm_key(row, "relation_type_gold", "relation", "relation_type")
    indices = [i for i, o in enumerate(objects) if str(o.object_id) == str(target_oid)]
    if not indices:
        log.warning("Target id %s not in scene %s objects; skipping row.", target_oid, scene_id)
        return None
    target_index = indices[0]
    sample = GroundingSample(
        scene_id=scene_id,
        utterance=utterance,
        target_object_id=str(target_oid),
        target_index=target_index,
        objects=objects,
        utterance_id=uid or f"{scene_id}::{abs(hash(utterance)) % (10**9)}",
        relation_type_gold=rel,
        tags={},
    )
    return apply_tags_to_sample(sample).model_dump()


def build_records_from_csv_and_scans(
    csv_path: Path,
    layout: ReferIt3DRawLayout,
    max_rows: int | None = None,
    max_scenes: int | None = None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    scenes_seen: set[str] = set()
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if max_rows is not None and len(records) >= max_rows:
                break
            sid = _norm_key(row, "scene_id", "scan_id", "scene")
            if not sid:
                continue
            if max_scenes is not None and sid not in scenes_seen and len(scenes_seen) >= max_scenes:
                continue
            try:
                objects = load_scene_objects_for_scene_id(layout.scans_dir, sid)
            except FileNotFoundError:
                log.warning("No aggregation for scene %s — skip row.", sid)
                continue
            rec = csv_row_to_record(row, objects, sid)
            if rec:
                records.append(rec)
                scenes_seen.add(sid)
    return records


def build_records_nr3d_hf_with_scans(
    nr3d_json_path: Path,
    layout: ReferIt3DRawLayout,
    max_rows: int | None = None,
    max_scenes: int | None = None,
    candidate_set: str = "full_scene",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """NR3D HF JSON + ScanNet-style aggregation per scene.

    ``candidate_set``:
    - ``full_scene``: all instances from aggregation (hard retrieval).
    - ``entity_only``: only NR3D ``entities`` (+ target if absent), geometry from aggregation rows.
    """
    import json

    if candidate_set not in {"full_scene", "entity_only"}:
        raise ValueError(f"candidate_set must be 'full_scene' or 'entity_only', got {candidate_set!r}")

    data = json.loads(nr3d_json_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array in {nr3d_json_path}")
    records: list[dict[str, Any]] = []
    scenes_seen: set[str] = set()
    skipped_no_meta = 0
    skipped_no_agg = 0
    skipped_no_target = 0
    skipped_entity_subset = 0
    for row in data:
        if not isinstance(row, dict):
            continue
        if max_rows is not None and len(records) >= max_rows:
            break
        meta = parse_nr3d_row_meta(row)
        if not meta:
            skipped_no_meta += 1
            continue
        sid = meta["scene_id"]
        if max_scenes is not None and sid not in scenes_seen and len(scenes_seen) >= max_scenes:
            continue
        try:
            full_objects = load_scene_objects_for_scene_id(layout.scans_dir, sid)
        except FileNotFoundError:
            skipped_no_agg += 1
            continue
        if candidate_set == "full_scene":
            objects = full_objects
        else:
            subset = scene_objects_entity_subset_from_aggregation(row, full_objects)
            if subset is None:
                skipped_entity_subset += 1
                continue
            objects = subset
        csv_like = {
            "scene_id": sid,
            "utterance": meta["utterance"],
            "target_object_id": meta["target_object_id"],
            "utterance_id": meta["utterance_id"],
        }
        rec = csv_row_to_record(csv_like, objects, sid)
        if rec:
            records.append(rec)
            scenes_seen.add(sid)
        else:
            skipped_no_target += 1
    stats = {
        "nr3d_rows_total": len(data),
        "records_kept": len(records),
        "unique_scenes_used": len(scenes_seen),
        "skipped_no_meta": skipped_no_meta,
        "skipped_no_aggregation": skipped_no_agg,
        "skipped_target_not_in_scene": skipped_no_target,
        "skipped_entity_subset_incomplete": skipped_entity_subset,
        "geometry_backed": True,
        "candidate_set": candidate_set,
    }
    return records, stats


def import_jsonl_records(path: Path, max_rows: int | None = None) -> list[dict[str, Any]]:
    raw = load_manifest_records(path)
    if max_rows is not None:
        raw = raw[:max_rows]
    out: list[dict[str, Any]] = []
    for r in raw:
        sample = GroundingSample.model_validate(r)
        sample = apply_tags_to_sample(sample)
        out.append(sample.model_dump())
    return out


def mock_debug_records(
    n_total: int = 64,
    n_scenes: int = 8,
    feat_dim: int = 256,
    objects_per_scene: int = 6,
) -> list[dict[str, Any]]:
    """Synthetic manifest records for fast iteration (no ScanNet files required)."""
    import numpy as np

    rng = np.random.default_rng(42)
    all_recs: list[dict[str, Any]] = []
    per = max(1, n_total // max(1, n_scenes))
    for si in range(n_scenes):
        scene_id = f"debug_scene_{si:04d}"
        for _ in range(per):
            if len(all_recs) >= n_total:
                break
            objs: list[SceneObject] = []
            for j in range(objects_per_scene):
                v = rng.standard_normal(feat_dim).astype(float)
                v = (v / (np.linalg.norm(v) + 1e-8)).tolist()
                objs.append(
                    SceneObject(
                        object_id=f"{si}_obj_{j}",
                        class_name="chair" if j % 3 != 1 else "table",
                        center=(float(j) * 0.2, 0.0, 0.0),
                        size=(0.4, 0.4, 0.8),
                        visibility_occlusion_proxy=0.95 - 0.02 * j,
                        feature_vector=v,
                    )
                )
            ti = int(rng.integers(0, objects_per_scene))
            utt = "Pick the chair left of the table near the wall."
            sample = GroundingSample(
                scene_id=scene_id,
                utterance=utt,
                target_object_id=objs[ti].object_id,
                target_index=ti,
                objects=objs,
                relation_type_gold="left-of",
                tags={},
            )
            sample = apply_tags_to_sample(sample)
            all_recs.append(sample.model_dump())
        if len(all_recs) >= n_total:
            break
    return all_recs[:n_total]


def build_from_combined_split_csv(
    csv_path: Path,
    layout: ReferIt3DRawLayout,
    max_rows: int | None = None,
    max_scenes: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Single CSV with `split` column: train | val | test."""
    train_r: list[dict[str, Any]] = []
    val_r: list[dict[str, Any]] = []
    test_r: list[dict[str, Any]] = []
    scenes_per: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    total = 0

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if max_rows is not None and total >= max_rows:
                break
            sp = (_norm_key(row, "split", "set", "partition") or "train").lower()
            if sp not in {"train", "val", "test"}:
                sp = "train"
            bucket = train_r if sp == "train" else val_r if sp == "val" else test_r
            sid = _norm_key(row, "scene_id", "scan_id", "scene")
            if not sid:
                continue
            if max_scenes is not None and sid not in scenes_per[sp] and len(scenes_per[sp]) >= max_scenes:
                continue
            try:
                objects = load_scene_objects_for_scene_id(layout.scans_dir, sid)
            except FileNotFoundError:
                log.warning("No aggregation for scene %s — skip.", sid)
                continue
            rec = csv_row_to_record(row, objects, sid)
            if rec:
                bucket.append(rec)
                scenes_per[sp].add(sid)
                total += 1
    return train_r, val_r, test_r


def split_records(
    records: list[dict[str, Any]],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    import random

    rng = random.Random(seed)
    idx = list(range(len(records)))
    rng.shuffle(idx)
    n = len(idx)
    n_tr = int(n * train_ratio)
    n_va = int(n * val_ratio)
    tr = [records[i] for i in idx[:n_tr]]
    va = [records[i] for i in idx[n_tr : n_tr + n_va]]
    te = [records[i] for i in idx[n_tr + n_va :]]
    return tr, va, te


def summarize_manifests(
    splits: dict[str, list[dict[str, Any]]],
    source: str,
) -> dict[str, Any]:
    hist: Counter[str] = Counter()
    scenes: set[str] = set()
    for split_name, rows in splits.items():
        hist[split_name] = len(rows)
        for r in rows:
            scenes.add(str(r["scene_id"]))
            rt = r.get("relation_type_gold") or "none"
            hist[f"rel::{rt}"] += 1
        for r in rows:
            for k, v in (r.get("tags") or {}).items():
                if v:
                    hist[f"tag::{k}"] += 1
    return {
        "source": source,
        "counts_by_split": {k: len(v) for k, v in splits.items()},
        "unique_scenes": len(scenes),
        "histogram": dict(hist),
    }


def write_split_manifests(
    out_dir: Path,
    train: list[dict[str, Any]],
    val: list[dict[str, Any]],
    test: list[dict[str, Any]],
    summary_extra: dict[str, Any] | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "train_manifest.jsonl", train)
    write_jsonl(out_dir / "val_manifest.jsonl", val)
    write_jsonl(out_dir / "test_manifest.jsonl", test)
    extra = summary_extra or {}
    summ = summarize_manifests({"train": train, "val": val, "test": test}, source=str(extra.get("source", "unknown")))
    summ.update(extra)
    write_json(out_dir / "dataset_summary.json", summ)
