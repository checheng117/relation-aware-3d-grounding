#!/usr/bin/env python3
"""Run COVER-3D coverage diagnostics without training.

This script is the fixed entry point for the first evidence loop after the
claim freeze. It analyzes one trusted backbone's predictions and scene/object
metadata to test whether annotated anchors are missed by sparse relation
selection proxies.

The sparse proxy used here is target-centric nearest-neighbor anchor selection:
for each sample, candidate anchors are ranked by Euclidean distance from the
target object center. This is intentionally a diagnostic proxy, not a trained
relation model. Dense reachability means all annotated anchors with recovered
geometry are reachable by all-pair candidate-anchor scoring.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MANIFEST = PROJECT_ROOT / "data/processed/scene_disjoint/official_scene_disjoint/test_manifest.jsonl"
DEFAULT_PREDICTIONS = PROJECT_ROOT / "outputs/20260409_learned_class_embedding/formal/eval_test_predictions.json"
DEFAULT_ANNOTATIONS = PROJECT_ROOT / "data/raw/referit3d/annotations/nr3d_annotations.json"
DEFAULT_GEOMETRY_DIR = PROJECT_ROOT / "data/geometry"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports/cover3d_coverage_diagnostics"

K_VALUES = (1, 3, 5, 10)


DIRECTIONAL_KEYWORDS = (
    "left of", "right of", "front of", "behind", "left-hand", "right-hand",
    "left side", "right side", "far left", "far right", "leftmost", "rightmost",
    "closest to", "farthest from", "nearest", "furthest",
)
SUPPORT_KEYWORDS = (
    "on the table", "on the desk", "on the shelf", "on the floor",
    "on the bed", "on the chair", "on the cabinet", "on the counter",
    "sitting on", "standing on", "lying on", "placed on", "resting on",
)
CONTAINER_KEYWORDS = (
    "in the cabinet", "in the box", "in the drawer", "in the closet",
    "inside", "within", "enclosed",
)
BETWEEN_KEYWORDS = ("between", "in between")
RELATIVE_KEYWORDS = (
    "next to", "near", "by", "beside", "adjacent", "against",
    "touching", "facing", "opposite",
)
SIZE_KEYWORDS = (
    "largest", "smallest", "biggest", "tallest", "shortest",
    "medium-sized", "large", "small", "big", "tiny",
)
COLOR_KEYWORDS = (
    "white", "black", "red", "blue", "green", "yellow", "brown",
    "gray", "grey", "orange", "pink", "purple",
)


@dataclass(frozen=True)
class GeometryScene:
    centers: dict[str, list[float]]
    labels: dict[str, str]


def norm_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().strip("'\"").lower())


def analyze_utterance(utterance: str) -> dict[str, Any]:
    text = norm_text(utterance)
    result = {
        "has_directional": any(kw in text for kw in DIRECTIONAL_KEYWORDS),
        "has_support": any(kw in text for kw in SUPPORT_KEYWORDS),
        "has_container": any(kw in text for kw in CONTAINER_KEYWORDS),
        "has_between": any(kw in text for kw in BETWEEN_KEYWORDS),
        "has_relative": any(kw in text for kw in RELATIVE_KEYWORDS),
        "has_size": any(kw in text for kw in SIZE_KEYWORDS),
        "has_color": any(kw in text for kw in COLOR_KEYWORDS),
    }
    if result["has_between"]:
        relation_type = "between"
    elif result["has_directional"]:
        relation_type = "directional"
    elif result["has_relative"]:
        relation_type = "relative"
    elif result["has_support"]:
        relation_type = "support"
    elif result["has_container"]:
        relation_type = "container"
    elif result["has_size"] or result["has_color"]:
        relation_type = "attribute"
    else:
        relation_type = "other"
    result["relation_type"] = relation_type
    result["is_relational"] = relation_type in {"between", "directional", "relative", "support", "container"}
    return result


def parse_entity(entity: str) -> dict[str, str]:
    parts = entity.split("_")
    if len(parts) < 2:
        return {"object_id": entity, "class_name": "unknown"}
    return {"object_id": parts[0], "class_name": "_".join(parts[1:])}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def load_annotation_indexes(path: Path) -> tuple[dict[tuple[str, str, str], dict[str, Any]], dict[tuple[str, str], list[dict[str, Any]]]]:
    annotations = json.load(path.open())
    exact: dict[tuple[str, str, str], dict[str, Any]] = {}
    by_target: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for ann in annotations:
        scene_id = str(ann["scene_id"])
        object_id = str(ann["object_id"])
        by_target[(scene_id, object_id)].append(ann)
        for desc in ann.get("descriptions", []):
            exact[(scene_id, object_id, norm_text(desc))] = ann
    return exact, by_target


def find_annotation(
    sample: dict[str, Any],
    exact: dict[tuple[str, str, str], dict[str, Any]],
    by_target: dict[tuple[str, str], list[dict[str, Any]]],
) -> tuple[dict[str, Any] | None, str]:
    scene_id = str(sample["scene_id"])
    target_id = str(sample["target_object_id"])
    text = norm_text(sample["utterance"])
    ann = exact.get((scene_id, target_id, text))
    if ann is not None:
        return ann, "exact_utterance"
    candidates = by_target.get((scene_id, target_id), [])
    if candidates:
        return candidates[0], "target_fallback"
    return None, "none"


def extract_anchor_entities(annotation: dict[str, Any] | None, target_id: str) -> list[dict[str, str]]:
    if not annotation:
        return []
    entities = annotation.get("entities", [])
    if len(entities) < 2:
        return []
    anchors = []
    seen = set()
    for ent in entities[1:]:
        parsed = parse_entity(ent)
        object_id = str(parsed["object_id"])
        if object_id == str(target_id) or object_id in seen:
            continue
        seen.add(object_id)
        anchors.append(parsed)
    return anchors


def load_geometry_scene(scene_id: str, geometry_dir: Path) -> GeometryScene | None:
    path = geometry_dir / f"{scene_id}_geometry.npz"
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=True)
    object_ids = [str(int(x)) for x in data["object_ids"]]
    centers_array = data["centers"]
    labels_array = data["labels"] if "labels" in data.files else np.array(["unknown"] * len(object_ids))
    centers = {obj_id: [float(v) for v in center] for obj_id, center in zip(object_ids, centers_array)}
    labels = {obj_id: str(label) for obj_id, label in zip(object_ids, labels_array)}
    return GeometryScene(centers=centers, labels=labels)


def euclidean(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def rank_anchors_by_target_distance(
    target_id: str,
    scene_geometry: GeometryScene | None,
) -> tuple[list[str], dict[str, int], dict[str, float]]:
    if scene_geometry is None or target_id not in scene_geometry.centers:
        return [], {}, {}
    target_center = scene_geometry.centers[target_id]
    rows = []
    for object_id, center in scene_geometry.centers.items():
        if object_id == target_id:
            continue
        rows.append((euclidean(target_center, center), object_id))
    rows.sort(key=lambda x: (x[0], int(x[1]) if x[1].isdigit() else x[1]))
    ranked = [object_id for _, object_id in rows]
    ranks = {object_id: i + 1 for i, object_id in enumerate(ranked)}
    distances = {object_id: float(distance) for distance, object_id in rows}
    return ranked, ranks, distances


def target_class_and_same_class_count(sample: dict[str, Any]) -> tuple[str | None, int]:
    target_id = str(sample["target_object_id"])
    target_class = None
    for obj in sample["objects"]:
        if str(obj["object_id"]) == target_id:
            target_class = obj.get("class_name")
            break
    if target_class is None:
        return None, 1
    return target_class, sum(1 for obj in sample["objects"] if obj.get("class_name") == target_class)


def subset_flags(stats: dict[str, Any]) -> dict[str, bool]:
    return {
        "all": True,
        "same_class_clutter": stats["same_class_count"] >= 3,
        "same_class_high_clutter": stats["same_class_count"] >= 5,
        "unique_class": stats["same_class_count"] == 1,
        "multi_anchor": stats["anchor_count"] >= 2,
        "single_anchor": stats["anchor_count"] == 1,
        "relative_position": stats["relation_type"] == "relative",
        "directional": stats["relation_type"] == "directional",
        "between": stats["relation_type"] == "between",
        "relational": stats["is_relational"],
        "dense_scene": stats["n_objects"] >= 50,
        "baseline_correct": stats["correct_at_1"],
        "baseline_wrong": not stats["correct_at_1"],
    }


def pct(num: int | float, den: int | float) -> float | None:
    if den == 0:
        return None
    return round(float(num) / float(den) * 100.0, 2)


def summarize_subset(samples: list[dict[str, Any]], predicate: Callable[[dict[str, Any]], bool]) -> dict[str, Any]:
    subset = [s for s in samples if predicate(s)]
    anchor_subset = [s for s in subset if s["anchor_count"] > 0]
    geom_subset = [s for s in anchor_subset if s["geometry_valid"] and s["anchor_geometry_count"] > 0]
    out: dict[str, Any] = {
        "count": len(subset),
        "pct_of_total": pct(len(subset), len(samples)),
        "baseline_acc_at_1": pct(sum(s["correct_at_1"] for s in subset), len(subset)),
        "baseline_acc_at_5": pct(sum(s["correct_at_5"] for s in subset), len(subset)),
        "anchor_annotated_count": len(anchor_subset),
        "geometry_evaluable_count": len(geom_subset),
        "dense_any_reachability": pct(sum(s["dense_any_reachable"] for s in geom_subset), len(geom_subset)),
        "dense_all_reachability": pct(sum(s["dense_all_reachable"] for s in geom_subset), len(geom_subset)),
        "mean_min_anchor_rank": None,
        "median_min_anchor_rank": None,
        "mean_anchor_rank": None,
    }
    min_ranks = [s["min_anchor_rank"] for s in geom_subset if s["min_anchor_rank"] is not None]
    all_ranks = [rank for s in geom_subset for rank in s["anchor_ranks"]]
    if min_ranks:
        out["mean_min_anchor_rank"] = round(sum(min_ranks) / len(min_ranks), 2)
        out["median_min_anchor_rank"] = round(sorted(min_ranks)[len(min_ranks) // 2], 2)
    if all_ranks:
        out["mean_anchor_rank"] = round(sum(all_ranks) / len(all_ranks), 2)
    for k in K_VALUES:
        out[f"any_coverage@{k}"] = pct(sum(s["any_anchor_covered_at"][str(k)] for s in geom_subset), len(geom_subset))
        out[f"all_coverage@{k}"] = pct(sum(s["all_anchors_covered_at"][str(k)] for s in geom_subset), len(geom_subset))
    return out


def make_casebook(samples: list[dict[str, Any]], k: int, limit: int) -> list[dict[str, Any]]:
    candidates = [
        s for s in samples
        if s["anchor_count"] > 0
        and s["geometry_valid"]
        and s["anchor_geometry_count"] > 0
        and not s["any_anchor_covered_at"][str(k)]
    ]
    candidates.sort(
        key=lambda s: (
            s["correct_at_1"],
            not s["is_multi_anchor"],
            -(s["same_class_count"]),
            s["min_anchor_rank"] if s["min_anchor_rank"] is not None else 9999,
        )
    )
    rows = []
    for s in candidates[:limit]:
        rows.append({
            "scene_id": s["scene_id"],
            "utterance": s["utterance"],
            "target_id": s["target_id"],
            "target_class": s["target_class"],
            "pred_top1": s["pred_top1"],
            "pred_top5": s["pred_top5"],
            "correct_at_1": s["correct_at_1"],
            "same_class_count": s["same_class_count"],
            "relation_type": s["relation_type"],
            "anchor_ids": s["anchor_ids"],
            "anchor_classes": s["anchor_classes"],
            "anchor_ranks": s["anchor_ranks"],
            "anchor_distances": s["anchor_distances"],
            "min_anchor_rank": s["min_anchor_rank"],
            "nearest_sparse_candidates": s["nearest_sparse_candidates"],
            "annotation_match": s["annotation_match"],
            "subset_tags": [name for name, value in s["subsets"].items() if value and name != "all"],
        })
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_subset_curve_csv(path: Path, subset_summary: dict[str, dict[str, Any]]) -> None:
    rows = []
    for subset, summary in subset_summary.items():
        for k in K_VALUES:
            rows.append({
                "subset": subset,
                "k": k,
                "count": summary["count"],
                "geometry_evaluable_count": summary["geometry_evaluable_count"],
                "baseline_acc_at_1": summary["baseline_acc_at_1"],
                "any_coverage": summary[f"any_coverage@{k}"],
                "all_coverage": summary[f"all_coverage@{k}"],
                "dense_any_reachability": summary["dense_any_reachability"],
                "dense_all_reachability": summary["dense_all_reachability"],
            })
    write_csv(path, rows)


def write_anchor_rank_histogram_csv(path: Path, samples: list[dict[str, Any]]) -> None:
    bins = [
        ("1", lambda r: r == 1),
        ("2-3", lambda r: 2 <= r <= 3),
        ("4-5", lambda r: 4 <= r <= 5),
        ("6-10", lambda r: 6 <= r <= 10),
        ("11-20", lambda r: 11 <= r <= 20),
        (">20", lambda r: r > 20),
    ]
    min_ranks = [s["min_anchor_rank"] for s in samples if s["min_anchor_rank"] is not None]
    all_ranks = [rank for s in samples for rank in s["anchor_ranks"]]
    rows = []
    for label, predicate in bins:
        min_count = sum(1 for rank in min_ranks if predicate(rank))
        all_count = sum(1 for rank in all_ranks if predicate(rank))
        rows.append({
            "rank_bin": label,
            "min_anchor_count": min_count,
            "min_anchor_pct": pct(min_count, len(min_ranks)),
            "all_anchor_count": all_count,
            "all_anchor_pct": pct(all_count, len(all_ranks)),
        })
    write_csv(path, rows)


def format_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}%"


def write_report(
    path: Path,
    summary: dict[str, Any],
    subset_summary: dict[str, dict[str, Any]],
    casebook: list[dict[str, Any]],
) -> None:
    all_summary = subset_summary["all"]
    lines = [
        "# COVER-3D Coverage Diagnostics Report",
        "",
        "**Date**: 2026-04-19",
        "**Training**: none. This report uses frozen ReferIt3DNet baseline predictions.",
        "",
        "## Diagnostic Definition",
        "",
        "This is the first direct evidence pass for the COVER-3D coverage hypothesis.",
        "Sparse coverage is measured with a target-centric nearest-neighbor anchor proxy: candidate anchors are ranked by Euclidean distance from the target object center. Dense reachability means all annotated anchors with recovered geometry are available to an all-pair candidate-anchor scorer.",
        "",
        "This does not prove a trained sparse relation model would rank anchors identically. It directly tests the common assumption that useful anchors are local/nearby enough for a sparse shortlist.",
        "",
        "## Inputs",
        "",
        f"- Manifest: `{summary['inputs']['manifest']}`",
        f"- Predictions: `{summary['inputs']['predictions']}`",
        f"- Entity annotations: `{summary['inputs']['annotations']}`",
        f"- Geometry dir: `{summary['inputs']['geometry_dir']}`",
        f"- Annotation matches: `{summary['annotation_match_counts']}`",
        "",
        "## Topline",
        "",
        f"- Samples analyzed: **{summary['total_samples']}**",
        f"- Baseline Acc@1 / Acc@5: **{format_pct(all_summary['baseline_acc_at_1'])} / {format_pct(all_summary['baseline_acc_at_5'])}**",
        f"- Anchor-annotated samples: **{all_summary['anchor_annotated_count']}** ({format_pct(pct(all_summary['anchor_annotated_count'], summary['total_samples']))})",
        f"- Geometry-evaluable anchor samples: **{all_summary['geometry_evaluable_count']}**",
        f"- Exact-utterance anchor samples: **{subset_summary['annotation_exact']['anchor_annotated_count']}**",
        f"- Target-fallback anchor samples: **{subset_summary['annotation_fallback']['anchor_annotated_count']}**",
        f"- Dense any-anchor reachability: **{format_pct(all_summary['dense_any_reachability'])}**",
        f"- Dense all-anchor reachability: **{format_pct(all_summary['dense_all_reachability'])}**",
        f"- Mean / median minimum anchor rank: **{all_summary['mean_min_anchor_rank']} / {all_summary['median_min_anchor_rank']}**",
        "",
        "## Coverage@k",
        "",
        "| k | Any Anchor Covered | All Anchors Covered |",
        "| ---: | ---: | ---: |",
    ]
    for k in K_VALUES:
        lines.append(
            f"| {k} | {format_pct(all_summary[f'any_coverage@{k}'])} | {format_pct(all_summary[f'all_coverage@{k}'])} |"
        )
    lines.extend([
        "",
        "## Subset Coverage Curves",
        "",
        "| Subset | Count | Anchor Eval | Acc@1 | Any@1 | Any@3 | Any@5 | Any@10 | Mean Min Rank |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    display_subsets = [
        "same_class_clutter",
        "same_class_high_clutter",
        "multi_anchor",
        "relative_position",
        "dense_scene",
        "baseline_wrong",
        "baseline_correct",
        "annotation_exact",
        "annotation_fallback",
    ]
    for name in display_subsets:
        s = subset_summary[name]
        lines.append(
            "| {name} | {count} | {anchor_eval} | {acc1} | {a1} | {a3} | {a5} | {a10} | {rank} |".format(
                name=name,
                count=s["count"],
                anchor_eval=s["geometry_evaluable_count"],
                acc1=format_pct(s["baseline_acc_at_1"]),
                a1=format_pct(s["any_coverage@1"]),
                a3=format_pct(s["any_coverage@3"]),
                a5=format_pct(s["any_coverage@5"]),
                a10=format_pct(s["any_coverage@10"]),
                rank=s["mean_min_anchor_rank"],
            )
        )
    lines.extend([
        "",
        "## Initial Interpretation",
        "",
    ])
    any5 = all_summary["any_coverage@5"]
    all5 = all_summary["all_coverage@5"]
    wrong_any5 = subset_summary["baseline_wrong"]["any_coverage@5"]
    multi_any5 = subset_summary["multi_anchor"]["any_coverage@5"]
    if any5 is not None:
        lines.append(f"- Under the nearest-neighbor sparse proxy, top-5 covers at least one annotated anchor in **{any5:.2f}%** of geometry-evaluable anchor samples.")
    if all5 is not None:
        lines.append(f"- Requiring all annotated anchors is stricter: top-5 covers all anchors in **{all5:.2f}%** of geometry-evaluable anchor samples.")
    if wrong_any5 is not None:
        lines.append(f"- For baseline-wrong samples with anchor geometry, top-5 any-anchor coverage is **{wrong_any5:.2f}%**.")
    if multi_any5 is not None:
        lines.append(f"- Multi-anchor samples are the stress case: top-5 any-anchor coverage is **{multi_any5:.2f}%**, while all-anchor coverage should be read separately in the JSON output.")
    lines.extend([
        "- These numbers are evidence about sparse geometric reachability, not method gains. They should decide whether the next step deserves calibrated reranker training.",
        "",
        "## Missed-Anchor Casebook Preview",
        "",
    ])
    if not casebook:
        lines.append("No missed-anchor cases found under the configured top-k.")
    else:
        lines.extend([
            "| Scene | Target | Correct | Min Rank | Anchors | Utterance |",
            "| --- | --- | --- | ---: | --- | --- |",
        ])
        for row in casebook[:10]:
            anchors = ", ".join(f"{a}:{c}" for a, c in zip(row["anchor_ids"], row["anchor_classes"]))
            utterance = row["utterance"].replace("|", "/")
            lines.append(
                f"| {row['scene_id']} | {row['target_id']} {row['target_class']} | {row['correct_at_1']} | {row['min_anchor_rank']} | {anchors} | {utterance} |"
            )
    lines.extend([
        "",
        "## Artifacts",
        "",
        "- `coverage_summary.json`: aggregate and subset metrics.",
        "- `subset_coverage_curves.csv`: coverage@k curves by subset.",
        "- `anchor_rank_histogram.csv`: anchor distance-rank histogram.",
        "- `per_sample_coverage.jsonl`: per-sample diagnostic records.",
        "- `missed_anchor_casebook_top5.json` / `.csv`: cases where sparse top-5 misses all annotated anchors.",
        "",
        "## Claim Status",
        "",
        "This report turns the coverage claim into a measurable proposition. It does not yet validate calibration or method gains.",
    ])
    path.write_text("\n".join(lines) + "\n")


def run_coverage_diagnostics(
    manifest_path: Path,
    predictions_path: Path,
    annotations_path: Path,
    geometry_dir: Path,
    output_dir: Path,
    casebook_k: int = 5,
    casebook_limit: int = 50,
) -> dict[str, Any]:
    samples = load_jsonl(manifest_path)
    predictions = json.load(predictions_path.open())
    exact_annotations, annotations_by_target = load_annotation_indexes(annotations_path)

    pred_lookup = {
        (str(pred["scene_id"]), str(pred["target_id"]), norm_text(pred.get("utterance", ""))): pred
        for pred in predictions
    }
    pred_fallback = {
        (str(pred["scene_id"]), str(pred["target_id"])): pred
        for pred in predictions
    }

    geometry_cache: dict[str, GeometryScene | None] = {}
    per_sample = []
    prediction_misses = 0
    annotation_match_counts = Counter()

    for sample in samples:
        scene_id = str(sample["scene_id"])
        target_id = str(sample["target_object_id"])
        pred = pred_lookup.get((scene_id, target_id, norm_text(sample["utterance"])))
        if pred is None:
            pred = pred_fallback.get((scene_id, target_id))
        if pred is None:
            prediction_misses += 1
            continue

        annotation, annotation_match = find_annotation(sample, exact_annotations, annotations_by_target)
        annotation_match_counts[annotation_match] += 1
        anchors = extract_anchor_entities(annotation, target_id)

        if scene_id not in geometry_cache:
            geometry_cache[scene_id] = load_geometry_scene(scene_id, geometry_dir)
        geometry = geometry_cache[scene_id]
        ranked_candidates, ranks, distances = rank_anchors_by_target_distance(target_id, geometry)

        target_class, same_class_count = target_class_and_same_class_count(sample)
        utterance_stats = analyze_utterance(sample["utterance"])

        anchor_ids = [str(anchor["object_id"]) for anchor in anchors]
        anchor_classes = [anchor["class_name"] for anchor in anchors]
        anchor_ranks = [ranks[anchor_id] for anchor_id in anchor_ids if anchor_id in ranks]
        anchor_distances = [round(distances[anchor_id], 4) for anchor_id in anchor_ids if anchor_id in distances]
        anchor_geometry_count = len(anchor_ranks)
        geometry_valid = geometry is not None and target_id in (geometry.centers if geometry else {})

        any_anchor_covered_at: dict[str, bool] = {}
        all_anchors_covered_at: dict[str, bool] = {}
        for k in K_VALUES:
            topk = set(ranked_candidates[:k])
            covered = [anchor_id for anchor_id in anchor_ids if anchor_id in topk]
            any_anchor_covered_at[str(k)] = bool(covered)
            all_anchors_covered_at[str(k)] = bool(anchor_ids) and all(anchor_id in topk for anchor_id in anchor_ids)

        stats = {
            "scene_id": scene_id,
            "utterance": sample["utterance"],
            "target_id": target_id,
            "target_index": sample["target_index"],
            "target_class": target_class,
            "n_objects": len(sample["objects"]),
            "same_class_count": same_class_count,
            "pred_top1": pred["pred_top1"],
            "pred_top5": pred["pred_top5"],
            "correct_at_1": bool(pred["correct_at_1"]),
            "correct_at_5": bool(pred["correct_at_5"]),
            "annotation_match": annotation_match,
            "anchor_count": len(anchor_ids),
            "anchor_ids": anchor_ids,
            "anchor_classes": anchor_classes,
            "anchor_geometry_count": anchor_geometry_count,
            "geometry_valid": geometry_valid,
            "dense_any_reachable": anchor_geometry_count > 0,
            "dense_all_reachable": bool(anchor_ids) and anchor_geometry_count == len(anchor_ids),
            "anchor_ranks": anchor_ranks,
            "anchor_distances": anchor_distances,
            "min_anchor_rank": min(anchor_ranks) if anchor_ranks else None,
            "mean_anchor_rank": round(sum(anchor_ranks) / len(anchor_ranks), 2) if anchor_ranks else None,
            "nearest_sparse_candidates": ranked_candidates[:casebook_k],
            "any_anchor_covered_at": any_anchor_covered_at,
            "all_anchors_covered_at": all_anchors_covered_at,
            "is_multi_anchor": len(anchor_ids) >= 2,
            **utterance_stats,
        }
        stats["subsets"] = subset_flags(stats)
        per_sample.append(stats)

    subset_predicates = {
        "all": lambda s: True,
        "same_class_clutter": lambda s: s["subsets"]["same_class_clutter"],
        "same_class_high_clutter": lambda s: s["subsets"]["same_class_high_clutter"],
        "unique_class": lambda s: s["subsets"]["unique_class"],
        "multi_anchor": lambda s: s["subsets"]["multi_anchor"],
        "single_anchor": lambda s: s["subsets"]["single_anchor"],
        "relative_position": lambda s: s["subsets"]["relative_position"],
        "directional": lambda s: s["subsets"]["directional"],
        "between": lambda s: s["subsets"]["between"],
        "relational": lambda s: s["subsets"]["relational"],
        "dense_scene": lambda s: s["subsets"]["dense_scene"],
        "baseline_wrong": lambda s: s["subsets"]["baseline_wrong"],
        "baseline_correct": lambda s: s["subsets"]["baseline_correct"],
        "annotation_exact": lambda s: s["annotation_match"] == "exact_utterance",
        "annotation_fallback": lambda s: s["annotation_match"] == "target_fallback",
    }
    subset_summary = {
        name: summarize_subset(per_sample, predicate)
        for name, predicate in subset_predicates.items()
    }
    casebook = make_casebook(per_sample, casebook_k, casebook_limit)

    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "inputs": {
            "manifest": str(manifest_path),
            "predictions": str(predictions_path),
            "annotations": str(annotations_path),
            "geometry_dir": str(geometry_dir),
        },
        "definitions": {
            "sparse_proxy": "target-centric nearest-neighbor anchors by recovered object-center distance",
            "dense_reachability": "annotated anchors with recovered geometry are available to all-pair candidate-anchor scoring",
            "coverage_at_k_any": "at least one annotated anchor appears in sparse top-k",
            "coverage_at_k_all": "all annotated anchors appear in sparse top-k",
        },
        "total_samples": len(per_sample),
        "prediction_misses": prediction_misses,
        "annotation_match_counts": dict(annotation_match_counts),
        "subsets": subset_summary,
    }

    with (output_dir / "coverage_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    with (output_dir / "per_sample_coverage.jsonl").open("w") as f:
        for row in per_sample:
            f.write(json.dumps(row) + "\n")
    with (output_dir / f"missed_anchor_casebook_top{casebook_k}.json").open("w") as f:
        json.dump(casebook, f, indent=2)
    write_csv(output_dir / f"missed_anchor_casebook_top{casebook_k}.csv", casebook)
    write_subset_curve_csv(output_dir / "subset_coverage_curves.csv", subset_summary)
    write_anchor_rank_histogram_csv(output_dir / "anchor_rank_histogram.csv", per_sample)
    write_report(output_dir / "coverage_diagnostics_report.md", summary, subset_summary, casebook)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="COVER-3D coverage diagnostics without training")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--geometry-dir", type=Path, default=DEFAULT_GEOMETRY_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--casebook-k", type=int, default=5)
    parser.add_argument("--casebook-limit", type=int, default=50)
    args = parser.parse_args()

    summary = run_coverage_diagnostics(
        manifest_path=args.manifest,
        predictions_path=args.predictions,
        annotations_path=args.annotations,
        geometry_dir=args.geometry_dir,
        output_dir=args.output_dir,
        casebook_k=args.casebook_k,
        casebook_limit=args.casebook_limit,
    )

    all_subset = summary["subsets"]["all"]
    print("COVER-3D coverage diagnostics complete")
    print(f"  samples: {summary['total_samples']}")
    print(f"  anchor-annotated: {all_subset['anchor_annotated_count']}")
    print(f"  geometry-evaluable: {all_subset['geometry_evaluable_count']}")
    print(f"  dense any reachability: {all_subset['dense_any_reachability']}%")
    for k in K_VALUES:
        print(f"  any coverage@{k}: {all_subset[f'any_coverage@{k}']}%")
    print(f"  report: {args.output_dir / 'coverage_diagnostics_report.md'}")


if __name__ == "__main__":
    main()
