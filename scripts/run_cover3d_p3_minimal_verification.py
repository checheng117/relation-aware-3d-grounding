#!/usr/bin/env python3
"""Minimal P3 verification for COVER-3D calibration.

This is an offline smoke diagnostic, not a trained method result. It compares
trusted base predictions against sparse/dense oracle-anchor geometry proxies and
a simple deterministic gate. The goal is to test whether relation evidence has a
benefit pool and a harm pool before investing in learned calibration.

The trusted baseline prediction file does not contain logits, so the default
mode uses a rank proxy derived from the stored top-5 predictions. If a future
prediction file contains `base_logits`, this script can be extended to use true
base margins directly.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "data/processed/scene_disjoint/official_scene_disjoint/test_manifest.jsonl"
DEFAULT_COVERAGE = PROJECT_ROOT / "reports/cover3d_coverage_diagnostics/per_sample_coverage.jsonl"
DEFAULT_GEOMETRY_DIR = PROJECT_ROOT / "data/geometry"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "reports/cover3d_p3_minimal"

SPARSE_K = 5
NO_CAL_LAMBDA = 0.50
MAX_GATE_LAMBDA = 0.50
TOP5_PROXY_WEIGHTS = (1.00, 0.75, 0.55, 0.35, 0.20)


def norm_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().strip("'\"").lower())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def pct(num: int | float, den: int | float) -> float | None:
    if den == 0:
        return None
    return round(float(num) / float(den) * 100.0, 2)


def fmt_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}%"


def euclidean(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def zscore(values: list[float]) -> list[float]:
    arr = np.array(values, dtype=float)
    finite = np.isfinite(arr)
    if not finite.any():
        return [0.0 for _ in values]
    mean = float(arr[finite].mean())
    std = float(arr[finite].std())
    if std < 1e-8:
        return [0.0 for _ in values]
    out = (arr - mean) / std
    out[~finite] = 0.0
    return [float(x) for x in out.tolist()]


def load_geometry(scene_id: str, geometry_dir: Path) -> dict[str, list[float]]:
    path = geometry_dir / f"{scene_id}_geometry.npz"
    if not path.exists():
        return {}
    data = np.load(path, allow_pickle=True)
    object_ids = [str(int(x)) for x in data["object_ids"]]
    centers = {
        object_id: [float(v) for v in center]
        for object_id, center in zip(object_ids, data["centers"])
    }
    data.close()
    return centers


def object_center(obj: dict[str, Any], geometry: dict[str, list[float]]) -> list[float] | None:
    object_id = str(obj["object_id"])
    if object_id in geometry:
        return geometry[object_id]
    center = obj.get("center")
    if center:
        return [float(v) for v in center]
    return None


def manifest_lookup(samples: list[dict[str, Any]]) -> dict[tuple[str, str, str], dict[str, Any]]:
    return {
        (str(sample["scene_id"]), str(sample["target_object_id"]), norm_text(sample["utterance"])): sample
        for sample in samples
    }


def base_rank_proxy(n_objects: int, pred_top1: int, pred_top5: list[int]) -> list[float]:
    scores = [0.0] * n_objects
    ordered = []
    if 0 <= int(pred_top1) < n_objects:
        ordered.append(int(pred_top1))
    for index in pred_top5:
        index = int(index)
        if 0 <= index < n_objects and index not in ordered:
            ordered.append(index)
    for rank, index in enumerate(ordered[: len(TOP5_PROXY_WEIGHTS)]):
        if 0 <= int(index) < n_objects:
            scores[int(index)] = TOP5_PROXY_WEIGHTS[rank]
    return scores


def relation_raw_score(
    candidate_center: list[float],
    anchor_centers: list[list[float]],
    utterance: str,
    relation_type: str,
) -> float:
    text = norm_text(utterance)
    if not anchor_centers:
        return 0.0

    distances = [euclidean(candidate_center, anchor_center) for anchor_center in anchor_centers]

    if "between" in text and len(anchor_centers) >= 2:
        midpoint = [
            float(sum(anchor_center[axis] for anchor_center in anchor_centers)) / len(anchor_centers)
            for axis in range(3)
        ]
        balance = float(np.std(distances)) if len(distances) > 1 else 0.0
        return -euclidean(candidate_center, midpoint) - 0.10 * balance

    if "farthest" in text or "furthest" in text:
        return min(distances)

    if "left of" in text or "left side" in text or "left-hand" in text or "leftmost" in text:
        return float(np.mean([anchor_center[0] - candidate_center[0] for anchor_center in anchor_centers]))

    if "right of" in text or "right side" in text or "right-hand" in text or "rightmost" in text:
        return float(np.mean([candidate_center[0] - anchor_center[0] for anchor_center in anchor_centers]))

    if "front of" in text or "in front" in text:
        return float(np.mean([candidate_center[1] - anchor_center[1] for anchor_center in anchor_centers]))

    if "behind" in text or "back of" in text:
        return float(np.mean([anchor_center[1] - candidate_center[1] for anchor_center in anchor_centers]))

    if relation_type in {"relative", "support", "container", "directional"}:
        return -min(distances)

    return -float(np.mean(distances))


def relation_scores_for_sample(
    sample: dict[str, Any],
    coverage_row: dict[str, Any],
    geometry: dict[str, list[float]],
    anchor_ids: list[str],
) -> list[float]:
    anchors = [geometry[anchor_id] for anchor_id in anchor_ids if anchor_id in geometry]
    if not anchors:
        return [0.0] * len(sample["objects"])

    raw_scores: list[float] = []
    for obj in sample["objects"]:
        center = object_center(obj, geometry)
        if center is None:
            raw_scores.append(float("-inf"))
        else:
            raw_scores.append(
                relation_raw_score(
                    candidate_center=center,
                    anchor_centers=anchors,
                    utterance=sample["utterance"],
                    relation_type=coverage_row.get("relation_type", "other"),
                )
            )

    finite_non_anchor = [
        score
        for obj, score in zip(sample["objects"], raw_scores)
        if np.isfinite(score) and str(obj["object_id"]) not in set(anchor_ids)
    ]
    if finite_non_anchor:
        floor = min(finite_non_anchor) - max(max(finite_non_anchor) - min(finite_non_anchor), 1.0)
        for idx, obj in enumerate(sample["objects"]):
            if str(obj["object_id"]) in set(anchor_ids):
                raw_scores[idx] = floor

    return zscore(raw_scores)


def top_indices(scores: list[float], base_scores: list[float], k: int = 5) -> list[int]:
    order = sorted(
        range(len(scores)),
        key=lambda idx: (-scores[idx], -base_scores[idx], idx),
    )
    return order[: min(k, len(order))]


def relation_margin(scores: list[float]) -> float:
    if len(scores) < 2:
        return 0.0
    ordered = sorted(scores, reverse=True)
    return float(ordered[0] - ordered[1])


def predicted_class_count(sample: dict[str, Any], pred_top1: int) -> int:
    if pred_top1 < 0 or pred_top1 >= len(sample["objects"]):
        return len(sample["objects"])
    pred_class = sample["objects"][pred_top1].get("class_name")
    return sum(1 for obj in sample["objects"] if obj.get("class_name") == pred_class)


def gate_lambda(
    relation_z: list[float],
    anchor_count: int,
    pred_class_count_value: int,
    max_anchor_count: int,
    max_pred_class_count: int,
) -> float:
    if anchor_count <= 0 or all(abs(score) < 1e-8 for score in relation_z):
        return 0.0

    rel_conf = sigmoid(relation_margin(relation_z))
    ambiguity = math.log1p(pred_class_count_value) / max(math.log1p(max_pred_class_count), 1e-8)
    ambiguity = max(0.0, min(1.0, ambiguity))
    anchor_uncertainty = math.log1p(anchor_count) / max(math.log1p(max_anchor_count), 1e-8)
    anchor_uncertainty = max(0.0, min(1.0, anchor_uncertainty))

    value = (
        MAX_GATE_LAMBDA
        * (0.25 + 0.75 * ambiguity)
        * (0.25 + 0.75 * rel_conf)
        * (1.0 - 0.50 * anchor_uncertainty)
    )
    return round(max(0.0, min(MAX_GATE_LAMBDA, value)), 6)


def metric_summary(rows: list[dict[str, Any]], pred_key: str, top5_key: str) -> dict[str, Any]:
    return {
        "count": len(rows),
        "acc_at_1": pct(sum(int(row[pred_key] == row["target_index"]) for row in rows), len(rows)),
        "acc_at_5": pct(sum(int(row["target_index"] in row[top5_key]) for row in rows), len(rows)),
    }


def transition_summary(rows: list[dict[str, Any]], pred_key: str) -> dict[str, Any]:
    base_correct = [row for row in rows if row["base_pred"] == row["target_index"]]
    base_wrong = [row for row in rows if row["base_pred"] != row["target_index"]]
    recovered = [row for row in base_wrong if row[pred_key] == row["target_index"]]
    harmed = [row for row in base_correct if row[pred_key] != row["target_index"]]
    return {
        "base_wrong_count": len(base_wrong),
        "base_correct_count": len(base_correct),
        "recovered_from_base_wrong": len(recovered),
        "recovered_pct_of_base_wrong": pct(len(recovered), len(base_wrong)),
        "harmed_from_base_correct": len(harmed),
        "harmed_pct_of_base_correct": pct(len(harmed), len(base_correct)),
        "net_correct_delta": len(recovered) - len(harmed),
    }


def subset_metrics(rows: list[dict[str, Any]], subset_names: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    variants = {
        "base": ("base_pred", "base_top5"),
        "sparse_no_cal": ("sparse_no_cal_pred", "sparse_no_cal_top5"),
        "dense_no_cal": ("dense_no_cal_pred", "dense_no_cal_top5"),
        "dense_simple_gate": ("dense_gate_pred", "dense_gate_top5"),
    }
    for subset in subset_names:
        subset_rows = [row for row in rows if row["subsets"].get(subset, False)]
        out[subset] = {
            name: metric_summary(subset_rows, pred_key, top5_key)
            for name, (pred_key, top5_key) in variants.items()
        }
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def casebook(rows: list[dict[str, Any]], limit: int = 80) -> list[dict[str, Any]]:
    selected = [
        row
        for row in rows
        if row["base_pred"] != row["target_index"]
        and (row["dense_no_cal_pred"] == row["target_index"] or row["dense_gate_pred"] == row["target_index"])
    ]
    harmed = [
        row
        for row in rows
        if row["base_pred"] == row["target_index"]
        and (row["dense_no_cal_pred"] != row["target_index"] or row["dense_gate_pred"] != row["target_index"])
    ]
    ordered = selected[: limit // 2] + harmed[: limit // 2]
    out = []
    for row in ordered:
        out.append({
            "scene_id": row["scene_id"],
            "utterance": row["utterance"],
            "target_index": row["target_index"],
            "target_id": row["target_id"],
            "anchor_ids": row["anchor_ids"],
            "relation_type": row["relation_type"],
            "subsets": ",".join(name for name, value in row["subsets"].items() if value and name != "all"),
            "base_pred": row["base_pred"],
            "sparse_no_cal_pred": row["sparse_no_cal_pred"],
            "dense_no_cal_pred": row["dense_no_cal_pred"],
            "dense_gate_pred": row["dense_gate_pred"],
            "gate_lambda": row["gate_lambda"],
            "relation_margin": row["relation_margin"],
            "pred_class_count": row["pred_class_count"],
            "event": (
                "recovered"
                if row["base_pred"] != row["target_index"]
                else "harmed"
            ),
        })
    return out


def write_report(path: Path, summary: dict[str, Any], case_rows: list[dict[str, Any]]) -> None:
    overall = summary["overall"]
    transitions = summary["transitions"]
    subset_summary = summary["subsets"]
    dense_transition = transitions["dense_no_cal"]
    gate_transition = transitions["dense_simple_gate"]
    gate_behavior = summary["gate_behavior"]
    lines = [
        "# COVER-3D Minimal P3 Verification",
        "",
        "**Date**: 2026-04-19",
        "**Training**: none.",
        "**Status**: offline rank-proxy / oracle-anchor geometry smoke test, not a learned COVER-3D result.",
        "",
        "## Executive Summary",
        "",
        "This diagnostic enters the minimal P3 stage without training a new model. It compares the trusted 30.79% ReferIt3DNet predictions against a sparse relation proxy, a dense uncalibrated relation proxy, and a dense relation proxy with a simple deterministic gate.",
        "",
        "The trusted prediction file does not contain logits, and a fresh logits export attempt did not reproduce the trusted baseline because the matching full-test BERT feature cache is not available. Therefore this report uses a rank proxy from the stored top-5 predictions rather than claiming true base-margin calibration.",
        "",
        "These results should be read as a low-risk calibration smoke test: they can reveal whether dense relation evidence has both recoveries and harms, but they do not prove calibration necessity or final method effectiveness.",
        "",
        f"Under this proxy, dense no-cal improves Acc@1 from **{fmt_pct(overall['base']['acc_at_1'])}** to **{fmt_pct(overall['dense_no_cal']['acc_at_1'])}**, with **{dense_transition['recovered_from_base_wrong']}** base-wrong recoveries and **{dense_transition['harmed_from_base_correct']}** base-correct harms. The simple gate prevents **{gate_behavior['dense_harms_prevented_by_gate']}** dense no-cal harms, but loses **{gate_behavior['dense_recoveries_lost_by_gate']}** dense no-cal recoveries, so the current gate is useful as a risk signal but not yet a finished calibration design.",
        "",
        "## Variants",
        "",
        "- `Base`: frozen trusted top-1/top-5 predictions.",
        f"- `Sparse no-cal`: base rank proxy plus relation proxy using only annotated anchors that are inside sparse top-{SPARSE_K} nearest-neighbor coverage.",
        "- `Dense no-cal`: base rank proxy plus relation proxy using all geometry-recovered annotated anchors.",
        "- `Dense simple gate`: dense proxy with a deterministic gate based on predicted-class ambiguity, anchor-count uncertainty, and relation-score margin.",
        "",
        "## Overall Metrics",
        "",
        "| Variant | Acc@1 | Acc@5 |",
        "| --- | ---: | ---: |",
    ]
    for name in ["base", "sparse_no_cal", "dense_no_cal", "dense_simple_gate"]:
        row = overall[name]
        lines.append(f"| {name} | {fmt_pct(row['acc_at_1'])} | {fmt_pct(row['acc_at_5'])} |")

    lines.extend([
        "",
        "## Base-Wrong Recovery vs Base-Correct Harm",
        "",
        "| Variant | Recovered | Harmed | Net Correct Delta |",
        "| --- | ---: | ---: | ---: |",
    ])
    for name in ["sparse_no_cal", "dense_no_cal", "dense_simple_gate"]:
        row = transitions[name]
        lines.append(
            f"| {name} | {row['recovered_from_base_wrong']} ({fmt_pct(row['recovered_pct_of_base_wrong'])}) | "
            f"{row['harmed_from_base_correct']} ({fmt_pct(row['harmed_pct_of_base_correct'])}) | "
            f"{row['net_correct_delta']:+d} |"
        )

    gate = summary["gate_behavior"]
    lines.extend([
        "",
        "## Gate Behavior",
        "",
        f"- Mean gate lambda: **{gate['mean_gate_lambda']}**",
        f"- Median gate lambda: **{gate['median_gate_lambda']}**",
        f"- Mean relation margin: **{gate['mean_relation_margin']}**",
        f"- Dense no-cal harms prevented by gate: **{gate['dense_harms_prevented_by_gate']}**",
        f"- Dense no-cal recoveries lost by gate: **{gate['dense_recoveries_lost_by_gate']}**",
        "",
        "## Hard-Subset Metrics",
        "",
        "| Subset | n | Base | Sparse no-cal | Dense no-cal | Dense gate |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for subset in [
        "same_class_clutter",
        "same_class_high_clutter",
        "multi_anchor",
        "relative_position",
        "relational",
        "dense_scene",
        "baseline_wrong",
        "baseline_correct",
        "sparse_any_missed_at_5",
        "sparse_all_incomplete_at_5",
    ]:
        rows = subset_summary[subset]
        n = rows["base"]["count"]
        lines.append(
            f"| {subset} | {n} | {fmt_pct(rows['base']['acc_at_1'])} | "
            f"{fmt_pct(rows['sparse_no_cal']['acc_at_1'])} | "
            f"{fmt_pct(rows['dense_no_cal']['acc_at_1'])} | "
            f"{fmt_pct(rows['dense_simple_gate']['acc_at_1'])} |"
        )

    lines.extend([
        "",
        "## Interpretation",
        "",
        "The useful question here is not whether this proxy beats the baseline overall. The useful question is whether dense relation evidence creates a measurable recovery pool and whether uncalibrated dense evidence also creates a harm pool on base-correct samples.",
        "",
        "If dense no-cal recovers hard cases but also harms base-correct cases, then the calibration hypothesis has a concrete target. If the simple gate prevents some harms without destroying all recoveries, then P3 becomes worth testing with real logits and a learned dense scorer.",
        "",
        "## Boundary",
        "",
        "These results do not establish that a learned dense relation scorer improves final grounding accuracy, nor do they prove calibration necessity. They only test whether dense oracle-anchor geometry evidence can expose the benefit/harm pattern that calibration is supposed to control.",
        "",
        "## Artifacts",
        "",
        "- `p3_minimal_summary.json`: aggregate metrics and transition counts.",
        "- `p3_minimal_per_sample.jsonl`: per-sample variant predictions and gate diagnostics.",
        "- `p3_minimal_casebook.json` / `.csv`: recovered and harmed examples.",
    ])
    if case_rows:
        lines.extend([
            "",
            "## Casebook Preview",
            "",
            "| Event | Scene | Target | Base | Dense no-cal | Dense gate | Gate | Utterance |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
        ])
        for row in case_rows[:12]:
            utterance = row["utterance"].replace("|", "/")
            lines.append(
                f"| {row['event']} | {row['scene_id']} | {row['target_id']} | "
                f"{row['base_pred']} | {row['dense_no_cal_pred']} | {row['dense_gate_pred']} | "
                f"{row['gate_lambda']} | {utterance} |"
            )

    path.write_text("\n".join(lines) + "\n")


def run_p3_minimal(
    manifest_path: Path,
    coverage_path: Path,
    geometry_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    manifest_samples = load_jsonl(manifest_path)
    samples_by_key = manifest_lookup(manifest_samples)
    coverage_rows = load_jsonl(coverage_path)
    geometry_cache: dict[str, dict[str, list[float]]] = {}

    max_anchor_count = max((row.get("anchor_count", 0) for row in coverage_rows), default=1)
    pred_class_counts = []
    for row in coverage_rows:
        sample = samples_by_key.get((row["scene_id"], row["target_id"], norm_text(row["utterance"])))
        if sample is None:
            continue
        pred_class_counts.append(predicted_class_count(sample, int(row["pred_top1"])))
    max_pred_class_count = max(pred_class_counts or [1])

    per_sample: list[dict[str, Any]] = []
    for row in coverage_rows:
        key = (row["scene_id"], row["target_id"], norm_text(row["utterance"]))
        sample = samples_by_key.get(key)
        if sample is None:
            continue

        scene_id = str(row["scene_id"])
        if scene_id not in geometry_cache:
            geometry_cache[scene_id] = load_geometry(scene_id, geometry_dir)
        geometry = geometry_cache[scene_id]

        n_objects = len(sample["objects"])
        target_index = int(row["target_index"])
        pred_top5 = [int(idx) for idx in row["pred_top5"]]
        base_scores = base_rank_proxy(n_objects, int(row["pred_top1"]), pred_top5)

        dense_anchor_ids = [str(anchor_id) for anchor_id in row.get("anchor_ids", []) if str(anchor_id) in geometry]
        sparse_top = set(str(anchor_id) for anchor_id in row.get("nearest_sparse_candidates", [])[:SPARSE_K])
        sparse_anchor_ids = [anchor_id for anchor_id in dense_anchor_ids if anchor_id in sparse_top]

        dense_relation_z = relation_scores_for_sample(sample, row, geometry, dense_anchor_ids)
        sparse_relation_z = relation_scores_for_sample(sample, row, geometry, sparse_anchor_ids)

        sparse_scores = [
            base + NO_CAL_LAMBDA * rel
            for base, rel in zip(base_scores, sparse_relation_z)
        ]
        dense_scores = [
            base + NO_CAL_LAMBDA * rel
            for base, rel in zip(base_scores, dense_relation_z)
        ]

        pred_class_count_value = predicted_class_count(sample, int(row["pred_top1"]))
        gate = gate_lambda(
            relation_z=dense_relation_z,
            anchor_count=len(dense_anchor_ids),
            pred_class_count_value=pred_class_count_value,
            max_anchor_count=max_anchor_count,
            max_pred_class_count=max_pred_class_count,
        )
        gated_scores = [
            base + gate * rel
            for base, rel in zip(base_scores, dense_relation_z)
        ]

        base_top5 = pred_top5[: min(5, n_objects)]
        sparse_top5 = top_indices(sparse_scores, base_scores, 5)
        dense_top5 = top_indices(dense_scores, base_scores, 5)
        gate_top5 = top_indices(gated_scores, base_scores, 5)

        subsets = dict(row.get("subsets", {}))
        subsets["sparse_any_missed_at_5"] = (
            row.get("anchor_count", 0) > 0
            and row.get("geometry_valid", False)
            and row.get("anchor_geometry_count", 0) > 0
            and not row.get("any_anchor_covered_at", {}).get(str(SPARSE_K), False)
        )
        subsets["sparse_all_incomplete_at_5"] = (
            row.get("anchor_count", 0) > 0
            and row.get("geometry_valid", False)
            and row.get("anchor_geometry_count", 0) > 0
            and not row.get("all_anchors_covered_at", {}).get(str(SPARSE_K), False)
        )

        per_sample.append({
            "scene_id": scene_id,
            "utterance": row["utterance"],
            "target_id": str(row["target_id"]),
            "target_index": target_index,
            "anchor_ids": dense_anchor_ids,
            "anchor_count": len(dense_anchor_ids),
            "relation_type": row.get("relation_type", "other"),
            "same_class_count": row.get("same_class_count"),
            "subsets": subsets,
            "base_pred": int(row["pred_top1"]),
            "base_top5": base_top5,
            "sparse_no_cal_pred": int(sparse_top5[0]) if sparse_top5 else -1,
            "sparse_no_cal_top5": sparse_top5,
            "dense_no_cal_pred": int(dense_top5[0]) if dense_top5 else -1,
            "dense_no_cal_top5": dense_top5,
            "dense_gate_pred": int(gate_top5[0]) if gate_top5 else -1,
            "dense_gate_top5": gate_top5,
            "gate_lambda": gate,
            "relation_margin": round(relation_margin(dense_relation_z), 6),
            "pred_class_count": pred_class_count_value,
            "sparse_anchor_count_used": len(sparse_anchor_ids),
            "dense_anchor_count_used": len(dense_anchor_ids),
        })

    variants = {
        "base": ("base_pred", "base_top5"),
        "sparse_no_cal": ("sparse_no_cal_pred", "sparse_no_cal_top5"),
        "dense_no_cal": ("dense_no_cal_pred", "dense_no_cal_top5"),
        "dense_simple_gate": ("dense_gate_pred", "dense_gate_top5"),
    }
    overall = {
        name: metric_summary(per_sample, pred_key, top5_key)
        for name, (pred_key, top5_key) in variants.items()
    }
    transitions = {
        "sparse_no_cal": transition_summary(per_sample, "sparse_no_cal_pred"),
        "dense_no_cal": transition_summary(per_sample, "dense_no_cal_pred"),
        "dense_simple_gate": transition_summary(per_sample, "dense_gate_pred"),
    }

    subset_names = [
        "same_class_clutter",
        "same_class_high_clutter",
        "multi_anchor",
        "relative_position",
        "relational",
        "dense_scene",
        "baseline_wrong",
        "baseline_correct",
        "sparse_any_missed_at_5",
        "sparse_all_incomplete_at_5",
    ]
    subset_summary = subset_metrics(per_sample, subset_names)

    dense_recovered = [
        row for row in per_sample
        if row["base_pred"] != row["target_index"] and row["dense_no_cal_pred"] == row["target_index"]
    ]
    dense_harmed = [
        row for row in per_sample
        if row["base_pred"] == row["target_index"] and row["dense_no_cal_pred"] != row["target_index"]
    ]
    gate_harmed = [
        row for row in per_sample
        if row["base_pred"] == row["target_index"] and row["dense_gate_pred"] != row["target_index"]
    ]
    gate_lost_recoveries = [
        row for row in dense_recovered
        if row["dense_gate_pred"] != row["target_index"]
    ]
    harms_prevented = [
        row for row in dense_harmed
        if row["dense_gate_pred"] == row["target_index"]
    ]
    gate_values = [row["gate_lambda"] for row in per_sample if row["gate_lambda"] > 0]
    relation_margins = [row["relation_margin"] for row in per_sample if row["dense_anchor_count_used"] > 0]
    gate_behavior = {
        "mean_gate_lambda": round(float(np.mean(gate_values)), 6) if gate_values else 0.0,
        "median_gate_lambda": round(float(np.median(gate_values)), 6) if gate_values else 0.0,
        "mean_relation_margin": round(float(np.mean(relation_margins)), 6) if relation_margins else 0.0,
        "dense_harms_prevented_by_gate": len(harms_prevented),
        "dense_recoveries_lost_by_gate": len(gate_lost_recoveries),
        "dense_no_cal_harm_count": len(dense_harmed),
        "dense_gate_harm_count": len(gate_harmed),
    }

    summary = {
        "inputs": {
            "manifest": str(manifest_path),
            "coverage": str(coverage_path),
            "geometry_dir": str(geometry_dir),
        },
        "strict_limitations": [
            "This is not a learned dense relation scorer.",
            "This uses oracle annotated anchors and recovered geometry.",
            "The trusted baseline prediction file lacks logits, so base margin is approximated by a top-5 rank proxy.",
            "A fresh logits export attempt did not reproduce the trusted 30.79% baseline and is excluded from this report.",
        ],
        "definitions": {
            "base_rank_proxy": "stored trusted top-5 receives fixed descending weights; all other objects receive zero base rank score",
            "sparse_relation_proxy": f"relation proxy using only annotated anchors present in sparse top-{SPARSE_K} nearest-neighbor coverage",
            "dense_relation_proxy": "relation proxy using all geometry-recovered annotated anchors",
            "simple_gate": "deterministic lambda from predicted-class ambiguity, anchor-count uncertainty, and relation-score margin",
        },
        "total_samples": len(per_sample),
        "overall": overall,
        "transitions": transitions,
        "subsets": subset_summary,
        "gate_behavior": gate_behavior,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "p3_minimal_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    with (output_dir / "p3_minimal_per_sample.jsonl").open("w") as f:
        for row in per_sample:
            f.write(json.dumps(row) + "\n")
    cases = casebook(per_sample)
    with (output_dir / "p3_minimal_casebook.json").open("w") as f:
        json.dump(cases, f, indent=2)
    write_csv(output_dir / "p3_minimal_casebook.csv", cases)
    write_report(output_dir / "p3_minimal_report.md", summary, cases)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="COVER-3D minimal P3 calibration proxy diagnostic")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--coverage", type=Path, default=DEFAULT_COVERAGE)
    parser.add_argument("--geometry-dir", type=Path, default=DEFAULT_GEOMETRY_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    summary = run_p3_minimal(
        manifest_path=args.manifest,
        coverage_path=args.coverage,
        geometry_dir=args.geometry_dir,
        output_dir=args.output_dir,
    )
    print("COVER-3D minimal P3 verification complete")
    for name, row in summary["overall"].items():
        print(f"  {name}: Acc@1={row['acc_at_1']} Acc@5={row['acc_at_5']}")
    dense = summary["transitions"]["dense_no_cal"]
    gate = summary["transitions"]["dense_simple_gate"]
    print(
        "  dense no-cal recovered/harmed/net: "
        f"{dense['recovered_from_base_wrong']}/"
        f"{dense['harmed_from_base_correct']}/"
        f"{dense['net_correct_delta']:+d}"
    )
    print(
        "  dense gate recovered/harmed/net: "
        f"{gate['recovered_from_base_wrong']}/"
        f"{gate['harmed_from_base_correct']}/"
        f"{gate['net_correct_delta']:+d}"
    )
    print(f"  report: {args.output_dir / 'p3_minimal_report.md'}")


if __name__ == "__main__":
    main()
