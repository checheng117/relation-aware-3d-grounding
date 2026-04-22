#!/usr/bin/env python3
"""Run a minimal real-logits Base/Dense-no-cal/Dense-calibrated entry check.

This is a pre-method readiness check, not a formal COVER-3D result. It consumes
baseline exports with real per-object logits and combines them with the existing
oracle-anchor geometry proxy used by the minimal P3 diagnostic.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_cover3d_p3_minimal_verification import (  # noqa: E402
    SPARSE_K,
    load_jsonl,
    manifest_lookup,
    norm_text,
    predicted_class_count,
    relation_margin,
    relation_scores_for_sample,
    transition_summary,
)


def pct(num: int | float, den: int | float) -> float | None:
    if den == 0:
        return None
    return round(float(num) / float(den) * 100.0, 2)


def fmt_pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}%"


def softmax(values: list[float]) -> list[float]:
    max_value = max(values)
    exps = [math.exp(v - max_value) for v in values]
    total = sum(exps)
    return [v / total for v in exps]


def entropy_from_logits(values: list[float]) -> float:
    probs = softmax(values)
    return -sum(p * math.log(max(p, 1e-12)) for p in probs)


def top_indices(scores: list[float], base_scores: list[float], k: int = 5) -> list[int]:
    order = sorted(
        range(len(scores)),
        key=lambda idx: (-float(scores[idx]), -float(base_scores[idx]), idx),
    )
    return order[: min(k, len(order))]


def base_margin(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    ordered = sorted(values, reverse=True)
    return float(ordered[0] - ordered[1])


def calibrated_lambda(
    logits: list[float],
    relation_z: list[float],
    anchor_count: int,
    pred_class_count_value: int,
    max_anchor_count: int,
    max_pred_class_count: int,
    max_lambda: float,
) -> float:
    if anchor_count <= 0 or all(abs(score) < 1e-8 for score in relation_z):
        return 0.0
    margin = base_margin(logits)
    base_uncertainty = 1.0 - (1.0 / (1.0 + math.exp(-margin)))
    base_entropy = entropy_from_logits(logits) / max(math.log(max(len(logits), 2)), 1e-8)
    relation_conf = 1.0 / (1.0 + math.exp(-relation_margin(relation_z)))
    ambiguity = math.log1p(pred_class_count_value) / max(math.log1p(max_pred_class_count), 1e-8)
    anchor_uncertainty = math.log1p(anchor_count) / max(math.log1p(max_anchor_count), 1e-8)
    value = (
        max_lambda
        * (0.25 + 0.75 * base_uncertainty)
        * (0.25 + 0.75 * base_entropy)
        * (0.25 + 0.75 * relation_conf)
        * (0.25 + 0.75 * ambiguity)
        * (1.0 - 0.50 * anchor_uncertainty)
    )
    return round(max(0.0, min(max_lambda, value)), 6)


def safe_load_geometry(scene_id: str, geometry_dir: Path) -> dict[str, list[float]]:
    """Load one scene geometry in a subprocess.

    The current Python/Numpy stack has shown intermittent in-process crashes
    while reading many .npz files. Subprocess isolation keeps this readiness
    entry deterministic and does not affect the training/evaluation path.
    """
    path = geometry_dir / f"{scene_id}_geometry.npz"
    if not path.exists():
        return {}
    code = (
        "import json, numpy as np; "
        f"p={str(path)!r}; "
        "d=np.load(p, allow_pickle=True); "
        "obj=[str(int(x)) for x in d['object_ids']]; "
        "cent={oid:[float(v) for v in c] for oid,c in zip(obj,d['centers'])}; "
        "d.close(); "
        "print(json.dumps(cent))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.returncode != 0:
        return {}
    return json.loads(proc.stdout)


def metric_summary(rows: list[dict[str, Any]], pred_key: str, top5_key: str) -> dict[str, Any]:
    return {
        "count": len(rows),
        "acc_at_1": pct(sum(int(row[pred_key] == row["target_index"]) for row in rows), len(rows)),
        "acc_at_5": pct(sum(int(row["target_index"] in row[top5_key]) for row in rows), len(rows)),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_report(path: Path, summary: dict[str, Any], case_rows: list[dict[str, Any]]) -> None:
    overall = summary["overall"]
    transitions = summary["transitions"]
    lines = [
        "# Real-Logits P3 Entry Readiness",
        "",
        "**Status**: pre-method infrastructure check, not a formal COVER-3D result.",
        "",
        "## Inputs",
        "",
        f"- Manifest: `{summary['inputs']['manifest']}`",
        f"- Predictions with logits: `{summary['inputs']['predictions']}`",
        f"- Coverage rows: `{summary['inputs']['coverage']}`",
        f"- Geometry dir: `{summary['inputs']['geometry_dir']}`",
        "",
        "## Overall",
        "",
        "| Variant | Acc@1 | Acc@5 |",
        "| --- | ---: | ---: |",
    ]
    for name in ["base", "sparse_no_cal", "dense_no_cal", "dense_calibrated"]:
        row = overall[name]
        lines.append(f"| {name} | {fmt_pct(row['acc_at_1'])} | {fmt_pct(row['acc_at_5'])} |")
    lines.extend([
        "",
        "## Recovery/Harm",
        "",
        "| Variant | Recovered | Harmed | Net Correct Delta |",
        "| --- | ---: | ---: | ---: |",
    ])
    for name in ["sparse_no_cal", "dense_no_cal", "dense_calibrated"]:
        row = transitions[name]
        lines.append(
            f"| {name} | {row['recovered_from_base_wrong']} | "
            f"{row['harmed_from_base_correct']} | {row['net_correct_delta']:+d} |"
        )
    lines.extend([
        "",
        "## Readiness Interpretation",
        "",
        "This entry proves that real exported base logits can drive the Base / Dense-no-cal / Dense-calibrated comparison path. The relation signal is still the existing oracle-anchor geometry proxy, so these numbers are not method evidence.",
        "",
        "## Artifacts",
        "",
        "- `real_logits_p3_summary.json`",
        "- `real_logits_p3_per_sample.jsonl`",
        "- `real_logits_p3_casebook.json` / `.csv`",
    ])
    if case_rows:
        lines.extend([
            "",
            "## Casebook Preview",
            "",
            "| Event | Scene | Target | Base | Dense no-cal | Dense calibrated | Gate | Utterance |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
        ])
        for row in case_rows[:12]:
            utterance = row["utterance"].replace("|", "/")
            lines.append(
                f"| {row['event']} | {row['scene_id']} | {row['target_id']} | "
                f"{row['base_pred']} | {row['dense_no_cal_pred']} | "
                f"{row['dense_calibrated_pred']} | {row['calibrated_lambda']} | {utterance} |"
            )
    path.write_text("\n".join(lines) + "\n")


def casebook(rows: list[dict[str, Any]], limit: int = 80) -> list[dict[str, Any]]:
    recovered = [
        row
        for row in rows
        if row["base_pred"] != row["target_index"]
        and (
            row["dense_no_cal_pred"] == row["target_index"]
            or row["dense_calibrated_pred"] == row["target_index"]
        )
    ]
    harmed = [
        row
        for row in rows
        if row["base_pred"] == row["target_index"]
        and (
            row["dense_no_cal_pred"] != row["target_index"]
            or row["dense_calibrated_pred"] != row["target_index"]
        )
    ]
    selected = recovered[: limit // 2] + harmed[: limit // 2]
    out = []
    for row in selected:
        out.append({
            "event": "recovered" if row["base_pred"] != row["target_index"] else "harmed",
            "scene_id": row["scene_id"],
            "utterance": row["utterance"],
            "target_id": row["target_id"],
            "target_index": row["target_index"],
            "anchor_ids": row["anchor_ids"],
            "base_pred": row["base_pred"],
            "sparse_no_cal_pred": row["sparse_no_cal_pred"],
            "dense_no_cal_pred": row["dense_no_cal_pred"],
            "dense_calibrated_pred": row["dense_calibrated_pred"],
            "base_margin": row["base_margin"],
            "base_entropy": row["base_entropy"],
            "relation_margin": row["relation_margin"],
            "calibrated_lambda": row["calibrated_lambda"],
        })
    return out


def run_entry(
    manifest_path: Path,
    coverage_path: Path,
    predictions_path: Path,
    geometry_dir: Path,
    output_dir: Path,
    dense_lambda: float,
) -> dict[str, Any]:
    manifest_samples = load_jsonl(manifest_path)
    samples_by_key = manifest_lookup(manifest_samples)
    coverage_rows = load_jsonl(coverage_path)
    predictions = json.load(predictions_path.open())
    pred_lookup = {
        (str(row["scene_id"]), str(row["target_id"]), norm_text(row["utterance"])): row
        for row in predictions
    }
    geometry_cache: dict[str, dict[str, list[float]]] = {}

    max_anchor_count = max((row.get("anchor_count", 0) for row in coverage_rows), default=1)
    pred_class_counts = []
    for row in coverage_rows:
        sample = samples_by_key.get((row["scene_id"], row["target_id"], norm_text(row["utterance"])))
        pred = pred_lookup.get((row["scene_id"], row["target_id"], norm_text(row["utterance"])))
        if sample is not None and pred is not None:
            pred_class_counts.append(predicted_class_count(sample, int(pred["pred_top1"])))
    max_pred_class_count = max(pred_class_counts or [1])

    per_sample = []
    skipped = 0
    for row in coverage_rows:
        key = (row["scene_id"], row["target_id"], norm_text(row["utterance"]))
        sample = samples_by_key.get(key)
        pred = pred_lookup.get(key)
        if sample is None or pred is None or "base_logits" not in pred:
            skipped += 1
            continue
        logits = [float(v) for v in pred["base_logits"]]
        if len(logits) != len(sample["objects"]):
            skipped += 1
            continue

        scene_id = str(row["scene_id"])
        if scene_id not in geometry_cache:
            geometry_cache[scene_id] = safe_load_geometry(scene_id, geometry_dir)
        geometry = geometry_cache[scene_id]

        target_index = int(row["target_index"])
        base_top5 = top_indices(logits, logits, 5)
        dense_anchor_ids = [str(anchor_id) for anchor_id in row.get("anchor_ids", []) if str(anchor_id) in geometry]
        sparse_top = set(str(anchor_id) for anchor_id in row.get("nearest_sparse_candidates", [])[:SPARSE_K])
        sparse_anchor_ids = [anchor_id for anchor_id in dense_anchor_ids if anchor_id in sparse_top]

        dense_relation_z = relation_scores_for_sample(sample, row, geometry, dense_anchor_ids)
        sparse_relation_z = relation_scores_for_sample(sample, row, geometry, sparse_anchor_ids)
        sparse_scores = [base + dense_lambda * rel for base, rel in zip(logits, sparse_relation_z)]
        dense_scores = [base + dense_lambda * rel for base, rel in zip(logits, dense_relation_z)]

        pred_class_count_value = predicted_class_count(sample, int(pred["pred_top1"]))
        gate = calibrated_lambda(
            logits=logits,
            relation_z=dense_relation_z,
            anchor_count=len(dense_anchor_ids),
            pred_class_count_value=pred_class_count_value,
            max_anchor_count=max_anchor_count,
            max_pred_class_count=max_pred_class_count,
            max_lambda=dense_lambda,
        )
        calibrated_scores = [base + gate * rel for base, rel in zip(logits, dense_relation_z)]
        sparse_top5 = top_indices(sparse_scores, logits, 5)
        dense_top5 = top_indices(dense_scores, logits, 5)
        calibrated_top5 = top_indices(calibrated_scores, logits, 5)

        per_sample.append({
            "scene_id": scene_id,
            "utterance": row["utterance"],
            "target_id": str(row["target_id"]),
            "target_index": target_index,
            "anchor_ids": dense_anchor_ids,
            "anchor_count": len(dense_anchor_ids),
            "relation_type": row.get("relation_type", "other"),
            "subsets": row.get("subsets", {}),
            "base_pred": int(base_top5[0]) if base_top5 else -1,
            "base_top5": base_top5,
            "sparse_no_cal_pred": int(sparse_top5[0]) if sparse_top5 else -1,
            "sparse_no_cal_top5": sparse_top5,
            "dense_no_cal_pred": int(dense_top5[0]) if dense_top5 else -1,
            "dense_no_cal_top5": dense_top5,
            "dense_calibrated_pred": int(calibrated_top5[0]) if calibrated_top5 else -1,
            "dense_calibrated_top5": calibrated_top5,
            "base_margin": round(base_margin(logits), 6),
            "base_entropy": round(entropy_from_logits(logits), 6),
            "relation_margin": round(relation_margin(dense_relation_z), 6),
            "calibrated_lambda": gate,
            "dense_lambda": dense_lambda,
            "sparse_anchor_count_used": len(sparse_anchor_ids),
            "dense_anchor_count_used": len(dense_anchor_ids),
        })

    variants = {
        "base": ("base_pred", "base_top5"),
        "sparse_no_cal": ("sparse_no_cal_pred", "sparse_no_cal_top5"),
        "dense_no_cal": ("dense_no_cal_pred", "dense_no_cal_top5"),
        "dense_calibrated": ("dense_calibrated_pred", "dense_calibrated_top5"),
    }
    overall = {
        name: metric_summary(per_sample, pred_key, top5_key)
        for name, (pred_key, top5_key) in variants.items()
    }
    transitions = {
        "sparse_no_cal": transition_summary(per_sample, "sparse_no_cal_pred"),
        "dense_no_cal": transition_summary(per_sample, "dense_no_cal_pred"),
        "dense_calibrated": transition_summary(per_sample, "dense_calibrated_pred"),
    }
    gate_values = [row["calibrated_lambda"] for row in per_sample]
    summary = {
        "status": "PASS" if per_sample else "FAIL",
        "inputs": {
            "manifest": str(manifest_path),
            "coverage": str(coverage_path),
            "predictions": str(predictions_path),
            "geometry_dir": str(geometry_dir),
        },
        "limitations": [
            "Uses real base logits from a clean export.",
            "Uses oracle-anchor geometry proxy relation scores; not a learned dense scorer.",
            "Intended only to prove the comparison entry point is runnable.",
        ],
        "dense_lambda": dense_lambda,
        "total_samples": len(per_sample),
        "skipped_rows": skipped,
        "overall": overall,
        "transitions": transitions,
        "gate_behavior": {
            "mean_calibrated_lambda": round(float(np.mean(gate_values)), 6) if gate_values else 0.0,
            "median_calibrated_lambda": round(float(np.median(gate_values)), 6) if gate_values else 0.0,
            "max_calibrated_lambda": round(float(np.max(gate_values)), 6) if gate_values else 0.0,
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "real_logits_p3_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    with (output_dir / "real_logits_p3_per_sample.jsonl").open("w") as f:
        for row in per_sample:
            f.write(json.dumps(row) + "\n")
    cases = casebook(per_sample)
    with (output_dir / "real_logits_p3_casebook.json").open("w") as f:
        json.dump(cases, f, indent=2)
    write_csv(output_dir / "real_logits_p3_casebook.csv", cases)
    write_report(output_dir / "real_logits_p3_report.md", summary, cases)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Real-logits P3 entry readiness check")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "data/processed/scene_disjoint/official_scene_disjoint/test_manifest.jsonl",
    )
    parser.add_argument(
        "--coverage",
        type=Path,
        default=PROJECT_ROOT / "reports/cover3d_coverage_diagnostics/per_sample_coverage.jsonl",
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--geometry-dir", type=Path, default=PROJECT_ROOT / "data/geometry")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dense-lambda", type=float, default=0.5)
    args = parser.parse_args()

    summary = run_entry(
        manifest_path=args.manifest,
        coverage_path=args.coverage,
        predictions_path=args.predictions,
        geometry_dir=args.geometry_dir,
        output_dir=args.output_dir,
        dense_lambda=args.dense_lambda,
    )
    print("real-logits P3 entry:", summary["status"])
    for name, row in summary["overall"].items():
        print(f"  {name}: Acc@1={row['acc_at_1']} Acc@5={row['acc_at_5']}")
    print(f"  report: {args.output_dir / 'real_logits_p3_report.md'}")
    raise SystemExit(0 if summary["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
