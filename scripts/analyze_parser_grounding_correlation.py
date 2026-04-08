#!/usr/bin/env python3
"""Analyze correlation between parser quality and grounding performance.

Reads Phase 3.5 experiment predictions and correlates parse quality with grounding outcomes.

Outputs:
- parser_to_grounding_analysis.json
- parser_to_grounding_analysis.md
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag3d.parsers.parse_quality import compute_parse_confidence_bucket
from rag3d.utils.logging import setup_logging

log = logging.getLogger(__name__)


def load_predictions(experiment_dir: Path) -> List[Dict[str, Any]]:
    """Load predictions from experiment directory."""
    pred_file = experiment_dir / "predictions.json"
    if pred_file.is_file():
        with pred_file.open("r", encoding="utf-8") as f:
            return json.load(f)
    return []


def compute_accuracy(predictions: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute accuracy metrics from predictions."""
    if not predictions:
        return {"acc@1": 0.0, "acc@5": 0.0}

    correct_at1 = 0
    correct_at5 = 0

    for pred in predictions:
        target_id = pred.get("target_id", "")
        pred_top1 = pred.get("pred_top1", "")
        pred_top5 = pred.get("pred_top5", [])

        if pred_top1 == target_id:
            correct_at1 += 1
        if target_id in pred_top5:
            correct_at5 += 1

    n = len(predictions)
    return {
        "acc@1": correct_at1 / n if n > 0 else 0.0,
        "acc@5": correct_at5 / n if n > 0 else 0.0,
    }


def group_by_parse_status(predictions: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group predictions by parse status."""
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for pred in predictions:
        status = pred.get("parse_status", "unknown")
        groups[status].append(pred)
    return dict(groups)


def group_by_confidence_bucket(predictions: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group predictions by parser confidence bucket."""
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for pred in predictions:
        confidence = pred.get("parser_confidence", 0.0)
        bucket = compute_parse_confidence_bucket(confidence)
        groups[bucket].append(pred)
    return dict(groups)


def group_by_fallback_triggered(predictions: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group predictions by fallback triggered."""
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for pred in predictions:
        triggered = pred.get("fallback_triggered", False)
        key = "fallback_triggered" if triggered else "no_fallback"
        groups[key].append(pred)
    return dict(groups)


def analyze_parser_grounding_correlation(
    output_dir: Path,
    experiments: List[str],
    predictions_dir: Path,
) -> Dict[str, Any]:
    """
    Analyze correlation between parser quality and grounding performance.

    Args:
        output_dir: Directory to export analysis
        experiments: List of experiment names
        predictions_dir: Base directory for experiment predictions

    Returns:
        Dict with analysis results
    """
    setup_logging()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results: Dict[str, Any] = {}

    for exp_name in experiments:
        exp_dir = predictions_dir / exp_name
        predictions = load_predictions(exp_dir)

        if not predictions:
            log.warning(f"No predictions found for {exp_name}")
            continue

        log.info(f"Analyzing {exp_name}: {len(predictions)} predictions")

        # Overall accuracy
        overall_acc = compute_accuracy(predictions)

        # By parse status
        status_groups = group_by_parse_status(predictions)
        status_acc = {}
        for status, preds in status_groups.items():
            status_acc[status] = {
                **compute_accuracy(preds),
                "count": len(preds),
            }

        # By confidence bucket
        confidence_groups = group_by_confidence_bucket(predictions)
        confidence_acc = {}
        for bucket, preds in confidence_groups.items():
            confidence_acc[bucket] = {
                **compute_accuracy(preds),
                "count": len(preds),
            }

        # By fallback triggered
        fallback_groups = group_by_fallback_triggered(predictions)
        fallback_acc = {}
        for key, preds in fallback_groups.items():
            fallback_acc[key] = {
                **compute_accuracy(preds),
                "count": len(preds),
            }

        all_results[exp_name] = {
            "overall": overall_acc,
            "by_parse_status": status_acc,
            "by_confidence_bucket": confidence_acc,
            "by_fallback_triggered": fallback_acc,
            "total_samples": len(predictions),
        }

    # Export JSON
    json_path = output_dir / "parser_to_grounding_analysis.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    log.info(f"Exported JSON to {json_path}")

    # Export Markdown
    md_path = output_dir / "parser_to_grounding_analysis.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Parser-to-Grounding Correlation Analysis\n\n")
        f.write(f"**Timestamp**: {all_results.get('timestamp', 'N/A')}\n\n")

        for exp_name, results in all_results.items():
            f.write(f"## {exp_name}\n\n")

            # Overall
            f.write("### Overall Accuracy\n\n")
            f.write(f"- Acc@1: {results['overall']['acc@1']:.4f}\n")
            f.write(f"- Acc@5: {results['overall']['acc@5']:.4f}\n")
            f.write(f"- Samples: {results['total_samples']}\n\n")

            # By parse status
            f.write("### Accuracy by Parse Status\n\n")
            f.write("| Status | Count | Acc@1 | Acc@5 |\n")
            f.write("|---|---|---|---|\n")
            for status, acc in sorted(results["by_parse_status"].items()):
                f.write(f"| {status} | {acc['count']} | {acc['acc@1']:.4f} | {acc['acc@5']:.4f} |\n")
            f.write("\n")

            # By confidence bucket
            f.write("### Accuracy by Confidence Bucket\n\n")
            f.write("| Bucket | Count | Acc@1 | Acc@5 |\n")
            f.write("|---|---|---|---|\n")
            for bucket in ["high", "medium", "low"]:
                if bucket in results["by_confidence_bucket"]:
                    acc = results["by_confidence_bucket"][bucket]
                    f.write(f"| {bucket} | {acc['count']} | {acc['acc@1']:.4f} | {acc['acc@5']:.4f} |\n")
            f.write("\n")

            # By fallback triggered
            if results.get("by_fallback_triggered"):
                f.write("### Accuracy by Fallback Triggered\n\n")
                f.write("| Condition | Count | Acc@1 | Acc@5 |\n")
                f.write("|---|---|---|---|\n")
                for key, acc in results["by_fallback_triggered"].items():
                    f.write(f"| {key} | {acc['count']} | {acc['acc@1']:.4f} | {acc['acc@5']:.4f} |\n")
                f.write("\n")

    log.info(f"Exported Markdown to {md_path}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Analyze parser-to-grounding correlation")
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        default=ROOT / "outputs/20260402_phase3_5_formal_rerun",
        help="Directory containing experiment predictions",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to export analysis",
    )
    parser.add_argument(
        "--experiments",
        type=str,
        nargs="+",
        default=[
            "heuristic_parser_no_fallback",
            "vlm_parser_no_fallback",
            "vlm_parser_hard_fallback",
            "vlm_parser_hybrid_fallback",
            "raw_text_relation_baseline",
            "attribute_only_baseline",
        ],
        help="Experiment names to analyze",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or args.predictions_dir

    analyze_parser_grounding_correlation(
        output_dir=output_dir,
        experiments=args.experiments,
        predictions_dir=args.predictions_dir,
    )


if __name__ == "__main__":
    main()