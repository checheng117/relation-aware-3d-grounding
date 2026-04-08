#!/usr/bin/env python3
"""Export case studies for 3D grounding analysis."""

import argparse
import sys
from pathlib import Path
import json
import numpy as np
from typing import Dict, List, Any

# Add src to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag3d.datasets.schema import GroundingSample, ObjectRecord, GroundingBatch
from rag3d.evaluation.metrics import compute_diagnostic_metrics
from rag3d.diagnostics.failure_taxonomy import apply_heuristic_hard_case_tags, generate_failure_summary
from rag3d.diagnostics.tagging import generate_hard_case_tags, summarize_hard_cases


def export_case_studies(
    predictions: List[Dict[str, Any]],
    targets: List[Dict[str, Any]],
    scores: List[List[float]],
    output_dir: Path,
    top_n: int = 5
):
    """
    Export case studies including successful cases, failures, and ambiguous cases.

    Args:
        predictions: List of prediction dictionaries
        targets: List of target dictionaries
        scores: List of confidence scores for each sample
        output_dir: Directory to save case studies
        top_n: Number of cases to include in each category
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Identify different types of cases
    successful_cases = []
    failed_cases = []
    low_margin_cases = []
    same_class_clutter_cases = []

    for i, (pred, target, sample_scores) in enumerate(zip(predictions, targets, scores)):
        target_id = target.get('target_id', target.get('target_obj_id', ''))
        pred_top1 = pred.get('pred_top1', '')

        # Calculate margin if scores available
        margin = float('inf')
        if sample_scores and target_id in pred.get('candidate_object_ids', []):
            target_idx = pred.get('candidate_object_ids', []).index(target_id)
            pred_idx = pred.get('candidate_object_ids', []).index(pred_top1) if pred_top1 in pred.get('candidate_object_ids', []) else -1

            if 0 <= pred_idx < len(sample_scores) and 0 <= target_idx < len(sample_scores):
                margin = abs(sample_scores[pred_idx] - sample_scores[target_idx])

        # Create case record
        case_record = {
            'index': i,
            'scene_id': target.get('scene_id', ''),
            'utterance': target.get('utterance', target.get('sentence', '')),
            'target_id': target_id,
            'predicted_id': pred_top1,
            'candidate_count': len(pred.get('candidate_object_ids', [])),
            'margin': margin,
            'correct': pred_top1 == target_id
        }

        # Categorize cases
        if pred_top1 == target_id:
            successful_cases.append(case_record)
        else:
            failed_cases.append(case_record)

        if margin < 0.1:  # Low margin indicates ambiguity
            low_margin_cases.append(case_record)

    # Sort cases by margin (for low margin cases) or by score (for successful cases)
    successful_cases.sort(key=lambda x: x['margin'])  # Lower margin successful cases are more interesting
    failed_cases.sort(key=lambda x: x['margin'], reverse=True)  # Higher margin failures show more confusion
    low_margin_cases.sort(key=lambda x: x['margin'])

    # Create case study reports
    case_reports = {
        'successful_cases': successful_cases[:top_n],
        'failed_cases': failed_cases[:top_n],
        'low_margin_cases': low_margin_cases[:top_n],
        'summary_statistics': {
            'total_samples': len(predictions),
            'accuracy': len([c for c in successful_cases]) / len(predictions) if predictions else 0,
            'average_margin': np.mean([c['margin'] for c in (successful_cases + failed_cases) if c['margin'] != float('inf')]) if (successful_cases + failed_cases) else 0,
            'low_margin_ratio': len(low_margin_cases) / len(predictions) if predictions else 0
        }
    }

    # Export case studies
    with open(output_dir / 'case_studies.json', 'w') as f:
        json.dump(case_reports, f, indent=2)

    # Create markdown report
    md_content = create_case_study_report(case_reports)
    with open(output_dir / 'case_studies.md', 'w') as f:
        f.write(md_content)

    print(f"Case studies exported to {output_dir}")
    print(f"  - Successful cases: {len(successful_cases)}")
    print(f"  - Failed cases: {len(failed_cases)}")
    print(f"  - Low margin cases: {len(low_margin_cases)}")


def create_case_study_report(case_reports: Dict[str, Any]) -> str:
    """Create a markdown report of case studies."""
    md = "# Case Studies Report\n\n"

    # Summary statistics
    summary = case_reports['summary_statistics']
    md += f"## Summary\n\n"
    md += f"- Total samples: {summary['total_samples']}\n"
    md += f"- Accuracy: {summary['accuracy']:.4f}\n"
    md += f"- Average margin: {summary['average_margin']:.4f}\n"
    md += f"- Low margin ratio: {summary['low_margin_ratio']:.4f}\n\n"

    # Successful cases
    md += "## Successful Cases\n\n"
    for i, case in enumerate(case_reports['successful_cases'][:3]):  # Show top 3
        md += f"### Case {i+1}: Scene {case['scene_id']}\n"
        md += f"- Utterance: \"{case['utterance']}\"\n"
        md += f"- Target: {case['target_id']}, Predicted: {case['predicted_id']}\n"
        md += f"- Margin: {case['margin']:.4f}, Candidates: {case['candidate_count']}\n\n"

    # Failed cases
    md += "## Failed Cases\n\n"
    for i, case in enumerate(case_reports['failed_cases'][:3]):  # Show top 3
        md += f"### Case {i+1}: Scene {case['scene_id']}\n"
        md += f"- Utterance: \"{case['utterance']}\"\n"
        md += f"- Target: {case['target_id']}, Predicted: {case['predicted_id']}\n"
        md += f"- Margin: {case['margin']:.4f}, Candidates: {case['candidate_count']}\n\n"

    # Low margin cases (ambiguous)
    md += "## Low Margin Cases (Ambiguous)\n\n"
    for i, case in enumerate(case_reports['low_margin_cases'][:3]):  # Show top 3
        md += f"### Case {i+1}: Scene {case['scene_id']}\n"
        md += f"- Utterance: \"{case['utterance']}\"\n"
        md += f"- Target: {case['target_id']}, Predicted: {case['predicted_id']}\n"
        md += f"- Margin: {case['margin']:.4f}, Candidates: {case['candidate_count']}\n\n"

    return md


def main():
    parser = argparse.ArgumentParser(description='Export case studies for 3D grounding analysis')
    parser.add_argument('--predictions-path', type=Path, required=True,
                        help='Path to predictions JSON file')
    parser.add_argument('--targets-path', type=Path, required=True,
                        help='Path to targets JSON file')
    parser.add_argument('--output-dir', type=Path,
                        default=ROOT / 'outputs' / 'case_studies',
                        help='Output directory for case studies')
    parser.add_argument('--top-n', type=int, default=5,
                        help='Number of cases to include in each category')

    args = parser.parse_args()

    # Load predictions and targets
    with open(args.predictions_path, 'r') as f:
        predictions = json.load(f)

    with open(args.targets_path, 'r') as f:
        targets = json.load(f)

    # Extract scores from predictions
    scores = []
    for pred in predictions:
        sample_scores = pred.get('confidence_scores', [])
        scores.append(sample_scores)

    # Export case studies
    export_case_studies(predictions, targets, scores, args.output_dir, args.top_n)

    # Create README for case studies
    readme_content = f"""# Case Studies for 3D Grounding Analysis

This directory contains various case studies analyzing the performance of the 3D grounding system.

## Contents

- `case_studies.json`: Complete case study data in JSON format
- `case_studies.md`: Human-readable markdown report
- `successful_cases/`: Details of correctly predicted samples
- `failed_cases/`: Details of incorrectly predicted samples
- `low_margin_cases/`: Samples with low confidence margins (ambiguous cases)

## Methodology

Cases are categorized based on:
- Prediction correctness (success vs failure)
- Confidence margins between predictions
- Candidate set sizes
- Spatial relation complexity (will be implemented in future versions)

For more details about the evaluation methodology, see the main project documentation.
"""

    readme_path = args.output_dir / 'README.md'
    with open(readme_path, 'w') as f:
        f.write(readme_content)

    print(f"Case study README created at: {readme_path}")


if __name__ == "__main__":
    main()