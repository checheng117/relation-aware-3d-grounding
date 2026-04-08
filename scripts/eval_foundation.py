#!/usr/bin/env python3
"""Foundation evaluation script for 3D grounding."""

import argparse
import sys
from pathlib import Path
import json
import torch
import numpy as np
from typing import List, Dict, Any

# Add src to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag3d.datasets.schema import GroundingSample, ObjectRecord, GroundingBatch
from rag3d.datasets.adapters import adapt_referit3d_sample_to_schema, adapt_object_record_to_schema
from rag3d.evaluation.metrics import (
    compute_overall_metrics,
    compute_stratified_metrics,
    compute_diagnostic_metrics,
    export_results_to_json,
    export_results_to_csv,
    export_results_to_markdown
)
from rag3d.evaluation.stratified_eval import compute_and_export_stratified_evaluation
from rag3d.diagnostics.failure_taxonomy import apply_heuristic_hard_case_tags, generate_failure_summary
from rag3d.diagnostics.tagging import generate_hard_case_tags, summarize_hard_cases
from rag3d.utils.config import load_yaml_config
from rag3d.datasets.referit3d import ReferIt3DManifestDataset


def main():
    parser = argparse.ArgumentParser(description='Foundation evaluation for 3D grounding')
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/eval/foundation_eval.yaml')
    parser.add_argument('--output-dir', type=Path, default=None)
    parser.add_argument('--data-path', type=Path, help='Path to evaluation data')
    parser.add_argument('--predictions-path', type=Path, help='Path to predictions file')
    args = parser.parse_args()

    # Load config
    config = load_yaml_config(args.config, base_dir=ROOT) if args.config.exists() else {}

    # Set up output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = ROOT / "outputs" / f"{timestamp}_foundation_eval"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data - either from file or using ReferIt3D dataset
    if args.data_path and args.data_path.exists():
        # Load from file
        with open(args.data_path, 'r') as f:
            raw_data = json.load(f)

        # Process the data according to your format
        samples = raw_data.get('samples', [])
        objects = raw_data.get('objects', [])
    else:
        # Use ReferIt3D dataset if available
        dataset_config_path = config.get('dataset_config', ROOT / 'configs/dataset/referit3d.yaml')
        if dataset_config_path.exists():
            dataset_config = load_yaml_config(dataset_config_path, base_dir=ROOT)
            # Use the correct class name
            dataset = ReferIt3DManifestDataset(
                manifest_path=Path(dataset_config.get('processed_dir', 'data/processed')) / f"{config.get('eval_split', 'val')}_manifest.jsonl",
            )

            # Convert dataset to our schema
            samples = []
            objects = []
            for i in range(min(config.get('max_samples', 100), len(dataset))):
                raw_sample = dataset[i]
                # Convert to schema format
                sample = adapt_referit3d_sample_to_schema(raw_sample.__dict__ if hasattr(raw_sample, '__dict__') else {}, [])
                samples.append(sample)
        else:
            # Create dummy data for testing
            print("No data source provided, creating dummy data for testing...")
            samples = []
            objects = []

            # Create some dummy samples for testing
            for i in range(5):
                sample = GroundingSample(
                    scene_id=f"scene_{i}",
                    utterance=f"This is a test utterance for object {i}",
                    target_id=f"obj_{i}_0",
                    candidate_object_ids=[f"obj_{i}_0", f"obj_{i}_1", f"obj_{i}_2"],
                    relation_tags=["left", "near"] if i % 2 == 0 else [],
                    difficulty_tags=["easy"] if i < 3 else ["hard"]
                )
                samples.append(sample)

    # Load predictions if provided
    if args.predictions_path and args.predictions_path.exists():
        with open(args.predictions_path, 'r') as f:
            predictions = json.load(f)
    else:
        # Generate dummy predictions for testing
        print("No predictions provided, creating dummy predictions for testing...")
        predictions = []
        targets = []

        for sample in samples:
            # Create dummy prediction data
            pred = {
                'scene_id': sample.scene_id,
                'target_id': sample.target_id,
                'pred_top1': sample.candidate_object_ids[0],  # Predict first candidate
                'pred_top5': sample.candidate_object_ids[:5],  # Top 5 candidates
                'candidate_object_ids': sample.candidate_object_ids,
                'confidence_scores': [0.9 - i*0.1 for i in range(len(sample.candidate_object_ids))]  # Dummy scores
            }
            predictions.append(pred)

            target = {
                'scene_id': sample.scene_id,
                'target_id': sample.target_id,
                'candidate_object_ids': sample.candidate_object_ids,
                'utterance': sample.utterance
            }
            targets.append(target)

    print(f"Evaluating on {len(predictions)} samples...")

    # Compute overall metrics
    print("Computing overall metrics...")
    overall_metrics = compute_overall_metrics(predictions, targets)
    print(f"Overall Acc@1: {overall_metrics['acc_at_1']:.4f}")
    print(f"Overall Acc@5: {overall_metrics['acc_at_5']:.4f}")

    # Compute diagnostic metrics
    print("Computing diagnostic metrics...")
    scores = [pred.get('confidence_scores', []) for pred in predictions]
    diagnostic_metrics = compute_diagnostic_metrics(predictions, targets, scores)
    print(f"Average target margin: {diagnostic_metrics['avg_target_margin']:.4f}")
    print(f"Failure rate: {diagnostic_metrics['failure_rate']:.4f}")

    # Apply hard case tags
    print("Applying hard case tags...")
    hard_case_tags = generate_hard_case_tags(targets, [[] for _ in targets])  # Empty objects for demo
    hard_case_summary = summarize_hard_cases(hard_case_tags)

    # Apply failure taxonomy tags
    print("Applying failure taxonomy...")
    failure_tags = apply_heuristic_hard_case_tags(samples, predictions, targets, scores)
    failure_summary = generate_failure_summary(failure_tags)

    # Compute stratified metrics
    print("Computing stratified metrics...")
    # For stratified metrics, we need to create tag info for each sample
    tag_info = []
    for target in targets:
        # Create a tag dictionary for each target
        tag_dict = {
            'relation_tags': [],
            'difficulty_tags': [],
            'same_class_clutter_count': 0,
            'relation_heavy': False,
            'attribute_dominant': False,
            'occlusion_heavy': False
        }
        tag_info.append(tag_dict)

    stratified_metrics = compute_stratified_metrics(predictions, targets, tag_info)

    # Compile all results
    results = {
        'overall': overall_metrics,
        'diagnostic': diagnostic_metrics,
        'stratified': stratified_metrics,
        'hard_case_summary': hard_case_summary,
        'failure_summary': failure_summary,
        'config': config
    }

    # Export results in multiple formats
    print(f"Exporting results to {output_dir}...")
    export_results_to_json(results, output_dir / 'evaluation_results.json')
    export_results_to_csv(results, output_dir / 'evaluation_results.csv')
    export_results_to_markdown(results, output_dir / 'evaluation_results.md')

    # Also export stratified results separately
    compute_and_export_stratified_evaluation(
        predictions,
        targets,
        output_dir / 'stratified',
        export_formats=['json', 'csv', 'markdown']
    )

    print(f"Evaluation complete. Results saved to {output_dir}")


if __name__ == "__main__":
    main()