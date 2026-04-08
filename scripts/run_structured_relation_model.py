#!/usr/bin/env python3
"""Training script for the structured relation model."""

import argparse
import sys
from pathlib import Path
import json
import torch
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Any, List

# Add src to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag3d.datasets.schema import GroundingSample, ObjectRecord, GroundingBatch
from rag3d.datasets.adapters import adapt_referit3d_sample_to_schema, adapt_object_record_to_schema
from rag3d.datasets.referit3d import ReferIt3DManifestDataset
from rag3d.relation_reasoner.structured_relation_model import StructuredRelationModel
from rag3d.parsers.structured_parser import StructuredParserInterface
from rag3d.training.runner import TrainingConfig, build_loaders, run_training_loop
from rag3d.utils.config import load_yaml_config
from rag3d.utils.logging import setup_logging
from rag3d.utils.seed import set_seed
from rag3d.datasets.collate import collate_grounding_samples
from rag3d.evaluation.metrics import (
    compute_overall_metrics,
    compute_diagnostic_metrics,
    export_results_to_json,
    export_results_to_markdown
)
from rag3d.evaluation.stratified_eval import compute_and_export_stratified_evaluation
from rag3d.diagnostics.failure_taxonomy import apply_heuristic_hard_case_tags, generate_failure_summary
from rag3d.diagnostics.tagging import generate_hard_case_tags, summarize_hard_cases


def _resolve(p: Path | None, base: Path) -> Path | None:
    if p is None:
        return None
    return p if p.is_absolute() else (base / p).resolve()


def _manifest_paths(tcfg: dict, dcfg: dict, base: Path) -> tuple[Path, Path | None]:
    mode = str(tcfg.get("mode", "real"))
    proc = Path(dcfg.get("processed_dir", "data/processed"))
    if not proc.is_absolute():
        proc = base / proc
    if mode == "debug":
        proc = proc / str(tcfg.get("debug_processed_subdir", "debug"))
    tr = tcfg.get("train_manifest")
    va = tcfg.get("val_manifest")
    train_p = _resolve(Path(tr), base) if tr else proc / "train_manifest.jsonl"
    val_p = _resolve(Path(va), base) if va else proc / "val_manifest.jsonl"
    return train_p, val_p if val_p.is_file() else None


def structured_relation_forward(model: StructuredRelationModel, batch: Dict[str, Any]):
    """
    Forward function for the structured relation model.
    """
    # For this initial implementation, we'll call the model directly without parsing
    # In a future enhancement, we can pass parsed utterances if available
    results = model(batch, parsed_list=None)
    return results['logits']


def run_structured_relation_model(config_path: Path, output_dir: Path, run_train: bool = True, run_eval: bool = True):
    """Run the structured relation model training and evaluation."""
    setup_logging()

    # Load configuration
    config = load_yaml_config(config_path, base_dir=ROOT)
    dataset_config = load_yaml_config(ROOT / config["dataset_config"], base_dir=ROOT)

    model_key = str(config.get("model", "structured_relation"))
    model_config = load_yaml_config(ROOT / "configs/model" / f"{model_key}.yaml", base_dir=ROOT)

    # Set seed
    set_seed(int(config.get("seed", 42)))
    device_str = str(config.get("device", "cpu"))
    device = torch.device(device_str if torch.cuda.is_available() and device_str == "cuda" else "cpu")

    # Initialize model
    if model_key == "structured_relation":
        model = StructuredRelationModel(
            int(model_config["object_dim"]),
            int(model_config["language_dim"]),
            int(model_config["hidden_dim"]),
            int(model_config["relation_dim"]),
            anchor_temperature=float(model_config.get("anchor_temperature", 1.0)),
            use_hierarchical_anchor=model_config.get("use_hierarchical_anchor", False),
            dropout=float(model_config.get("dropout", 0.1)),
        )
        forward_fn = structured_relation_forward
    else:
        raise ValueError(f"Unsupported model type: {model_key}")

    # Prepare training config
    training_config = TrainingConfig(
        epochs=int(config.get("epochs", 5)),
        batch_size=int(config.get("batch_size", 8)),
        lr=float(config.get("lr", 1e-4)),
        weight_decay=float(config.get("weight_decay", 0.01)),
        seed=int(config.get("seed", 42)),
        feat_dim=int(model_config["object_dim"]),
        checkpoint_dir=_resolve(Path(config["checkpoint_dir"]), ROOT) or ROOT / "outputs/checkpoints",
        metrics_path=_resolve(Path(config.get("metrics_file", "outputs/metrics/train_metrics.jsonl")), ROOT)
        or ROOT / "outputs/metrics/train_metrics.jsonl",
        device=device_str if device.type == "cuda" else "cpu",
        debug_max_batches=config.get("debug_max_batches"),
        loss=dict(config.get("loss") or {}),
    )

    print(f"Using device: {training_config.device}")
    print(f"Training epochs: {training_config.epochs}")
    print(f"Learning rate: {training_config.lr}")

    # Training
    if run_train:
        print("Starting structured relation model training...")

        # Create checkpoint directory
        training_config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Get manifest paths
        train_path, val_path = _manifest_paths(config, dataset_config, ROOT)

        if not train_path.is_file():
            print(f"WARNING: Train manifest missing: {train_path}")
            print("Consider running data preparation first.")
            # For demo purposes, we'll skip training if data is missing
            run_train = False
        else:
            print(f"Training on: {train_path}")
            print(f"Validating on: {val_path or 'None'}")

            # Build data loaders
            train_loader, val_loader = build_loaders(
                train_path,
                val_path,
                training_config,
                num_workers=int(config.get("num_workers", 0)),
            )

            # Run training
            run_training_loop(
                model, train_loader, val_loader, training_config, forward_fn, parser=None, model_name="structured_relation_model"
            )

            # Save final model
            final_checkpoint_path = training_config.checkpoint_dir / "structured_relation_model_final.pt"
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': config,
                'model_config': model_config
            }, final_checkpoint_path)
            print(f"Final model saved to: {final_checkpoint_path}")

    # Evaluation
    if run_eval:
        print("Starting structured relation model evaluation...")

        # Load trained model if available, otherwise initialize randomly
        if run_train:
            checkpoint_path = training_config.checkpoint_dir / "structured_relation_model_final.pt"
            if checkpoint_path.exists():
                checkpoint = torch.load(checkpoint_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Loaded trained model from: {checkpoint_path}")
            else:
                print("Trained model not found, using randomly initialized model")

        model = model.to(device)
        model.eval()

        # Load evaluation data
        eval_split = config.get("eval_split", "val")
        dataset_config_path = ROOT / config["dataset_config"]
        if dataset_config_path.exists():
            dataset_cfg = load_yaml_config(dataset_config_path, base_dir=ROOT)
            manifest_path = Path(dataset_cfg.get('processed_dir', 'data/processed')) / f"{eval_split}_manifest.jsonl"

            if manifest_path.exists():
                dataset = ReferIt3DManifestDataset(manifest_path=manifest_path)
                print(f"Loaded {len(dataset)} samples from {eval_split} split")
            else:
                print(f"Manifest not found at {manifest_path}, using mock data for evaluation")
                dataset = None
        else:
            print("Dataset config not found, using mock data for evaluation")
            dataset = None

        if dataset and len(dataset) > 0:
            # Create data loader for evaluation
            eval_loader = DataLoader(
                dataset,
                batch_size=min(training_config.batch_size, len(dataset)),  # Prevent batch size issues
                shuffle=False,
                collate_fn=collate_grounding_samples
            )

            # Perform inference
            all_predictions = []
            all_targets = []
            all_scores = []
            all_anchor_info = []  # Store anchor-related information

            with torch.no_grad():
                for batch_idx, batch in enumerate(eval_loader):
                    if training_config.debug_max_batches and batch_idx >= training_config.debug_max_batches:
                        break

                    tensors = batch.to_tensors(training_config.feat_dim, device=device)

                    # Forward pass
                    model_results = model(tensors, parsed_list=None)
                    logits = model_results['logits']

                    # Get predictions
                    probs = torch.softmax(logits, dim=-1)

                    # Store predictions and targets
                    for i in range(len(batch.samples)):
                        sample = batch.samples[i]

                        # Apply mask to probabilities
                        sample_logits = logits[i]
                        sample_probs = probs[i]
                        sample_mask = tensors["object_mask"][i]

                        masked_probs = sample_probs.clone()
                        masked_probs[~sample_mask] = float('-inf')

                        # Get top predictions
                        top1_idx = torch.argmax(masked_probs)
                        top5_values, top5_indices = torch.topk(masked_probs, k=min(5, len(sample_mask)), largest=True)

                        # Extract anchor information for this sample
                        anchor_dist = model_results['anchor_dist'][i]
                        anchor_entropy = model_results['anchor_entropy'][i]
                        top_anchor_id = model_results['top_anchor_id'][i]

                        pred_record = {
                            'scene_id': sample.scene_id,
                            'target_id': sample.target_id,
                            'pred_top1': str(top1_idx.item()),
                            'pred_top5': [str(idx.item()) for idx in top5_indices],
                            'candidate_object_ids': [str(j) for j in range(len(sample_mask))],
                            'confidence_scores': sample_probs[sample_mask].tolist(),
                            'model_type': 'structured_relation',
                            'anchor_entropy': float(anchor_entropy.item()),
                            'top_anchor_id': int(top_anchor_id.item()),
                            'anchor_confidence': float(anchor_dist[top_anchor_id].item())
                        }

                        target_record = {
                            'scene_id': sample.scene_id,
                            'target_id': sample.target_id,
                            'candidate_object_ids': [str(j) for j in range(len(sample_mask))],
                            'utterance': sample.utterance
                        }

                        all_predictions.append(pred_record)
                        all_targets.append(target_record)
                        all_scores.append(sample_probs[sample_mask].tolist())

                        # Store anchor information
                        all_anchor_info.append({
                            'anchor_distribution': anchor_dist.cpu().numpy(),
                            'entropy': float(anchor_entropy.item()),
                            'top_anchor': int(top_anchor_id.item()),
                            'max_prob': float(anchor_dist[top_anchor_id].item())
                        })

            print(f"Evaluated {len(all_predictions)} samples")

            # Compute metrics
            overall_metrics = compute_overall_metrics(all_predictions, all_targets)
            diagnostic_metrics = compute_diagnostic_metrics(all_predictions, all_targets, all_scores)

            print(f"Overall Acc@1: {overall_metrics['acc_at_1']:.4f}")
            print(f"Overall Acc@5: {overall_metrics['acc_at_5']:.4f}")
            print(f"Average target margin: {diagnostic_metrics['avg_target_margin']:.4f}")
            print(f"Failure rate: {diagnostic_metrics['failure_rate']:.4f}")

            # Add anchor-specific metrics to results
            avg_anchor_entropy = np.mean([info['entropy'] for info in all_anchor_info])
            print(f"Average anchor entropy: {avg_anchor_entropy:.4f}")

            # Compile results
            results = {
                'overall': overall_metrics,
                'diagnostic': diagnostic_metrics,
                'config': config,
                'model_config': model_config,
                'predictions': all_predictions[:10],  # Include first 10 predictions as sample
                'model_type': 'structured_relation',
                'anchor_info': {
                    'avg_entropy': avg_anchor_entropy,
                    'sample_count': len(all_anchor_info)
                }
            }

            # Export results
            eval_output_dir = output_dir / "structured_relation_model"
            eval_output_dir.mkdir(parents=True, exist_ok=True)

            export_results_to_json(results, eval_output_dir / 'structured_relation_model_results.json')
            export_results_to_markdown(results, eval_output_dir / 'structured_relation_model_results.md')

            # Export stratified evaluation
            compute_and_export_stratified_evaluation(
                all_predictions,
                all_targets,
                eval_output_dir / 'stratified',
                export_formats=['json', 'markdown']
            )

            print(f"Evaluation results saved to: {eval_output_dir}")
        else:
            print("Skipping evaluation due to missing or empty dataset")


def main():
    parser = argparse.ArgumentParser(description='Structured relation model for 3D grounding')
    parser.add_argument('--config', type=Path,
                       default=ROOT / 'configs/train/structured_relation_model.yaml',
                       help='Path to configuration file')
    parser.add_argument('--output-dir', type=Path,
                       default=None,
                       help='Output directory for results')
    parser.add_argument('--no-train', action='store_true',
                       help='Skip training, only run evaluation')
    parser.add_argument('--no-eval', action='store_true',
                       help='Skip evaluation, only run training')

    args = parser.parse_args()

    # Set up output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = ROOT / "outputs" / f"{timestamp}_structured_relation_model"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Check if config exists, if not create a basic one
    if not args.config.exists():
        print(f"Config file {args.config} does not exist, creating a basic config...")
        create_basic_config(args.config)

    # Run the model
    run_structured_relation_model(
        args.config,
        output_dir,
        run_train=not args.no_train,
        run_eval=not args.no_eval
    )


def create_basic_config(config_path: Path):
    """Create a basic configuration file for the structured relation model."""
    basic_config = {
        "dataset_config": "configs/dataset/referit3d.yaml",
        "model": "structured_relation",
        "epochs": 5,
        "batch_size": 8,
        "lr": 1e-4,
        "weight_decay": 0.01,
        "seed": 42,
        "device": "cpu",
        "checkpoint_dir": "outputs/checkpoints/structured_relation_model",
        "metrics_file": "outputs/metrics/structured_relation_model_metrics.jsonl",
        "eval_split": "val",
        "debug_max_batches": 10  # For testing purposes
    }

    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, 'w') as f:
        import yaml
        yaml.dump(basic_config, f, default_flow_style=False)

    print(f"Created basic config at: {config_path}")


if __name__ == "__main__":
    main()