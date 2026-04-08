#!/usr/bin/env python3
"""Phase 3 Parser Ablation Experiment Runner.

Runs controlled comparisons of:
1. Parser source: heuristic vs cached_vlm
2. Fallback mode: none vs hard vs hybrid
3. Baseline comparison: raw_text_relation baseline

All experiments share the same evaluation pipeline and export metrics.
"""

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag3d.datasets.collate import make_grounding_collate_fn
from rag3d.datasets.referit3d import ReferIt3DManifestDataset
from rag3d.datasets.schemas import ParsedUtterance
from rag3d.evaluation.metrics import compute_overall_metrics, compute_diagnostic_metrics
from rag3d.parsers import build_parser_from_config
from rag3d.parsers.parse_quality import classify_parse_quality_batch, compute_parse_confidence_bucket
from rag3d.relation_reasoner.fallback_controller import FallbackController, build_fallback_controller_from_config
from rag3d.relation_reasoner.structured_relation_model import StructuredRelationModel
from rag3d.relation_reasoner.model import RawTextRelationModel, AttributeOnlyModel
from rag3d.utils.config import load_yaml_config
from rag3d.utils.logging import setup_logging
from rag3d.utils.seed import set_seed

log = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for a single parser ablation experiment."""

    name: str
    parser_source: str
    fallback_mode: str
    confidence_threshold: float
    hybrid_blend_factor: float
    model_type: str  # structured_relation / raw_text_relation / attribute_only


@dataclass
class ExperimentResult:
    """Results from a single experiment run."""

    config: ExperimentConfig
    overall_metrics: Dict[str, float]
    diagnostic_metrics: Dict[str, float]
    parse_statistics: Optional[Dict[str, Any]] = None
    fallback_statistics: Optional[Dict[str, Any]] = None
    predictions: List[Dict[str, Any]] = None


PARSER_ABLATION_CONFIGS: List[ExperimentConfig] = [
    # Parser source comparison
    ExperimentConfig(
        name="heuristic_parser_no_fallback",
        parser_source="heuristic",
        fallback_mode="none",
        confidence_threshold=0.5,
        hybrid_blend_factor=0.5,
        model_type="structured_relation",
    ),
    ExperimentConfig(
        name="vlm_parser_no_fallback",
        parser_source="cached_vlm",
        fallback_mode="none",
        confidence_threshold=0.5,
        hybrid_blend_factor=0.5,
        model_type="structured_relation",
    ),
    # Fallback comparison (VLM parser)
    ExperimentConfig(
        name="vlm_parser_hard_fallback",
        parser_source="cached_vlm",
        fallback_mode="hard",
        confidence_threshold=0.5,
        hybrid_blend_factor=0.5,
        model_type="structured_relation",
    ),
    ExperimentConfig(
        name="vlm_parser_hybrid_fallback",
        parser_source="cached_vlm",
        fallback_mode="hybrid",
        confidence_threshold=0.5,
        hybrid_blend_factor=0.5,
        model_type="structured_relation",
    ),
    # Baseline comparison
    ExperimentConfig(
        name="raw_text_relation_baseline",
        parser_source="heuristic",
        fallback_mode="none",
        confidence_threshold=0.5,
        hybrid_blend_factor=0.5,
        model_type="raw_text_relation",
    ),
    ExperimentConfig(
        name="attribute_only_baseline",
        parser_source="heuristic",
        fallback_mode="none",
        confidence_threshold=0.5,
        hybrid_blend_factor=0.5,
        model_type="attribute_only",
    ),
]


def build_model_from_config(
    exp_config: ExperimentConfig,
    model_config: Dict[str, Any],
    fallback_controller: Optional[FallbackController] = None,
) -> torch.nn.Module:
    """Build model instance from experiment config."""

    if exp_config.model_type == "structured_relation":
        return StructuredRelationModel(
            object_dim=int(model_config["object_dim"]),
            language_dim=int(model_config["language_dim"]),
            hidden_dim=int(model_config["hidden_dim"]),
            relation_dim=int(model_config["relation_dim"]),
            anchor_temperature=float(model_config.get("anchor_temperature", 1.0)),
            use_hierarchical_anchor=model_config.get("use_hierarchical_anchor", False),
            dropout=float(model_config.get("dropout", 0.1)),
            fallback_controller=fallback_controller,
        )
    elif exp_config.model_type == "raw_text_relation":
        return RawTextRelationModel(
            object_dim=int(model_config["object_dim"]),
            language_dim=int(model_config["language_dim"]),
            hidden_dim=int(model_config["hidden_dim"]),
            relation_dim=int(model_config["relation_dim"]),
            dropout=float(model_config.get("dropout", 0.1)),
        )
    elif exp_config.model_type == "attribute_only":
        return AttributeOnlyModel(
            object_dim=int(model_config["object_dim"]),
            language_dim=int(model_config["language_dim"]),
            hidden_dim=int(model_config["hidden_dim"]),
            dropout=float(model_config.get("dropout", 0.1)),
        )
    else:
        raise ValueError(f"Unknown model_type: {exp_config.model_type}")


def run_single_experiment(
    exp_config: ExperimentConfig,
    dataset_config: Dict[str, Any],
    model_config: Dict[str, Any],
    eval_manifest_path: Path,
    output_dir: Path,
    device: torch.device,
    batch_size: int = 8,
    debug_max_batches: Optional[int] = None,
) -> ExperimentResult:
    """Run a single experiment and return results."""

    log.info(f"Running experiment: {exp_config.name}")

    # Build parser
    parser = build_parser_from_config(
        parser_source=exp_config.parser_source,
        cache_dir=ROOT / "data/parser_cache",
    )

    # Build fallback controller
    fallback_controller = None
    if exp_config.model_type == "structured_relation":
        fallback_controller = FallbackController(
            mode=exp_config.fallback_mode,
            confidence_threshold=exp_config.confidence_threshold,
            hybrid_blend_factor=exp_config.hybrid_blend_factor,
        )

    # Build model
    model = build_model_from_config(exp_config, model_config, fallback_controller)
    model = model.to(device)
    model.eval()

    # Load dataset
    dataset = ReferIt3DManifestDataset(manifest_path=eval_manifest_path)
    log.info(f"Loaded {len(dataset)} samples for evaluation")

    # Create data loader
    feat_dim = int(model_config["object_dim"])
    collate_fn = make_grounding_collate_fn(feat_dim, attach_features=True)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # Run evaluation
    all_predictions: List[Dict[str, Any]] = []
    all_targets: List[Dict[str, Any]] = []
    all_scores: List[List[float]] = []
    all_parsed: List[ParsedUtterance] = []
    all_fallback_decisions: List[Any] = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if debug_max_batches is not None and batch_idx >= debug_max_batches:
                break

            # Move tensors to device
            tensors = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            # Parse utterances for structured model
            samples = batch.get("samples_ref", [])
            parsed_list: List[ParsedUtterance] = []
            for s in samples:
                parsed = parser.parse(s.utterance)
                parsed_list.append(parsed)
                all_parsed.append(parsed)

            # Forward pass
            if exp_config.model_type == "structured_relation":
                results = model(tensors, parsed_list=parsed_list)
                logits = results["logits"]
                fallback_decisions = results.get("fallback_decisions", [])
                all_fallback_decisions.extend(fallback_decisions)
            else:
                logits = model(tensors)

            # Compute predictions
            probs = torch.softmax(logits, dim=-1)
            mask = tensors["object_mask"]

            for i in range(len(samples)):
                sample = samples[i]
                sample_probs = probs[i]
                sample_mask = mask[i]

                # Mask invalid objects
                masked_probs = sample_probs.clone()
                masked_probs[~sample_mask] = float("-inf")

                # Get top predictions
                top1_idx = torch.argmax(masked_probs)
                top5_indices = torch.topk(masked_probs, k=min(5, sample_mask.sum().item()))[1]

                pred_record = {
                    "scene_id": sample.scene_id,
                    "target_id": sample.target_object_id,
                    "pred_top1": str(top1_idx.item()),
                    "pred_top5": [str(idx.item()) for idx in top5_indices],
                    "utterance": sample.utterance,
                    "parser_source": parsed_list[i].parse_source,
                    "parse_status": parsed_list[i].parse_status,
                    "parser_confidence": parsed_list[i].parser_confidence,
                }

                # Add fallback info for structured model
                if exp_config.model_type == "structured_relation" and i < len(fallback_decisions):
                    dec = fallback_decisions[i]
                    pred_record["fallback_triggered"] = dec.should_fallback
                    pred_record["fallback_reason"] = dec.reason
                    pred_record["structured_weight"] = dec.structured_weight
                    pred_record["raw_text_weight"] = dec.raw_text_weight

                all_predictions.append(pred_record)
                all_targets.append({
                    "scene_id": sample.scene_id,
                    "target_id": sample.target_object_id,
                    "utterance": sample.utterance,
                })
                all_scores.append(sample_probs[sample_mask].tolist())

    # Compute metrics
    overall_metrics = compute_overall_metrics(all_predictions, all_targets)
    diagnostic_metrics = compute_diagnostic_metrics(all_predictions, all_targets, all_scores)

    # Compute parse statistics
    parse_stats = classify_parse_quality_batch(all_parsed)
    confidence_buckets = {"high": 0, "medium": 0, "low": 0}
    for p in all_parsed:
        bucket = compute_parse_confidence_bucket(p.parser_confidence)
        confidence_buckets[bucket] += 1

    # Compute fallback statistics
    fallback_stats = None
    if exp_config.model_type == "structured_relation" and all_fallback_decisions:
        controller = FallbackController(
            mode=exp_config.fallback_mode,
            confidence_threshold=exp_config.confidence_threshold,
        )
        fallback_stats = controller.get_statistics(all_fallback_decisions)

    # Export predictions
    pred_output_dir = output_dir / exp_config.name
    pred_output_dir.mkdir(parents=True, exist_ok=True)
    with (pred_output_dir / "predictions.json").open("w") as f:
        json.dump(all_predictions, f, indent=2)

    log.info(
        f"Experiment {exp_config.name}: Acc@1={overall_metrics['acc_at_1']:.4f}, "
        f"Acc@5={overall_metrics['acc_at_5']:.4f}"
    )

    return ExperimentResult(
        config=exp_config,
        overall_metrics=overall_metrics,
        diagnostic_metrics=diagnostic_metrics,
        parse_statistics={"status_counts": parse_stats, "confidence_buckets": confidence_buckets},
        fallback_statistics=fallback_stats,
        predictions=all_predictions[:100],  # Sample for report
    )


def run_phase3_ablation(
    output_dir: Path,
    eval_manifest_path: Optional[Path] = None,
    batch_size: int = 8,
    device_str: str = "cuda",
    debug_max_batches: Optional[int] = None,
    configs: Optional[List[ExperimentConfig]] = None,
) -> List[ExperimentResult]:
    """Run all Phase 3 parser ablation experiments."""

    setup_logging()
    set_seed(42)

    device = torch.device(device_str if torch.cuda.is_available() and device_str == "cuda" else "cpu")
    log.info(f"Using device: {device}")

    # Load configs
    dataset_config = load_yaml_config(ROOT / "configs/dataset/referit3d.yaml")
    model_config = load_yaml_config(ROOT / "configs/model/relation_aware.yaml")

    # Resolve eval manifest path
    if eval_manifest_path is None:
        processed_dir = ROOT / dataset_config.get("processed_dir", "data/processed")
        eval_manifest_path = processed_dir / "val_manifest.jsonl"

    if not eval_manifest_path.is_file():
        log.error(f"Eval manifest not found: {eval_manifest_path}")
        raise FileNotFoundError(f"Eval manifest not found: {eval_manifest_path}")

    # Use provided configs or default
    exp_configs = configs or PARSER_ABLATION_CONFIGS

    results: List[ExperimentResult] = []
    for exp_config in exp_configs:
        result = run_single_experiment(
            exp_config=exp_config,
            dataset_config=dataset_config,
            model_config=model_config,
            eval_manifest_path=eval_manifest_path,
            output_dir=output_dir,
            device=device,
            batch_size=batch_size,
            debug_max_batches=debug_max_batches,
        )
        results.append(result)

    # Export comparison summary
    summary_path = output_dir / "phase3_comparison_summary.json"
    summary = {
        "timestamp": datetime.now().isoformat(),
        "eval_manifest": str(eval_manifest_path),
        "device": str(device),
        "results": [
            {
                "name": r.config.name,
                "model_type": r.config.model_type,
                "parser_source": r.config.parser_source,
                "fallback_mode": r.config.fallback_mode,
                "overall_metrics": r.overall_metrics,
                "diagnostic_metrics": r.diagnostic_metrics,
                "parse_statistics": r.parse_statistics,
                "fallback_statistics": r.fallback_statistics,
            }
            for r in results
        ],
    }
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    # Export markdown summary
    md_path = output_dir / "phase3_comparison_summary.md"
    with md_path.open("w") as f:
        f.write("# Phase 3 Parser Ablation Comparison\n\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n\n")
        f.write(f"Eval manifest: `{eval_manifest_path}`\n\n")
        f.write("## Results Summary\n\n")
        f.write("| Experiment | Model | Parser | Fallback | Acc@1 | Acc@5 |\n")
        f.write("|---|---|---|---|---|---|\n")
        for r in results:
            f.write(
                f"| {r.config.name} | {r.config.model_type} | "
                f"{r.config.parser_source} | {r.config.fallback_mode} | "
                f"{r.overall_metrics['acc_at_1']:.4f} | "
                f"{r.overall_metrics['acc_at_5']:.4f} |\n"
            )
        f.write("\n## Parse Statistics\n\n")
        for r in results:
            if r.parse_statistics:
                f.write(f"### {r.config.name}\n\n")
                f.write(f"- Parse status counts: {r.parse_statistics['status_counts']}\n")
                f.write(f"- Confidence buckets: {r.parse_statistics['confidence_buckets']}\n\n")
        f.write("\n## Fallback Statistics\n\n")
        for r in results:
            if r.fallback_statistics:
                f.write(f"### {r.config.name}\n\n")
                for k, v in r.fallback_statistics.items():
                    f.write(f"- {k}: {v}\n")
                f.write("\n")

    log.info(f"Exported summary to {summary_path} and {md_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Phase 3 Parser Ablation Experiment")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for results",
    )
    parser.add_argument(
        "--eval-manifest",
        type=Path,
        default=None,
        help="Path to eval manifest (default: data/processed/val_manifest.jsonl)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (cuda/cpu)",
    )
    parser.add_argument(
        "--debug",
        type=int,
        default=None,
        help="Max batches for debug mode",
    )
    args = parser.parse_args()

    # Resolve output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = ROOT / "outputs" / f"{timestamp}_phase3_parser_ablation"

    output_dir.mkdir(parents=True, exist_ok=True)

    run_phase3_ablation(
        output_dir=output_dir,
        eval_manifest_path=args.eval_manifest,
        batch_size=args.batch_size,
        device_str=args.device,
        debug_max_batches=args.debug,
    )


if __name__ == "__main__":
    main()