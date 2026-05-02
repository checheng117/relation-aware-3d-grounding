#!/usr/bin/env python3
"""COVER-3D Round-1 Formal Training and Evaluation.

This script trains COVER-3D modules (DenseRelationModule and CalibratedFusionGate)
using real features extracted from the clean sorted-vocabulary baseline.

Training approach:
- Load real base logits and object embeddings from extracted exports
- Train only COVER-3D modules (base model remains frozen/unmodified)
- Use scene-disjoint splits
- Evaluate on hard subsets
- Track harmed/recovered cases

Variants:
- Base: Clean baseline anchor (no COVER-3D, just re-confirm metrics)
- Dense-no-cal: Train DenseRelationModule only, no calibration gate
- Dense-calibrated: Train both DenseRelationModule and CalibratedFusionGate

Usage:
    python scripts/train_cover3d_round1.py --variant dense-no-cal --epochs 10
    python scripts/train_cover3d_round1.py --variant dense-calibrated --epochs 10

Date: 2026-04-19
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag3d.models.cover3d_model import Cover3DModel
from rag3d.models.cover3d_dense_relation import DenseRelationModule
from rag3d.models.cover3d_calibration import CalibratedFusionGate
from rag3d.utils.logging import setup_logging
from rag3d.utils.seed import set_seed

log = logging.getLogger(__name__)


# ============================================================================
# Data Loading
# ============================================================================

def load_clean_predictions(predictions_path: Path) -> List[Dict[str, Any]]:
    """Load clean baseline predictions as list (same ordering as embeddings)."""
    with predictions_path.open() as f:
        data = json.load(f)
    log.info(f"Loaded {len(data)} clean predictions from {predictions_path}")
    return data


def merge_embeddings_with_clean_logits(
    embeddings: List[Dict[str, Any]],
    clean_preds: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Replace embedding logits with clean baseline logits by index matching.

    Since both files have same ordering (verified), we can match by index directly
    instead of using key lookup (which loses duplicates).
    """
    merged = []
    mismatches = 0

    for i, emb in enumerate(embeddings):
        if i < len(clean_preds):
            clean_pred = clean_preds[i]
            # Verify it's the same sample
            if emb["scene_id"] != clean_pred["scene_id"]:
                mismatches += 1
                log.warning(f"Sample {i}: scene_id mismatch emb={emb['scene_id']} clean={clean_pred['scene_id']}")

            # Use clean baseline logits instead of re-run logits
            merged.append({
                **emb,
                "base_logits": clean_pred["base_logits"],
                "pred_top1": clean_pred["pred_top1"],
                "correct_at_1": clean_pred["correct_at_1"],
            })
        else:
            merged.append(emb)

    log.info(f"Replaced {len(embeddings)} logits with clean baseline by index, {mismatches} mismatches")
    return merged


def load_embeddings(embeddings_path: Path) -> List[Dict[str, Any]]:
    """Load extracted embeddings from JSON file."""
    with embeddings_path.open() as f:
        data = json.load(f)
    log.info(f"Loaded {len(data)} embeddings from {embeddings_path}")
    return data


def load_predictions(predictions_path: Path) -> List[Dict[str, Any]]:
    """Load existing predictions with logits."""
    with predictions_path.open() as f:
        data = json.load(f)
    log.info(f"Loaded {len(data)} predictions from {predictions_path}")
    return data


def load_coverage(coverage_path: Path) -> Dict[str, Dict[str, Any]]:
    """Load coverage diagnostics."""
    coverage = {}
    with coverage_path.open() as f:
        for line in f:
            row = json.loads(line)
            key = (row["scene_id"], row["target_id"], row["utterance"])
            coverage[key] = row
    log.info(f"Loaded {len(coverage)} coverage entries from {coverage_path}")
    return coverage


def merge_embeddings_with_coverage(
    embeddings: List[Dict[str, Any]],
    coverage: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Merge embeddings with coverage information for hard-subset tagging."""
    merged = []
    for emb in embeddings:
        key = (emb["scene_id"], emb["target_id"], emb["utterance"])
        cov = coverage.get(key, {})
        merged.append({
            **emb,
            "anchor_ids": cov.get("anchor_ids", []),
            "anchor_count": cov.get("anchor_count", 0),
            "subsets": cov.get("subsets", {}),
            "relation_type": cov.get("relation_type", "other"),
        })
    return merged


def create_sample_key(sample: Dict[str, Any]) -> Tuple[str, str, str]:
    """Create unique key for sample matching."""
    return (
        str(sample.get("scene_id", "")),
        str(sample.get("target_id", "")),
        sample.get("utterance", "").strip().lower(),
    )


# ============================================================================
# Training Dataset
# ============================================================================

class EmbeddingDataset(torch.utils.data.Dataset):
    """Dataset from extracted embeddings."""

    def __init__(self, samples: List[Dict[str, Any]]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Object embeddings
        obj_emb = np.array(sample["object_embeddings"], dtype=np.float32)
        object_embeddings = torch.from_numpy(obj_emb)  # [N, 320]

        # Language features
        lang_feat = np.array(sample["lang_features"], dtype=np.float32)
        lang_features = torch.from_numpy(lang_feat)  # [256]

        # Base logits - use clean baseline logits directly
        base_logits = torch.tensor(sample["base_logits"], dtype=torch.float32)  # [N]

        # Target
        target_index = sample["target_index"]
        if isinstance(target_index, list):
            target_index = target_index[0]  # Handle list-wrapped target
        target_index = int(target_index)

        # Verify logits are correct
        logits_len = len(sample["base_logits"])
        assert len(obj_emb) == logits_len, f"Object count mismatch at {idx}: {len(obj_emb)} vs {logits_len}"

        # Metadata
        num_objects = len(obj_emb)
        object_mask = torch.ones(num_objects, dtype=torch.bool)

        return {
            "object_embeddings": object_embeddings,
            "lang_features": lang_features,
            "base_logits": base_logits,
            "target_index": target_index,
            "object_mask": object_mask,
            "sample_idx": idx,
            "scene_id": sample["scene_id"],
            "utterance": sample["utterance"],
            "anchor_ids": sample.get("anchor_ids", []),
            "anchor_count": sample.get("anchor_count", 0),
            "subsets": sample.get("subsets", {}),
        }


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate batch with variable-size objects."""
    B = len(batch)
    max_n = max(len(s["object_embeddings"]) for s in batch)

    object_embeddings = torch.zeros(B, max_n, 320)
    lang_features = torch.zeros(B, 256)
    base_logits = torch.zeros(B, max_n)
    object_mask = torch.zeros(B, max_n, dtype=torch.bool)
    target_index = torch.zeros(B, dtype=torch.long)
    sample_indices = []

    for i, sample in enumerate(batch):
        N = len(sample["object_embeddings"])
        object_embeddings[i, :N] = sample["object_embeddings"]
        lang_features[i] = sample["lang_features"]
        base_logits[i, :N] = sample["base_logits"]
        object_mask[i, :N] = sample["object_mask"]
        target_index[i] = sample["target_index"]
        sample_indices.append(sample["sample_idx"])

    return {
        "object_embeddings": object_embeddings,
        "lang_features": lang_features,
        "base_logits": base_logits,
        "object_mask": object_mask,
        "target_index": target_index,
        "sample_indices": sample_indices,
        "batch_ref": batch,  # Keep references for metadata
    }


# ============================================================================
# Models
# ============================================================================

class Cover3DReranker(nn.Module):
    """COVER-3D reranker that works on pre-extracted embeddings."""

    def __init__(
        self,
        variant: str = "dense-calibrated",  # "base", "dense-no-cal", "dense-calibrated", "dense-calibrated-v2"
        object_dim: int = 320,
        language_dim: int = 256,
        geometry_dim: int = 6,
        relation_hidden_dim: int = 256,
        relation_chunk_size: int = 16,
        fusion_hidden_dim: int = 32,
        fusion_min_gate: float = 0.1,
        fusion_max_gate: float = 0.9,
        fusion_init_bias: float = 0.5,
        fusion_gate_prior_weight: float = 1.0,
    ):
        super().__init__()

        self.variant = variant
        self.object_dim = object_dim
        self.language_dim = language_dim

        if variant == "base":
            # Base variant: no COVER-3D modules
            self.dense_relation = None
            self.calibration = None
            log.info("Base variant: no COVER-3D modules")

        elif variant == "dense-no-cal":
            # Dense-no-cal: only dense relation module (baseline anchor)
            self.dense_relation = DenseRelationModule(
                object_dim=object_dim,
                language_dim=language_dim,
                geometry_dim=geometry_dim,
                hidden_dim=relation_hidden_dim,
                chunk_size=relation_chunk_size,
                use_geometry=False,  # No geometry in extracted data
                aggregation="weighted",
                use_focal=False,
            )
            self.calibration = None
            log.info("Dense-no-cal variant: DenseRelationModule only (baseline anchor)")

        elif variant == "dense-v2-attpool":
            # Dense-v2-AttPool: attention-based aggregation
            # Uses attention pooling to select relevant anchors instead of simple weighted sum
            self.dense_relation = DenseRelationModule(
                object_dim=object_dim,
                language_dim=language_dim,
                geometry_dim=geometry_dim,
                hidden_dim=relation_hidden_dim,
                chunk_size=relation_chunk_size,
                use_geometry=False,
                aggregation="attention",
                use_attention=True,
                attention_heads=4,
                attention_hidden_dim=128,
                use_focal=False,
            )
            self.calibration = None
            log.info("Dense-v2-AttPool variant: attention-based aggregation for better anchor selection")

        elif variant == "dense-v3-geo":
            # Dense-v3-Geo: geometry-enhanced features
            # Uses explicit geometric features (distance, direction) for better spatial reasoning
            self.dense_relation = DenseRelationModule(
                object_dim=object_dim,
                language_dim=language_dim,
                geometry_dim=geometry_dim,
                hidden_dim=relation_hidden_dim,
                chunk_size=relation_chunk_size,
                use_geometry=True,  # Enable geometry features
                aggregation="weighted",
                use_focal=False,
            )
            self.calibration = None
            log.info("Dense-v3-Geo variant: geometry-enhanced pair features")

        elif variant == "dense-v4-hardneg":
            # Dense-v4-HardNeg: hard-negative aware training
            # Uses focal weighting to emphasize hard cases (clutter, multi-anchor)
            self.dense_relation = DenseRelationModule(
                object_dim=object_dim,
                language_dim=language_dim,
                geometry_dim=geometry_dim,
                hidden_dim=relation_hidden_dim,
                chunk_size=relation_chunk_size,
                use_geometry=False,
                aggregation="hybrid",  # Hybrid: max + weighted for better signal
                use_focal=True,
                focal_gamma=2.0,
            )
            self.calibration = None
            log.info("Dense-v4-HardNeg variant: focal weighting for hard-negative training")

        elif variant in ["dense-calibrated", "dense-calibrated-v2"]:
            # Dense-calibrated: both modules
            # v2 adds gate prior regularization and balanced init_bias
            self.dense_relation = DenseRelationModule(
                object_dim=object_dim,
                language_dim=language_dim,
                geometry_dim=geometry_dim,
                hidden_dim=relation_hidden_dim,
                chunk_size=relation_chunk_size,
                use_geometry=False,
            )

            # v2 uses balanced init_bias and stronger gate prior regularization
            # Prior weight 1.0 to counteract CE gradient pushing gate to bounds
            init_bias = fusion_init_bias if variant == "dense-calibrated-v2" else 0.3
            prior_weight = fusion_gate_prior_weight if variant == "dense-calibrated-v2" else 0.0

            self.calibration = CalibratedFusionGate(
                signal_dim=4,
                hidden_dim=fusion_hidden_dim,
                min_gate=fusion_min_gate,
                max_gate=fusion_max_gate,
                init_bias=init_bias,
                gate_prior_weight=prior_weight,
                gate_prior_target=0.5,
            )
            log.info(f"{variant} variant: DenseRelationModule + CalibratedFusionGate (init_bias={init_bias}, prior_weight={prior_weight})")

        else:
            raise ValueError(f"Unknown variant: {variant}")

    def forward(
        self,
        base_logits: torch.Tensor,  # [B, N]
        object_embeddings: torch.Tensor,  # [B, N, D]
        lang_features: torch.Tensor,  # [B, D]
        object_mask: torch.Tensor,  # [B, N]
    ) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        B, N = base_logits.shape
        device = base_logits.device

        if self.variant == "base":
            # Base: just return base logits, but mask padded positions
            # CRITICAL: Padded positions are zeros, which can be > negative logits
            # causing incorrect argmax on samples with all negative logits
            fused_logits = base_logits.masked_fill(~object_mask, float("-inf"))
            return {
                "fused_logits": fused_logits,
                "relation_scores": torch.zeros(B, N, device=device),
                "gate_values": torch.ones(B, device=device),
            }

        # Compute relation scores
        relation_result = self.dense_relation(
            object_embeddings=object_embeddings,
            utterance_features=lang_features,
            candidate_mask=object_mask,
        )
        relation_scores = relation_result["relation_scores"]  # [B, N]

        if self.variant in ["dense-no-cal", "dense-v2-attpool", "dense-v3-geo", "dense-v4-hardneg"]:
            # Dense-no-cal and variants: simple addition with fixed lambda
            dense_lambda = 0.5
            fused_logits = base_logits + dense_lambda * relation_scores
            fused_logits = fused_logits.masked_fill(~object_mask, float("-inf"))
            return {
                "fused_logits": fused_logits,
                "relation_scores": relation_scores,
                "gate_values": torch.ones(B, device=device) * dense_lambda,
            }

        # Dense-calibrated: calibrated fusion
        # Compute calibration signals
        # 1. Base margin
        base_sorted = base_logits.sort(dim=-1, descending=True).values
        base_margin = base_sorted[:, 0] - base_sorted[:, 1]  # [B]

        # 2. Relation margin
        relation_sorted = relation_scores.sort(dim=-1, descending=True).values
        relation_margin = relation_sorted[:, 0] - relation_sorted[:, 1]  # [B]

        # 3. Anchor entropy (from relation scores softmax)
        relation_probs = F.softmax(relation_scores.masked_fill(~object_mask, float("-inf")), dim=-1)
        anchor_entropy = -(relation_probs * relation_probs.clamp(min=1e-8).log()).sum(dim=-1)  # [B]

        # Compute gate using CalibratedFusionGate API
        # Use additive fusion like dense-no-cal, but with learned gate
        gate_result = self.calibration(
            base_logits=base_logits,
            relation_scores=relation_scores,
            anchor_posterior=relation_probs,  # Use softmax as proxy
            anchor_entropy=anchor_entropy,
            base_margin=base_margin,
            relation_margin=relation_margin,
            candidate_mask=object_mask,
        )
        gate_values = gate_result["gate_values"]  # [B]

        # CRITICAL FIX: Prevent -inf gradient explosion in fusion
        # relation_scores contains -inf for masked positions
        # gate * -inf produces -inf gradient that explodes gate MLP
        # Solution: Use clamped relation_scores for fusion gradient,
        # then apply mask to final logits for correct argmax

        # Use clamped relation scores for stable gradient computation
        safe_relation = relation_scores.clamp(min=-1e6)

        # Additive fusion with safe relation scores
        fused_logits = base_logits + gate_values.unsqueeze(-1) * safe_relation

        # Apply candidate mask for correct argmax (masked positions = -inf)
        fused_logits = fused_logits.masked_fill(~object_mask, float("-inf"))

        return {
            "fused_logits": fused_logits,
            "relation_scores": relation_scores,
            "gate_values": gate_values,
            "base_margin": base_margin,
            "relation_margin": relation_margin,
            "anchor_entropy": anchor_entropy,
        }

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        total = 0
        if self.dense_relation is not None:
            total += sum(p.numel() for p in self.dense_relation.parameters())
        if self.calibration is not None:
            total += sum(p.numel() for p in self.calibration.parameters())
        return total


# ============================================================================
# Training
# ============================================================================

def train_epoch(
    model: Cover3DReranker,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    focal_gamma: float = 0.0,  # Focal weighting gamma (0 = standard CE)
) -> Dict[str, float]:
    """Train one epoch."""
    model.train()

    total_loss = 0.0
    total_ce_loss = 0.0
    total_gate_loss = 0.0
    total_focal_loss = 0.0
    total_correct = 0
    total_samples = 0
    all_gate_values = []

    for batch_idx, batch in enumerate(loader):
        object_embeddings = batch["object_embeddings"].to(device)
        lang_features = batch["lang_features"].to(device)
        base_logits = batch["base_logits"].to(device)
        object_mask = batch["object_mask"].to(device)
        target_index = batch["target_index"].to(device)

        # Forward
        result = model(
            base_logits=base_logits,
            object_embeddings=object_embeddings,
            lang_features=lang_features,
            object_mask=object_mask,
        )

        fused_logits = result["fused_logits"]

        # Loss = CE loss + focal weighting (if applicable) + gate prior loss (if applicable)
        # Standard cross-entropy
        ce_loss = F.cross_entropy(fused_logits, target_index, reduction='none')  # [B]

        # Focal weighting: down-weight easy samples
        # pt = exp(-loss) = probability of correct class
        # focal_weight = (1 - pt)^gamma
        if focal_gamma > 0:
            # Compute softmax probability of correct class
            log_probs = F.log_softmax(fused_logits, dim=-1)  # [B, N]
            pt = torch.exp(log_probs.gather(1, target_index.unsqueeze(1)))  # [B, 1]
            focal_weight = (1 - pt) ** focal_gamma  # [B, 1]
            ce_loss = (ce_loss * focal_weight.squeeze()).mean()  # Weighted mean
            focal_loss = ce_loss.item() - F.cross_entropy(fused_logits, target_index).item()
        else:
            ce_loss = ce_loss.mean()
            focal_loss = 0.0

        # Add gate prior regularization if model has calibration with prior
        gate_loss = torch.zeros(1, device=device)
        if model.calibration is not None and hasattr(model.calibration, 'compute_gate_prior_loss'):
            gate_values = result.get("gate_values", None)
            if gate_values is not None:
                gate_loss = model.calibration.compute_gate_prior_loss(gate_values)

        loss = ce_loss + gate_loss

        # Backward
        optimizer.zero_grad()
        loss.backward()
        # CRITICAL: Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Metrics
        pred_top1 = fused_logits.argmax(dim=-1)
        total_loss += loss.item() * len(target_index)
        total_ce_loss += ce_loss.item() * len(target_index)
        total_gate_loss += gate_loss.item() * len(target_index)
        total_focal_loss += focal_loss * len(target_index)
        total_correct += (pred_top1 == target_index).sum().item()
        total_samples += len(target_index)

        if "gate_values" in result:
            all_gate_values.extend(result["gate_values"].cpu().tolist())

    return {
        "loss": total_loss / total_samples,
        "ce_loss": total_ce_loss / total_samples,
        "gate_loss": total_gate_loss / total_samples,
        "focal_loss": total_focal_loss / total_samples,
        "acc": total_correct / total_samples * 100,
        "gate_mean": np.mean(all_gate_values) if all_gate_values else 1.0,
    }


def evaluate(
    model: Cover3DReranker,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Dict[str, Any]:
    """Evaluate model."""
    model.eval()

    predictions = []
    total_correct_1 = 0
    total_correct_5 = 0
    total_samples = 0
    all_gate_values = []
    all_relation_margins = []

    with torch.no_grad():
        for batch in loader:
            object_embeddings = batch["object_embeddings"].to(device)
            lang_features = batch["lang_features"].to(device)
            base_logits = batch["base_logits"].to(device)
            object_mask = batch["object_mask"].to(device)
            target_index = batch["target_index"].to(device)

            result = model(
                base_logits=base_logits,
                object_embeddings=object_embeddings,
                lang_features=lang_features,
                object_mask=object_mask,
            )

            fused_logits = result["fused_logits"]
            gate_values = result.get("gate_values", torch.ones(len(target_index), device=device))

            # Predictions
            pred_top1 = fused_logits.argmax(dim=-1)
            pred_top5 = fused_logits.topk(5, dim=-1).indices

            # Accuracy
            total_correct_1 += (pred_top1 == target_index).sum().item()
            for i in range(len(target_index)):
                if target_index[i] in pred_top5[i]:
                    total_correct_5 += 1
            total_samples += len(target_index)

            # Diagnostics
            all_gate_values.extend(gate_values.cpu().tolist())
            if "relation_margin" in result:
                all_relation_margins.extend(result["relation_margin"].cpu().tolist())

            # Store predictions
            for i, sample in enumerate(batch["batch_ref"]):
                # CRITICAL: Apply object_mask before argmax to avoid padding artifact
                # Padded positions are filled with 0.0, which can be > negative logits
                # causing incorrect argmax on samples with all negative logits
                masked_logits = base_logits[i].masked_fill(~object_mask[i], float("-inf"))
                base_pred = masked_logits.argmax().item()
                fused_pred = pred_top1[i].item()

                predictions.append({
                    "scene_id": sample["scene_id"],
                    "utterance": sample["utterance"],
                    "target_index": sample["target_index"],
                    "base_pred": base_pred,
                    "fused_pred": fused_pred,
                    "base_correct": base_pred == sample["target_index"],
                    "fused_correct": fused_pred == sample["target_index"],
                    "gate_value": gate_values[i].item(),
                    "anchor_count": sample.get("anchor_count", 0),
                    "subsets": sample.get("subsets", {}),
                })

    return {
        "acc_at_1": total_correct_1 / total_samples * 100,
        "acc_at_5": total_correct_5 / total_samples * 100,
        "total_samples": total_samples,
        "gate_mean": np.mean(all_gate_values),
        "gate_std": np.std(all_gate_values),
        "relation_margin_mean": np.mean(all_relation_margins) if all_relation_margins else 0.0,
        "predictions": predictions,
    }


def compute_hard_subset_metrics(predictions: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Compute metrics for hard subsets."""
    subsets = {
        "overall": {"correct": 0, "total": 0},
        "same_class_clutter_ge3": {"correct": 0, "total": 0},
        "same_class_clutter_ge5": {"correct": 0, "total": 0},
        "multi_anchor": {"correct": 0, "total": 0},
        "relative_position": {"correct": 0, "total": 0},
        "easy": {"correct": 0, "total": 0},
    }

    for pred in predictions:
        subsets["overall"]["total"] += 1
        if pred["fused_correct"]:
            subsets["overall"]["correct"] += 1

        subset_tags = pred.get("subsets", {})
        if subset_tags.get("same_class_clutter_ge3", False):
            subsets["same_class_clutter_ge3"]["total"] += 1
            if pred["fused_correct"]:
                subsets["same_class_clutter_ge3"]["correct"] += 1

        if subset_tags.get("same_class_clutter_ge5", False):
            subsets["same_class_clutter_ge5"]["total"] += 1
            if pred["fused_correct"]:
                subsets["same_class_clutter_ge5"]["correct"] += 1

        if subset_tags.get("multi_anchor", False) or pred["anchor_count"] >= 1:
            subsets["multi_anchor"]["total"] += 1
            if pred["fused_correct"]:
                subsets["multi_anchor"]["correct"] += 1

        if subset_tags.get("relative_position", False):
            subsets["relative_position"]["total"] += 1
            if pred["fused_correct"]:
                subsets["relative_position"]["correct"] += 1

        # Easy = no hard subsets
        is_easy = not any([
            subset_tags.get("same_class_clutter_ge3", False),
            subset_tags.get("same_class_clutter_ge5", False),
            subset_tags.get("multi_anchor", False),
            pred["anchor_count"] >= 1,
        ])
        if is_easy:
            subsets["easy"]["total"] += 1
            if pred["fused_correct"]:
                subsets["easy"]["correct"] += 1

    # Compute accuracies
    metrics = {}
    for name, counts in subsets.items():
        if counts["total"] > 0:
            metrics[name] = {
                "acc": counts["correct"] / counts["total"] * 100,
                "correct": counts["correct"],
                "total": counts["total"],
            }
        else:
            metrics[name] = {"acc": 0.0, "correct": 0, "total": 0}

    return metrics


def compute_harmed_recovered(predictions: List[Dict[str, Any]]) -> Dict[str, int]:
    """Compute harmed/recovered cases."""
    recovered = 0  # base wrong -> fused correct
    harmed = 0  # base correct -> fused wrong

    for pred in predictions:
        if not pred["base_correct"] and pred["fused_correct"]:
            recovered += 1
        elif pred["base_correct"] and not pred["fused_correct"]:
            harmed += 1

    return {
        "recovered": recovered,
        "harmed": harmed,
        "net": recovered - harmed,
    }


# ============================================================================
# Main Training Loop
# ============================================================================

def run_training(
    variant: str,
    train_samples: List[Dict[str, Any]],
    test_samples: List[Dict[str, Any]],
    device: torch.device,
    epochs: int = 10,
    batch_size: int = 16,
    lr: float = 1e-4,
    output_dir: Path = Path("outputs/cover3d_round1"),
    seed: int = 42,
) -> Dict[str, Any]:
    """Run full training and evaluation."""

    set_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Starting Round-1 training: variant={variant}, epochs={epochs}")

    # Configure variant-specific hyperparameters
    focal_gamma = 0.0  # Default: no focal weighting
    if variant == "dense-v4-hardneg":
        focal_gamma = 2.0  # Focal weighting for hard-negative training
        log.info(f"Using focal weighting with gamma={focal_gamma}")

    # Create datasets
    train_dataset = EmbeddingDataset(train_samples)
    test_dataset = EmbeddingDataset(test_samples)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    log.info(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}")

    # Create model
    model = Cover3DReranker(variant=variant)
    model = model.to(device)

    trainable_params = model.count_parameters()
    log.info(f"Trainable parameters: {trainable_params}")

    if trainable_params == 0:
        # Base variant: no training needed
        log.info("Base variant: skipping training, direct evaluation")
        history = []
    else:
        # Training
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

        history = []
        best_acc = 0.0

        for epoch in range(epochs):
            train_metrics = train_epoch(model, train_loader, optimizer, device, epoch, focal_gamma)
            scheduler.step()

            # Log gate_loss if available (v2 variant)
            if "gate_loss" in train_metrics and train_metrics["gate_loss"] > 0:
                log.info(
                    f"Epoch {epoch+1}/{epochs}: "
                    f"Loss={train_metrics['loss']:.4f} (CE={train_metrics['ce_loss']:.4f}, Gate={train_metrics['gate_loss']:.4f}), "
                    f"Acc={train_metrics['acc']:.2f}%, "
                    f"Gate={train_metrics['gate_mean']:.3f}"
                )
            elif "focal_loss" in train_metrics and train_metrics["focal_loss"] > 0:
                log.info(
                    f"Epoch {epoch+1}/{epochs}: "
                    f"Loss={train_metrics['loss']:.4f} (CE={train_metrics['ce_loss']:.4f}, Focal={train_metrics['focal_loss']:.4f}), "
                    f"Acc={train_metrics['acc']:.2f}%, "
                    f"Gate={train_metrics['gate_mean']:.3f}"
                )
            else:
                log.info(
                    f"Epoch {epoch+1}/{epochs}: "
                    f"Loss={train_metrics['loss']:.4f}, "
                    f"Acc={train_metrics['acc']:.2f}%, "
                    f"Gate={train_metrics['gate_mean']:.3f}"
                )

            history.append({
                "epoch": epoch + 1,
                "train_loss": train_metrics["loss"],
                "train_acc": train_metrics["acc"],
                "gate_mean": train_metrics["gate_mean"],
                "gate_loss": train_metrics.get("gate_loss", 0.0),
            })

            # Save checkpoint
            if train_metrics["acc"] > best_acc:
                best_acc = train_metrics["acc"]
                torch.save(model.state_dict(), output_dir / "best_model.pt")

    # Evaluation
    log.info("Evaluating on test set...")
    eval_result = evaluate(model, test_loader, device)

    log.info(f"Test Acc@1: {eval_result['acc_at_1']:.2f}%")
    log.info(f"Test Acc@5: {eval_result['acc_at_5']:.2f}%")
    log.info(f"Gate mean: {eval_result['gate_mean']:.3f}")

    # Hard subset metrics
    hard_metrics = compute_hard_subset_metrics(eval_result["predictions"])

    log.info("Hard subset metrics:")
    for name, m in hard_metrics.items():
        if m["total"] > 0:
            log.info(f"  {name}: {m['acc']:.2f}% ({m['correct']}/{m['total']})")

    # Harmed/recovered
    harm_recover = compute_harmed_recovered(eval_result["predictions"])
    log.info(f"Recovered: {harm_recover['recovered']}, Harmed: {harm_recover['harmed']}, Net: {harm_recover['net']}")

    # Save results
    result_summary = {
        "variant": variant,
        "epochs": epochs,
        "seed": seed,
        "acc_at_1": eval_result["acc_at_1"],
        "acc_at_5": eval_result["acc_at_5"],
        "gate_mean": eval_result["gate_mean"],
        "gate_std": eval_result["gate_std"],
        "hard_subsets": hard_metrics,
        "harmed_recovered": harm_recover,
        "trainable_params": trainable_params,
        "history": history,
    }

    with (output_dir / f"{variant}_results.json").open("w") as f:
        json.dump(result_summary, f, indent=2)

    with (output_dir / f"{variant}_predictions.json").open("w") as f:
        json.dump(eval_result["predictions"], f, indent=2)

    log.info(f"Saved results to {output_dir}")

    return result_summary


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="COVER-3D Round-1 Training")
    parser.add_argument("--variant", type=str, default="dense-no-cal",
        choices=["base", "dense-no-cal", "dense-calibrated", "dense-calibrated-v2",
                 "dense-v2-attpool", "dense-v3-geo", "dense-v4-hardneg"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=Path,
        default=ROOT / "outputs/cover3d_round1")
    parser.add_argument("--train-embeddings", type=Path,
        default=ROOT / "outputs/20260420_clean_sorted_vocab_baseline/embeddings/train_embeddings.json")
    parser.add_argument("--test-embeddings", type=Path,
        default=ROOT / "outputs/20260420_clean_sorted_vocab_baseline/embeddings/test_embeddings.json")
    parser.add_argument("--coverage", type=Path,
        default=ROOT / "reports/pre_method_clean_coverage_diagnostics/per_sample_coverage.jsonl")
    parser.add_argument("--skip-coverage-merge", action="store_true")

    args = parser.parse_args()

    setup_logging()

    # Device
    if args.device == "cuda" and not torch.cuda.is_available():
        log.warning("CUDA not available, using CPU")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    log.info(f"COVER-3D Round-1: variant={args.variant}")
    log.info(f"Device: {device}")

    # CRITICAL: Load clean baseline predictions for correct logits
    clean_test_path = ROOT / "outputs/20260420_clean_sorted_vocab_baseline/formal_logits/eval_test_predictions.json"
    clean_train_path = ROOT / "outputs/20260420_clean_sorted_vocab_baseline/formal_logits/eval_val_predictions.json"

    clean_test_preds = []
    clean_train_preds = []

    if clean_test_path.exists():
        clean_test_preds = load_clean_predictions(clean_test_path)
    else:
        log.warning(f"Clean test predictions not found at {clean_test_path}")

    if args.variant != "base" and clean_train_path.exists():
        clean_train_preds = load_clean_predictions(clean_train_path)

    # Load embeddings
    if args.train_embeddings.exists():
        train_embeddings = load_embeddings(args.train_embeddings)
    else:
        log.warning(f"Train embeddings not found at {args.train_embeddings}")
        train_embeddings = []

    if args.test_embeddings.exists():
        test_embeddings = load_embeddings(args.test_embeddings)
    else:
        log.warning(f"Test embeddings not found at {args.test_embeddings}")
        test_embeddings = []

    # CRITICAL: Replace logits with clean baseline logits by index
    # ONLY for test embeddings - train embeddings use re-run logits (no train predictions file)
    if clean_test_preds:
        log.info("Replacing test logits with clean baseline logits by index")
        test_embeddings = merge_embeddings_with_clean_logits(test_embeddings, clean_test_preds)

    # Note: Train embeddings use re-run logits from extraction (no clean train predictions available)
    # This is acceptable for Dense training since relation_scores are learned independently
    if clean_train_preds and train_embeddings and len(clean_train_preds) == len(train_embeddings):
        # Only replace if sizes match (they should be from same split)
        log.info("Replacing train logits with clean baseline logits by index")
        train_embeddings = merge_embeddings_with_clean_logits(train_embeddings, clean_train_preds)
    else:
        log.info(f"Train logits NOT replaced: train_embeddings={len(train_embeddings)}, clean_train={len(clean_train_preds)} (different splits)")

    # Merge with coverage
    if not args.skip_coverage_merge and args.coverage.exists():
        coverage = load_coverage(args.coverage)
        train_embeddings = merge_embeddings_with_coverage(train_embeddings, coverage)
        test_embeddings = merge_embeddings_with_coverage(test_embeddings, coverage)

    # For Base variant, we only need test embeddings (no training)
    if args.variant == "base":
        if len(test_embeddings) == 0:
            log.error("Test embeddings not available. Run extract_clean_baseline_embeddings.py first.")
            raise SystemExit(1)
        # Use test embeddings as "train" for dataset creation (no actual training)
        if len(train_embeddings) == 0:
            log.info("Base variant: using test embeddings for evaluation-only run")
            train_embeddings = test_embeddings  # Dummy train set (no training happens)
    else:
        # Other variants need both train and test
        if len(train_embeddings) == 0 or len(test_embeddings) == 0:
            log.error("Both train and test embeddings required for training variants.")
            log.error("Run extract_clean_baseline_embeddings.py for both splits first.")
            raise SystemExit(1)

    # Run training
    result = run_training(
        variant=args.variant,
        train_samples=train_embeddings,
        test_samples=test_embeddings,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        output_dir=args.output_dir,
        seed=args.seed,
    )

    log.info("=" * 60)
    log.info(f"ROUND-1 {args.variant.upper()} COMPLETE")
    log.info(f"Acc@1: {result['acc_at_1']:.2f}%, Acc@5: {result['acc_at_5']:.2f}%")
    log.info(f"Recovered: {result['harmed_recovered']['recovered']}, Harmed: {result['harmed_recovered']['harmed']}")
    log.info("=" * 60)

    return result


if __name__ == "__main__":
    main()