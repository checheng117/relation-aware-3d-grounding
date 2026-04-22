"""COVER-3D Unified Model: Coverage-Calibrated Relational Reranking.

This module combines all COVER-3D components into an end-to-end reranker:

1. DenseRelationModule: all-pair candidate-anchor relation evidence
2. SoftAnchorPosteriorModule: soft anchor distribution (no hard decisions)
3. CalibratedFusionGate: signal-dependent logit fusion

The model wraps around any base 3D grounding model and reranks predictions
using dense relation evidence, with calibration to prevent degradation.

Key design:
- Model-agnostic: works with any base model providing logits + embeddings
- Dense coverage: all N² pairs, no sparse approximation
- Calibrated fusion: bounded gate prevents collapse
- Diagnostics-rich: emits calibration signals for analysis

Usage:
    model = Cover3DModel(config)
    output = model.forward(
        base_logits=base_logits,
        object_embeddings=object_embeddings,
        utterance_features=utterance_features,
        object_geometry=object_geometry,
    )
    final_logits = output["fused_logits"]
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from rag3d.models.cover3d_wrapper import (
    Cover3DInput,
    Cover3DOutput,
    Cover3DWrapper,
)
from rag3d.models.cover3d_dense_relation import DenseRelationModule
from rag3d.models.cover3d_anchor_posterior import SoftAnchorPosteriorModule
from rag3d.models.cover3d_calibration import CalibratedFusionGate

log = logging.getLogger(__name__)


class Cover3DModel(nn.Module):
    """COVER-3D: Coverage-Calibrated Relational Reranking for 3D Grounding.

    Full implementation combining all modules. This is the final model
    for Phase 2/3 training and evaluation.

    Architecture:
    ┌─────────────────────────────────────────────────┐
    │                    COVER-3D                      │
    ├─────────────────────────────────────────────────┤
    │  Input:                                         │
    │  - base_logits [B, N]                           │
    │  - object_embeddings [B, N, D]                  │
    │  - utterance_features [B, L]                    │
    │  - object_geometry [B, N, G] (optional)         │
    │                                                 │
    │  Modules:                                       │
    │  1. DenseRelationModule                         │
    │     → relation_scores [B, N]                    │
    │     → relation_context [B, N, D]                │
    │                                                 │
    │  2. SoftAnchorPosteriorModule                   │
    │     → anchor_posterior [B, N]                   │
    │     → anchor_entropy [B]                        │
    │                                                 │
    │  3. CalibratedFusionGate                        │
    │     → fused_logits [B, N]                       │
    │     → gate_values [B]                           │
    │                                                 │
    │  Output:                                        │
    │  - fused_logits [B, N]                          │
    │  - diagnostics dict                             │
    └─────────────────────────────────────────────────┘
    """

    def __init__(
        self,
        # Dimension parameters
        object_dim: int = 320,
        language_dim: int = 256,
        geometry_dim: int = 6,
        class_dim: int = 64,
        # Relation module parameters
        relation_hidden_dim: int = 256,
        relation_mlp_layers: int = 2,
        relation_dropout: float = 0.1,
        relation_chunk_size: int = 16,
        relation_temperature: float = 1.0,
        use_geometry: bool = True,
        relation_residual: bool = True,
        relation_residual_scale: float = 0.1,
        # Anchor posterior parameters
        anchor_hidden_dim: int = 128,
        anchor_temperature: float = 1.0,
        anchor_min_entropy: float = 0.5,
        anchor_prior_weight: float = 0.2,
        anchor_use_class: bool = True,
        anchor_dropout: float = 0.1,
        # Fusion gate parameters
        fusion_signal_dim: int = 4,
        fusion_hidden_dim: int = 32,
        fusion_min_gate: float = 0.1,
        fusion_max_gate: float = 0.9,
        fusion_init_bias: float = 0.3,
        fusion_dropout: float = 0.1,
        # General parameters
        dropout: float = 0.1,
        emit_diagnostics: bool = True,
        **kwargs,
    ):
        super().__init__()

        self.object_dim = object_dim
        self.language_dim = language_dim
        self.geometry_dim = geometry_dim
        self.class_dim = class_dim
        self.emit_diagnostics = emit_diagnostics

        # === Dense Relation Module ===
        self.dense_relation = DenseRelationModule(
            object_dim=object_dim,
            language_dim=language_dim,
            geometry_dim=geometry_dim,
            hidden_dim=relation_hidden_dim,
            mlp_layers=relation_mlp_layers,
            dropout=relation_dropout,
            chunk_size=relation_chunk_size,
            temperature=relation_temperature,
            use_geometry=use_geometry,
            use_residual=relation_residual,
            residual_scale=relation_residual_scale,
        )

        # === Soft Anchor Posterior Module ===
        self.anchor_posterior = SoftAnchorPosteriorModule(
            object_dim=object_dim,
            language_dim=language_dim,
            class_dim=class_dim,
            hidden_dim=anchor_hidden_dim,
            temperature=anchor_temperature,
            min_entropy_threshold=anchor_min_entropy,
            prior_weight=anchor_prior_weight,
            use_class_features=anchor_use_class,
            dropout=anchor_dropout,
        )

        # === Calibrated Fusion Gate ===
        self.fusion_gate = CalibratedFusionGate(
            signal_dim=fusion_signal_dim,
            hidden_dim=fusion_hidden_dim,
            min_gate=fusion_min_gate,
            max_gate=fusion_max_gate,
            init_bias=fusion_init_bias,
            dropout=fusion_dropout,
        )

        # Create wrapper interface
        self.wrapper = Cover3DWrapper(
            object_dim=object_dim,
            language_dim=language_dim,
            geometry_dim=geometry_dim,
            class_dim=class_dim,
            emit_diagnostics=emit_diagnostics,
        )

        # Connect wrapper to sub-modules
        self.wrapper.set_modules(
            dense_relation_module=self.dense_relation,
            anchor_posterior_module=self.anchor_posterior,
            calibration_module=self.fusion_gate,
        )

        log.info(
            f"Cover3DModel initialized: "
            f"object_dim={object_dim}, language_dim={language_dim}, "
            f"relation_chunk_size={relation_chunk_size}, "
            f"fusion_gate=[{fusion_min_gate}, {fusion_max_gate}]"
        )

    def forward(
        self,
        base_logits: torch.Tensor,  # [B, N]
        object_embeddings: torch.Tensor,  # [B, N, D]
        utterance_features: torch.Tensor,  # [B, L] or [B, D]
        object_geometry: Optional[torch.Tensor] = None,  # [B, N, G]
        candidate_mask: Optional[torch.Tensor] = None,  # [B, N]
        object_class_features: Optional[torch.Tensor] = None,  # [B, N, C]
        anchor_priors: Optional[torch.Tensor] = None,  # [B, N]
        parser_confidence: Optional[torch.Tensor] = None,  # [B]
        return_intermediate: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """Forward pass: apply COVER-3D reranking.

        Args:
            base_logits: base model prediction logits
            object_embeddings: object feature embeddings
            utterance_features: language features
            object_geometry: spatial features (optional)
            candidate_mask: valid object mask
            object_class_features: class semantic features
            anchor_priors: external anchor prior (weak signal)
            parser_confidence: parser confidence
            return_intermediate: whether to return intermediate signals

        Returns:
            dict with:
            - fused_logits: [B, N] final predictions
            - diagnostics: calibration and coverage stats
            - intermediate signals (if return_intermediate)
        """
        # Build input bundle
        inputs = Cover3DInput(
            base_logits=base_logits,
            object_embeddings=object_embeddings,
            utterance_features=utterance_features,
            object_geometry=object_geometry,
            candidate_mask=candidate_mask,
            object_class_features=object_class_features,
            anchor_priors=anchor_priors,
            parser_confidence=parser_confidence,
        )

        # Run through wrapper
        output = self.wrapper.rerank(inputs, return_intermediate=return_intermediate)

        # Convert to dict format
        result = {
            "fused_logits": output.reranked_logits,
            "diagnostics": output.diagnostics,
        }

        if return_intermediate:
            result["dense_relation_scores"] = output.dense_relation_scores
            result["anchor_posterior"] = output.anchor_posterior
            result["anchor_entropy"] = output.anchor_entropy
            result["top_anchor_margin"] = output.top_anchor_margin
            result["base_margin"] = output.base_margin
            result["relation_margin"] = output.relation_margin
            result["gate_values"] = output.gate_values

        return result

    def forward_from_input(self, inputs: Cover3DInput) -> Cover3DOutput:
        """Forward pass from Cover3DInput bundle.

        Convenience method for using the wrapper interface.
        """
        return self.wrapper.rerank(inputs)

    def get_coverage_stats(
        self,
        pair_weights: torch.Tensor,  # [B, N, N]
        anchor_posterior: torch.Tensor,  # [B, N]
        top_k_values: list = [1, 3, 5, 10],
    ) -> Dict[str, Any]:
        """Compute coverage statistics for analysis.

        Measures how much of the anchor posterior mass is covered
        by top-k highest relation weights.

        This is for Phase 1-style coverage analysis.
        """
        B, N = anchor_posterior.shape

        coverage_stats = {}

        for k in top_k_values:
            if k > N:
                k = N

            # Get top-k relation weights for each candidate i
            top_k_weights, _ = pair_weights.topk(k, dim=-1)  # [B, N, k]

            # Sum top-k weights
            top_k_sum = top_k_weights.sum(dim=-1)  # [B, N]

            # Coverage: for each object, what fraction of anchor posterior
            # is covered by top-k relation weights
            # This is approximate since anchor_posterior is over all objects
            coverage = top_k_sum.mean().item()

            coverage_stats[f"coverage_at_{k}"] = coverage

        return coverage_stats

    def count_parameters(self) -> Dict[str, int]:
        """Count trainable parameters per module."""
        counts = {
            "dense_relation": sum(p.numel() for p in self.dense_relation.parameters()),
            "anchor_posterior": sum(p.numel() for p in self.anchor_posterior.parameters()),
            "fusion_gate": sum(p.numel() for p in self.fusion_gate.parameters()),
            "total": sum(p.numel() for p in self.parameters()),
        }
        return counts


def create_cover3d_model_from_config(config: Dict[str, Any]) -> Cover3DModel:
    """Factory function to create COVER-3D model from config dict.

    Args:
        config: configuration dictionary with model parameters

    Returns:
        Cover3DModel instance
    """
    return Cover3DModel(**config)