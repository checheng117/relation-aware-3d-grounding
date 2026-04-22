"""COVER-3D Wrapper Interface: Model-Agnostic Reranking Entry Point.

This module provides the base interface for wrapping any 3D grounding model
with COVER-3D's coverage-calibrated relational reranking.

Key design:
- Model-agnostic: works with any base model that outputs logits + embeddings
- Clean interface: minimal assumptions about base model internals
- Diagnostics-rich: outputs detailed calibration signals for analysis
- No hard parser dependency

Usage:
    wrapper = Cover3DWrapper(config)
    result = wrapper.rerank(
        base_logits=base_logits,
        object_embeddings=object_embeddings,
        object_geometry=object_geometry,
        candidate_mask=candidate_mask,
        utterance_features=utterance_features,
    )
    final_logits = result['reranked_logits']
    diagnostics = result['diagnostics']
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol, runtime_checkable

import torch
import torch.nn as nn

log = logging.getLogger(__name__)


@dataclass
class Cover3DInput:
    """Input bundle for COVER-3D reranking.

    All tensors are expected to have batch dimension [B, ...].
    """

    # Required inputs
    base_logits: torch.Tensor  # [B, N] - base model prediction logits
    object_embeddings: torch.Tensor  # [B, N, D] - object feature embeddings
    utterance_features: torch.Tensor  # [B, L] or [B, D] - language features

    # Optional inputs (may be fallback/placeholder)
    object_geometry: Optional[torch.Tensor] = None  # [B, N, G] - spatial info (centers, sizes)
    candidate_mask: Optional[torch.Tensor] = None  # [B, N] - valid object mask

    # Metadata inputs (optional)
    object_class_ids: Optional[torch.Tensor] = None  # [B, N] - semantic class indices
    object_class_features: Optional[torch.Tensor] = None  # [B, N, C] - class embeddings
    scene_ids: Optional[list] = None  # [B] - scene identifiers
    utterances: Optional[list] = None  # [B] - raw text strings

    # Prior inputs (optional weak signals, not hard decisions)
    anchor_priors: Optional[torch.Tensor] = None  # [B, N] - soft anchor prior distribution
    relation_type_prior: Optional[torch.Tensor] = None  # [B, K] - relation type distribution
    parser_confidence: Optional[torch.Tensor] = None  # [B] - parser confidence score

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Cover3DInput":
        """Create input bundle from dictionary."""
        return cls(
            base_logits=data.get("base_logits"),
            object_embeddings=data.get("object_embeddings"),
            utterance_features=data.get("utterance_features"),
            object_geometry=data.get("object_geometry"),
            candidate_mask=data.get("candidate_mask"),
            object_class_ids=data.get("object_class_ids"),
            object_class_features=data.get("object_class_features"),
            scene_ids=data.get("scene_ids"),
            utterances=data.get("utterances"),
            anchor_priors=data.get("anchor_priors"),
            relation_type_prior=data.get("relation_type_prior"),
            parser_confidence=data.get("parser_confidence"),
        )


@dataclass
class Cover3DOutput:
    """Output bundle from COVER-3D reranking.

    Contains reranked predictions and rich diagnostics.
    """

    # Primary output
    reranked_logits: torch.Tensor  # [B, N] - final prediction logits

    # Diagnostics (always emitted)
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    # Core signals (for analysis)
    dense_relation_scores: Optional[torch.Tensor] = None  # [B, N]
    anchor_posterior: Optional[torch.Tensor] = None  # [B, N]
    anchor_entropy: Optional[torch.Tensor] = None  # [B]
    top_anchor_margin: Optional[torch.Tensor] = None  # [B]
    base_margin: Optional[torch.Tensor] = None  # [B]
    relation_margin: Optional[torch.Tensor] = None  # [B]
    gate_values: Optional[torch.Tensor] = None  # [B]

    # Coverage stats (if computed)
    coverage_at_k: Optional[Dict[int, float]] = None  # {k: pct_covered}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "reranked_logits": self.reranked_logits,
            "diagnostics": self.diagnostics,
            "dense_relation_scores": self.dense_relation_scores,
            "anchor_posterior": self.anchor_posterior,
            "anchor_entropy": self.anchor_entropy,
            "top_anchor_margin": self.top_anchor_margin,
            "base_margin": self.base_margin,
            "relation_margin": self.relation_margin,
            "gate_values": self.gate_values,
            "coverage_at_k": self.coverage_at_k,
        }


@runtime_checkable
class BaseModelProtocol(Protocol):
    """Protocol for base models compatible with COVER-3D wrapper.

    Any model implementing this interface can be wrapped.
    """

    def forward(
        self,
        point_features: torch.Tensor,
        lang_features: torch.Tensor,
        class_features: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Base model forward pass returning logits.

        Args:
            point_features: [B, N, D_p] - point cloud features
            lang_features: [B, D_l] - language features
            class_features: [B, N, D_c] - class semantic features

        Returns:
            logits: [B, N] - prediction logits for each object
        """
        ...


class Cover3DWrapper(nn.Module):
    """Model-agnostic COVER-3D reranking wrapper.

    This is the entry point for COVER-3D. It wraps any base model
    and applies coverage-calibrated relational reranking.

    Architecture:
    1. Receive base model outputs (logits + embeddings)
    2. Compute dense relation evidence (all-pair coverage)
    3. Estimate soft anchor posterior (no hard decisions)
    4. Calibrate fusion gate based on entropy/margin signals
    5. Return reranked logits + diagnostics

    Key constraints:
    - No hard parser decisions (soft distributions only)
    - Dense coverage (no sparse top-k approximation)
    - Calibrated gates (bounded, no collapse)
    """

    def __init__(
        self,
        # Dimension parameters
        object_dim: int = 320,
        language_dim: int = 256,
        geometry_dim: int = 6,
        class_dim: int = 64,
        hidden_dim: int = 256,
        # Relation module parameters
        relation_chunk_size: int = 16,
        relation_mlp_layers: int = 2,
        relation_dropout: float = 0.1,
        relation_temperature: float = 1.0,
        # Anchor posterior parameters
        anchor_softmax_temperature: float = 1.0,
        anchor_min_entropy: float = 0.1,
        # Fusion gate parameters
        gate_hidden_dim: int = 64,
        gate_min_value: float = 0.1,
        gate_max_value: float = 0.9,
        gate_init_bias: float = 0.3,
        # Mode flags
        use_geometry: bool = True,
        use_class_for_anchor: bool = True,
        emit_diagnostics: bool = True,
    ):
        super().__init__()

        self.object_dim = object_dim
        self.language_dim = language_dim
        self.geometry_dim = geometry_dim
        self.class_dim = class_dim
        self.hidden_dim = hidden_dim
        self.emit_diagnostics = emit_diagnostics

        log.info(
            f"Cover3DWrapper: object_dim={object_dim}, language_dim={language_dim}, "
            f"geometry_dim={geometry_dim}, hidden_dim={hidden_dim}"
        )

        # Note: Sub-modules are NOT instantiated here.
        # This wrapper is a pure interface module.
        # The actual COVER-3D model (cover3d_model.py) instantiates sub-modules.
        # This design allows using the wrapper as a protocol/interface without
        # tying it to specific module implementations.

        # Placeholder for wrapped sub-modules (set by Cover3DModel)
        self._dense_relation_module = None
        self._anchor_posterior_module = None
        self._calibration_module = None

    def set_modules(
        self,
        dense_relation_module: Optional[nn.Module] = None,
        anchor_posterior_module: Optional[nn.Module] = None,
        calibration_module: Optional[nn.Module] = None,
    ):
        """Set the sub-modules for full COVER-3D functionality.

        Called by Cover3DModel during composition.
        """
        self._dense_relation_module = dense_relation_module
        self._anchor_posterior_module = anchor_posterior_module
        self._calibration_module = calibration_module

    def validate_inputs(self, inputs: Cover3DInput) -> Dict[str, Any]:
        """Validate input tensors and compute basic statistics.

        Returns validation results and shape info.
        """
        validation = {
            "valid": True,
            "warnings": [],
            "shapes": {},
        }

        # Check required inputs
        if inputs.base_logits is None:
            validation["valid"] = False
            validation["warnings"].append("base_logits is None")
        else:
            validation["shapes"]["base_logits"] = inputs.base_logits.shape

        if inputs.object_embeddings is None:
            validation["valid"] = False
            validation["warnings"].append("object_embeddings is None")
        else:
            validation["shapes"]["object_embeddings"] = inputs.object_embeddings.shape

        if inputs.utterance_features is None:
            validation["valid"] = False
            validation["warnings"].append("utterance_features is None")
        else:
            validation["shapes"]["utterance_features"] = inputs.utterance_features.shape

        # Check optional inputs
        if inputs.object_geometry is None:
            validation["warnings"].append("object_geometry is None (will use fallback)")
        else:
            validation["shapes"]["object_geometry"] = inputs.object_geometry.shape

        if inputs.candidate_mask is None:
            validation["warnings"].append("candidate_mask is None (will assume all valid)")

        # Check shape consistency
        if validation["valid"]:
            B = inputs.base_logits.shape[0]
            N = inputs.base_logits.shape[1]

            if inputs.object_embeddings.shape[0] != B:
                validation["warnings"].append(
                    f"Batch size mismatch: base_logits B={B}, embeddings B={inputs.object_embeddings.shape[0]}"
                )
            if inputs.object_embeddings.shape[1] != N:
                validation["warnings"].append(
                    f"Object count mismatch: base_logits N={N}, embeddings N={inputs.object_embeddings.shape[1]}"
                )

        return validation

    def compute_base_margin(self, base_logits: torch.Tensor) -> torch.Tensor:
        """Compute base prediction margin (top1 - top2 gap).

        Higher margin = more confident base prediction.
        """
        sorted_logits, _ = torch.sort(base_logits, dim=-1, descending=True)
        margin = sorted_logits[:, 0] - sorted_logits[:, 1]
        return margin

    def compute_relation_margin(self, relation_scores: torch.Tensor) -> torch.Tensor:
        """Compute relation score margin.

        Higher margin = relation evidence is concentrated.
        """
        sorted_scores, _ = torch.sort(relation_scores, dim=-1, descending=True)
        margin = sorted_scores[:, 0] - sorted_scores[:, 1]
        return margin

    def rerank(
        self,
        inputs: Cover3DInput,
        return_intermediate: bool = True,
    ) -> Cover3DOutput:
        """Apply COVER-3D reranking to base model outputs.

        This is the main entry point for reranking.

        Args:
            inputs: Input bundle with base logits, embeddings, etc.
            return_intermediate: Whether to return intermediate signals

        Returns:
            Cover3DOutput with reranked logits and diagnostics
        """
        # Validate inputs
        validation = self.validate_inputs(inputs)
        if not validation["valid"]:
            raise ValueError(f"Invalid inputs: {validation['warnings']}")

        # Extract tensors
        base_logits = inputs.base_logits
        object_embeddings = inputs.object_embeddings
        utterance_features = inputs.utterance_features
        object_geometry = inputs.object_geometry
        candidate_mask = inputs.candidate_mask

        B, N = base_logits.shape

        # Initialize diagnostics
        diagnostics = {
            "validation": validation,
            "batch_size": B,
            "num_objects": N,
        }

        # If sub-modules not set, return base logits unchanged
        # (This allows wrapper to be used as a protocol without full implementation)
        if self._dense_relation_module is None:
            log.warning("Dense relation module not set, returning base logits unchanged")
            return Cover3DOutput(
                reranked_logits=base_logits,
                diagnostics=diagnostics,
            )

        # === Dense Relation Evidence ===
        relation_result = self._dense_relation_module(
            object_embeddings=object_embeddings,
            object_geometry=object_geometry,
            utterance_features=utterance_features,
            candidate_mask=candidate_mask,
        )
        dense_relation_scores = relation_result["relation_scores"]

        diagnostics["dense_relation_stats"] = {
            "mean": dense_relation_scores.mean().item(),
            "std": dense_relation_scores.std().item(),
            "min": dense_relation_scores.min().item(),
            "max": dense_relation_scores.max().item(),
        }

        # === Soft Anchor Posterior ===
        if self._anchor_posterior_module is not None:
            anchor_result = self._anchor_posterior_module(
                utterance_features=utterance_features,
                object_embeddings=object_embeddings,
                object_class_features=inputs.object_class_features,
                candidate_mask=candidate_mask,
                anchor_priors=inputs.anchor_priors,
            )
            anchor_posterior = anchor_result["anchor_posterior"]
            anchor_entropy = anchor_result["anchor_entropy"]
            top_anchor_margin = anchor_result["top_anchor_margin"]
        else:
            # Fallback: uniform anchor posterior
            anchor_posterior = torch.ones(B, N, device=base_logits.device) / N
            anchor_entropy = torch.log(torch.tensor(N)).expand(B).to(base_logits.device)
            top_anchor_margin = torch.zeros(B, device=base_logits.device)

        diagnostics["anchor_stats"] = {
            "mean_entropy": anchor_entropy.mean().item(),
            "min_entropy": anchor_entropy.min().item(),
            "max_entropy": anchor_entropy.max().item(),
        }

        # === Calibrated Fusion ===
        base_margin = self.compute_base_margin(base_logits)
        relation_margin = self.compute_relation_margin(dense_relation_scores)

        if self._calibration_module is not None:
            fusion_result = self._calibration_module(
                base_logits=base_logits,
                relation_scores=dense_relation_scores,
                anchor_posterior=anchor_posterior,
                anchor_entropy=anchor_entropy,
                base_margin=base_margin,
                relation_margin=relation_margin,
                candidate_mask=candidate_mask,
            )
            reranked_logits = fusion_result["fused_logits"]
            gate_values = fusion_result["gate_values"]
        else:
            # Fallback: simple weighted average (gate=0.3)
            gate = 0.3
            reranked_logits = (1 - gate) * base_logits + gate * dense_relation_scores
            gate_values = torch.full((B,), gate, device=base_logits.device)

        diagnostics["fusion_stats"] = {
            "mean_gate": gate_values.mean().item(),
            "min_gate": gate_values.min().item(),
            "max_gate": gate_values.max().item(),
            "base_margin_mean": base_margin.mean().item(),
            "relation_margin_mean": relation_margin.mean().item(),
        }

        # Build output
        output = Cover3DOutput(
            reranked_logits=reranked_logits,
            diagnostics=diagnostics,
            dense_relation_scores=dense_relation_scores if return_intermediate else None,
            anchor_posterior=anchor_posterior if return_intermediate else None,
            anchor_entropy=anchor_entropy if return_intermediate else None,
            top_anchor_margin=top_anchor_margin if return_intermediate else None,
            base_margin=base_margin if return_intermediate else None,
            relation_margin=relation_margin if return_intermediate else None,
            gate_values=gate_values if return_intermediate else None,
        )

        return output

    def forward(self, inputs: Cover3DInput) -> Cover3DOutput:
        """Forward pass alias for rerank method."""
        return self.rerank(inputs)