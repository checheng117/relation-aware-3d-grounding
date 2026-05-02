"""COVER-3D Soft Anchor Posterior Module: Uncertainty-Aware Anchor Estimation.

This module computes a soft distribution over potential anchor objects,
avoiding hard parser decisions while incorporating multiple weak signals.

Key design:
- Soft distribution: no hard anchor assignment (all objects have non-zero probability)
- Multi-source evidence: combines language, class features, and optional priors
- Entropy tracking: measures anchor uncertainty for calibration
- Parser priors as weak signals: structured parser outputs are priors, not decisions

Architecture:
1. Project utterance features to anchor query space
2. Match against object embeddings/class features
3. Combine with optional priors (parser outputs, relation priors)
4. Softmax to get posterior distribution
5. Compute entropy, top-anchor margin for calibration signals

Output:
- anchor_posterior [B, N]: soft distribution over objects as potential anchors
- anchor_entropy [B]: uncertainty measure (high = unsure about anchor)
- top_anchor_margin [B]: concentration measure (high = confident anchor)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)


class SoftAnchorPosteriorModule(nn.Module):
    """Soft anchor posterior estimation with uncertainty tracking.

    Computes a probability distribution over scene objects indicating
    which are likely to be relational anchors mentioned in the utterance.

    Key constraint: NO hard decisions. All objects receive non-zero
    probability, and entropy is tracked for calibration.

    Parameters:
    - object_dim: dimension of object embeddings
    - language_dim: dimension of utterance features
    - class_dim: dimension of class semantic features
    - hidden_dim: projection hidden dimension
    - temperature: softmax temperature (higher = softer)
    - min_entropy_threshold: entropy below this indicates confident anchor
    - prior_weight: weight for external priors (0-1, low = weak prior influence)
    """

    def __init__(
        self,
        object_dim: int = 320,
        language_dim: int = 256,
        class_dim: int = 64,
        hidden_dim: int = 128,
        temperature: float = 1.0,
        min_entropy_threshold: float = 0.5,
        prior_weight: float = 0.2,
        use_class_features: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.object_dim = object_dim
        self.language_dim = language_dim
        self.class_dim = class_dim
        self.hidden_dim = hidden_dim
        self.temperature = temperature
        self.min_entropy_threshold = min_entropy_threshold
        self.prior_weight = prior_weight
        self.use_class_features = use_class_features

        # Language → anchor query projection
        self.anchor_query_proj = nn.Sequential(
            nn.Linear(language_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Object → anchor key projection
        # Input may be embeddings only, or embeddings + class features
        if use_class_features:
            object_key_dim = object_dim + class_dim
        else:
            object_key_dim = object_dim

        self.anchor_key_proj = nn.Sequential(
            nn.Linear(object_key_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Prior integration gate
        # Controls how much external priors influence posterior
        if prior_weight > 0:
            self.prior_gate = nn.Sequential(
                nn.Linear(1, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.Sigmoid(),
            )
        else:
            self.prior_gate = None

        log.info(
            f"SoftAnchorPosteriorModule: object_dim={object_dim}, "
            f"language_dim={language_dim}, temperature={temperature}, "
            f"prior_weight={prior_weight}"
        )

    def compute_entropy(self, probs: torch.Tensor) -> torch.Tensor:
        """Compute entropy of probability distribution.

        Args:
            probs: [B, N] probability distribution

        Returns:
            entropy: [B] entropy value (higher = more uncertain)
        """
        # Entropy: H(p) = -sum(p * log(p))
        # Clamp to avoid log(0)
        probs_clamped = probs.clamp(min=1e-10)
        entropy = -(probs_clamped * probs_clamped.log()).sum(dim=-1)
        return entropy

    def compute_top_margin(self, probs: torch.Tensor) -> torch.Tensor:
        """Compute margin between top-1 and top-2 anchor probabilities.

        Args:
            probs: [B, N] probability distribution

        Returns:
            margin: [B] margin value (higher = more concentrated)
        """
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        margin = sorted_probs[:, 0] - sorted_probs[:, 1]
        return margin

    def forward(
        self,
        utterance_features: torch.Tensor,  # [B, L] or [B, D]
        object_embeddings: torch.Tensor,  # [B, N, D]
        object_class_features: Optional[torch.Tensor] = None,  # [B, N, C]
        candidate_mask: Optional[torch.Tensor] = None,  # [B, N]
        anchor_priors: Optional[torch.Tensor] = None,  # [B, N] external priors
        parser_confidence: Optional[torch.Tensor] = None,  # [B] parser confidence
        **kwargs,
    ) -> Dict[str, Any]:
        """Forward pass: compute soft anchor posterior.

        Args:
            utterance_features: language representation
            object_embeddings: object features from base model
            object_class_features: class semantic features (optional)
            candidate_mask: valid object mask
            anchor_priors: external anchor prior distribution (weak signal)
            parser_confidence: confidence of parser if used

        Returns:
            dict with:
            - anchor_posterior: [B, N] soft anchor distribution
            - anchor_entropy: [B] uncertainty measure
            - top_anchor_margin: [B] concentration measure
            - diagnostics: dict with computation stats
        """
        B, N, D = object_embeddings.shape
        device = object_embeddings.device

        # Handle utterance features dimension
        if utterance_features.dim() == 2:
            # [B, L] → use as-is
            lang_feat = utterance_features
        else:
            # Unexpected shape, flatten
            lang_feat = utterance_features.view(B, -1)

        # Project language to anchor query
        anchor_query = self.anchor_query_proj(lang_feat)  # [B, hidden_dim]

        # Build object key features
        if self.use_class_features and object_class_features is not None:
            # Concatenate embeddings and class features
            object_key_input = torch.cat([
                object_embeddings,
                object_class_features,
            ], dim=-1)  # [B, N, D+C]
        else:
            object_key_input = object_embeddings  # [B, N, D]

        # Project objects to anchor keys
        anchor_keys = self.anchor_key_proj(object_key_input)  # [B, N, hidden_dim]

        # Compute anchor-utterance matching scores
        # anchor_query: [B, hidden_dim]
        # anchor_keys: [B, N, hidden_dim]
        match_scores = torch.bmm(
            anchor_keys,  # [B, N, hidden_dim]
            anchor_query.unsqueeze(2),  # [B, hidden_dim, 1]
        ).squeeze(-1)  # [B, N]

        # Apply temperature scaling
        match_scores = match_scores / self.temperature

        # Apply candidate mask
        if candidate_mask is not None:
            match_scores = match_scores.masked_fill(~candidate_mask, float("-inf"))

        # Convert to probability (base posterior)
        base_posterior = F.softmax(match_scores, dim=-1)  # [B, N]

        # Integrate external priors if available
        if anchor_priors is not None and self.prior_weight > 0 and self.prior_gate is not None:
            # Prior should be a distribution [B, N]
            # Normalize prior if needed
            if anchor_priors.sum(dim=-1).mean() != 1.0:
                anchor_priors = F.softmax(anchor_priors, dim=-1)

            # Compute gate value based on parser confidence
            if parser_confidence is not None:
                # Higher confidence → stronger prior influence
                gate_input = parser_confidence.unsqueeze(-1)  # [B, 1]
                gate_value = self.prior_gate(gate_input).squeeze(-1)  # [B]
            else:
                # Default gate: use prior_weight directly
                gate_value = torch.full((B,), self.prior_weight, device=device)

            # Combine: posterior = (1-gate) * base + gate * prior
            anchor_posterior = (1 - gate_value.unsqueeze(-1)) * base_posterior + \
                               gate_value.unsqueeze(-1) * anchor_priors
        else:
            anchor_posterior = base_posterior

        # Compute calibration signals
        anchor_entropy = self.compute_entropy(anchor_posterior)  # [B]
        top_anchor_margin = self.compute_top_margin(anchor_posterior)  # [B]

        # Normalize entropy by log(N) for comparable scale across scenes
        max_entropy = torch.log(torch.tensor(N, dtype=torch.float, device=device))
        normalized_entropy = anchor_entropy / max_entropy

        # Identify confident vs uncertain anchors
        is_confident_anchor = normalized_entropy < self.min_entropy_threshold

        # Build diagnostics
        diagnostics = {
            "mean_entropy": anchor_entropy.mean().item(),
            "normalized_entropy_mean": normalized_entropy.mean().item(),
            "min_entropy": anchor_entropy.min().item(),
            "max_entropy": anchor_entropy.max().item(),
            "mean_margin": top_anchor_margin.mean().item(),
            "confident_anchor_pct": is_confident_anchor.float().mean().item(),
            "max_possible_entropy": max_entropy.item(),
            "num_objects": N,
            "prior_integrated": anchor_priors is not None,
        }

        # Top anchor identification (for diagnostics)
        top_anchor_idx = anchor_posterior.argmax(dim=-1)  # [B]
        top_anchor_prob = anchor_posterior.max(dim=-1).values  # [B]

        diagnostics["top_anchor_prob_mean"] = top_anchor_prob.mean().item()
        diagnostics["top_anchor_prob_min"] = top_anchor_prob.min().item()

        # NaN/inf checks
        if torch.isnan(anchor_posterior).any():
            log.warning("NaN in anchor_posterior")
            diagnostics["has_nan"] = True
            anchor_posterior = torch.nan_to_num(anchor_posterior, nan=1.0/N)

        return {
            "anchor_posterior": anchor_posterior,  # [B, N]
            "anchor_entropy": anchor_entropy,  # [B]
            "normalized_entropy": normalized_entropy,  # [B]
            "top_anchor_margin": top_anchor_margin,  # [B]
            "top_anchor_idx": top_anchor_idx,  # [B]
            "top_anchor_prob": top_anchor_prob,  # [B]
            "is_confident_anchor": is_confident_anchor,  # [B]
            "match_scores": match_scores,  # [B, N] (before softmax)
            "diagnostics": diagnostics,
        }


class AnchorPosteriorModule(SoftAnchorPosteriorModule):
    """Alias for SoftAnchorPosteriorModule (backward compatibility)."""

    pass