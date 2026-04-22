"""COVER-3D Calibrated Fusion Gate: Signal-Dependent Logit Combination.

This module fuses base model logits with dense relation scores using
a calibration gate that depends on multiple confidence signals.

Key design:
- Signal-dependent gate: gate value depends on entropy, margins, confidence
- Bounded gate: [min_gate, max_gate] to prevent collapse
- No collapse to base-only or relation-only (always mixed)
- Calibration: high anchor entropy → less relation influence
- Calibration: high base margin → more base influence

Architecture:
1. Collect calibration signals:
   - base_margin (base model confidence)
   - anchor_entropy (anchor uncertainty)
   - relation_margin (relation concentration)
   - optional parser_confidence
2. Compute gate value via MLP: gate ∈ [min, max]
3. Fuse: fused = (1-gate) * base_logits + gate * relation_scores
4. Emit diagnostics for analysis

Output:
- fused_logits [B, N]: combined predictions
- gate_values [B]: gate per sample
- calibration diagnostics

Key constraint: Gate bounded [0.1, 0.9] prevents:
- gate=0: always base (relation branch useless)
- gate=1: always relation (base model ignored)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)


class CalibratedFusionGate(nn.Module):
    """Calibrated fusion gate for combining base and relation scores.

    Computes a gate value that determines how much relation evidence
    influences the final prediction. Gate is signal-dependent and bounded.

    Parameters:
    - signal_dim: dimension of calibration signal vector
    - hidden_dim: MLP hidden dimension
    - min_gate: minimum gate value (prevents full base reliance)
    - max_gate: maximum gate value (prevents full relation reliance)
    - init_bias: initial gate bias (default 0.5 = balanced)
    - dropout: dropout rate
    - gate_prior_weight: L2 regularization toward gate=0.5 (default 0.0 = no reg)

    Gate computation:
    - gate_mlp(signal_vector) ∈ [min_gate, max_gate]
    - fused = (1-gate) * base_logits + gate * relation_scores

    Signals (input to gate MLP):
    - base_margin: [B] base model prediction confidence
    - anchor_entropy: [B] anchor uncertainty (high → less relation)
    - relation_margin: [B] relation evidence concentration
    - parser_confidence: [B] optional parser confidence
    """

    def __init__(
        self,
        signal_dim: int = 4,
        hidden_dim: int = 32,
        min_gate: float = 0.1,
        max_gate: float = 0.9,
        init_bias: float = 0.5,
        dropout: float = 0.1,
        use_entropy_gate: bool = True,
        use_margin_gate: bool = True,
        gate_prior_weight: float = 0.0,
        gate_prior_target: float = 0.5,
    ):
        super().__init__()

        self.signal_dim = signal_dim
        self.min_gate = min_gate
        self.max_gate = max_gate
        self.use_entropy_gate = use_entropy_gate
        self.use_margin_gate = use_margin_gate
        self.gate_prior_weight = gate_prior_weight
        self.gate_prior_target = gate_prior_target

        # Gate MLP - reduced capacity for v2
        self.gate_mlp = nn.Sequential(
            nn.Linear(signal_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),  # Output ∈ [0, 1]
        )

        # Initialize bias toward balanced gate (init_bias=0.5)
        # Apply scaling: output = sigmoid(raw) * (max - min) + min
        # To get init_bias, set raw such that sigmoid(raw) = (init_bias - min) / (max - min)
        target_sigmoid = (init_bias - min_gate) / (max_gate - min_gate)
        init_raw = torch.log(torch.tensor(target_sigmoid / (1 - target_sigmoid + 1e-10)))

        # Set bias in last linear layer
        with torch.no_grad():
            self.gate_mlp[-2].bias.fill_(init_raw.item())

        log.info(
            f"CalibratedFusionGate: min_gate={min_gate}, max_gate={max_gate}, "
            f"init_bias={init_bias}, gate_prior_weight={gate_prior_weight}, signal_dim={signal_dim}"
        )

    def compute_gate_prior_loss(self, gate_values: torch.Tensor) -> torch.Tensor:
        """Compute L2 regularization loss toward gate prior target.

        Args:
            gate_values: [B] gate values

        Returns:
            scalar loss tensor
        """
        if self.gate_prior_weight == 0.0:
            return torch.zeros(1, device=gate_values.device)

        prior_loss = ((gate_values - self.gate_prior_target) ** 2).mean()
        return self.gate_prior_weight * prior_loss

    def build_signal_vector(
        self,
        base_margin: torch.Tensor,  # [B]
        anchor_entropy: torch.Tensor,  # [B]
        relation_margin: torch.Tensor,  # [B]
        parser_confidence: Optional[torch.Tensor] = None,  # [B]
    ) -> torch.Tensor:
        """Build calibration signal vector.

        Returns [B, signal_dim] tensor.
        """
        B = base_margin.shape[0]
        device = base_margin.device

        # Normalize signals to [0, 1] range
        # base_margin: higher = more confident → scale to 0-1
        # anchor_entropy: higher = more uncertain → invert (1 - normalized)
        # relation_margin: higher = more concentrated → scale to 0-1

        # Base margin: sigmoid normalization
        base_signal = torch.sigmoid(base_margin)

        # Anchor entropy: normalize by max possible entropy
        # Assume max_entropy ≈ log(100) ≈ 4.6 for typical scenes
        max_entropy_approx = 4.6
        entropy_normalized = anchor_entropy / max_entropy_approx
        # Invert: high entropy → low signal (less relation influence)
        entropy_signal = 1 - entropy_normalized.clamp(0, 1)

        # Relation margin: sigmoid normalization
        relation_signal = torch.sigmoid(relation_margin)

        # Parser confidence: if available, use directly
        if parser_confidence is not None:
            parser_signal = parser_confidence.clamp(0, 1)
        else:
            parser_signal = torch.zeros(B, device=device)

        # Stack signals
        signal_vector = torch.stack([
            base_signal,
            entropy_signal,
            relation_signal,
            parser_signal,
        ], dim=-1)  # [B, 4]

        return signal_vector

    def compute_gate(
        self,
        signal_vector: torch.Tensor,  # [B, signal_dim]
    ) -> torch.Tensor:
        """Compute bounded gate value.

        Returns gate ∈ [min_gate, max_gate]
        """
        # Raw gate from MLP: [B, 1]
        raw_gate = self.gate_mlp(signal_vector).squeeze(-1)  # [B]

        # Scale to bounds
        gate = raw_gate * (self.max_gate - self.min_gate) + self.min_gate

        return gate

    def forward(
        self,
        base_logits: torch.Tensor,  # [B, N]
        relation_scores: torch.Tensor,  # [B, N]
        anchor_posterior: torch.Tensor,  # [B, N]
        anchor_entropy: torch.Tensor,  # [B]
        base_margin: torch.Tensor,  # [B]
        relation_margin: torch.Tensor,  # [B]
        candidate_mask: Optional[torch.Tensor] = None,  # [B, N]
        parser_confidence: Optional[torch.Tensor] = None,  # [B]
        **kwargs,
    ) -> Dict[str, Any]:
        """Forward pass: fuse base and relation scores with calibrated gate.

        Args:
            base_logits: base model predictions
            relation_scores: dense relation evidence
            anchor_posterior: anchor distribution (for potential use)
            anchor_entropy: anchor uncertainty
            base_margin: base model confidence
            relation_margin: relation concentration
            candidate_mask: valid object mask
            parser_confidence: optional parser confidence

        Returns:
            dict with:
            - fused_logits: [B, N] combined predictions
            - gate_values: [B] gate per sample
            - diagnostics: calibration stats
        """
        B, N = base_logits.shape
        device = base_logits.device

        # Build signal vector
        signal_vector = self.build_signal_vector(
            base_margin=base_margin,
            anchor_entropy=anchor_entropy,
            relation_margin=relation_margin,
            parser_confidence=parser_confidence,
        )  # [B, signal_dim]

        # Compute gate
        gate_values = self.compute_gate(signal_vector)  # [B]

        # Fuse logits
        # fused = (1-gate) * base + gate * relation
        gate_exp = gate_values.unsqueeze(-1)  # [B, 1]
        fused_logits = (1 - gate_exp) * base_logits + gate_exp * relation_scores

        # Apply candidate mask
        if candidate_mask is not None:
            fused_logits = fused_logits.masked_fill(~candidate_mask, float("-inf"))

        # Alternative fusion: use anchor posterior to weight relation scores
        # relation_weighted = relation_scores * anchor_posterior
        # This emphasizes relation evidence for objects that are potential anchors
        # Optional, not used by default

        # Compute diagnostics
        diagnostics = {
            "gate_mean": gate_values.mean().item(),
            "gate_std": gate_values.std().item(),
            "gate_min": gate_values.min().item(),
            "gate_max": gate_values.max().item(),
            "base_margin_mean": base_margin.mean().item(),
            "relation_margin_mean": relation_margin.mean().item(),
            "anchor_entropy_mean": anchor_entropy.mean().item(),
            "signal_base_mean": signal_vector[:, 0].mean().item(),
            "signal_entropy_mean": signal_vector[:, 1].mean().item(),
            "signal_relation_mean": signal_vector[:, 2].mean().item(),
            "num_fused": B,
            "num_objects": N,
        }

        # Analyze gate behavior
        high_base_influence = (gate_values < 0.2).float().mean().item()  # mostly base
        high_relation_influence = (gate_values > 0.8).float().mean().item()  # mostly relation
        balanced_influence = ((gate_values >= 0.4) & (gate_values <= 0.6)).float().mean().item()

        diagnostics["high_base_pct"] = high_base_influence
        diagnostics["high_relation_pct"] = high_relation_influence
        diagnostics["balanced_pct"] = balanced_influence

        # Check for collapse
        if diagnostics["high_base_pct"] > 0.95:
            log.warning("Gate collapse detected: >95% samples using mostly base")
            diagnostics["gate_collapse_warning"] = True

        if diagnostics["high_relation_pct"] > 0.95:
            log.warning("Gate collapse detected: >95% samples using mostly relation")
            diagnostics["gate_collapse_warning"] = True

        # NaN/inf checks
        if torch.isnan(fused_logits).any():
            log.warning("NaN in fused_logits")
            diagnostics["has_nan"] = True
            fused_logits = torch.nan_to_num(fused_logits, nan=0.0)

        # Prediction comparison
        base_pred = base_logits.argmax(dim=-1)  # [B]
        fused_pred = fused_logits.argmax(dim=-1)  # [B]
        pred_changed = (base_pred != fused_pred).float().mean().item()

        diagnostics["prediction_changed_pct"] = pred_changed

        return {
            "fused_logits": fused_logits,  # [B, N]
            "gate_values": gate_values,  # [B]
            "signal_vector": signal_vector,  # [B, signal_dim]
            "diagnostics": diagnostics,
        }


class FusionGate(CalibratedFusionGate):
    """Alias for CalibratedFusionGate (backward compatibility)."""

    pass