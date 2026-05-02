"""Implicit Pairwise Relation Module for 3D Object Grounding.

This module models object-object relationships implicitly through:
- Pairwise geometry features (relative position, size)
- Language-conditioned relation scoring
- Attention-based aggregation

No parser dependency. Relations are learned from data.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)


class PairwiseRelationModule(nn.Module):
    """Implicit pairwise relation module for 3D object grounding.

    Architecture:
    1. Compute pairwise geometry features (relative position, size)
    2. Compute pairwise feature differences
    3. Score each pair conditioned on language embedding
    4. Aggregate relation context via softmax attention
    5. Fuse with original features via residual addition

    This is a minimal, lightweight approach that:
    - Does NOT use explicit parser outputs
    - Does NOT use graph neural networks
    - Uses simple 2-layer MLP for relation scoring
    - Uses softmax attention for aggregation
    """

    def __init__(
        self,
        object_dim: int = 256,  # Object feature dimension (from baseline encoder)
        language_dim: int = 256,  # Language embedding dimension
        geometry_dim: int = 6,  # Relative position (3) + relative size (3)
        hidden_dim: int = 256,  # MLP hidden dimension
        num_mlp_layers: int = 2,  # MLP depth
        dropout: float = 0.1,
        use_residual: bool = True,  # Use residual addition (preferred for stability)
        use_gate: bool = False,  # Use learned gate (fallback if residual insufficient)
        gate_init: float = 0.1,  # Initialize gate to small value for stability
        temperature: float = 1.0,  # Softmax temperature
    ):
        super().__init__()

        self.object_dim = object_dim
        self.language_dim = language_dim
        self.geometry_dim = geometry_dim
        self.hidden_dim = hidden_dim
        self.use_residual = use_residual
        self.use_gate = use_gate
        self.temperature = temperature

        # Input to relation MLP:
        # - object_i [object_dim]
        # - object_j [object_dim]
        # - relative_position [3]
        # - relative_size [3]
        # - language_embedding [language_dim]
        pair_input_dim = 2 * object_dim + geometry_dim + language_dim

        # Build relation score MLP
        mlp_layers = []
        for layer_idx in range(num_mlp_layers):
            in_dim = pair_input_dim if layer_idx == 0 else hidden_dim
            out_dim = hidden_dim if layer_idx < num_mlp_layers - 1 else 1
            mlp_layers.append(nn.Linear(in_dim, out_dim))
            if layer_idx < num_mlp_layers - 1:
                mlp_layers.append(nn.ReLU())
                mlp_layers.append(nn.Dropout(dropout))

        self.relation_mlp = nn.Sequential(*mlp_layers)

        # Optional: learned gate for fusion
        if use_gate and not use_residual:
            # Single scalar gate per object (shared across all objects)
            self.gate = nn.Parameter(torch.tensor(gate_init))
            log.info(f"Using learned gate initialized to {gate_init}")
        else:
            self.gate = None

        log.info(
            f"PairwiseRelationModule: object_dim={object_dim}, "
            f"language_dim={language_dim}, hidden_dim={hidden_dim}, "
            f"use_residual={use_residual}, use_gate={use_gate}"
        )

    def forward(
        self,
        object_features: torch.Tensor,  # [B, N, D]
        language_embedding: torch.Tensor,  # [B, D]
        centers: torch.Tensor,  # [B, N, 3]
        sizes: torch.Tensor,  # [B, N, 3]
        object_mask: Optional[torch.Tensor] = None,  # [B, N]
        scene_diameter: Optional[torch.Tensor] = None,  # [B] or scalar
    ) -> dict:
        """
        Compute relation-aware object features.

        Args:
            object_features: [B, N, D] object features from baseline encoder
            language_embedding: [B, D] language embedding
            centers: [B, N, 3] object centers (world coordinates)
            sizes: [B, N, 3] object sizes (bounding box dimensions)
            object_mask: [B, N] boolean mask for valid objects
            scene_diameter: [B] scene diameter for normalization (optional)

        Returns:
            dict containing:
            - enhanced_features: [B, N, D] relation-enhanced object features
            - relation_weights: [B, N, N] attention weights (for analysis)
            - gate_value: float, gate value if using gate
        """
        B, N, D = object_features.shape

        # Step 1: Compute pairwise geometry features
        # Relative position: center_i - center_j
        # Use broadcasting: centers[:, :, None, :] - centers[:, None, :, :]
        rel_position = centers.unsqueeze(2) - centers.unsqueeze(1)  # [B, N, N, 3]

        # Relative size: size_i - size_j
        rel_size = sizes.unsqueeze(2) - sizes.unsqueeze(1)  # [B, N, N, 3]

        # Normalize by scene scale
        if scene_diameter is not None:
            if isinstance(scene_diameter, (int, float)):
                scale = scene_diameter
            else:
                # scene_diameter is [B]
                scale = scene_diameter.unsqueeze(1).unsqueeze(2)  # [B, 1, 1]
            rel_position = rel_position / (scale + 1e-6)
        else:
            # Default normalization (assume ~5m scene diameter)
            rel_position = rel_position / 5.0

        # Normalize size differences (assume ~1m max size)
        rel_size = rel_size / 2.0

        # Concatenate geometry features
        rel_geometry = torch.cat([rel_position, rel_size], dim=-1)  # [B, N, N, 6]

        # Step 2: Compute pairwise feature differences
        # This captures semantic similarity between objects
        feat_i = object_features.unsqueeze(2)  # [B, N, 1, D]
        feat_j = object_features.unsqueeze(1)  # [B, 1, N, D]

        # Step 3: Build pair input with language conditioning
        # Expand language embedding to all pairs
        lang_expanded = language_embedding.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, D_lang]
        lang_expanded = lang_expanded.expand(B, N, N, -1)  # [B, N, N, D_lang]

        # Concatenate all pair features
        pair_input = torch.cat([
            feat_i.expand(B, N, N, D),  # object_i
            feat_j.expand(B, N, N, D),  # object_j
            rel_geometry,  # relative position + size
            lang_expanded,  # language embedding
        ], dim=-1)  # [B, N, N, pair_input_dim]

        # Step 4: Compute relation scores via MLP
        relation_scores = self.relation_mlp(pair_input).squeeze(-1)  # [B, N, N]

        # Apply temperature scaling
        relation_scores = relation_scores / self.temperature

        # Apply mask: invalid pairs should not receive attention
        if object_mask is not None:
            # Create pairwise mask: mask_i AND mask_j
            mask_i = object_mask.unsqueeze(2)  # [B, N, 1]
            mask_j = object_mask.unsqueeze(1)  # [B, 1, N]
            pair_mask = mask_i & mask_j  # [B, N, N]

            # Mask out invalid pairs (set to large negative before softmax)
            relation_scores = relation_scores.masked_fill(~pair_mask, -1e9)

        # Step 5: Compute attention weights via softmax
        # For each object i, aggregate over all j
        relation_weights = F.softmax(relation_scores, dim=2)  # [B, N, N]

        # Step 6: Aggregate relation context
        # weighted sum of neighbor features for each object
        # relation_weights[i, j] = how much object i attends to object j
        relation_context = torch.einsum('bnj,bjd->bnd', relation_weights, object_features)  # [B, N, D]

        # Step 7: Fuse with original features
        if self.use_residual:
            # Simple residual addition (preferred for stability)
            enhanced_features = object_features + relation_context
        elif self.gate is not None:
            # Learned gate fusion
            gate_value = torch.sigmoid(self.gate)
            enhanced_features = gate_value * relation_context + (1 - gate_value) * object_features
        else:
            # Default: just use relation context (not recommended)
            enhanced_features = relation_context

        # Return outputs
        result = {
            "enhanced_features": enhanced_features,
            "relation_weights": relation_weights,
        }

        if self.gate is not None:
            result["gate_value"] = torch.sigmoid(self.gate).item()

        return result


class PairwiseRelationModuleLight(nn.Module):
    """Even lighter version: only relative position + simple scoring.

    For maximum stability and minimal overhead.
    """

    def __init__(
        self,
        object_dim: int = 256,
        language_dim: int = 256,
        hidden_dim: int = 128,  # Smaller
        use_residual: bool = True,
    ):
        super().__init__()

        self.use_residual = use_residual

        # Simplified input: object_i + object_j + relative_position + language
        pair_input_dim = 2 * object_dim + 3 + language_dim

        self.relation_mlp = nn.Sequential(
            nn.Linear(pair_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        object_features: torch.Tensor,  # [B, N, D]
        language_embedding: torch.Tensor,  # [B, D]
        centers: torch.Tensor,  # [B, N, 3]
        object_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        B, N, D = object_features.shape

        # Relative position only
        rel_position = centers.unsqueeze(2) - centers.unsqueeze(1)  # [B, N, N, 3]
        rel_position = rel_position / 5.0  # Normalize

        # Expand features
        feat_i = object_features.unsqueeze(2).expand(B, N, N, D)
        feat_j = object_features.unsqueeze(1).expand(B, N, N, D)
        lang_expanded = language_embedding.unsqueeze(1).unsqueeze(2).expand(B, N, N, -1)

        # Concatenate
        pair_input = torch.cat([feat_i, feat_j, rel_position, lang_expanded], dim=-1)

        # Score and softmax
        relation_scores = self.relation_mlp(pair_input).squeeze(-1)  # [B, N, N]

        if object_mask is not None:
            pair_mask = object_mask.unsqueeze(2) & object_mask.unsqueeze(1)
            relation_scores = relation_scores.masked_fill(~pair_mask, -1e9)

        relation_weights = F.softmax(relation_scores, dim=2)

        # Aggregate
        relation_context = torch.einsum('bnj,bjd->bnd', relation_weights, object_features)

        # Fuse
        if self.use_residual:
            enhanced_features = object_features + relation_context
        else:
            enhanced_features = relation_context

        return {
            "enhanced_features": enhanced_features,
            "relation_weights": relation_weights,
        }