"""Chunked Dense Implicit Pairwise Relation Module for 3D Object Grounding.

This module computes dense pairwise relations (all N² pairs) but processes
them in memory-safe chunks to avoid O(N²) memory spikes.

Key design:
- Numerical equivalence to dense v1 (same coverage, same operations)
- Memory-safe: peak memory ~N/C times lower than dense
- Chunked computation along j dimension (neighbors)
- Accumulates scores, then applies softmax over full N

This is a pure engineering stabilization of v1 without changing semantics.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)


class ChunkedDensePairwiseRelationModule(nn.Module):
    """Chunked dense pairwise relation module.

    Preserves full N² coverage like v1, but computes in chunks to avoid
    memory spikes. Numerically equivalent to dense version within FP tolerance.

    Architecture:
    1. Compute pairwise geometry features (relative position, size)
    2. For each chunk of j neighbors:
       - Compute pairwise features for all i vs chunk_j
       - Score each pair conditioned on language
    3. Accumulate scores into full [B, N, N] tensor
    4. Apply softmax over all N (preserves dense semantics)
    5. Aggregate relation context via weighted sum
    6. Fuse with original features via residual addition

    Memory: O(N×chunk_size) instead of O(N²)
    Coverage: Full N² (same as dense)
    """

    def __init__(
        self,
        object_dim: int = 320,
        language_dim: int = 256,
        geometry_dim: int = 6,
        hidden_dim: int = 256,
        num_mlp_layers: int = 2,
        dropout: float = 0.1,
        use_residual: bool = True,
        use_gate: bool = False,
        gate_init: float = 0.1,
        temperature: float = 1.0,
        chunk_size: int = 8,  # Number of j neighbors per chunk
    ):
        super().__init__()

        self.object_dim = object_dim
        self.language_dim = language_dim
        self.geometry_dim = geometry_dim
        self.hidden_dim = hidden_dim
        self.use_residual = use_residual
        self.use_gate = use_gate
        self.temperature = temperature
        self.chunk_size = chunk_size

        # Pair input dimension (same as dense v1)
        pair_input_dim = 2 * object_dim + geometry_dim + language_dim

        # Build relation score MLP (same as v1)
        mlp_layers = []
        for layer_idx in range(num_mlp_layers):
            in_dim = pair_input_dim if layer_idx == 0 else hidden_dim
            out_dim = hidden_dim if layer_idx < num_mlp_layers - 1 else 1
            mlp_layers.append(nn.Linear(in_dim, out_dim))
            if layer_idx < num_mlp_layers - 1:
                mlp_layers.append(nn.ReLU())
                mlp_layers.append(nn.Dropout(dropout))

        self.relation_mlp = nn.Sequential(*mlp_layers)

        # Optional learned gate (same as v1)
        if use_gate and not use_residual:
            self.gate = nn.Parameter(torch.tensor(gate_init))
            log.info(f"Using learned gate initialized to {gate_init}")
        else:
            self.gate = None

        log.info(
            f"ChunkedDensePairwiseRelationModule: object_dim={object_dim}, "
            f"language_dim={language_dim}, hidden_dim={hidden_dim}, "
            f"chunk_size={chunk_size}, use_residual={use_residual}"
        )

    def forward(
        self,
        object_features: torch.Tensor,  # [B, N, D]
        language_embedding: torch.Tensor,  # [B, D_lang]
        centers: torch.Tensor,  # [B, N, 3]
        sizes: torch.Tensor,  # [B, N, 3]
        object_mask: Optional[torch.Tensor] = None,  # [B, N]
        scene_diameter: Optional[float] = 5.0,
    ) -> dict:
        """
        Compute relation-aware object features using chunked dense computation.

        Args:
            object_features: [B, N, D] object features
            language_embedding: [B, D_lang] language embedding
            centers: [B, N, 3] object centers
            sizes: [B, N, 3] object sizes
            object_mask: [B, N] boolean mask
            scene_diameter: scene scale for normalization

        Returns:
            dict containing:
            - enhanced_features: [B, N, D] relation-enhanced features
            - relation_weights: [B, N, N] full attention weights (dense semantics)
        """
        B, N, D = object_features.shape
        device = object_features.device

        # Initialize full score tensor (will accumulate chunk scores)
        relation_scores = torch.zeros(B, N, N, device=device)

        # Chunk over j dimension
        for j_start in range(0, N, self.chunk_size):
            j_end = min(j_start + self.chunk_size, N)
            j_chunk_size = j_end - j_start

            # Get chunk of j neighbors
            feat_j_chunk = object_features[:, j_start:j_end, :]  # [B, C, D]
            center_j_chunk = centers[:, j_start:j_end, :]  # [B, C, 3]
            size_j_chunk = sizes[:, j_start:j_end, :]  # [B, C, 3]

            # Compute relative geometry: i vs chunk_j
            # rel_position[i, j_chunk] = center_i - center_{j_chunk}
            rel_position = centers.unsqueeze(2) - center_j_chunk.unsqueeze(1)  # [B, N, C, 3]
            rel_size = sizes.unsqueeze(2) - size_j_chunk.unsqueeze(1)  # [B, N, C, 3]

            # Normalize geometry (same as v1)
            if scene_diameter is not None:
                rel_position = rel_position / (scene_diameter + 1e-6)
            else:
                rel_position = rel_position / 5.0
            rel_size = rel_size / 2.0

            # Concatenate geometry
            rel_geometry = torch.cat([rel_position, rel_size], dim=-1)  # [B, N, C, 6]

            # Build pair input (chunked)
            feat_i = object_features.unsqueeze(2).expand(B, N, j_chunk_size, D)  # [B, N, C, D]
            feat_j = feat_j_chunk.unsqueeze(1).expand(B, N, j_chunk_size, D)  # [B, N, C, D]

            # Language conditioning
            lang_expanded = language_embedding.unsqueeze(1).unsqueeze(2).expand(B, N, j_chunk_size, -1)  # [B, N, C, D_lang]

            # Concatenate all features (same as dense v1)
            pair_input = torch.cat([
                feat_i,  # [B, N, C, D]
                feat_j,  # [B, N, C, D]
                rel_geometry,  # [B, N, C, 6]
                lang_expanded,  # [B, N, C, D_lang]
            ], dim=-1)  # [B, N, C, pair_input_dim]

            # Compute scores for this chunk
            chunk_scores = self.relation_mlp(pair_input).squeeze(-1)  # [B, N, C]

            # Store in full score tensor
            relation_scores[:, :, j_start:j_end] = chunk_scores

        # Now we have full [B, N, N] scores (dense coverage achieved)

        # Apply temperature
        relation_scores = relation_scores / self.temperature

        # Apply mask (same as v1)
        if object_mask is not None:
            mask_i = object_mask.unsqueeze(2)  # [B, N, 1]
            mask_j = object_mask.unsqueeze(1)  # [B, 1, N]
            pair_mask = mask_i & mask_j  # [B, N, N]
            relation_scores = relation_scores.masked_fill(~pair_mask, -1e9)

        # Softmax over full N (dense semantics preserved)
        relation_weights = F.softmax(relation_scores, dim=2)  # [B, N, N]

        # Aggregate relation context (same as v1)
        relation_context = torch.einsum('bnj,bjd->bnd', relation_weights, object_features)  # [B, N, D]

        # Residual fusion (same as v1)
        if self.use_residual:
            enhanced_features = object_features + relation_context
        elif self.gate is not None:
            gate_value = torch.sigmoid(self.gate)
            enhanced_features = gate_value * relation_context + (1 - gate_value) * object_features
        else:
            enhanced_features = relation_context

        result = {
            "enhanced_features": enhanced_features,
            "relation_weights": relation_weights,  # Full [B, N, N] like dense
        }

        if self.gate is not None:
            result["gate_value"] = torch.sigmoid(self.gate).item()

        return result


class ChunkedDensePairwiseRelationModuleOptimized(nn.Module):
    """Optimized version with gradient checkpointing for larger N."""

    def __init__(
        self,
        object_dim: int = 320,
        language_dim: int = 256,
        hidden_dim: int = 256,
        chunk_size: int = 8,
        use_residual: bool = True,
        use_gradient_checkpoint: bool = False,
    ):
        super().__init__()

        self.use_residual = use_residual
        self.chunk_size = chunk_size
        self.use_gradient_checkpoint = use_gradient_checkpoint

        pair_input_dim = 2 * object_dim + 6 + language_dim

        self.relation_mlp = nn.Sequential(
            nn.Linear(pair_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        log.info(f"ChunkedDensePairwiseRelationModuleOptimized: chunk_size={chunk_size}")

    def _compute_chunk(self, object_features, language_embedding, centers, sizes, j_start, j_end):
        """Compute relations for one chunk of j neighbors."""
        B, N, D = object_features.shape
        j_chunk_size = j_end - j_start

        feat_j_chunk = object_features[:, j_start:j_end, :]
        center_j_chunk = centers[:, j_start:j_end, :]
        size_j_chunk = sizes[:, j_start:j_end, :]

        rel_position = (centers.unsqueeze(2) - center_j_chunk.unsqueeze(1)) / 5.0
        rel_size = (sizes.unsqueeze(2) - size_j_chunk.unsqueeze(1)) / 2.0
        rel_geometry = torch.cat([rel_position, rel_size], dim=-1)

        feat_i = object_features.unsqueeze(2).expand(B, N, j_chunk_size, D)
        feat_j = feat_j_chunk.unsqueeze(1).expand(B, N, j_chunk_size, D)
        lang_expanded = language_embedding.unsqueeze(1).unsqueeze(2).expand(B, N, j_chunk_size, -1)

        pair_input = torch.cat([feat_i, feat_j, rel_geometry, lang_expanded], dim=-1)
        chunk_scores = self.relation_mlp(pair_input).squeeze(-1)

        return chunk_scores

    def forward(self, object_features, language_embedding, centers, sizes, object_mask=None):
        B, N, D = object_features.shape
        device = object_features.device

        relation_scores = torch.zeros(B, N, N, device=device)

        for j_start in range(0, N, self.chunk_size):
            j_end = min(j_start + self.chunk_size, N)

            if self.use_gradient_checkpoint and self.training:
                chunk_scores = torch.utils.checkpoint.checkpoint(
                    self._compute_chunk,
                    object_features, language_embedding, centers, sizes, j_start, j_end,
                    use_reentrant=False,
                )
            else:
                chunk_scores = self._compute_chunk(
                    object_features, language_embedding, centers, sizes, j_start, j_end
                )

            relation_scores[:, :, j_start:j_end] = chunk_scores

        if object_mask is not None:
            pair_mask = object_mask.unsqueeze(2) & object_mask.unsqueeze(1)
            relation_scores = relation_scores.masked_fill(~pair_mask, -1e9)

        relation_weights = F.softmax(relation_scores, dim=2)
        relation_context = torch.einsum('bnj,bjd->bnd', relation_weights, object_features)

        if self.use_residual:
            enhanced_features = object_features + relation_context
        else:
            enhanced_features = relation_context

        return {
            "enhanced_features": enhanced_features,
            "relation_weights": relation_weights,
        }