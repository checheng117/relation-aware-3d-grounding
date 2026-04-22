"""Sparse Implicit Pairwise Relation Module for 3D Object Grounding.

This module models object-object relationships with sparse top-k neighbors:
- Only compute pairwise features for k nearest neighbors
- Reduces memory from O(N²) to O(N×k)
- Preserves spatial locality while enabling stable training

No parser dependency. Relations are learned from data.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)


class SparsePairwiseRelationModule(nn.Module):
    """Sparse pairwise relation module using top-k nearest neighbors.

    Memory-safe version that avoids dense O(N²) computation.

    Architecture:
    1. Compute pairwise distances (for neighbor selection)
    2. Select top-k nearest neighbors for each object
    3. Compute pairwise features only for selected neighbors
    4. Score each pair conditioned on language embedding
    5. Aggregate relation context via softmax over k neighbors
    6. Fuse with original features via residual addition

    Memory: O(N × k) instead of O(N²)
    """

    def __init__(
        self,
        object_dim: int = 320,  # Object feature dimension
        language_dim: int = 256,  # Language embedding dimension
        hidden_dim: int = 256,  # MLP hidden dimension
        topk: int = 5,  # Number of neighbors per object
        dropout: float = 0.1,
        use_residual: bool = True,
        temperature: float = 1.0,
        include_size: bool = True,  # Include relative size in geometry
    ):
        super().__init__()

        self.object_dim = object_dim
        self.language_dim = language_dim
        self.hidden_dim = hidden_dim
        self.topk = topk
        self.use_residual = use_residual
        self.temperature = temperature
        self.include_size = include_size

        # Geometry dimension: position (3) + optionally size (3)
        self.geometry_dim = 3 + (3 if include_size else 0)

        # Input to relation MLP:
        # - object_i [object_dim]
        # - object_j [object_dim]
        # - relative_position [3]
        # - relative_size [3] (optional)
        # - language_embedding [language_dim]
        pair_input_dim = 2 * object_dim + self.geometry_dim + language_dim

        # Build relation score MLP (2 layers)
        self.relation_mlp = nn.Sequential(
            nn.Linear(pair_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        log.info(
            f"SparsePairwiseRelationModule: object_dim={object_dim}, "
            f"language_dim={language_dim}, hidden_dim={hidden_dim}, "
            f"topk={topk}, use_residual={use_residual}"
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
        Compute relation-aware object features using sparse top-k neighbors.

        Args:
            object_features: [B, N, D] object features from baseline encoder
            language_embedding: [B, D_lang] language embedding
            centers: [B, N, 3] object centers (world coordinates)
            sizes: [B, N, 3] object sizes (bounding box dimensions)
            object_mask: [B, N] boolean mask for valid objects
            scene_diameter: scene diameter for distance normalization

        Returns:
            dict containing:
            - enhanced_features: [B, N, D] relation-enhanced object features
            - relation_weights: [B, N, k] sparse attention weights
            - neighbor_indices: [B, N, k] indices of selected neighbors
        """
        B, N, D = object_features.shape
        k = min(self.topk, N - 1)  # Can't have more neighbors than N-1

        # Step 1: Compute pairwise distances
        # distances[i, j] = ||center_i - center_j||
        diff = centers.unsqueeze(2) - centers.unsqueeze(1)  # [B, N, N, 3]
        distances = (diff ** 2).sum(dim=-1)  # [B, N, N]

        # Normalize distances by scene diameter
        if scene_diameter is not None:
            distances = distances / (scene_diameter ** 2)

        # Step 2: Mask out self (distance = 0) and invalid objects
        if object_mask is not None:
            # Mask invalid pairs
            mask_i = object_mask.unsqueeze(2)  # [B, N, 1]
            mask_j = object_mask.unsqueeze(1)  # [B, 1, N]
            pair_mask = mask_i & mask_j  # [B, N, N]
            distances = distances.masked_fill(~pair_mask, float('inf'))

        # Mask out self (distance to self is 0, set to inf)
        # Create diagonal mask
        diag_mask = torch.eye(N, device=distances.device).unsqueeze(0).expand(B, -1, -1).bool()
        distances = distances.masked_fill(diag_mask, float('inf'))

        # Step 3: Select top-k nearest neighbors for each object
        # neighbor_indices[i] = indices of k nearest neighbors for object i
        neighbor_distances, neighbor_indices = torch.topk(
            distances, k=k, dim=2, largest=False  # Smallest = nearest
        )  # [B, N, k] each

        # Step 4: Gather neighbor features
        # neighbor_features[b, i, m] = object_features[b, neighbor_indices[b, i, m]]
        neighbor_features = self._gather_neighbors(object_features, neighbor_indices)  # [B, N, k, D]
        neighbor_centers = self._gather_neighbors(centers, neighbor_indices)  # [B, N, k, 3]
        neighbor_sizes = self._gather_neighbors(sizes, neighbor_indices)  # [B, N, k, 3]

        # Step 5: Compute relative geometry for selected pairs
        # rel_position[i, m] = center_i - center_{neighbor_m}
        rel_position = centers.unsqueeze(2) - neighbor_centers  # [B, N, k, 3]

        # Normalize relative position
        if scene_diameter is not None:
            rel_position = rel_position / (scene_diameter + 1e-6)
        else:
            rel_position = rel_position / 5.0

        # Relative size
        if self.include_size:
            rel_size = sizes.unsqueeze(2) - neighbor_sizes  # [B, N, k, 3]
            rel_size = rel_size / 2.0  # Normalize
            rel_geometry = torch.cat([rel_position, rel_size], dim=-1)  # [B, N, k, 6]
        else:
            rel_geometry = rel_position  # [B, N, k, 3]

        # Step 6: Build sparse pair input
        # For each object i, pair_input[i, m] = features for (i, neighbor_m)
        feat_i = object_features.unsqueeze(2).expand(B, N, k, D)  # [B, N, k, D] (repeated)
        feat_j = neighbor_features  # [B, N, k, D]

        # Expand language embedding
        lang_expanded = language_embedding.unsqueeze(1).unsqueeze(2).expand(B, N, k, -1)  # [B, N, k, D_lang]

        # Concatenate
        pair_input = torch.cat([
            feat_i,  # object_i [B, N, k, D]
            feat_j,  # neighbor_j [B, N, k, D]
            rel_geometry,  # relative geometry [B, N, k, geom_dim]
            lang_expanded,  # language [B, N, k, D_lang]
        ], dim=-1)  # [B, N, k, pair_input_dim]

        # Step 7: Compute relation scores
        relation_scores = self.relation_mlp(pair_input).squeeze(-1)  # [B, N, k]

        # Apply temperature
        relation_scores = relation_scores / self.temperature

        # Step 8: Handle invalid neighbors (those selected but masked)
        if object_mask is not None:
            # Check which neighbors are actually valid
            neighbor_valid = self._gather_neighbors(object_mask.unsqueeze(-1), neighbor_indices).squeeze(-1)  # [B, N, k]
            relation_scores = relation_scores.masked_fill(~neighbor_valid, -1e9)

        # Step 9: Softmax over k neighbors
        relation_weights = F.softmax(relation_scores, dim=2)  # [B, N, k]

        # Step 10: Aggregate relation context (sparse weighted sum)
        # relation_context[i] = sum_m (relation_weights[i, m] * neighbor_features[i, m])
        relation_context = torch.einsum('bnk,bnkd->bnd', relation_weights, neighbor_features)  # [B, N, D]

        # Step 11: Residual fusion
        if self.use_residual:
            enhanced_features = object_features + relation_context
        else:
            enhanced_features = relation_context

        return {
            "enhanced_features": enhanced_features,
            "relation_weights": relation_weights,
            "neighbor_indices": neighbor_indices,
        }

    def _gather_neighbors(self, tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        """Gather neighbor values using indices.

        Args:
            tensor: [B, N, ...] source tensor
            indices: [B, N, k] neighbor indices

        Returns:
            gathered: [B, N, k, ...] neighbor values
        """
        B, N = indices.shape[:2]
        k = indices.shape[2]

        # Expand indices for gathering
        # indices_expanded: [B, N, k, ...] with same trailing dims as tensor
        if tensor.dim() > 2:
            trailing_dims = tensor.shape[2:]
            indices_expanded = indices.unsqueeze(-1).expand(B, N, k, *trailing_dims)
        else:
            indices_expanded = indices

        # Gather
        gathered = torch.gather(
            tensor.unsqueeze(2).expand(B, N, k, *tensor.shape[2:]),
            dim=1,
            index=indices_expanded,
        )

        return gathered


class SparsePairwiseRelationModuleV2(nn.Module):
    """Alternative implementation with efficient chunked computation.

    Uses chunked processing to further reduce memory spikes.
    """

    def __init__(
        self,
        object_dim: int = 320,
        language_dim: int = 256,
        hidden_dim: int = 256,
        topk: int = 5,
        dropout: float = 0.1,
        use_residual: bool = True,
        chunk_size: int = 8,  # Process objects in chunks
    ):
        super().__init__()

        self.topk = topk
        self.use_residual = use_residual
        self.chunk_size = chunk_size

        # Relation MLP
        pair_input_dim = 2 * object_dim + 6 + language_dim
        self.relation_mlp = nn.Sequential(
            nn.Linear(pair_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        log.info(f"SparsePairwiseRelationModuleV2: topk={topk}, chunk_size={chunk_size}")

    def forward(
        self,
        object_features: torch.Tensor,
        language_embedding: torch.Tensor,
        centers: torch.Tensor,
        sizes: torch.Tensor,
        object_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        B, N, D = object_features.shape
        k = min(self.topk, N - 1)

        # Compute distances and select neighbors
        diff = centers.unsqueeze(2) - centers.unsqueeze(1)
        distances = (diff ** 2).sum(dim=-1)

        # Mask
        if object_mask is not None:
            pair_mask = object_mask.unsqueeze(2) & object_mask.unsqueeze(1)
            distances = distances.masked_fill(~pair_mask, float('inf'))

        diag_mask = torch.eye(N, device=distances.device).unsqueeze(0).expand(B, -1, -1).bool()
        distances = distances.masked_fill(diag_mask, float('inf'))

        neighbor_indices = torch.topk(distances, k=k, dim=2, largest=False)[1]

        # Process in chunks to reduce memory
        relation_context = torch.zeros(B, N, D, device=object_features.device)
        relation_weights_all = torch.zeros(B, N, k, device=object_features.device)

        for chunk_start in range(0, N, self.chunk_size):
            chunk_end = min(chunk_start + self.chunk_size, N)
            chunk_size_actual = chunk_end - chunk_start

            # Get neighbors for this chunk
            chunk_indices = neighbor_indices[:, chunk_start:chunk_end, :]  # [B, chunk, k]

            # Gather chunk data
            chunk_features = object_features[:, chunk_start:chunk_end, :]  # [B, chunk, D]
            chunk_centers = centers[:, chunk_start:chunk_end, :]  # [B, chunk, 3]
            chunk_sizes = sizes[:, chunk_start:chunk_end, :]  # [B, chunk, 3]

            neighbor_feats = torch.gather(
                object_features.unsqueeze(1).expand(B, N, N, D),
                dim=2,
                index=chunk_indices.unsqueeze(-1).expand(B, chunk_size_actual, k, D),
            )  # [B, chunk, k, D]

            neighbor_centers_chunk = torch.gather(
                centers.unsqueeze(1).expand(B, N, N, 3),
                dim=2,
                index=chunk_indices.unsqueeze(-1).expand(B, chunk_size_actual, k, 3),
            )

            neighbor_sizes_chunk = torch.gather(
                sizes.unsqueeze(1).expand(B, N, N, 3),
                dim=2,
                index=chunk_indices.unsqueeze(-1).expand(B, chunk_size_actual, k, 3),
            )

            # Compute relative geometry
            rel_pos = (chunk_centers.unsqueeze(2) - neighbor_centers_chunk) / 5.0
            rel_size = (chunk_sizes.unsqueeze(2) - neighbor_sizes_chunk) / 2.0
            rel_geom = torch.cat([rel_pos, rel_size], dim=-1)

            # Build pair input
            feat_i = chunk_features.unsqueeze(2).expand(B, chunk_size_actual, k, D)
            feat_j = neighbor_feats
            lang_exp = language_embedding.unsqueeze(1).unsqueeze(2).expand(B, chunk_size_actual, k, -1)

            pair_input = torch.cat([feat_i, feat_j, rel_geom, lang_exp], dim=-1)

            # Score and aggregate
            scores = self.relation_mlp(pair_input).squeeze(-1)  # [B, chunk, k]
            weights = F.softmax(scores, dim=2)

            context_chunk = torch.einsum('bck,bckd->bcd', weights, neighbor_feats)

            relation_context[:, chunk_start:chunk_end, :] = context_chunk
            relation_weights_all[:, chunk_start:chunk_end, :] = weights

        # Fuse
        if self.use_residual:
            enhanced = object_features + relation_context
        else:
            enhanced = relation_context

        return {
            "enhanced_features": enhanced,
            "relation_weights": relation_weights_all,
            "neighbor_indices": neighbor_indices,
        }