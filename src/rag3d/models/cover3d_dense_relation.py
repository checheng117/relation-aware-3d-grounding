"""COVER-3D Dense Relation Module: All-Pair Candidate-Anchor Relation Evidence.

This module computes dense pairwise relation scores across ALL candidate-anchor pairs,
using chunked processing to avoid O(N²) memory spikes.

Key design:
- Dense coverage: all N² pairs scored (no sparse top-k approximation)
- Chunked processing: memory-safe, O(N × chunk_size) peak memory
- Language-conditioned: relation scores depend on utterance semantics
- Geometric-aware: uses spatial features when available, embeddings otherwise

Architecture:
1. Build pairwise features: [obj_i, obj_j, relative_geom, language]
2. Score each pair via MLP conditioned on language
3. Chunk along j dimension to avoid N² allocation
4. Aggregate: softmax over all j → weighted sum → per-object relation score

Output:
- dense_relation_scores [B, N]: relation evidence per candidate object
- relation_evidence: intermediate tensors for diagnostics
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)


class DenseRelationModule(nn.Module):
    """Dense pairwise relation scoring with chunked computation.

    Computes relation evidence for all N² candidate-anchor pairs,
    but processes in chunks to avoid memory spikes.

    Memory: O(N × chunk_size) instead of O(N²)
    Coverage: Full N² (same as dense, no approximation)

    Parameters:
    - object_dim: dimension of object embeddings
    - language_dim: dimension of utterance features
    - geometry_dim: dimension of spatial features (default 6: 3 center + 3 size)
    - hidden_dim: MLP hidden dimension
    - mlp_layers: number of MLP layers
    - dropout: dropout rate
    - chunk_size: number of j objects per chunk (controls memory)
    - temperature: softmax temperature for aggregation
    - use_geometry: whether to use geometric features (fallback to embeddings if False)
    - aggregation: aggregation method ('weighted', 'max', 'hybrid', 'attention')
    - use_focal: whether to use focal weighting for hard-negative training
    - focal_gamma: focal weighting gamma (default 2.0)
    """

    def __init__(
        self,
        object_dim: int = 320,
        language_dim: int = 256,
        geometry_dim: int = 6,
        hidden_dim: int = 256,
        mlp_layers: int = 2,
        dropout: float = 0.1,
        chunk_size: int = 16,
        temperature: float = 1.0,
        use_geometry: bool = True,
        use_residual: bool = True,
        residual_scale: float = 0.1,
        aggregation: str = "weighted",  # 'weighted', 'max', 'hybrid', 'attention'
        use_focal: bool = False,
        focal_gamma: float = 2.0,
        use_attention: bool = False,
        attention_heads: int = 4,
        attention_hidden_dim: int = 128,
    ):
        super().__init__()

        self.object_dim = object_dim
        self.language_dim = language_dim
        self.geometry_dim = geometry_dim
        self.hidden_dim = hidden_dim
        self.chunk_size = chunk_size
        self.temperature = temperature
        self.use_geometry = use_geometry
        self.use_residual = use_residual
        self.residual_scale = residual_scale
        self.aggregation = aggregation
        self.use_focal = use_focal
        self.focal_gamma = focal_gamma
        self.use_attention = use_attention
        self.attention_hidden_dim = attention_hidden_dim

        # Pair input dimension
        # obj_i + obj_j + relative_geometry + language
        if use_geometry:
            pair_input_dim = 2 * object_dim + geometry_dim + language_dim
        else:
            # Fallback: use embedding difference as proxy
            pair_input_dim = 2 * object_dim + language_dim

        # Build relation score MLP with stable initialization
        mlp = []
        for layer_idx in range(mlp_layers):
            in_dim = pair_input_dim if layer_idx == 0 else hidden_dim
            out_dim = hidden_dim if layer_idx < mlp_layers - 1 else 1
            layer = nn.Linear(in_dim, out_dim)
            # Xavier initialization with small gain for stability
            nn.init.xavier_uniform_(layer.weight, gain=0.1)
            nn.init.zeros_(layer.bias)
            mlp.append(layer)
            if layer_idx < mlp_layers - 1:
                mlp.append(nn.GELU())  # Smoother than ReLU, better numerical stability
                mlp.append(nn.Dropout(dropout))

        # No final tanh - let scores be unbounded for better expressivity
        # mlp.append(nn.Tanh())  # REMOVED for v2

        self.relation_mlp = nn.Sequential(*mlp)

        # Language projection (for conditioning)
        self.language_proj = nn.Linear(language_dim, hidden_dim)

        # Object projection (for residual-style relation context)
        if use_residual:
            self.output_proj = nn.Linear(object_dim, object_dim)
        else:
            self.output_proj = None

        # Attention pooling (for 'attention' aggregation)
        if use_attention or aggregation == "attention":
            self.attention_query = nn.Parameter(torch.randn(attention_hidden_dim))
            # Project relation_context (object_dim) to attention space
            self.attention_proj = nn.Linear(object_dim, attention_hidden_dim)
            self.attention_key = nn.Linear(attention_hidden_dim, attention_hidden_dim)
            self.attention_value = nn.Linear(attention_hidden_dim, attention_hidden_dim)
            self.aggregation_mlp = nn.Linear(attention_hidden_dim, 1)
            log.info(f"DenseRelationModule: attention pooling enabled (heads={attention_heads}, attention_hidden_dim={attention_hidden_dim})")

        log.info(
            f"DenseRelationModule: object_dim={object_dim}, language_dim={language_dim}, "
            f"geometry_dim={geometry_dim}, chunk_size={chunk_size}, "
            f"pair_input_dim={pair_input_dim}, aggregation={aggregation}, "
            f"use_focal={use_focal}, focal_gamma={focal_gamma}"
        )

    def compute_pairwise_features(
        self,
        obj_i: torch.Tensor,  # [B, N, D]
        obj_j: torch.Tensor,  # [B, chunk_size, D]
        geom_i: Optional[torch.Tensor],  # [B, N, G]
        geom_j: Optional[torch.Tensor],  # [B, chunk_size, G]
        lang: torch.Tensor,  # [B, L] or [B, D]
    ) -> torch.Tensor:
        """Build pairwise feature tensor for scoring.

        Returns [B, N, chunk_size, pair_dim]
        """
        B, N, D = obj_i.shape
        chunk_size = obj_j.shape[1]

        # Expand for pairwise combination
        # obj_i: [B, N, D] → [B, N, chunk_size, D]
        obj_i_exp = obj_i.unsqueeze(2).expand(-1, -1, chunk_size, -1)

        # obj_j: [B, chunk_size, D] → [B, N, chunk_size, D]
        obj_j_exp = obj_j.unsqueeze(1).expand(-1, N, -1, -1)

        # Language: [B, L] → [B, N, chunk_size, L]
        if lang.dim() == 2:
            lang_exp = lang.unsqueeze(1).unsqueeze(2).expand(-1, N, chunk_size, -1)
        else:
            lang_exp = lang.unsqueeze(1).unsqueeze(2).expand(-1, N, chunk_size, -1)

        # Geometric features (if available)
        if self.use_geometry and geom_i is not None and geom_j is not None:
            # geom_i: [B, N, G] → [B, N, chunk_size, G]
            geom_i_exp = geom_i.unsqueeze(2).expand(-1, -1, chunk_size, -1)

            # geom_j: [B, chunk_size, G] → [B, N, chunk_size, G]
            geom_j_exp = geom_j.unsqueeze(1).expand(-1, N, -1, -1)

            # Relative geometry: diff + product (for size)
            # For centers: relative_position = center_j - center_i
            # For sizes: can use product or ratio
            relative_geom = geom_j_exp - geom_i_exp  # [B, N, chunk_size, G]

            # Concatenate all features
            pair_features = torch.cat([
                obj_i_exp,
                obj_j_exp,
                relative_geom,
                lang_exp,
            ], dim=-1)
        else:
            # Fallback: use embedding difference
            embedding_diff = obj_j_exp - obj_i_exp  # [B, N, chunk_size, D]

            pair_features = torch.cat([
                obj_i_exp,
                obj_j_exp,
                lang_exp,
            ], dim=-1)

        return pair_features

    def forward_chunk(
        self,
        obj_embeddings: torch.Tensor,  # [B, N, D]
        obj_geometry: Optional[torch.Tensor],  # [B, N, G]
        lang_features: torch.Tensor,  # [B, L]
        j_start: int,
        j_end: int,
    ) -> torch.Tensor:
        """Process one chunk of j objects.

        Returns relation scores for all i vs chunk_j: [B, N, chunk_size]
        """
        B, N, D = obj_embeddings.shape
        chunk_size = j_end - j_start

        # Extract chunk j
        obj_j = obj_embeddings[:, j_start:j_end, :]  # [B, chunk_size, D]

        if obj_geometry is not None:
            geom_j = obj_geometry[:, j_start:j_end, :]  # [B, chunk_size, G]
        else:
            geom_j = None

        # Build pairwise features
        pair_features = self.compute_pairwise_features(
            obj_i=obj_embeddings,
            obj_j=obj_j,
            geom_i=obj_geometry,
            geom_j=geom_j,
            lang=lang_features,
        )  # [B, N, chunk_size, pair_dim]

        # Score via MLP
        # [B, N, chunk_size, pair_dim] → [B, N, chunk_size, 1]
        scores = self.relation_mlp(pair_features)  # [B, N, chunk_size, 1]
        scores = scores.squeeze(-1)  # [B, N, chunk_size]

        return scores

    def forward(
        self,
        object_embeddings: torch.Tensor,  # [B, N, D]
        object_geometry: Optional[torch.Tensor] = None,  # [B, N, G]
        utterance_features: Optional[torch.Tensor] = None,  # [B, L]
        candidate_mask: Optional[torch.Tensor] = None,  # [B, N]
        **kwargs,
    ) -> Dict[str, Any]:
        """Forward pass: compute dense relation scores.

        Args:
            object_embeddings: [B, N, D] object features
            object_geometry: [B, N, G] spatial features (optional)
            utterance_features: [B, L] language features
            candidate_mask: [B, N] valid object mask

        Returns:
            dict with:
            - relation_scores: [B, N] per-object relation evidence
            - relation_context: [B, N, D] relation context vectors (optional)
            - diagnostics: dict with computation stats
        """
        B, N, D = object_embeddings.shape
        device = object_embeddings.device

        # CRITICAL: L2 normalize embeddings for numerical stability
        # Prevents large embedding magnitudes from causing MLP explosion
        object_embeddings_norm = F.normalize(object_embeddings, p=2, dim=-1)

        # Handle missing utterance features
        if utterance_features is None:
            # Use zeros as fallback (will produce neutral relation scores)
            utterance_features = torch.zeros(B, self.language_dim, device=device)
            log.warning("utterance_features is None, using zeros fallback")
        else:
            # Also normalize language features
            utterance_features = F.normalize(utterance_features, p=2, dim=-1)

        # Handle missing geometry
        if object_geometry is None and self.use_geometry:
            # Use zeros as fallback
            object_geometry = torch.zeros(B, N, self.geometry_dim, device=device)
            log.warning("object_geometry is None, using zeros fallback")

        # Determine chunk count
        chunk_size = min(self.chunk_size, N)
        num_chunks = (N + chunk_size - 1) // chunk_size

        # Process chunks and accumulate scores
        # We compute scores for all N×N pairs but in chunks
        all_pair_scores = torch.zeros(B, N, N, device=device)

        for chunk_idx in range(num_chunks):
            j_start = chunk_idx * chunk_size
            j_end = min(j_start + chunk_size, N)

            chunk_scores = self.forward_chunk(
                obj_embeddings=object_embeddings_norm,  # Use normalized embeddings
                obj_geometry=object_geometry,
                lang_features=utterance_features,  # Use normalized language features
                j_start=j_start,
                j_end=j_end,
            )  # [B, N, actual_chunk_size]

            # Store in full score tensor
            all_pair_scores[:, :, j_start:j_end] = chunk_scores

        # Apply candidate mask (mask invalid objects as low relation score)
        if candidate_mask is not None:
            # Mask: [B, N] → [B, N, N]
            mask_exp = candidate_mask.unsqueeze(1).expand(-1, N, -1)  # mask j
            mask_i = candidate_mask.unsqueeze(2).expand(-1, -1, N)  # mask i
            full_mask = mask_exp & mask_i
            all_pair_scores = all_pair_scores.masked_fill(~full_mask, float("-inf"))

        # Aggregate: for each candidate i, compute relation evidence from all j
        # Support multiple aggregation modes: 'weighted', 'max', 'hybrid', 'attention'

        if self.temperature != 1.0:
            scaled_scores = all_pair_scores / self.temperature
        else:
            scaled_scores = all_pair_scores

        # Compute softmax weights for weighted/attention aggregation
        # CRITICAL: Numerically stable softmax that handles all-masked rows
        max_scores = scaled_scores.max(dim=-1, keepdim=True).values  # [B, N, 1]
        all_masked_rows = (max_scores == float("-inf"))  # [B, N, 1]

        # Safe exponent: for masked rows, set exp_scores to 0 instead of NaN
        safe_diff = scaled_scores - max_scores.masked_fill(all_masked_rows, 0.0)
        exp_scores = torch.exp(safe_diff)  # [B, N, N]

        # Explicitly zero out fully masked rows
        exp_scores = exp_scores.masked_fill(all_masked_rows, 0.0)

        # Mask exp_scores for valid positions
        if candidate_mask is not None:
            mask_exp = candidate_mask.unsqueeze(1).expand(-1, N, -1)  # [B, N, N]
            exp_scores = exp_scores.masked_fill(~mask_exp, 0.0)

        # Sum and normalize
        sum_exp = exp_scores.sum(dim=-1, keepdim=True)  # [B, N, 1]
        pair_weights = exp_scores / (sum_exp.clamp(min=1e-8))  # [B, N, N]

        # Set weights to 0 for fully masked rows (where sum_exp was 0)
        zero_weight_rows = (sum_exp < 1e-8).squeeze(-1)  # [B, N]
        if zero_weight_rows.any():
            pair_weights[zero_weight_rows] = 0.0

        # Weighted sum of j embeddings for relation context
        relation_context = torch.bmm(
            pair_weights,  # [B, N, N]
            object_embeddings_norm,  # [B, N, D] normalized
        )  # [B, N, D]

        # Compute per-object relation score based on aggregation mode
        if self.aggregation == "max":
            # Max relation score: captures strongest relation signal
            relation_scores, _ = all_pair_scores.max(dim=-1)  # [B, N]

        elif self.aggregation == "weighted":
            # Weighted aggregation: smooth, robust to noise
            # CRITICAL: all_pair_scores contains -inf for masked positions
            # pair_weights is 0 for masked positions, but 0 * -inf = NaN
            safe_pair_scores = all_pair_scores.clone()
            safe_pair_scores[torch.isinf(all_pair_scores)] = 0.0
            relation_scores = (pair_weights * safe_pair_scores).sum(dim=-1)  # [B, N]

        elif self.aggregation == "hybrid":
            # Hybrid: combine max (strongest signal) and weighted (context)
            relation_scores_max, _ = all_pair_scores.max(dim=-1)  # [B, N]
            safe_pair_scores = all_pair_scores.clone()
            safe_pair_scores[torch.isinf(all_pair_scores)] = 0.0
            relation_scores_weighted = (pair_weights * safe_pair_scores).sum(dim=-1)  # [B, N]
            # Use max as primary, weighted as refinement
            relation_scores = relation_scores_max + 0.1 * relation_scores_weighted

        elif self.aggregation == "attention":
            # Attention pooling: learn to select relevant anchors for each candidate
            # For each candidate i, attend over all anchors j
            if hasattr(self, 'attention_query') and self.attention_query is not None:
                # Project relation_context to attention space
                relation_proj = self.attention_proj(relation_context)  # [B, N, attention_hidden_dim]

                # Self-attention style: each candidate queries all anchors
                # Q: from relation_proj, K: from relation_proj, V: from relation_proj
                query = self.attention_query.unsqueeze(0).expand(B, N, -1)  # [B, N, attention_hidden_dim]
                key = self.attention_key(relation_proj)  # [B, N, attention_hidden_dim]
                value = self.attention_value(relation_proj)  # [B, N, attention_hidden_dim]

                # Attention scores: for each i, score all j
                # We use pair_weights (from softmax over pair scores) as attention weights
                # This combines score-based attention with learned value aggregation

                # Compute learned value aggregation
                # attn_weights: [B, N, N] from pair_weights
                attn_output = torch.bmm(
                    pair_weights,  # [B, N, N] - weight each anchor j for candidate i
                    value,  # [B, N, attention_hidden_dim]
                )  # [B, N, attention_hidden_dim]

                # Combine with query to get per-candidate relation score
                # Use gating: gate = sigmoid(query + attn_output)
                gate = torch.sigmoid(query + attn_output)  # [B, N, attention_hidden_dim]

                # Project to scalar
                relation_scores = self.aggregation_mlp(gate * query).squeeze(-1)  # [B, N]

                # Add pair score signal
                safe_pair_scores = all_pair_scores.clone()
                safe_pair_scores[torch.isinf(all_pair_scores)] = 0.0
                relation_scores = relation_scores + 0.1 * (pair_weights * safe_pair_scores).sum(dim=-1)
            else:
                # Fallback to weighted aggregation if attention not initialized
                safe_pair_scores = all_pair_scores.clone()
                safe_pair_scores[torch.isinf(all_pair_scores)] = 0.0
                relation_scores = (pair_weights * safe_pair_scores).sum(dim=-1)  # [B, N]

        else:
            # Default to weighted aggregation
            safe_pair_scores = all_pair_scores.clone()
            safe_pair_scores[torch.isinf(all_pair_scores)] = 0.0
            relation_scores = (pair_weights * safe_pair_scores).sum(dim=-1)  # [B, N]

        # Apply mask to final scores
        if candidate_mask is not None:
            relation_scores = relation_scores.masked_fill(~candidate_mask, float("-inf"))

        # Optional residual addition (for compatibility with v3)
        if self.use_residual and self.output_proj is not None:
            enhanced_embeddings = object_embeddings + self.residual_scale * self.output_proj(relation_context)
        else:
            enhanced_embeddings = object_embeddings

        # Build diagnostics
        diagnostics = {
            "num_chunks": num_chunks,
            "chunk_size": chunk_size,
            "total_pairs": N * N,
            "peak_memory_pairs": N * chunk_size,
            "relation_score_mean": relation_scores.mean().item(),
            "relation_score_std": relation_scores.std().item(),
            "relation_score_min": relation_scores.min().item(),
            "relation_score_max": relation_scores.max().item(),
            "pair_weight_entropy": pair_weights.sum(dim=-1).entropy().mean().item() if hasattr(pair_weights, 'entropy') else -pair_weights.sum(dim=-1).log().mean().item(),
        }

        # Handle NaN/inf checks
        if torch.isnan(relation_scores).any():
            log.warning("NaN in relation_scores")
            diagnostics["has_nan"] = True
            relation_scores = torch.nan_to_num(relation_scores, nan=0.0)

        if torch.isinf(relation_scores).any():
            log.warning("Inf in relation_scores (likely from mask)")
            diagnostics["has_inf"] = True

        return {
            "relation_scores": relation_scores,  # [B, N]
            "relation_context": relation_context,  # [B, N, D]
            "enhanced_embeddings": enhanced_embeddings,  # [B, N, D]
            "pair_weights": pair_weights,  # [B, N, N] (for coverage analysis)
            "all_pair_scores": all_pair_scores,  # [B, N, N] (for coverage analysis)
            "diagnostics": diagnostics,
        }


class ChunkedDenseRelationModule(DenseRelationModule):
    """Alias for DenseRelationModule (backward compatibility)."""

    pass