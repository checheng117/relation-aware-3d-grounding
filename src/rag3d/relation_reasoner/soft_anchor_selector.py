"""Soft anchor selector for structured 3D grounding."""
from typing import Optional, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from rag3d.relation_reasoner.anchor_selector import soft_anchor_distribution


class SoftAnchorSelector(nn.Module):
    """Module for selecting soft anchor distributions in structured reasoning."""

    def __init__(self, object_dim: int, language_dim: int, hidden_dim: int, temperature: float = 1.0):
        super().__init__()
        self.temperature = temperature

        # Projection layers for combining object and language features
        self.anchor_query_proj = nn.Linear(language_dim, hidden_dim)
        self.object_proj = nn.Linear(object_dim, hidden_dim)

        # MLP for final anchor scoring
        self.anchor_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )

    def forward(
        self,
        object_features: torch.Tensor,  # [B, N, object_dim]
        anchor_query: torch.Tensor,    # [B, language_dim]
        mask: torch.Tensor             # [B, N] boolean mask
    ) -> Dict[str, Any]:
        """
        Compute soft anchor distribution for structured reasoning.

        Args:
            object_features: Object features [B, N, object_dim]
            anchor_query: Language features for anchor query [B, language_dim]
            mask: Boolean mask indicating valid objects [B, N]

        Returns:
            Dictionary containing:
            - anchor_distribution: Soft probability distribution over objects [B, N]
            - top_anchor_id: Index of highest-probability anchor for each batch [B]
            - anchor_entropy: Entropy of anchor distribution [B]
            - anchor_scores: Raw anchor scores before softmax [B, N]
        """
        batch_size, num_objects, object_dim = object_features.shape

        # Project object features
        projected_objects = self.object_proj(object_features)  # [B, N, hidden_dim]

        # Project anchor query
        projected_query = self.anchor_query_proj(anchor_query)  # [B, hidden_dim]
        expanded_query = projected_query.unsqueeze(1).expand(-1, num_objects, -1)  # [B, N, hidden_dim]

        # Combine object and query features
        combined_features = torch.cat([projected_objects, expanded_query], dim=-1)  # [B, N, 2*hidden_dim]

        # Score each object as potential anchor
        raw_scores = self.anchor_scorer(combined_features).squeeze(-1)  # [B, N]

        # Apply mask to invalid objects
        masked_scores = raw_scores.masked_fill(~mask, float("-inf"))

        # Compute soft anchor distribution using softmax
        anchor_distribution = F.softmax(masked_scores / self.temperature, dim=-1)

        # Calculate entropy of the anchor distribution (measure of uncertainty)
        epsilon = 1e-8
        entropy = -torch.sum(anchor_distribution * torch.log(anchor_distribution + epsilon), dim=-1)

        # Get top anchor IDs (most probable anchors)
        top_anchor_ids = torch.argmax(anchor_distribution, dim=-1)

        return {
            'anchor_distribution': anchor_distribution,
            'top_anchor_id': top_anchor_ids,
            'anchor_entropy': entropy,
            'anchor_scores': masked_scores,
            'raw_scores': raw_scores  # Unmasked scores before applying mask
        }

    def compute_anchor_confidence(self, anchor_entropy: torch.Tensor, max_entropy: float = None) -> torch.Tensor:
        """
        Compute confidence in anchor selection based on entropy.

        Args:
            anchor_entropy: Entropy of anchor distributions [B]
            max_entropy: Maximum possible entropy (log(num_objects)) if known

        Returns:
            Confidence scores between 0 and 1 [B]
        """
        if max_entropy is None:
            # Calculate max entropy as log(N) for each batch
            max_entropy = torch.log(torch.tensor(anchor_entropy.size(0), dtype=torch.float))

        # Confidence is inversely related to entropy (higher entropy = lower confidence)
        # Normalize to [0, 1] range
        confidence = 1.0 - (anchor_entropy / max_entropy.clamp(min=1.0))
        return torch.clamp(confidence, 0.0, 1.0)


class HierarchicalAnchorSelector(nn.Module):
    """Extended anchor selector with hierarchical processing capabilities."""

    def __init__(
        self,
        object_dim: int,
        language_dim: int,
        hidden_dim: int,
        temperature: float = 1.0,
        use_attention: bool = True
    ):
        super().__init__()
        self.use_attention = use_attention
        self.temperature = temperature

        if use_attention:
            # Multi-head attention for more sophisticated anchor selection
            self.attention = nn.MultiheadAttention(
                embed_dim=object_dim,
                num_heads=min(8, object_dim // 32),  # At least 32 dims per head
                batch_first=True
            )
            self.query_proj = nn.Linear(language_dim, object_dim)

        # Fallback to the basic soft anchor selector
        self.basic_selector = SoftAnchorSelector(object_dim, language_dim, hidden_dim, temperature)

    def forward(
        self,
        object_features: torch.Tensor,  # [B, N, object_dim]
        anchor_query: torch.Tensor,    # [B, language_dim]
        mask: torch.Tensor             # [B, N] boolean mask
    ) -> Dict[str, Any]:
        """
        Forward pass with hierarchical anchor selection.

        Args:
            object_features: Object features [B, N, object_dim]
            anchor_query: Language features for anchor query [B, language_dim]
            mask: Boolean mask indicating valid objects [B, N]

        Returns:
            Dictionary containing anchor selection results
        """
        if self.use_attention:
            # Use attention mechanism to select anchors
            query = self.query_proj(anchor_query).unsqueeze(1)  # [B, 1, object_dim]

            # Apply attention between query and object features
            attn_output, attn_weights = self.attention(
                query, object_features, object_features,
                key_padding_mask=~mask
            )

            # Use attention weights as anchor distribution
            anchor_scores = attn_weights.squeeze(1)  # [B, N]

            # Apply temperature scaling and masking
            masked_scores = anchor_scores.masked_fill(~mask, float("-inf"))
            anchor_distribution = F.softmax(masked_scores / self.temperature, dim=-1)

            # Calculate entropy
            epsilon = 1e-8
            entropy = -torch.sum(anchor_distribution * torch.log(anchor_distribution + epsilon), dim=-1)

            # Get top anchor IDs
            top_anchor_ids = torch.argmax(anchor_distribution, dim=-1)

            return {
                'anchor_distribution': anchor_distribution,
                'top_anchor_id': top_anchor_ids,
                'anchor_entropy': entropy,
                'anchor_scores': masked_scores,
                'attention_weights': attn_weights,
                'raw_scores': anchor_scores
            }
        else:
            # Fall back to basic selector
            return self.basic_selector(object_features, anchor_query, mask)