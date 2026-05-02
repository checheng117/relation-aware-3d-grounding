"""Structured relation model for 3D grounding with explicit anchor selection."""
from typing import Any, Dict, Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from rag3d.datasets.schemas import ParsedUtterance
from rag3d.encoders.object_encoder import ObjectMLPEncoder
from rag3d.relation_reasoner.attribute_scorer import AttributeScorer
from rag3d.relation_reasoner.relation_scorer import PairwiseRelationScorer
from rag3d.relation_reasoner.text_encoding import TextHashEncoder, StructuredTextEncoder
from rag3d.relation_reasoner.soft_anchor_selector import SoftAnchorSelector, HierarchicalAnchorSelector
from rag3d.relation_reasoner.fallback_controller import FallbackController, FallbackDecision
from rag3d.parsers.structured_parser import StructuredParserInterface
from rag3d.parsers.parse_quality import validate_parse_quality


class StructuredRelationModel(nn.Module):
    """
    Structured relation model that separates anchor selection from relation scoring.
    Implements the full pipeline: objects + utterance -> structured parse -> anchor distribution -> target scores

    Phase 3 extension: supports fallback modes for raw-text scoring when parse quality is low.
    """

    def __init__(
        self,
        object_dim: int,
        language_dim: int,
        hidden_dim: int,
        relation_dim: int,
        anchor_temperature: float = 1.0,
        use_hierarchical_anchor: bool = False,
        dropout: float = 0.1,
        fallback_controller: Optional[FallbackController] = None,
    ):
        super().__init__()

        # Object encoding
        self.object_enc = ObjectMLPEncoder(object_dim, hidden_dim, dropout)

        # Text encoding for structured parsing
        self.text_enc = TextHashEncoder(dim=language_dim)
        self.struct_text_enc = StructuredTextEncoder(dim=language_dim)

        # Attribute scoring component
        self.attr = AttributeScorer(hidden_dim, language_dim)

        # Relation scoring component
        self.rel = PairwiseRelationScorer(hidden_dim, relation_dim, language_dim)

        # Soft anchor selector component
        if use_hierarchical_anchor:
            self.anchor_selector = HierarchicalAnchorSelector(
                object_dim, language_dim, hidden_dim, anchor_temperature
            )
        else:
            self.anchor_selector = SoftAnchorSelector(
                object_dim, language_dim, hidden_dim, anchor_temperature
            )

        # Gate for balancing attribute and relation components
        self.gate = nn.Linear(language_dim, 1)

        # Temperature for anchor selection
        self.anchor_temperature = anchor_temperature

        # Phase 3: Fallback controller
        self.fallback_controller = fallback_controller

    def forward(
        self,
        batch: Dict[str, Any],
        parsed_list: Optional[List[ParsedUtterance]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with structured reasoning and optional fallback.

        Args:
            batch: Batch data with object_features, object_mask, raw_texts
            parsed_list: Optional pre-parsed utterances

        Returns:
            Dictionary containing:
            - logits: Final scores for each object [B, N]
            - anchor_dist: Anchor distribution [B, N]
            - anchor_entropy: Anchor selection entropy [B]
            - top_anchor_id: Most probable anchor index [B]
            - s_attr: Attribute-only scores [B, N]
            - s_rel: Relation scores [B, N]
            - s_structured: Structured scores (attr + rel) [B, N]
            - s_raw_text: Raw-text relation scores [B, N]
            - fallback_decisions: List of FallbackDecision per sample
            - structured_weights: Tensor of structured weights [B]
            - raw_text_weights: Tensor of raw-text weights [B]
        """
        obj = batch["object_features"]  # [B, N, object_dim]
        mask = batch["object_mask"]     # [B, N]
        texts = batch["raw_texts"]      # List of strings

        batch_size, num_objects = obj.shape[:2]
        device = obj.device

        # Encode object features
        h_o = self.object_enc(obj)  # [B, N, hidden_dim]

        # Encode language features (raw text embedding)
        h_t = self.text_enc(texts)  # [B, language_dim]

        # Compute attribute scores (target-like attributes)
        s_attr = self.attr(h_o, h_t)  # [B, N]

        # Compute raw-text relation scores (uniform anchor for fallback)
        uniform_anchor = mask.float() / (mask.float().sum(dim=1, keepdim=True).clamp_min(1.0))
        r_ij_raw = self.rel(h_o, h_t)  # [B, N, N]
        s_raw_text = s_attr + torch.einsum("bj,bij->bi", uniform_anchor, r_ij_raw) * torch.sigmoid(self.gate(h_t))

        # Compute structured scores
        if parsed_list is not None and len(parsed_list) > 0:
            # Use structured information from parsed utterances
            anchor_query = self._get_anchor_query_from_parsed(parsed_list, h_t)
        else:
            # Fall back to using the raw language embedding as anchor query
            anchor_query = h_t  # [B, language_dim]

        # Compute anchor distribution using soft anchor selector
        anchor_results = self.anchor_selector(h_o, anchor_query, mask)
        p_anchor = anchor_results['anchor_distribution']  # [B, N]

        # Compute relation scores using anchor distribution
        s_rel = torch.einsum("bj,bij->bi", p_anchor, r_ij_raw)  # [B, N]

        # Combine attribute and relation scores with gating
        g = torch.sigmoid(self.gate(h_t))  # [B, 1]
        s_structured = s_attr + g * s_rel  # [B, N]

        # Apply fallback blending if controller is present
        fallback_decisions: List[FallbackDecision] = []
        structured_weights = torch.ones(batch_size, device=device)
        raw_text_weights = torch.zeros(batch_size, device=device)

        if self.fallback_controller is not None and parsed_list is not None:
            # Make fallback decisions for each sample
            fallback_decisions = self.fallback_controller.decide_batch(parsed_list)

            # Extract weights as tensors
            for i, dec in enumerate(fallback_decisions):
                structured_weights[i] = dec.structured_weight
                raw_text_weights[i] = dec.raw_text_weight

        # Blend structured and raw-text scores
        logits = structured_weights.unsqueeze(1) * s_structured + raw_text_weights.unsqueeze(1) * s_raw_text

        # Apply mask to invalid objects
        logits = logits.masked_fill(~mask, float("-inf"))

        # Return results dictionary
        return {
            'logits': logits,
            'anchor_dist': p_anchor,
            'anchor_entropy': anchor_results['anchor_entropy'],
            'top_anchor_id': anchor_results['top_anchor_id'],
            's_attr': s_attr,
            's_rel': s_rel,
            's_structured': s_structured,
            's_raw_text': s_raw_text,
            'anchor_scores': anchor_results['anchor_scores'],
            'raw_anchor_scores': anchor_results['raw_scores'],
            'fallback_decisions': fallback_decisions,
            'structured_weights': structured_weights,
            'raw_text_weights': raw_text_weights,
        }

    def _get_anchor_query_from_parsed(
        self,
        parsed_list: List[ParsedUtterance],
        fallback_lang_emb: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate anchor query from parsed utterances.
        """
        b = len(parsed_list)
        lang_dim = fallback_lang_emb.shape[-1]
        device = fallback_lang_emb.device

        # For now, fall back to the original language embedding if parsing not available
        # In the future, we could use structured information from parsed_list
        anchor_queries = []

        for i in range(b):
            if i < len(parsed_list) and parsed_list[i] is not None:
                # Use structured parsing info to create anchor query
                # For now, just return the language embedding
                anchor_queries.append(fallback_lang_emb[i])
            else:
                anchor_queries.append(fallback_lang_emb[i])

        return torch.stack(anchor_queries, dim=0)  # [B, language_dim]

    def get_anchor_selection_info(
        self,
        batch: Dict[str, Any],
        parsed_list: Optional[List[ParsedUtterance]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Get detailed anchor selection information without full forward pass.
        """
        obj = batch["object_features"]
        mask = batch["object_mask"]
        texts = batch["raw_texts"]

        h_o = self.object_enc(obj)

        # Get language embedding
        h_t = self.text_enc(texts)

        # Get anchor query
        if parsed_list is not None and len(parsed_list) > 0:
            anchor_query = self._get_anchor_query_from_parsed(parsed_list, h_t)
        else:
            anchor_query = h_t

        # Compute anchor distribution
        anchor_results = self.anchor_selector(h_o, anchor_query, mask)

        return anchor_results


def compute_anchor_quality_metrics(anchor_entropy: torch.Tensor, anchor_distribution: torch.Tensor) -> Dict[str, float]:
    """
    Compute quality metrics for anchor selection.

    Args:
        anchor_entropy: Entropy of anchor distributions [B]
        anchor_distribution: Anchor probability distributions [B, N]

    Returns:
        Dictionary with anchor quality metrics
    """
    with torch.no_grad():
        metrics = {
            'avg_anchor_entropy': float(anchor_entropy.mean().item()),
            'std_anchor_entropy': float(anchor_entropy.std().item()),
            'max_anchor_prob': float(anchor_distribution.max(dim=1)[0].mean().item()),  # Avg max prob across batches
            'top1_certainty': float((anchor_distribution.max(dim=1)[0] > 0.8).float().mean().item()),  # Fraction with high confidence
        }
    return metrics