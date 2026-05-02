"""Relation-Aware ReferIt3DNet v2 - BERT Span-Grounded Semantics.

This is the second iteration of the relation-aware model that addresses
the key failure mode of v1: random embeddings for parsed words.

v1 Failure:
- StructuredTextEncoder used TextHashEncoder with random embeddings
- Target/anchor/relation vectors had no semantic grounding
- Gate α stayed low because relation branch signal was weak/noisy

v2 Fix:
- SpanTextEncoder uses BERT span embeddings from the original utterance
- Target/anchor/relation vectors are semantically grounded
- Gate α should learn to trust the relation branch more

Key Changes from v1:
- Replace StructuredTextEncoder with SpanTextEncoder
- Parser outputs aligned to token spans in utterance
- BERT token embeddings extracted for each span
- Span-found rates tracked as diagnostics

Preserved from v1:
- Base model: ReferIt3DNet (unchanged)
- Parser: HeuristicParser (unchanged)
- Object encoder: ObjectMLPEncoder
- Relation scorer: PairwiseRelationScorer
- Anchor selection: soft_anchor_distribution
- Fusion: learned gate α
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

# Add repro path for ReferIt3DNet import
_REPRO_PATH = Path(__file__).resolve().parents[3] / "repro" / "referit3d_baseline" / "src"
if str(_REPRO_PATH) not in sys.path:
    sys.path.insert(0, str(_REPRO_PATH))

# Import ReferIt3DNet from repro
try:
    from referit3d_net import ReferIt3DNet
except ImportError:
    ReferIt3DNet = None

# Import existing components
from rag3d.parsers.heuristic_parser import HeuristicParser
from rag3d.parsers.span_alignment import (
    align_batch_utterances,
    get_span_found_rates,
    UtteranceSpanAlignment,
)
from rag3d.datasets.schemas import ParsedUtterance
from rag3d.encoders.object_encoder import ObjectMLPEncoder

# Import v2 text encoder (BERT span-grounded)
from rag3d.models.span_text_encoder import SpanTextEncoder

# Import relation reasoning components
from rag3d.relation_reasoner.anchor_selector import soft_anchor_distribution
from rag3d.relation_reasoner.relation_scorer import PairwiseRelationScorer

log = logging.getLogger(__name__)


class RelationAwareReferIt3DNetV2(nn.Module):
    """ReferIt3DNet with BERT Span-Grounded Relation-Aware scoring.

    Architecture:
    1. Base: ReferIt3DNet (trusted baseline, 30.79% Test Acc@1)
    2. Parser: HeuristicParser (extracts target/anchor/relation)
    3. Span Alignment: Maps parsed text to token spans in utterance
    4. SpanTextEncoder: Extracts BERT span embeddings (NOT random!)
    5. Relation scorer: Computes pairwise relation scores
    6. Fusion: Learned gate α combining base + relation scores

    Key improvement over v1:
    - v1: random embeddings for parsed words (TextHashEncoder)
    - v2: BERT span embeddings (SpanTextEncoder)
    """

    def __init__(
        self,
        # Base model configuration
        base_model: Optional[nn.Module] = None,
        point_input_dim: int = 256,
        point_hidden_dim: int = 128,
        point_output_dim: int = 256,
        lang_input_dim: int = 768,
        lang_hidden_dim: int = 256,
        lang_output_dim: int = 256,
        fusion_dim: int = 512,
        dropout: float = 0.1,
        encoder_type: str = "simple_point",
        use_learned_class_embedding: bool = False,
        num_object_classes: int = 516,
        class_embed_dim: int = 64,
        # Relation branch configuration
        relation_hidden_dim: int = 256,
        relation_dim: int = 64,
        anchor_temperature: float = 1.0,
        # Text encoder configuration (v2 specific)
        bert_model_name: str = "distilbert-base-uncased",
        freeze_bert: bool = True,
        # Fusion configuration
        gate_dim: int = 256,
        initial_gate_bias: float = -1.0,  # sigmoid(-1) ≈ 0.27
    ):
        super().__init__()

        if ReferIt3DNet is None:
            raise ImportError("ReferIt3DNet not available. Ensure repro/referit3d_baseline/src is in path.")

        # Store dimensions
        self.lang_input_dim = lang_input_dim
        self.lang_output_dim = lang_output_dim

        # Build or use provided base model
        if base_model is not None:
            self.base_model = base_model
        else:
            self.base_model = ReferIt3DNet(
                point_input_dim=point_input_dim,
                point_hidden_dim=point_hidden_dim,
                point_output_dim=point_output_dim,
                lang_input_dim=lang_input_dim,
                lang_hidden_dim=lang_hidden_dim,
                lang_output_dim=lang_output_dim,
                fusion_dim=fusion_dim,
                dropout=dropout,
                encoder_type=encoder_type,
                use_learned_class_embedding=use_learned_class_embedding,
                num_object_classes=num_object_classes,
                class_embed_dim=class_embed_dim,
            )

        self.encoder_type = encoder_type

        # Parser for preprocessing
        self.parser = HeuristicParser()

        # === v2 KEY CHANGE: BERT Span Text Encoder ===
        self.span_text_encoder = SpanTextEncoder(
            bert_model_name=bert_model_name,
            output_dim=lang_output_dim,
            freeze_bert=freeze_bert,
        )
        log.info(f"Using SpanTextEncoder (BERT: {bert_model_name})")

        # Object encoder for relation branch
        self.object_encoder = ObjectMLPEncoder(
            in_dim=point_output_dim,
            hidden_dim=relation_hidden_dim,
            dropout=dropout,
        )

        # Pairwise relation scorer
        self.relation_scorer = PairwiseRelationScorer(
            hidden_dim=relation_hidden_dim,
            rel_dim=relation_dim,
            lang_dim=lang_output_dim,
        )

        # Anchor temperature
        self.anchor_temperature = anchor_temperature

        # Fusion gate
        self.gate = nn.Linear(self.lang_input_dim, 1)
        self.gate.bias.data.fill_(initial_gate_bias)
        log.info(f"Gate initialized with bias={initial_gate_bias}, sigmoid ≈ {torch.sigmoid(torch.tensor(initial_gate_bias)).item():.3f}")

        # Diagnostic accumulators
        self._alpha_history = []
        self._span_found_history = []

    def parse_utterances(
        self,
        utterances: List[str],
    ) -> List[ParsedUtterance]:
        """Parse utterances using HeuristicParser."""
        return [self.parser.parse(utt) for utt in utterances]

    def forward(
        self,
        # Base model inputs
        points: torch.Tensor,
        object_mask: torch.Tensor,
        text_features: Optional[torch.Tensor] = None,
        class_features: Optional[torch.Tensor] = None,
        class_indices: Optional[torch.Tensor] = None,
        # Relation branch inputs
        utterances: Optional[List[str]] = None,
        parsed_list: Optional[List[ParsedUtterance]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass combining base and relation scores.

        Args:
            points: [B, N, ...] object point features
            object_mask: [B, N] boolean mask
            text_features: [B, D] pre-computed text features (BERT CLS)
            class_features: [B, N, 250] class semantic features
            class_indices: [B, N] class indices
            utterances: list of raw utterance strings (REQUIRED for v2)
            parsed_list: pre-parsed ParsedUtterance objects

        Returns:
            Dict containing logits, base_logits, relation_logits, gate_alpha,
            span_masks, anchor_dist, and diagnostic info
        """
        B, N = points.shape[:2]
        device = points.device

        # === BASE MODEL FORWARD ===
        base_output = self.base_model(
            points=points,
            object_mask=object_mask,
            text_features=text_features,
            class_features=class_features,
            class_indices=class_indices,
        )
        base_logits = base_output["logits"]  # [B, N]

        # === RELATION BRANCH (v2: BERT Span-Grounded) ===
        relation_logits = torch.zeros(B, N, device=device, dtype=points.dtype)
        anchor_dist = torch.zeros(B, N, device=device, dtype=points.dtype)
        gate_alpha = torch.zeros(B, 1, device=device, dtype=points.dtype)
        span_masks = torch.zeros(B, 3, device=device, dtype=torch.bool)

        # Parse utterances if not provided
        if parsed_list is None and utterances is not None:
            parsed_list = self.parse_utterances(utterances)

        if parsed_list is not None and utterances is not None and len(parsed_list) > 0:
            # Get object features
            if points.dim() == 3:
                obj_features = points
            else:
                obj_features = base_output["obj_features"]

            # Encode objects
            h_o = self.object_encoder(obj_features)  # [B, N, hidden_dim]

            # === v2 KEY: Extract BERT span embeddings ===
            # This replaces the random embeddings from v1's StructuredTextEncoder
            q_t, q_a, q_r, span_masks = self.span_text_encoder.forward(
                utterances, parsed_list
            )
            # q_t, q_a, q_r: [B, output_dim] - BERT-grounded embeddings!
            # span_masks: [B, 3] - found status for each component

            # Soft anchor distribution
            anchor_dist = soft_anchor_distribution(
                h_o, q_a, object_mask, temperature=self.anchor_temperature
            )

            # Pairwise relation scoring
            R_ij = self.relation_scorer(h_o, q_r)  # [B, N, N]

            # Aggregate
            relation_logits = torch.einsum("bj,bij->bi", anchor_dist, R_ij)  # [B, N]

            # === FUSION ===
            if text_features is not None:
                gate_input = text_features
            else:
                gate_input = base_output["lang_features"]

            gate_alpha = torch.sigmoid(self.gate(gate_input))  # [B, 1]

            # Track diagnostics
            self._alpha_history.append(gate_alpha.mean().item())
            span_found_batch = span_masks.float().mean(dim=0)  # [3] - rates per component
            self._span_found_history.append(span_found_batch.tolist())

        # Combine base and relation scores
        logits = base_logits + gate_alpha * relation_logits

        # Apply mask
        logits = logits.masked_fill(~object_mask, float("-inf"))

        return {
            "logits": logits,
            "base_logits": base_logits,
            "relation_logits": relation_logits,
            "gate_alpha": gate_alpha,
            "span_masks": span_masks,
            "anchor_dist": anchor_dist,
            "obj_features": base_output.get("obj_features"),
            "lang_features": base_output.get("lang_features"),
        }

    def load_base_weights(self, checkpoint_path: str) -> None:
        """Load weights from pre-trained ReferIt3DNet checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        self.base_model.load_state_dict(state_dict, strict=False)
        log.info(f"Loaded base weights from {checkpoint_path}")

    def get_diagnostic_stats(self) -> Dict[str, Any]:
        """Get accumulated diagnostic statistics.

        Returns:
            Dict with alpha statistics, span-found rates, etc.
        """
        if not self._alpha_history:
            return {}

        alphas = self._alpha_history
        span_rates = self._span_found_history

        # Compute statistics
        alpha_mean = sum(alphas) / len(alphas)
        alpha_min = min(alphas)
        alpha_max = max(alphas)

        # Span found rates
        if span_rates:
            target_rates = [r[0] for r in span_rates]
            anchor_rates = [r[1] for r in span_rates]
            relation_rates = [r[2] for r in span_rates]

            span_stats = {
                "target_found_rate": sum(target_rates) / len(target_rates),
                "anchor_found_rate": sum(anchor_rates) / len(anchor_rates),
                "relation_found_rate": sum(relation_rates) / len(relation_rates),
                "all_found_rate": sum(1 for r in span_rates if all(v > 0.9 for v in r)) / len(span_rates),
            }
        else:
            span_stats = {}

        return {
            "alpha_mean": alpha_mean,
            "alpha_min": alpha_min,
            "alpha_max": alpha_max,
            "alpha_count": len(alphas),
            "span_stats": span_stats,
        }

    def reset_diagnostics(self) -> None:
        """Reset diagnostic accumulators."""
        self._alpha_history = []
        self._span_found_history = []


def build_relation_aware_v2_model(config: Dict[str, Any]) -> RelationAwareReferIt3DNetV2:
    """Build RelationAwareReferIt3DNetV2 from config.

    Args:
        config: Configuration dict

    Returns:
        RelationAwareReferIt3DNetV2 model
    """
    model_config = config.get("model", {})
    relation_config = config.get("relation_branch", {})
    text_config = config.get("text_encoder", {})
    fusion_config = config.get("fusion", {})

    base_checkpoint = model_config.get("base_checkpoint", None)

    model = RelationAwareReferIt3DNetV2(
        # Base model params
        point_input_dim=model_config.get("point_input_dim", 256),
        point_hidden_dim=model_config.get("point_hidden_dim", 128),
        point_output_dim=model_config.get("point_output_dim", 256),
        lang_input_dim=model_config.get("lang_input_dim", 768),
        lang_hidden_dim=model_config.get("lang_hidden_dim", 256),
        lang_output_dim=model_config.get("lang_output_dim", 256),
        fusion_dim=model_config.get("fusion_dim", 512),
        dropout=model_config.get("dropout", 0.1),
        encoder_type=model_config.get("encoder_type", "simple_point"),
        use_learned_class_embedding=model_config.get("use_learned_class_embedding", False),
        num_object_classes=model_config.get("num_classes", 516),
        class_embed_dim=model_config.get("class_embed_dim", 64),
        # Relation branch params
        relation_hidden_dim=relation_config.get("hidden_dim", 256),
        relation_dim=relation_config.get("relation_dim", 64),
        anchor_temperature=relation_config.get("anchor_temperature", 1.0),
        # Text encoder params (v2 specific)
        bert_model_name=text_config.get("bert_model_name", "distilbert-base-uncased"),
        freeze_bert=text_config.get("freeze_bert", True),
        # Fusion params
        gate_dim=fusion_config.get("gate_dim", 256),
        initial_gate_bias=fusion_config.get("initial_gate_bias", -1.0),
    )

    if base_checkpoint:
        model.load_base_weights(base_checkpoint)
        log.info(f"Loaded pre-trained base from {base_checkpoint}")

    return model