"""Relation-Aware ReferIt3DNet - Baseline with relation scoring branch.

This model extends the trusted ReferIt3DNet baseline with a lightweight
relation-aware scoring branch that uses structured language parsing.

Architecture:
1. Base: ReferIt3DNet (unchanged, optionally pre-trained)
2. Parser: HeuristicParser (extracts target/anchor/relation)
3. Relation branch:
   - StructuredTextEncoder for parsed components
   - ObjectMLPEncoder for object features
   - soft_anchor_distribution for anchor selection
   - PairwiseRelationScorer for pairwise relation scoring
4. Fusion: learned gate combining base + relation scores
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add repro path for ReferIt3DNet import
_REPRO_PATH = Path(__file__).resolve().parents[3] / "repro" / "referit3d_baseline" / "src"
if str(_REPRO_PATH) not in sys.path:
    sys.path.insert(0, str(_REPRO_PATH))

# Import ReferIt3DNet from repro
try:
    from referit3d_net import ReferIt3DNet
except ImportError:
    # Fallback: define a placeholder for cases where ReferIt3DNet is not available
    ReferIt3DNet = None

# Import from existing components
from rag3d.parsers.heuristic_parser import HeuristicParser
from rag3d.datasets.schemas import ParsedUtterance
from rag3d.encoders.object_encoder import ObjectMLPEncoder
from rag3d.relation_reasoner.text_encoding import StructuredTextEncoder
from rag3d.relation_reasoner.anchor_selector import soft_anchor_distribution
from rag3d.relation_reasoner.relation_scorer import PairwiseRelationScorer

log = logging.getLogger(__name__)


class RelationAwareReferIt3DNet(nn.Module):
    """ReferIt3DNet with relation-aware scoring branch.

    This model:
    1. Uses ReferIt3DNet as the base scorer (trusted baseline)
    2. Parses utterances to extract target/anchor/relation structure
    3. Computes relation scores between candidate-anchor pairs
    4. Fuses base and relation scores with a learned gate

    The key insight is that relational queries like "the chair next to the table"
    require reasoning about:
    - Target: chair (what we want)
    - Anchor: table (reference object)
    - Relation: next-to (spatial relationship)

    The base model treats this as a single embedding. We add explicit
    structured reasoning on top.
    """

    def __init__(
        self,
        # Base model configuration
        base_model: Optional[nn.Module] = None,
        # If base_model is None, build from these params
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
        # Fusion configuration
        gate_dim: int = 256,
        initial_gate_bias: float = -1.0,  # sigmoid(-1) ≈ 0.27, starts with low relation weight
    ):
        super().__init__()

        if ReferIt3DNet is None:
            raise ImportError("ReferIt3DNet not available. Ensure repro/referit3d_baseline/src is in path.")

        # Store dimensions for later use
        self.lang_input_dim = lang_input_dim

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
        self.lang_output_dim = lang_output_dim

        # Parser (for preprocessing, not part of forward)
        self.parser = HeuristicParser()

        # Relation branch components
        # Object encoder for relation branch (processes object features)
        self.object_encoder = ObjectMLPEncoder(
            in_dim=point_output_dim,
            hidden_dim=relation_hidden_dim,
            dropout=dropout,
        )

        # Structured text encoder for parsed components
        self.structured_text_encoder = StructuredTextEncoder(
            dim=lang_output_dim,
        )

        # Pairwise relation scorer
        self.relation_scorer = PairwiseRelationScorer(
            hidden_dim=relation_hidden_dim,
            rel_dim=relation_dim,
            lang_dim=lang_output_dim,
        )

        # Anchor temperature
        self.anchor_temperature = anchor_temperature

        # Fusion gate: learned weight for combining base + relation
        # Gate takes BERT language features (768-dim) and outputs scalar α
        self.gate = nn.Linear(self.lang_input_dim, 1)
        # Initialize bias to start with low relation weight
        self.gate.bias.data.fill_(initial_gate_bias)
        log.info(f"Gate initialized with bias={initial_gate_bias}, sigmoid ≈ {torch.sigmoid(torch.tensor(initial_gate_bias)).item():.3f}")

    def parse_utterances(
        self,
        utterances: List[str],
    ) -> List[ParsedUtterance]:
        """Parse utterances using HeuristicParser.

        Args:
            utterances: list of raw text strings

        Returns:
            List of ParsedUtterance objects
        """
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
            object_mask: [B, N] boolean mask for valid objects
            text_features: [B, D] pre-computed text features (BERT)
            class_features: [B, N, 250] class semantic features
            class_indices: [B, N] class indices for learned embedding
            utterances: list of raw utterance strings (for parsing)
            parsed_list: pre-parsed ParsedUtterance objects (optional)

        Returns:
            Dict containing:
            - logits: [B, N] final scores
            - base_logits: [B, N] base model scores
            - relation_logits: [B, N] relation branch scores
            - gate_alpha: [B, 1] fusion gate values
            - anchor_dist: [B, N] anchor distributions
        """
        B, N = points.shape[:2]

        # === BASE MODEL FORWARD ===
        base_output = self.base_model(
            points=points,
            object_mask=object_mask,
            text_features=text_features,
            class_features=class_features,
            class_indices=class_indices,
        )
        base_logits = base_output["logits"]  # [B, N]

        # === RELATION BRANCH ===
        # Only compute relation scores if we have parsed utterances
        relation_logits = torch.zeros(B, N, device=points.device, dtype=points.dtype)
        anchor_dist = torch.zeros(B, N, device=points.device, dtype=points.dtype)
        gate_alpha = torch.zeros(B, 1, device=points.device, dtype=points.dtype)

        # Parse utterances if not provided
        if parsed_list is None and utterances is not None:
            parsed_list = self.parse_utterances(utterances)

        if parsed_list is not None and len(parsed_list) > 0:
            # Get object features for relation branch
            # For SimplePointEncoder mode, points are already features [B, N, D]
            if points.dim() == 3:
                obj_features = points  # [B, N, D]
            else:
                # For PointNet++ mode, points are raw XYZ [B, N, P, 3]
                # Use base model's point encoder output
                obj_features = base_output["obj_features"]  # [B, N, D]

            # Encode objects for relation branch
            h_o = self.object_encoder(obj_features)  # [B, N, hidden_dim]

            # Extract parsed components
            target_heads = []
            anchor_heads = []
            rel_types = []
            for i in range(B):
                if i < len(parsed_list) and parsed_list[i] is not None:
                    p = parsed_list[i]
                    target_heads.append(p.target_head or "object")
                    anchor_heads.append(p.anchor_head or "object")
                    rel_types.append(" ".join(p.relation_types) if p.relation_types else "none")
                else:
                    target_heads.append("object")
                    anchor_heads.append("object")
                    rel_types.append("none")

            # Encode structured text
            q_t, q_a, q_r = self.structured_text_encoder.forward_batch_from_parsed(
                target_heads, anchor_heads, rel_types
            )

            # Soft anchor distribution
            anchor_dist = soft_anchor_distribution(
                h_o, q_a, object_mask, temperature=self.anchor_temperature
            )

            # Pairwise relation scoring
            R_ij = self.relation_scorer(h_o, q_r)  # [B, N, N]

            # Aggregate: s_rel[i] = Σ_j p_anchor[j] * R_ij
            relation_logits = torch.einsum("bj,bij->bi", anchor_dist, R_ij)  # [B, N]

            # === FUSION ===
            # Gate based on language features
            if text_features is not None:
                gate_input = text_features  # [B, D]
            else:
                # Fallback: use language features from base model
                gate_input = base_output["lang_features"]  # [B, D]

            gate_alpha = torch.sigmoid(self.gate(gate_input))  # [B, 1]

        # Combine base and relation scores
        logits = base_logits + gate_alpha * relation_logits

        # Apply mask
        logits = logits.masked_fill(~object_mask, float("-inf"))

        return {
            "logits": logits,
            "base_logits": base_logits,
            "relation_logits": relation_logits,
            "gate_alpha": gate_alpha,
            "anchor_dist": anchor_dist,
            "obj_features": base_output.get("obj_features"),
            "lang_features": base_output.get("lang_features"),
        }

    def load_base_weights(self, checkpoint_path: str) -> None:
        """Load weights from pre-trained ReferIt3DNet checkpoint.

        Args:
            checkpoint_path: path to .pt checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        # Load into base model
        self.base_model.load_state_dict(state_dict, strict=False)
        log.info(f"Loaded base weights from {checkpoint_path}")


def build_relation_aware_model(config: Dict[str, Any]) -> RelationAwareReferIt3DNet:
    """Build RelationAwareReferIt3DNet from config.

    Args:
        config: configuration dict

    Returns:
        RelationAwareReferIt3DNet model
    """
    model_config = config.get("model", {})
    relation_config = config.get("relation_branch", {})
    fusion_config = config.get("fusion", {})

    # Check for pre-trained checkpoint
    base_checkpoint = model_config.get("base_checkpoint", None)

    model = RelationAwareReferIt3DNet(
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
        # Fusion params
        gate_dim=fusion_config.get("gate_dim", 256),
        initial_gate_bias=fusion_config.get("initial_gate_bias", -1.0),
    )

    # Load pre-trained weights if provided
    if base_checkpoint:
        model.load_base_weights(base_checkpoint)
        log.info(f"Loaded pre-trained base from {base_checkpoint}")

    return model