"""Relation-Aware Implicit v1 Model for 3D Object Grounding.

This model wraps the ReferIt3DNet baseline and adds an implicit pairwise
relation module that models object-object interactions conditioned on language.

Key design:
- Baseline encoder produces object features (trusted)
- Relation module adds relation-aware context (auxiliary)
- Residual fusion preserves baseline stability
- No parser dependency

Goal: +1% Test Acc@1 over 30.79% baseline without breaking stability.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

# Import baseline model
import sys
from pathlib import Path

# Add repro baseline path
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "repro" / "referit3d_baseline" / "src"))

from referit3d_net import ReferIt3DNet, SimplePointEncoder, PointNetPPEncoder

from rag3d.models.relation_module import PairwiseRelationModule, PairwiseRelationModuleLight

log = logging.getLogger(__name__)


class RelationAwareImplicitV1(nn.Module):
    """Relation-Aware Implicit v1 model.

    Architecture:
    1. Baseline ReferIt3DNet encoder → object_features [B, N, D]
    2. PairwiseRelationModule → relation_context [B, N, D]
    3. Residual fusion: enhanced = object_features + relation_context
    4. Classification head (same as baseline)

    Integration approach:
    - Insert relation module AFTER baseline encoder
    - Use residual addition (not replacement) for stability
    - Pass geometry (centers, sizes) to relation module
    """

    def __init__(
        self,
        # Baseline parameters (copied from ReferIt3DNet)
        point_input_dim: int = 256,
        point_hidden_dim: int = 128,
        point_output_dim: int = 256,
        lang_input_dim: int = 768,
        lang_hidden_dim: int = 256,
        lang_output_dim: int = 256,
        fusion_dim: int = 512,
        num_classes: int = 1,
        dropout: float = 0.1,
        encoder_type: str = "simple_point",
        pointnetpp_num_points: int = 1024,
        class_feature_dim: int = 250,
        use_class_semantics: bool = True,
        use_learned_class_embedding: bool = False,
        num_object_classes: int = 516,
        class_embed_dim: int = 64,
        # Relation module parameters
        relation_hidden_dim: int = 256,
        relation_mlp_layers: int = 2,
        relation_dropout: float = 0.1,
        use_residual: bool = True,  # CRITICAL for stability
        use_gate: bool = False,
        gate_init: float = 0.1,
        relation_temperature: float = 1.0,
        use_light_relation: bool = False,  # Use lighter relation module
    ):
        super().__init__()

        self.encoder_type = encoder_type
        self.point_output_dim = point_output_dim
        self.use_class_semantics = use_class_semantics
        self.use_learned_class_embedding = use_learned_class_embedding
        self.class_embed_dim = class_embed_dim
        self.lang_output_dim = lang_output_dim

        # Build baseline encoder (same as ReferIt3DNet)
        if use_learned_class_embedding and encoder_type == "simple_point":
            self.class_embedding = nn.Embedding(num_object_classes, class_embed_dim)
            log.info(f"Using learned class embedding: {num_object_classes} classes -> {class_embed_dim} dims")
        else:
            self.class_embedding = None

        if encoder_type == "pointnetpp":
            self.point_encoder = PointNetPPEncoder(
                input_channels=3,
                num_points=pointnetpp_num_points,
                output_dim=point_output_dim,
            )
            log.info(f"Using PointNet++ encoder")

            if use_class_semantics:
                self.class_encoder = nn.Sequential(
                    nn.Linear(class_feature_dim, 128),
                    nn.ReLU(),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                )
        else:
            self.point_encoder = SimplePointEncoder(
                input_dim=point_input_dim,
                hidden_dim=point_hidden_dim,
                output_dim=point_output_dim,
            )
            log.info(f"Using SimplePointEncoder")

        # Language encoder
        self.lang_encoder = nn.Sequential(
            nn.Linear(lang_input_dim, lang_hidden_dim),
            nn.ReLU(),
            nn.Linear(lang_hidden_dim, lang_output_dim),
        )

        # Determine object feature dimension after class embedding
        if encoder_type == "pointnetpp" and use_class_semantics:
            obj_feat_dim = point_output_dim + 64
        elif use_learned_class_embedding and encoder_type == "simple_point":
            obj_feat_dim = point_output_dim + class_embed_dim
        else:
            obj_feat_dim = point_output_dim

        # Relation module
        # Input: object_features [B, N, obj_feat_dim]
        # Output: relation_context [B, N, obj_feat_dim]
        if use_light_relation:
            self.relation_module = PairwiseRelationModuleLight(
                object_dim=obj_feat_dim,
                language_dim=lang_output_dim,
                hidden_dim=relation_hidden_dim // 2,  # Lighter
                use_residual=use_residual,
            )
            log.info("Using PairwiseRelationModuleLight")
        else:
            self.relation_module = PairwiseRelationModule(
                object_dim=obj_feat_dim,
                language_dim=lang_output_dim,
                hidden_dim=relation_hidden_dim,
                num_mlp_layers=relation_mlp_layers,
                dropout=relation_dropout,
                use_residual=use_residual,
                use_gate=use_gate,
                gate_init=gate_init,
                temperature=relation_temperature,
            )
            log.info(f"Using PairwiseRelationModule with residual={use_residual}")

        # Fusion input dimension (same as baseline)
        fusion_input_dim = obj_feat_dim + lang_output_dim

        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Classification head
        self.classifier = nn.Linear(fusion_dim, num_classes)

    def forward(
        self,
        points: torch.Tensor,
        object_mask: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        text_features: Optional[torch.Tensor] = None,
        class_features: Optional[torch.Tensor] = None,
        class_indices: Optional[torch.Tensor] = None,
        centers: Optional[torch.Tensor] = None,  # NEW: for relation module
        sizes: Optional[torch.Tensor] = None,  # NEW: for relation module
        scene_diameter: Optional[torch.Tensor] = None,  # NEW: for normalization
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with implicit relation modeling.

        Args:
            points: [B, N, ...] object point features
            object_mask: [B, N] boolean mask
            text_features: [B, 768] BERT features
            class_features: [B, N, 250] class hash features (PointNet++ mode)
            class_indices: [B, N] class indices (SimplePointEncoder mode)
            centers: [B, N, 3] object centers (REQUIRED for relation module)
            sizes: [B, N, 3] object sizes (REQUIRED for relation module)
            scene_diameter: [B] scene diameter for normalization

        Returns:
            Dict containing:
            - logits: [B, N] per-object scores
            - obj_features: [B, N, D] object features
            - lang_features: [B, D] language features
            - relation_weights: [B, N, N] attention weights
            - gate_value: float (if using gate)
        """
        B, N = points.shape[:2]

        # Encode objects
        obj_features = self.point_encoder(points, object_mask)  # [B, N, D]

        # Add class semantics
        if self.encoder_type == "pointnetpp" and self.use_class_semantics:
            if class_features is None:
                class_features = torch.zeros(B, N, 250, device=points.device, dtype=points.dtype)
            class_embed = self.class_encoder(class_features)  # [B, N, 64]
            obj_features = torch.cat([obj_features, class_embed], dim=-1)

        elif self.use_learned_class_embedding and self.class_embedding is not None:
            if class_indices is None:
                class_indices = torch.zeros(B, N, dtype=torch.long, device=points.device)
            class_embed = self.class_embedding(class_indices)  # [B, N, class_embed_dim]
            obj_features = torch.cat([obj_features, class_embed], dim=-1)

        # Encode language
        lang_features = self.lang_encoder(text_features)  # [B, lang_output_dim]

        # === RELATION MODULE ===
        # Extract centers and sizes from input or from points features

        # If centers/sizes not provided, extract from points features
        # (SimplePointEncoder mode: channels 0-2 are center, 3-5 are size)
        if centers is None and self.encoder_type == "simple_point":
            centers = points[:, :, 0:3] * 5.0  # Unnormalize
        if sizes is None and self.encoder_type == "simple_point":
            sizes = points[:, :, 3:6] * 2.0  # Unnormalize

        # Fallback: use zeros if still None
        if centers is None:
            centers = torch.zeros(B, N, 3, device=points.device)
            log.warning("centers not provided, using zeros")
        if sizes is None:
            sizes = torch.ones(B, N, 3, device=points.device)
            log.warning("sizes not provided, using ones")

        # Apply relation module
        relation_output = self.relation_module(
            object_features=obj_features,  # [B, N, obj_feat_dim]
            language_embedding=lang_features,  # [B, lang_output_dim]
            centers=centers,  # [B, N, 3]
            sizes=sizes,  # [B, N, 3]
            object_mask=object_mask,  # [B, N]
            scene_diameter=scene_diameter,
        )

        # Get enhanced features (baseline + relation context via residual)
        enhanced_features = relation_output["enhanced_features"]  # [B, N, obj_feat_dim]
        relation_weights = relation_output["relation_weights"]  # [B, N, N]

        # === FUSION AND CLASSIFICATION ===
        # Broadcast language to each object and concatenate
        lang_expanded = lang_features.unsqueeze(1).expand(-1, N, -1)  # [B, N, lang_output_dim]
        fused_input = torch.cat([enhanced_features, lang_expanded], dim=-1)  # [B, N, obj_feat_dim + lang_dim]

        # Fusion MLP
        fused_features = self.fusion(fused_input)  # [B, N, fusion_dim]

        # Classification
        logits = self.classifier(fused_features).squeeze(-1)  # [B, N]

        # Apply mask
        logits = logits.masked_fill(~object_mask, float("-inf"))

        # Return outputs
        result = {
            "logits": logits,
            "obj_features": enhanced_features,  # Relation-enhanced features
            "base_obj_features": obj_features,  # Baseline features (before relation)
            "lang_features": lang_features,
            "relation_weights": relation_weights,
        }

        if "gate_value" in relation_output:
            result["gate_value"] = relation_output["gate_value"]

        return result


def build_relation_aware_implicit_v1(config: Dict[str, Any]) -> RelationAwareImplicitV1:
    """Build RelationAwareImplicitV1 model from config."""
    model_config = config.get("model", {})

    return RelationAwareImplicitV1(
        # Baseline params
        point_input_dim=model_config.get("point_input_dim", 256),
        point_hidden_dim=model_config.get("point_hidden_dim", 128),
        point_output_dim=model_config.get("point_output_dim", 256),
        lang_input_dim=model_config.get("lang_input_dim", 768),
        lang_hidden_dim=model_config.get("lang_hidden_dim", 256),
        lang_output_dim=model_config.get("lang_output_dim", 256),
        fusion_dim=model_config.get("fusion_dim", 512),
        dropout=model_config.get("dropout", 0.1),
        encoder_type=model_config.get("encoder_type", "simple_point"),
        pointnetpp_num_points=model_config.get("pointnetpp_num_points", 1024),
        use_class_semantics=model_config.get("use_class_semantics", True),
        use_learned_class_embedding=model_config.get("use_learned_class_embedding", False),
        num_object_classes=model_config.get("num_classes", 516),
        class_embed_dim=model_config.get("class_embed_dim", 64),
        # Relation module params
        relation_hidden_dim=model_config.get("relation_hidden_dim", 256),
        relation_mlp_layers=model_config.get("relation_mlp_layers", 2),
        relation_dropout=model_config.get("relation_dropout", 0.1),
        use_residual=model_config.get("use_residual", True),  # Default: True
        use_gate=model_config.get("use_gate", False),
        gate_init=model_config.get("gate_init", 0.1),
        relation_temperature=model_config.get("relation_temperature", 1.0),
        use_light_relation=model_config.get("use_light_relation", False),
    )