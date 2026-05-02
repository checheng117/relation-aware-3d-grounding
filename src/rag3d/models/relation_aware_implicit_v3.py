"""Relation-Aware Implicit v3 Model for 3D Object Grounding.

This model uses chunked dense pairwise relations to preserve v1's
full coverage semantics while avoiding memory spikes.

Key design:
- Baseline encoder produces object features (trusted)
- Chunked relation module computes dense relations safely
- Residual fusion preserves baseline stability
- No parser dependency
- Numerically equivalent to v1

Goal: Recover v1's +0.47% improvement with stable training.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "repro" / "referit3d_baseline" / "src"))

from referit3d_net import ReferIt3DNet, SimplePointEncoder, PointNetPPEncoder

from rag3d.models.relation_module_v3 import ChunkedDensePairwiseRelationModule

log = logging.getLogger(__name__)


class RelationAwareImplicitV3(nn.Module):
    """Relation-Aware Implicit v3 model with chunked dense pairwise relations.

    Architecture (same as v1):
    1. Baseline ReferIt3DNet encoder → object_features [B, N, D]
    2. ChunkedDensePairwiseRelationModule → relation_context [B, N, D]
       (computes dense relations in memory-safe chunks)
    3. Residual fusion: enhanced = object_features + relation_context
    4. Classification head (same as baseline)

    Key difference from v1:
    - Relation module computes in chunks, not full tensor allocation
    - Same dense coverage (all N² pairs)
    - Same numerical semantics
    - Stable memory usage
    """

    def __init__(
        self,
        # Baseline parameters (same as v1)
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
        # Relation module parameters (same as v1)
        relation_hidden_dim: int = 256,
        relation_mlp_layers: int = 2,
        relation_dropout: float = 0.1,
        use_residual: bool = True,
        use_gate: bool = False,
        gate_init: float = 0.1,
        relation_temperature: float = 1.0,
        # Chunking parameters (NEW for v3)
        chunk_size: int = 8,  # Number of j neighbors per chunk
    ):
        super().__init__()

        self.encoder_type = encoder_type
        self.point_output_dim = point_output_dim
        self.use_class_semantics = use_class_semantics
        self.use_learned_class_embedding = use_learned_class_embedding
        self.class_embed_dim = class_embed_dim
        self.lang_output_dim = lang_output_dim
        self.chunk_size = chunk_size

        # Build baseline encoder (same as v1)
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
            log.info("Using PointNet++ encoder")

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
            log.info("Using SimplePointEncoder")

        # Language encoder (same as v1)
        self.lang_encoder = nn.Sequential(
            nn.Linear(lang_input_dim, lang_hidden_dim),
            nn.ReLU(),
            nn.Linear(lang_hidden_dim, lang_output_dim),
        )

        # Determine object feature dimension (same as v1)
        if encoder_type == "pointnetpp" and use_class_semantics:
            obj_feat_dim = point_output_dim + 64
        elif use_learned_class_embedding and encoder_type == "simple_point":
            obj_feat_dim = point_output_dim + class_embed_dim
        else:
            obj_feat_dim = point_output_dim

        # Chunked dense relation module (NEW for v3)
        self.relation_module = ChunkedDensePairwiseRelationModule(
            object_dim=obj_feat_dim,
            language_dim=lang_output_dim,
            hidden_dim=relation_hidden_dim,
            num_mlp_layers=relation_mlp_layers,
            dropout=relation_dropout,
            use_residual=use_residual,
            use_gate=use_gate,
            gate_init=gate_init,
            temperature=relation_temperature,
            chunk_size=chunk_size,
        )
        log.info(f"Using ChunkedDensePairwiseRelationModule with chunk_size={chunk_size}")

        # Fusion layer (same as v1)
        fusion_input_dim = obj_feat_dim + lang_output_dim

        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Classification head (same as v1)
        self.classifier = nn.Linear(fusion_dim, num_classes)

    def forward(
        self,
        points: torch.Tensor,
        object_mask: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        text_features: Optional[torch.Tensor] = None,
        class_features: Optional[torch.Tensor] = None,
        class_indices: Optional[torch.Tensor] = None,
        centers: Optional[torch.Tensor] = None,
        sizes: Optional[torch.Tensor] = None,
        scene_diameter: Optional[float] = 5.0,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with chunked dense relation modeling.

        Same as v1, outputs should be equivalent.
        """
        B, N = points.shape[:2]

        # Encode objects (same as v1)
        obj_features = self.point_encoder(points, object_mask)

        # Add class semantics (same as v1)
        if self.encoder_type == "pointnetpp" and self.use_class_semantics:
            if class_features is None:
                class_features = torch.zeros(B, N, 250, device=points.device, dtype=points.dtype)
            class_embed = self.class_encoder(class_features)
            obj_features = torch.cat([obj_features, class_embed], dim=-1)

        elif self.use_learned_class_embedding and self.class_embedding is not None:
            if class_indices is None:
                class_indices = torch.zeros(B, N, dtype=torch.long, device=points.device)
            class_embed = self.class_embedding(class_indices)
            obj_features = torch.cat([obj_features, class_embed], dim=-1)

        # Encode language (same as v1)
        lang_features = self.lang_encoder(text_features)

        # === CHUNKED RELATION MODULE ===
        # Extract centers and sizes (same as v1)
        if centers is None and self.encoder_type == "simple_point":
            centers = points[:, :, 0:3] * 5.0
        if sizes is None and self.encoder_type == "simple_point":
            sizes = points[:, :, 3:6] * 2.0

        if centers is None:
            centers = torch.zeros(B, N, 3, device=points.device)
            log.warning("centers not provided, using zeros")
        if sizes is None:
            sizes = torch.ones(B, N, 3, device=points.device)
            log.warning("sizes not provided, using ones")

        # Apply chunked relation module
        relation_output = self.relation_module(
            object_features=obj_features,
            language_embedding=lang_features,
            centers=centers,
            sizes=sizes,
            object_mask=object_mask,
            scene_diameter=scene_diameter,
        )

        enhanced_features = relation_output["enhanced_features"]
        relation_weights = relation_output["relation_weights"]  # Full [B, N, N]

        # === FUSION AND CLASSIFICATION (same as v1) ===
        lang_expanded = lang_features.unsqueeze(1).expand(-1, N, -1)
        fused_input = torch.cat([enhanced_features, lang_expanded], dim=-1)

        fused_features = self.fusion(fused_input)
        logits = self.classifier(fused_features).squeeze(-1)

        logits = logits.masked_fill(~object_mask, float("-inf"))

        return {
            "logits": logits,
            "obj_features": enhanced_features,
            "base_obj_features": obj_features,
            "lang_features": lang_features,
            "relation_weights": relation_weights,
        }


def build_relation_aware_implicit_v3(config: Dict[str, Any]) -> RelationAwareImplicitV3:
    """Build RelationAwareImplicitV3 model from config."""
    model_config = config.get("model", {})

    return RelationAwareImplicitV3(
        # Baseline params (same as v1)
        point_input_dim=model_config.get("point_input_dim", 256),
        point_hidden_dim=model_config.get("point_hidden_dim", 128),
        point_output_dim=model_config.get("point_output_dim", 256),
        lang_input_dim=model_config.get("lang_input_dim", 768),
        lang_hidden_dim=model_config.get("lang_hidden_dim", 256),
        lang_output_dim=model_config.get("lang_output_dim", 256),
        fusion_dim=model_config.get("fusion_dim", 512),
        dropout=model_config.get("dropout", 0.1),
        encoder_type=model_config.get("encoder_type", "simple_point"),
        use_class_semantics=model_config.get("use_class_semantics", True),
        use_learned_class_embedding=model_config.get("use_learned_class_embedding", False),
        num_object_classes=model_config.get("num_classes", 516),
        class_embed_dim=model_config.get("class_embed_dim", 64),
        # Relation params (same as v1)
        relation_hidden_dim=model_config.get("relation_hidden_dim", 256),
        relation_mlp_layers=model_config.get("relation_mlp_layers", 2),
        relation_dropout=model_config.get("relation_dropout", 0.1),
        use_residual=model_config.get("use_residual", True),
        use_gate=model_config.get("use_gate", False),
        gate_init=model_config.get("gate_init", 0.1),
        relation_temperature=model_config.get("relation_temperature", 1.0),
        # Chunking params (NEW for v3)
        chunk_size=model_config.get("chunk_size", 8),
    )