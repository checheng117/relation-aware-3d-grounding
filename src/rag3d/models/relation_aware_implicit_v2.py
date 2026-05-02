"""Relation-Aware Implicit v2 Model for 3D Object Grounding.

This model uses sparse pairwise relations (top-k neighbors) to reduce memory
from O(N²) to O(N×k), enabling stable full training.

Key improvements over v1:
- Sparse top-k neighbor selection instead of dense O(N²)
- Memory-safe for full 30-epoch training
- Preserves v1's positive results (+0.47%)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

# Import baseline model
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "repro" / "referit3d_baseline" / "src"))

from referit3d_net import ReferIt3DNet, SimplePointEncoder, PointNetPPEncoder

from rag3d.models.relation_module_v2 import SparsePairwiseRelationModule

log = logging.getLogger(__name__)


class RelationAwareImplicitV2(nn.Module):
    """Relation-Aware Implicit v2 model with sparse pairwise relations.

    Architecture:
    1. Baseline ReferIt3DNet encoder → object_features [B, N, D]
    2. SparsePairwiseRelationModule → relation_context [B, N, D]
       (uses top-k nearest neighbors, not all N² pairs)
    3. Residual fusion: enhanced = object_features + relation_context
    4. Classification head (same as baseline)

    Key difference from v1:
    - Memory: O(N×k) instead of O(N²)
    - Training: Stable for full 30 epochs
    - Coverage: Local spatial neighbors (top-k by distance)
    """

    def __init__(
        self,
        # Baseline parameters
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
        # Sparse relation module parameters (NEW)
        relation_topk: int = 5,  # Number of neighbors per object
        relation_hidden_dim: int = 256,
        relation_dropout: float = 0.1,
        use_residual: bool = True,
        relation_temperature: float = 1.0,
        include_size: bool = True,  # Include relative size in geometry
    ):
        super().__init__()

        self.encoder_type = encoder_type
        self.point_output_dim = point_output_dim
        self.use_class_semantics = use_class_semantics
        self.use_learned_class_embedding = use_learned_class_embedding
        self.class_embed_dim = class_embed_dim
        self.lang_output_dim = lang_output_dim
        self.relation_topk = relation_topk

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

        # Sparse relation module (key difference from v1)
        self.relation_module = SparsePairwiseRelationModule(
            object_dim=obj_feat_dim,
            language_dim=lang_output_dim,
            hidden_dim=relation_hidden_dim,
            topk=relation_topk,
            dropout=relation_dropout,
            use_residual=use_residual,
            temperature=relation_temperature,
            include_size=include_size,
        )
        log.info(f"Using SparsePairwiseRelationModule with topk={relation_topk}")

        # Fusion layer
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
        centers: Optional[torch.Tensor] = None,
        sizes: Optional[torch.Tensor] = None,
        scene_diameter: Optional[float] = 5.0,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with sparse implicit relation modeling.

        Args:
            points: [B, N, ...] object point features
            object_mask: [B, N] boolean mask
            text_features: [B, 768] BERT features
            class_features: [B, N, 250] class hash features (PointNet++ mode)
            class_indices: [B, N] class indices (SimplePointEncoder mode)
            centers: [B, N, 3] object centers
            sizes: [B, N, 3] object sizes
            scene_diameter: scene diameter for normalization

        Returns:
            Dict containing:
            - logits: [B, N] per-object scores
            - obj_features: [B, N, D] relation-enhanced object features
            - base_obj_features: [B, N, D] baseline features
            - lang_features: [B, D] language features
            - relation_weights: [B, N, k] sparse attention weights
            - neighbor_indices: [B, N, k] selected neighbor indices
        """
        B, N = points.shape[:2]

        # Encode objects
        obj_features = self.point_encoder(points, object_mask)

        # Add class semantics
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

        # Encode language
        lang_features = self.lang_encoder(text_features)

        # === SPARSE RELATION MODULE ===
        # Extract centers and sizes if not provided
        if centers is None and self.encoder_type == "simple_point":
            centers = points[:, :, 0:3] * 5.0  # Unnormalize
        if sizes is None and self.encoder_type == "simple_point":
            sizes = points[:, :, 3:6] * 2.0  # Unnormalize

        if centers is None:
            centers = torch.zeros(B, N, 3, device=points.device)
            log.warning("centers not provided, using zeros")
        if sizes is None:
            sizes = torch.ones(B, N, 3, device=points.device)
            log.warning("sizes not provided, using ones")

        # Apply sparse relation module
        relation_output = self.relation_module(
            object_features=obj_features,
            language_embedding=lang_features,
            centers=centers,
            sizes=sizes,
            object_mask=object_mask,
            scene_diameter=scene_diameter,
        )

        enhanced_features = relation_output["enhanced_features"]
        relation_weights = relation_output["relation_weights"]  # [B, N, k]
        neighbor_indices = relation_output["neighbor_indices"]  # [B, N, k]

        # === FUSION AND CLASSIFICATION ===
        lang_expanded = lang_features.unsqueeze(1).expand(-1, N, -1)
        fused_input = torch.cat([enhanced_features, lang_expanded], dim=-1)

        fused_features = self.fusion(fused_input)
        logits = self.classifier(fused_features).squeeze(-1)

        # Apply mask
        logits = logits.masked_fill(~object_mask, float("-inf"))

        return {
            "logits": logits,
            "obj_features": enhanced_features,
            "base_obj_features": obj_features,
            "lang_features": lang_features,
            "relation_weights": relation_weights,  # [B, N, k] sparse
            "neighbor_indices": neighbor_indices,
        }


def build_relation_aware_implicit_v2(config: Dict[str, Any]) -> RelationAwareImplicitV2:
    """Build RelationAwareImplicitV2 model from config."""
    model_config = config.get("model", {})

    return RelationAwareImplicitV2(
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
        use_class_semantics=model_config.get("use_class_semantics", True),
        use_learned_class_embedding=model_config.get("use_learned_class_embedding", False),
        num_object_classes=model_config.get("num_classes", 516),
        class_embed_dim=model_config.get("class_embed_dim", 64),
        # Sparse relation params
        relation_topk=model_config.get("relation_topk", 5),
        relation_hidden_dim=model_config.get("relation_hidden_dim", 256),
        relation_dropout=model_config.get("relation_dropout", 0.1),
        use_residual=model_config.get("use_residual", True),
        relation_temperature=model_config.get("relation_temperature", 1.0),
        include_size=model_config.get("include_size", True),
    )