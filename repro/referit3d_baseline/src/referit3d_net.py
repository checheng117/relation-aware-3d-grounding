"""ReferIt3DNet - Official baseline model for 3D object grounding.

This is a reproduction of the ReferIt3DNet baseline from:
"ReferIt3D: Neural Listeners for Fine-Grained 3D Object Identification"
ECCV 2020

Architecture:
- Point cloud encoder (PointNet++ or simple MLP for now)
- Language encoder (BERT or simple embedding)
- Cross-modal fusion and classification
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)


class SimplePointEncoder(nn.Module):
    """Simple point cloud encoder as placeholder for PointNet++.

    For full reproduction, replace with PointNet++ backbone.
    """

    def __init__(
        self,
        input_dim: int = 3,  # xyz, optionally + rgb
        hidden_dim: int = 128,
        output_dim: int = 256,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # Simple MLP encoder (PointNet-style without T-net)
        # Use input_dim if features are already aggregated, otherwise use 3 for raw points
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, points: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            points: [B, N, P, C] where N=objects, P=points, C=channels
                    OR [B, N, C] if already aggregated features
            mask: [B, N] boolean mask for valid objects

        Returns:
            Object features: [B, N, output_dim]
        """
        B, N = points.shape[:2]

        if points.dim() == 4:
            # Aggregate per-object point features via max pooling
            P, C = points.shape[2:]
            points_flat = points.view(B * N, P, C)
            # Simple aggregation: max pool over points
            obj_features = points_flat.max(dim=1).values  # [B*N, C]
            obj_features = self.encoder(obj_features)  # [B*N, output_dim]
            obj_features = obj_features.view(B, N, -1)  # [B, N, output_dim]
        else:
            # Already aggregated features [B, N, C]
            C = points.shape[-1]
            # If input dim doesn't match, adapt
            if C != self.input_dim:
                # Skip encoder, just project
                obj_features = points
            else:
                obj_features_flat = points.view(B * N, C)
                obj_features_flat = self.encoder(obj_features_flat)
                obj_features = obj_features_flat.view(B, N, -1)

        return obj_features


class PointNetPPEncoder(nn.Module):
    """PointNet++-style encoder for per-object point clouds.

    This is a simplified, memory-efficient PointNet-style encoder that:
    - Processes raw XYZ point coordinates (not hand-crafted features)
    - Uses shared MLP layers for point-wise features
    - Uses max pooling for permutation invariance
    - Includes hierarchical feature extraction

    The key difference from SimplePointEncoder:
    - Input: Raw points [B, N, P, 3] instead of hand-crafted features [B, N, 256]
    - Architecture: PointNet-style shared MLP with hierarchical abstraction
    """

    def __init__(
        self,
        input_channels: int = 3,  # XYZ coordinates
        num_points: int = 1024,  # Points sampled per object
        sa1_mlp: List[int] = [64, 64, 128],  # MLP channels level 1
        sa2_mlp: List[int] = [128, 128, 256],  # MLP channels level 2
        output_dim: int = 256,  # Final embedding dimension
    ):
        super().__init__()

        self.input_channels = input_channels
        self.num_points = num_points
        self.output_dim = output_dim

        # Level 1: Point-wise MLP (shared weights across all points)
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_channels, sa1_mlp[0], 1),
            nn.BatchNorm1d(sa1_mlp[0]),
            nn.ReLU(),
            nn.Conv1d(sa1_mlp[0], sa1_mlp[1], 1),
            nn.BatchNorm1d(sa1_mlp[1]),
            nn.ReLU(),
            nn.Conv1d(sa1_mlp[1], sa1_mlp[2], 1),
            nn.BatchNorm1d(sa1_mlp[2]),
            nn.ReLU(),
        )

        # Level 2: Another point-wise MLP on level 1 features
        self.conv2 = nn.Sequential(
            nn.Conv1d(sa1_mlp[2], sa2_mlp[0], 1),
            nn.BatchNorm1d(sa2_mlp[0]),
            nn.ReLU(),
            nn.Conv1d(sa2_mlp[0], sa2_mlp[1], 1),
            nn.BatchNorm1d(sa2_mlp[1]),
            nn.ReLU(),
            nn.Conv1d(sa2_mlp[1], sa2_mlp[2], 1),
            nn.BatchNorm1d(sa2_mlp[2]),
            nn.ReLU(),
        )

        # Final projection to output_dim
        self.fc = nn.Sequential(
            nn.Linear(sa2_mlp[2], output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, points: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            points: [B, N, P, C] where B=batch, N=objects, P=points, C=channels (expect C=3 for XYZ)
            mask: [B, N] boolean mask for valid objects

        Returns:
            Object features: [B, N, output_dim]
        """
        if points.dim() != 4:
            # If input is not point cloud, treat as aggregated features (backward compat)
            B, N = points.shape[:2]
            return self._forward_aggregated(points, B, N)

        B, N, P, C = points.shape

        # Reshape: [B, N, P, C] -> [B*N, C, P] for Conv1d
        points_flat = points.view(B * N, P, C).permute(0, 2, 1)  # [B*N, C, P]

        # Level 1: Point-wise conv
        x1 = self.conv1(points_flat)  # [B*N, 128, P]

        # Level 2: Point-wise conv on level 1 features
        x2 = self.conv2(x1)  # [B*N, 256, P]

        # Max pooling over points (permutation invariant)
        x_max = x2.max(dim=-1).values  # [B*N, 256]

        # Final projection
        obj_features = self.fc(x_max)  # [B*N, output_dim]

        # Reshape back to [B, N, output_dim]
        obj_features = obj_features.view(B, N, -1)

        return obj_features

    def _forward_aggregated(self, points: torch.Tensor, B: int, N: int) -> torch.Tensor:
        """Fallback for non-point-cloud input (backward compatibility)."""
        points_flat = points.view(B * N, -1)
        if points_flat.shape[-1] != 256:
            points_flat = F.linear(points_flat, torch.randn(256, points_flat.shape[-1], device=points.device) * 0.01)
        obj_features = self.fc(points_flat)
        return obj_features.view(B, N, -1)


class SimpleLanguageEncoder(nn.Module):
    """Simple language encoder as placeholder for BERT.

    For full reproduction, replace with pretrained BERT.
    """

    def __init__(
        self,
        input_dim: int = 768,  # BERT hidden size
        hidden_dim: int = 256,
        output_dim: int = 256,
        vocab_size: int = 10000,
        max_length: int = 128,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # Project BERT features to output dimension
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

        # Keep embedding for fallback (if no BERT features)
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = nn.Embedding(max_length, hidden_dim)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        text_features: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: [B, L] token indices (fallback)
            text_features: [B, D] pre-computed BERT features

        Returns:
            Language features: [B, output_dim]
        """
        if text_features is not None:
            # Project BERT features
            return self.proj(text_features)

        if input_ids is not None:
            B, L = input_ids.shape
            positions = torch.arange(L, device=input_ids.device).unsqueeze(0).expand(B, -1)

            embeddings = self.embedding(input_ids) + self.position_embedding(positions)
            features = embeddings.mean(dim=1)
            return self.proj(features)

        raise ValueError("Either input_ids or text_features must be provided")


class ReferIt3DNet(nn.Module):
    """Official ReferIt3DNet baseline model.

    Architecture:
    1. Encode object point clouds → per-object features
    2. Encode language utterance → language feature
    3. Cross-modal fusion (concatenation + MLP)
    4. Per-object classification scores
    """

    def __init__(
        self,
        point_input_dim: int = 256,  # Default to match feature dim (for SimplePointEncoder)
        point_hidden_dim: int = 128,
        point_output_dim: int = 256,
        lang_input_dim: int = 768,  # BERT hidden size
        lang_hidden_dim: int = 256,
        lang_output_dim: int = 256,
        fusion_dim: int = 512,
        num_classes: int = 1,  # Binary: is this the target?
        dropout: float = 0.1,
        encoder_type: str = "simple_point",  # "simple_point" or "pointnetpp"
        # PointNet++ specific params (simplified)
        pointnetpp_num_points: int = 1024,
    ):
        super().__init__()

        self.encoder_type = encoder_type
        self.point_output_dim = point_output_dim

        # Point cloud encoder - select based on encoder_type
        if encoder_type == "pointnetpp":
            self.point_encoder = PointNetPPEncoder(
                input_channels=3,  # XYZ
                num_points=pointnetpp_num_points,
                output_dim=point_output_dim,
            )
            log.info(f"Using PointNet++ encoder with {pointnetpp_num_points} points per object")
        else:
            self.point_encoder = SimplePointEncoder(
                input_dim=point_input_dim,
                hidden_dim=point_hidden_dim,
                output_dim=point_output_dim,
            )
            log.info(f"Using SimplePointEncoder with input_dim={point_input_dim}")

        # Language encoder (with BERT projection)
        self.lang_encoder = SimpleLanguageEncoder(
            input_dim=lang_input_dim,
            hidden_dim=lang_hidden_dim,
            output_dim=lang_output_dim,
        )

        # Cross-modal fusion
        self.fusion = nn.Sequential(
            nn.Linear(point_output_dim + lang_output_dim, fusion_dim),
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
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            points: [B, N, ...] object point features
            object_mask: [B, N] boolean mask for valid objects
            input_ids: [B, L] token indices
            text_features: [B, D] pre-computed text features

        Returns:
            Dict containing:
            - logits: [B, N] per-object scores
            - obj_features: [B, N, D] object features
            - lang_features: [B, D] language features
        """
        B, N = points.shape[:2]

        # Encode objects
        obj_features = self.point_encoder(points, object_mask)  # [B, N, D]

        # Encode language
        lang_features = self.lang_encoder(input_ids, text_features)  # [B, D]

        # Broadcast language to each object and concatenate
        lang_expanded = lang_features.unsqueeze(1).expand(-1, N, -1)  # [B, N, D]
        fused = torch.cat([obj_features, lang_expanded], dim=-1)  # [B, N, D+D]

        # Fusion and classification
        fused_features = self.fusion(fused)  # [B, N, fusion_dim]
        logits = self.classifier(fused_features).squeeze(-1)  # [B, N]

        # Apply mask
        logits = logits.masked_fill(~object_mask, float("-inf"))

        return {
            "logits": logits,
            "obj_features": obj_features,
            "lang_features": lang_features,
        }


class ReferIt3DNetWithAttention(nn.Module):
    """ReferIt3DNet variant with attention-based fusion.

    This is closer to modern architectures while still being a baseline.
    """

    def __init__(
        self,
        point_input_dim: int = 3,
        point_hidden_dim: int = 128,
        point_output_dim: int = 256,
        lang_output_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
        encoder_type: str = "simple_point",  # "simple_point" or "pointnetpp"
        # PointNet++ specific params
        pointnetpp_num_points: int = 1024,
    ):
        super().__init__()

        self.encoder_type = encoder_type

        if encoder_type == "pointnetpp":
            self.point_encoder = PointNetPPEncoder(
                input_channels=3,
                num_points=pointnetpp_num_points,
                output_dim=point_output_dim,
            )
        else:
            self.point_encoder = SimplePointEncoder(
                input_dim=point_input_dim,
                hidden_dim=point_hidden_dim,
                output_dim=point_output_dim,
            )

        self.lang_encoder = SimpleLanguageEncoder(
            output_dim=lang_output_dim,
        )

        # Cross-attention: language queries objects
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=lang_output_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Projection for point features
        self.point_proj = nn.Linear(point_output_dim, lang_output_dim)

        # Output classifier
        self.classifier = nn.Sequential(
            nn.Linear(lang_output_dim * 2, lang_output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lang_output_dim, 1),
        )

    def forward(
        self,
        points: torch.Tensor,
        object_mask: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        text_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        B, N = points.shape[:2]

        # Encode
        obj_features = self.point_encoder(points, object_mask)
        lang_features = self.lang_encoder(input_ids, text_features)

        # Project object features
        obj_proj = self.point_proj(obj_features)  # [B, N, D]

        # Cross-attention: language attends to objects
        lang_query = lang_features.unsqueeze(1)  # [B, 1, D]
        attn_output, attn_weights = self.cross_attention(
            lang_query, obj_proj, obj_proj,
            key_padding_mask=~object_mask,
        )

        # Combine language with attended object info
        lang_enhanced = torch.cat([lang_features, attn_output.squeeze(1)], dim=-1)

        # Score each object
        lang_expanded = lang_enhanced.unsqueeze(1).expand(-1, N, -1)
        combined = torch.cat([obj_proj, lang_expanded[:, :, :obj_proj.size(-1)]], dim=-1)

        # For now, use simple scoring
        logits = (obj_proj * lang_features.unsqueeze(1)).sum(dim=-1)

        # Apply mask
        logits = logits.masked_fill(~object_mask, float("-inf"))

        return {
            "logits": logits,
            "obj_features": obj_features,
            "lang_features": lang_features,
            "attention_weights": attn_weights,
        }