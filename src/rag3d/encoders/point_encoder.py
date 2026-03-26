from __future__ import annotations

import torch
import torch.nn as nn


class PointNetToyEncoder(nn.Module):
    """Minimal point-set encoder (placeholder for real point backbones)."""

    def __init__(self, in_channels: int = 3, hidden_dim: int = 128, out_dim: int = 256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        # points: [B, K, C] -> max pool over K
        h = self.mlp(points)
        return h.max(dim=1).values
