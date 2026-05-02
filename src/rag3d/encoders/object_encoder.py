from __future__ import annotations

import torch
import torch.nn as nn


class ObjectMLPEncoder(nn.Module):
    """Map per-object feature vectors to a hidden representation."""

    def __init__(self, in_dim: int, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ObjectMLPEncoderWithGeomContext(nn.Module):
    """Concatenate synthetic/visual features with honest box + quality flags (``geom_dim`` channels)."""

    def __init__(
        self,
        in_dim: int,
        geom_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.geom_dim = geom_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim + geom_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor, geom: torch.Tensor | None = None) -> torch.Tensor:
        if geom is None:
            geom = torch.zeros(
                x.size(0),
                x.size(1),
                self.geom_dim,
                device=x.device,
                dtype=x.dtype,
            )
        return self.net(torch.cat([x, geom], dim=-1))
