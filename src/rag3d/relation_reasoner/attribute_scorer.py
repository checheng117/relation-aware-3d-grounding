from __future__ import annotations

import torch
import torch.nn as nn


class AttributeScorer(nn.Module):
    def __init__(self, hidden_dim: int, lang_dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim + lang_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, object_h: torch.Tensor, lang_h: torch.Tensor) -> torch.Tensor:
        """object_h [B,N,H], lang_h [B,H] -> scores [B,N]."""
        b, n, h = object_h.shape
        lang = lang_h.unsqueeze(1).expand(-1, n, -1)
        x = torch.cat([object_h, lang], dim=-1)
        return self.mlp(x).squeeze(-1)
