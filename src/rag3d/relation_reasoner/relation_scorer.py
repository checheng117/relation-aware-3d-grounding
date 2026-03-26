from __future__ import annotations

import torch
import torch.nn as nn


class PairwiseRelationScorer(nn.Module):
    """R_ij = score for (candidate i, anchor j) given relation query."""

    def __init__(self, hidden_dim: int, rel_dim: int, lang_dim: int) -> None:
        super().__init__()
        in_dim = hidden_dim * 2 + rel_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.rel_proj = nn.Linear(lang_dim, rel_dim)

    def forward(self, object_h: torch.Tensor, rel_query: torch.Tensor) -> torch.Tensor:
        """object_h [B,N,H], rel_query [B,H] -> R [B,N,N]."""
        b, n, h = object_h.shape
        rq = self.rel_proj(rel_query)  # [B, rel_dim]
        oi = object_h.unsqueeze(2).expand(b, n, n, h)
        oj = object_h.unsqueeze(1).expand(b, n, n, h)
        rq_e = rq.view(b, 1, 1, -1).expand(b, n, n, -1)
        x = torch.cat([oi, oj, rq_e], dim=-1)
        return self.mlp(x).squeeze(-1)
