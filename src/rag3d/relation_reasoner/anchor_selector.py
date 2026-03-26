from __future__ import annotations

import torch
import torch.nn.functional as F


def soft_anchor_distribution(
    object_h: torch.Tensor,
    anchor_query: torch.Tensor,
    mask: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """object_h: [B,N,H], anchor_query: [B,H], mask: [B,N] bool -> p: [B,N]."""
    logits = (object_h * anchor_query.unsqueeze(1)).sum(-1) / max(temperature, 1e-6)
    logits = logits.masked_fill(~mask, float("-inf"))
    return F.softmax(logits, dim=-1)
