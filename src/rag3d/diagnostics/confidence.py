from __future__ import annotations

import torch


def logits_to_confidence_masked(logits_row: torch.Tensor, mask_row: torch.Tensor, pred_index: int) -> float:
    """Softmax confidence over valid positions for predicted index."""
    row = logits_row.masked_fill(~mask_row, float("-inf"))
    p = torch.softmax(row, dim=-1)
    return float(p[pred_index].item())


def anchor_entropy(p: torch.Tensor, mask: torch.Tensor) -> float:
    pm = p[mask]
    pm = pm[pm > 0]
    return float((-(pm * pm.log())).sum().item()) if pm.numel() else 0.0


def target_margin(logits: torch.Tensor, mask: torch.Tensor) -> float:
    vals = logits[mask].float()
    if vals.numel() < 2:
        return float("inf")
    top2 = torch.topk(vals, k=min(2, vals.numel())).values
    if top2.numel() < 2:
        return float("inf")
    return float(top2[0] - top2[1])
