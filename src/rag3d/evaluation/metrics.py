from __future__ import annotations

import torch


def accuracy_at_k(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, k: int) -> float:
    """Masked top-k accuracy, batch mean."""
    logits = logits.masked_fill(~mask, float("-inf"))
    topk = logits.topk(k, dim=-1).indices
    correct = (topk == target.unsqueeze(1)).any(dim=1)
    return float(correct.float().mean().item())


def logit_top12_margin(logits_row: torch.Tensor, mask_row: torch.Tensor) -> float:
    vals = logits_row[mask_row]
    if vals.numel() < 2:
        return 1e6
    top2 = torch.topk(vals.float(), k=min(2, vals.numel())).values
    if top2.numel() < 2:
        return 1e6
    return float(top2[0] - top2[1])


def per_sample_correct_at1(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> list[bool]:
    row = logits.masked_fill(~mask, float("-inf"))
    pred = row.argmax(dim=-1)
    return [bool(pred[i].item() == target[i].item()) for i in range(logits.size(0))]


def per_sample_correct_at5(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> list[bool]:
    row = logits.masked_fill(~mask, float("-inf"))
    out: list[bool] = []
    for i in range(logits.size(0)):
        k = min(5, int(mask[i].sum().item()))
        if k < 1:
            out.append(False)
            continue
        topk = row[i].topk(k).indices
        out.append(bool((topk == target[i]).any().item()))
    return out
