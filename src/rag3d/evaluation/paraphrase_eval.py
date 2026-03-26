from __future__ import annotations

import torch


def paraphrase_consistency_score(
    logits_list: list[torch.Tensor],
    target_index: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, float]:
    """Fraction of samples where argmax matches across paraphrase views."""
    if len(logits_list) < 2:
        return {"paraphrase_target_agreement": 1.0, "paraphrase_mean_acc@1": 0.0}
    preds = [l.argmax(dim=-1) for l in logits_list]
    agree = torch.ones(logits_list[0].size(0), dtype=torch.bool, device=logits_list[0].device)
    for p in preds[1:]:
        agree &= p == preds[0]
    accs = []
    for l in logits_list:
        hit = l.argmax(dim=-1) == target_index
        accs.append(hit.float().mean().item())
    return {
        "paraphrase_target_agreement": float(agree.float().mean().item()),
        "paraphrase_mean_acc@1": float(sum(accs) / len(accs)),
    }


def anchor_distribution_drift(p_list: list[torch.Tensor], mask: torch.Tensor) -> float:
    """Mean L1 drift vs first anchor distribution (per-sample then mean)."""
    if len(p_list) < 2:
        return 0.0
    ref = p_list[0]
    total = 0.0
    b = ref.size(0)
    for i in range(1, len(p_list)):
        diff = (p_list[i] - ref).abs().masked_fill(~mask, 0.0).sum(dim=-1)
        denom = mask.float().sum(dim=-1).clamp_min(1.0)
        total += float((diff / denom).mean().item())
    return total / (len(p_list) - 1)
