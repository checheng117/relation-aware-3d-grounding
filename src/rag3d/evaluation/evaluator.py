from __future__ import annotations

from typing import Any

import torch

from rag3d.evaluation.metrics import accuracy_at_k
from rag3d.evaluation.paraphrase_eval import paraphrase_consistency_score
from rag3d.evaluation.stratified_eval import stratified_accuracy


class Evaluator:
    def __init__(self, device: torch.device | str = "cpu") -> None:
        self.device = torch.device(device)

    def evaluate_batch(
        self,
        logits: torch.Tensor,
        target_index: torch.Tensor,
        mask: torch.Tensor,
        meta: list[dict[str, Any]] | None = None,
    ) -> dict[str, float]:
        logits = logits.detach()
        target_index = target_index.detach()
        mask = mask.detach()
        out = {
            "acc@1": accuracy_at_k(logits, target_index, mask, 1),
            "acc@5": accuracy_at_k(logits, target_index, mask, min(5, logits.size(1))),
        }
        if meta:
            out.update(stratified_accuracy(logits, target_index, mask, meta))
        return out

    def paraphrase_eval(
        self,
        logits_list: list[torch.Tensor],
        target_index: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict[str, float]:
        return paraphrase_consistency_score(logits_list, target_index, mask)
