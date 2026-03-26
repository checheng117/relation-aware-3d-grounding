from __future__ import annotations

import torch

from rag3d.evaluation.metrics import accuracy_at_k


def test_accuracy_at_1():
    logits = torch.tensor([[0.0, 1.0, 0.0], [2.0, 0.0, 0.0]])
    target = torch.tensor([1, 0])
    mask = torch.tensor([[True, True, True], [True, True, False]])
    assert accuracy_at_k(logits, target, mask, 1) == 1.0
