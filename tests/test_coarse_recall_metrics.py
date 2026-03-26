"""Stage-1 recall metric helpers."""

from __future__ import annotations

import torch

from rag3d.evaluation.coarse_recall import (
    aggregate_recall_at_ks,
    gold_rank_in_scene,
    per_sample_recall_at_k,
    recall_bucket,
)


def test_gold_rank_and_recall_at_k() -> None:
    # Object 2 should be top-1
    logits = torch.tensor([[0.0, 0.5, 2.0, 0.1]])
    mask = torch.tensor([[True, True, True, True]])
    target = torch.tensor([2])
    ranks = gold_rank_in_scene(logits, target, mask)
    assert ranks[0] == 0
    assert per_sample_recall_at_k(logits, target, mask, 1) == [True]
    assert per_sample_recall_at_k(logits, target, mask, 5) == [True]
    agg = aggregate_recall_at_ks(logits, target, mask, (1, 5))
    assert agg["recall@1"] == 1.0
    assert recall_bucket(0) == "top1"


def test_recall_at_k_when_gold_is_second() -> None:
    logits = torch.tensor([[3.0, 2.0, 1.0]])
    mask = torch.ones(1, 3, dtype=torch.bool)
    target = torch.tensor([1])
    ranks = gold_rank_in_scene(logits, target, mask)
    assert ranks[0] == 1
    assert per_sample_recall_at_k(logits, target, mask, 1) == [False]
    assert per_sample_recall_at_k(logits, target, mask, 2) == [True]


def test_compute_batch_training_loss_ranking_margin() -> None:
    from rag3d.relation_reasoner.losses import compute_batch_training_loss

    logits = torch.tensor([[2.0, 3.0, 1.0]], requires_grad=True)
    mask = torch.ones(1, 3, dtype=torch.bool)
    target = torch.tensor([0])
    loss = compute_batch_training_loss(
        logits,
        target,
        mask,
        {"ranking_margin": {"enabled": True, "margin": 0.5, "lambda": 1.0}},
        None,
        None,
    )
    loss.backward()
    assert loss.item() > 0.0
