"""Reproducible ranking for promoting coarse checkpoints to K-aligned rerank training.

The legacy rule sorted primarily by recall@20, then recall@10 (see ``old_promote_key``).
The shortlist-aligned score weights metrics that match the inference path: gold in the
coarse top-K shortlist and coarse-only accuracy when logits are restricted to that shortlist.
"""

from __future__ import annotations

from typing import Any


def old_promote_key(row: dict[str, Any]) -> tuple[float, float]:
    """Sort key used by ``promote_coarse_opt_rerank.py`` (higher is better)."""
    return (float(row.get("recall@20", 0.0)), float(row.get("recall@10", 0.0)))


def mrr_proxy_from_median_rank(gold_rank_median: float) -> float:
    """Map median 0-based gold rank to (0,1]; lower rank → higher score."""
    g = float(gold_rank_median)
    if g < 0 or g != g:  # nan
        return 0.0
    return 1.0 / (1.0 + g)


def shortlist_aligned_score(
    recall_at_10: float,
    coarse_target_in_topk_k: float,
    coarse_topk_acc_at_1: float,
    gold_rank_median: float,
    recall_at_20: float,
) -> float:
    """Weighted score for K=10 rerank promotion (higher is better).

    Components (all in [0, 1] except MRR proxy, also in (0,1]):

    - ``recall@10``: long-tail coarse recall from stage-1 sweep (same definition as
      ``eval_coarse_stage1_metrics``).
    - ``coarse_target_in_topk_k``: gold ∈ top-K shortlist rate from
      ``eval_coarse_topk_attribute`` with the same K as rerank (inference-aligned).
    - ``coarse_topk_acc_at_1``: coarse argmax restricted to the top-K mask (shortlist
      usefulness for the identity reranker).
    - MRR proxy from median gold rank (secondary shape signal).
    - ``recall@20``: tie-break / long-tail safety (small weight).

    Weights sum to 1.0 (constants in this function); see repository README for rationale.
    """
    r10 = float(recall_at_10)
    hit = float(coarse_target_in_topk_k)
    acc_k = float(coarse_topk_acc_at_1)
    mrr = mrr_proxy_from_median_rank(float(gold_rank_median))
    r20 = float(recall_at_20)
    return (
        0.28 * r10
        + 0.30 * hit
        + 0.24 * acc_k
        + 0.10 * mrr
        + 0.08 * r20
    )


def assign_ranks(rows: list[dict[str, Any]], key_fn: Any, reverse: bool = True) -> dict[str, int]:
    """Return 1-based ranks for ``row['name']`` after sorting by ``key_fn(row)``."""
    sorted_rows = sorted(rows, key=key_fn, reverse=reverse)
    return {str(r["name"]): i + 1 for i, r in enumerate(sorted_rows)}

