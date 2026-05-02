"""Tests for shortlist-aligned coarse promotion scoring."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag3d.evaluation.shortlist_promote import assign_ranks, old_promote_key, shortlist_aligned_score


def test_shortlist_score_ordering_spatial_vs_hardneg() -> None:
    """Synthetic: higher k10 hit + topK acc should beat high recall@20 alone."""
    s_hardneg = shortlist_aligned_score(0.33, 0.23, 0.02, 13.0, 0.71)
    s_spatial = shortlist_aligned_score(0.35, 0.30, 0.04, 17.0, 0.53)
    assert s_spatial > s_hardneg


def test_old_key_prefers_recall20() -> None:
    a = {"recall@20": 0.7, "recall@10": 0.32}
    b = {"recall@20": 0.6, "recall@10": 0.40}
    assert old_promote_key(a) > old_promote_key(b)


def test_assign_ranks_stable() -> None:
    rows = [
        {"name": "a", "promote_score_shortlist": 0.5},
        {"name": "b", "promote_score_shortlist": 0.9},
        {"name": "c", "promote_score_shortlist": 0.5},
    ]
    r = assign_ranks(rows, key_fn=lambda x: float(x["promote_score_shortlist"]))
    assert r == {"b": 1, "a": 2, "c": 3}
