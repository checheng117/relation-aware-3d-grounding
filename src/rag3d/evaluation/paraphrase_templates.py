"""Deterministic, relation-preserving surface paraphrases for robustness evaluation (no LLM)."""

from __future__ import annotations


def relation_preserving_paraphrases(utterance: str, max_variants: int = 4) -> list[str]:
    """Return ``utterance`` plus cheap rephrasings (deduped, capped at ``max_variants``)."""
    u = utterance.strip()
    if not u:
        return []
    out: list[str] = []
    seen: set[str] = set()

    def add(x: str) -> None:
        s = x.strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)

    add(u)
    low = u.lower()
    if low.startswith("the "):
        rest = u[4:]
        add("this " + rest)
        add("Pick the " + rest)
    elif not low.startswith("pick "):
        add("Pick " + low)
    if u[-1] not in ".!?":
        add(u + ".")
    return out[:max_variants]
