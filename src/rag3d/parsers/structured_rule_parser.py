"""Deterministic structured-style parser (template-heavy) for ablations vs :class:`HeuristicParser`.

Uses different segmentation and confidence rules so relation-aware training sees a distinct parse
distribution without calling external APIs. This is **not** an LLM: it is an alternate rule baseline.
"""

from __future__ import annotations

import re

from rag3d.datasets.schemas import ParsedUtterance
from rag3d.parsers.base import BaseParser

# Single primary relation only (first hit wins), unlike HeuristicParser which accumulates all matches.
_REL_PRIORITY = [
    (re.compile(r"\bleft of\b|\bleft to\b", re.I), "left-of"),
    (re.compile(r"\bright of\b|\bright to\b", re.I), "right-of"),
    (re.compile(r"\bin front of\b", re.I), "front-of"),
    (re.compile(r"\bfront of\b", re.I), "front-of"),
    (re.compile(r"\bbehind\b|\bback of\b", re.I), "behind"),
    (re.compile(r"\bnext to\b|\bbeside\b", re.I), "next-to"),
    (re.compile(r"\bnear\b", re.I), "next-to"),
    (re.compile(r"\bbetween\b", re.I), "between"),
    (re.compile(r"\bon top of\b|\babove\b", re.I), "above"),
    (re.compile(r"\bunder\b|\bbelow\b", re.I), "below"),
]


class StructuredRuleParser(BaseParser):
    """Template-first parse: one relation, anchor from ``... of (the) X``, looser target head."""

    def parse(self, raw_text: str) -> ParsedUtterance:
        text = raw_text.strip()
        relation_types: list[str] = ["none"]
        for rx, name in _REL_PRIORITY:
            if rx.search(text):
                relation_types = [name]
                break

        anchor_head = None
        m_anchor = re.search(
            r"(?:left of|right of|in front of|behind|next to|near|beside|on top of|above|below)\s+(?:the|a|an)?\s*([a-z0-9\-]+)",
            text,
            re.I,
        )
        if m_anchor:
            anchor_head = m_anchor.group(1).strip().lower()

        target_head = None
        target_modifiers: list[str] = []
        m_t = re.match(
            r"^\s*(?:the|a|an)\s+([a-z0-9\-]+(?:\s+[a-z0-9\-]+){0,3})\b",
            text,
            re.I,
        )
        if m_t:
            chunk = m_t.group(1).strip().lower()
            parts = chunk.split()
            if parts:
                target_head = parts[-1]
                target_modifiers = parts[:-1]

        if target_head is None:
            parts = re.findall(r"[A-Za-z]{3,}", text)
            if parts:
                target_head = parts[0].lower()

        if relation_types == ["none"]:
            conf = 0.38
        elif anchor_head and target_head:
            conf = 0.82
        elif anchor_head or target_head:
            conf = 0.58
        else:
            conf = 0.42

        warnings: list[str] = []
        if relation_types == ["none"]:
            warnings.append("no_relation_pattern")
        return ParsedUtterance(
            raw_text=text,
            target_head=target_head,
            target_modifiers=target_modifiers,
            anchor_head=anchor_head,
            relation_types=relation_types,
            parser_confidence=conf,
            paraphrase_set=[],
            parse_source="structured_rule",
            parse_warnings=warnings,
        )
