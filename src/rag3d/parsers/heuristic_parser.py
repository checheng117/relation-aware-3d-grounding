"""Rule-based fallback parser for smoke tests and offline development."""

from __future__ import annotations

import re

from rag3d.datasets.schemas import ParsedUtterance
from rag3d.parsers.base import BaseParser

_REL_PATTERNS = [
    (re.compile(r"\bleft of\b|\bleft to\b", re.I), "left-of"),
    (re.compile(r"\bright of\b|\bright to\b", re.I), "right-of"),
    (re.compile(r"\bin front of\b|\bfront of\b", re.I), "front-of"),
    (re.compile(r"\bbehind\b|\bback of\b", re.I), "behind"),
    (re.compile(r"\bnext to\b|\bbeside\b|\bnear\b", re.I), "next-to"),
    (re.compile(r"\bbetween\b", re.I), "between"),
    (re.compile(r"\bon top of\b|\babove\b", re.I), "above"),
    (re.compile(r"\bunder\b|\bbelow\b", re.I), "below"),
]


class HeuristicParser(BaseParser):
    def parse(self, raw_text: str) -> ParsedUtterance:
        text = raw_text.strip()
        relation_types: list[str] = []
        for rx, name in _REL_PATTERNS:
            if rx.search(text):
                relation_types.append(name)
        anchor_head = None
        target_head = None
        target_modifiers: list[str] = []

        m = re.search(
            r"the\s+([a-z0-9\- ]+?)\s+(?:that is|which is)?\s*(?:left|right|front|behind|next|near|beside|between|on|under)",
            text,
            re.I,
        )
        if m:
            target_head = m.group(1).strip().split()[-1]

        m2 = re.search(
            r"(?:left of|right of|in front of|behind|next to|near|beside)\s+(?:the\s+)?([a-z0-9\-]+)",
            text,
            re.I,
        )
        if m2:
            anchor_head = m2.group(1).strip()

        if target_head is None:
            parts = text.replace(".", "").split()
            for i, w in enumerate(parts):
                if w.lower() in {"the", "a", "an", "pick", "choose", "select", "object"}:
                    continue
                if w.isalpha() and len(w) > 2:
                    target_head = w.lower()
                    target_modifiers = [p.lower() for p in parts[:i] if p.isalpha()]
                    break

        if not relation_types:
            relation_types = ["none"]
        conf = 0.9 if anchor_head and target_head and relation_types != ["none"] else 0.45
        warnings: list[str] = []
        if conf < 0.5:
            warnings.append("low_parse_confidence")
        if not anchor_head and relation_types != ["none"]:
            warnings.append("missing_anchor_head")
        return ParsedUtterance(
            raw_text=text,
            target_head=target_head,
            target_modifiers=target_modifiers,
            anchor_head=anchor_head,
            relation_types=relation_types,
            parser_confidence=conf,
            paraphrase_set=[],
            parse_source="heuristic",
            parse_warnings=warnings,
        )
