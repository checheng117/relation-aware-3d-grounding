"""Optional Hugging Face–backed structured parser (graceful degradation).

This module never prints or logs tokens. Real VLM/LLM structured decoding is a TODO;
when implemented, use `get_hf_token()` only inside API calls and avoid logging headers.
"""

from __future__ import annotations

import logging
import os

from rag3d.datasets.schemas import ParsedUtterance
from rag3d.parsers.base import BaseParser
from rag3d.parsers.heuristic_parser import HeuristicParser
from rag3d.utils.env import get_hf_token

log = logging.getLogger(__name__)


class HFStructuredParser(BaseParser):
    """Placeholder: falls back to heuristic unless `HF_STRUCTURED_PARSER_ENABLED=1`."""

    def __init__(self) -> None:
        self._fallback = HeuristicParser()
        self._enabled = os.environ.get("HF_STRUCTURED_PARSER_ENABLED", "").lower() in {"1", "true", "yes"}

    def parse(self, raw_text: str) -> ParsedUtterance:
        if not self._enabled:
            return self._fallback.parse(raw_text)
        tok = get_hf_token()
        if not tok:
            log.warning("HF structured parser enabled but HF_TOKEN missing; using heuristic fallback.")
            return self._fallback.parse(raw_text)
        # TODO: call a small HF model with structured JSON output; validate into ParsedUtterance.
        _ = tok  # token used only for future API client initialization (do not log)
        log.info("HF structured parser not fully implemented; using heuristic fallback.")
        p = self._fallback.parse(raw_text)
        return p.model_copy(
            update={"parse_source": "hf_structured_stub", "parse_warnings": list(p.parse_warnings) + ["hf_stub_fallback"]}
        )
