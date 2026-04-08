"""VLM Parser Adapter for loading pre-cached structured parses.

This adapter loads VLM-generated parses from offline cache only.
It does NOT make live API calls to any VLM service.

Cache structure:
    data/parser_cache/vlm/*.json

Each cache file contains a JSON record with fields matching ParsedUtterance schema.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from rag3d.datasets.schemas import ParsedUtterance
from rag3d.parsers.base import BaseParser
from rag3d.parsers.parse_quality import validate_parse_quality

log = logging.getLogger(__name__)


class VlmParserAdapter(BaseParser):
    """
    Offline-only VLM parser that loads from pre-cached JSON files.

    Usage:
        1. Pre-generate VLM parses using external script
        2. Save to data/parser_cache/vlm/<hash>.json
        3. This adapter loads and validates them

    If cache file missing or invalid, returns ParsedUtterance with parse_status="missing".
    """

    def __init__(self, cache_dir: Path | None = None, strict_validation: bool = False) -> None:
        """
        Initialize VLM parser adapter.

        Args:
            cache_dir: Root cache directory. Defaults to data/parser_cache
            strict_validation: If True, raise on malformed JSON. If False, return invalid parse.
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path("data/parser_cache")
        self.vlm_cache_dir = self.cache_dir / "vlm"
        self.strict_validation = strict_validation

        # Ensure cache directory exists
        self.vlm_cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_key(self, raw_text: str) -> str:
        """Generate SHA256 hash key for utterance."""
        return hashlib.sha256(raw_text.encode("utf-8")).hexdigest()

    def _cache_path(self, raw_text: str) -> Path:
        """Get cache file path for utterance."""
        return self.vlm_cache_dir / f"{self._cache_key(raw_text)}.json"

    def parse(self, raw_text: str) -> ParsedUtterance:
        """
        Load cached VLM parse for utterance.

        Args:
            raw_text: Utterance to parse

        Returns:
            ParsedUtterance with:
            - parse_source="cached_vlm" if found
            - parse_status validated by quality checker
            - parse_status="missing" if cache file not found
            - parse_status="invalid" if JSON malformed
        """
        cache_path = self._cache_path(raw_text)

        if not cache_path.is_file():
            log.debug("VLM parse cache missing for: %s (key=%s)", raw_text[:50], cache_path.stem[:12])
            return ParsedUtterance(
                raw_text=raw_text,
                parse_source="cached_vlm",
                parse_status="missing",
                parse_warnings=["vlm_cache_missing"],
                parser_confidence=0.0,
            )

        try:
            with cache_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            log.warning("VLM parse cache JSON decode error: %s", e)
            if self.strict_validation:
                raise
            return ParsedUtterance(
                raw_text=raw_text,
                parse_source="cached_vlm",
                parse_status="invalid",
                parse_warnings=["malformed_json"],
                parser_confidence=0.0,
            )

        # Validate required fields
        try:
            parsed = ParsedUtterance.model_validate(data)
        except Exception as e:
            log.warning("VLM parse cache validation error: %s", e)
            if self.strict_validation:
                raise
            return ParsedUtterance(
                raw_text=raw_text,
                parse_source="cached_vlm",
                parse_status="invalid",
                parse_warnings=["schema_validation_failed"],
                parser_confidence=0.0,
            )

        # Ensure raw_text matches
        if parsed.raw_text != raw_text:
            log.warning("VLM parse cache raw_text mismatch: cached='%s' vs input='%s'",
                        parsed.raw_text[:50], raw_text[:50])
            parsed.raw_text = raw_text
            parsed.parse_warnings.append("raw_text_mismatch")

        # Set source and validate quality
        parsed.parse_source = "cached_vlm"
        parsed.parse_status = validate_parse_quality(parsed)

        return parsed

    def parse_batch(self, raw_texts: list[str]) -> list[ParsedUtterance]:
        """
        Load cached VLM parses for batch of utterances.

        Args:
            raw_texts: List of utterances

        Returns:
            List of ParsedUtterance records
        """
        return [self.parse(t) for t in raw_texts]

    def cache_parse(self, raw_text: str, parsed: ParsedUtterance) -> None:
        """
        Save a parse to VLM cache (for pre-generation scripts).

        Args:
            raw_text: Original utterance
            parsed: ParsedUtterance to cache
        """
        cache_path = self._cache_path(raw_text)

        # Ensure source is set
        parsed.parse_source = "cached_vlm"

        # Write to cache
        with cache_path.open("w", encoding="utf-8") as f:
            f.write(parsed.model_dump_json(indent=2))

        log.info("Cached VLM parse for: %s (key=%s)", raw_text[:50], cache_path.stem[:12])


def build_parser_from_config(
    parser_source: str,
    cache_dir: Path | None = None,
    inner_parser: BaseParser | None = None,
) -> BaseParser:
    """
    Build parser instance from config string.

    Args:
        parser_source: "heuristic", "cached_vlm", "structured_rule"
        cache_dir: Cache directory for cached parsers
        inner_parser: Inner parser for CachedParser wrapper

    Returns:
        BaseParser instance

    Raises:
        ValueError: Unknown parser_source
    """
    from rag3d.parsers.heuristic_parser import HeuristicParser
    from rag3d.parsers.structured_rule_parser import StructuredRuleParser
    from rag3d.parsers.cached_parser import CachedParser

    cache_dir = cache_dir or Path("data/parser_cache")

    if parser_source == "heuristic":
        return CachedParser(HeuristicParser(), cache_dir / "heuristic")
    elif parser_source == "structured_rule":
        return CachedParser(StructuredRuleParser(), cache_dir / "heuristic")
    elif parser_source == "cached_vlm":
        return VlmParserAdapter(cache_dir)
    else:
        raise ValueError(f"Unknown parser_source: {parser_source}")