"""Structured parser interface for 3D grounding."""
from typing import List, Dict, Any, Optional
from .parser_cache import ParsedUtterance, ParserCache, get_parsed_utterance


class StructuredParserInterface:
    """Interface for structured parsing with cache support."""

    def __init__(self, cache_dir: Optional[str] = "data/parser_cache"):
        self.parser_cache = ParserCache(cache_dir) if cache_dir else None

    def parse_utterances(self, texts: List[str]) -> List[ParsedUtterance]:
        """
        Parse a batch of utterances into structured representations.

        Args:
            texts: List of utterances to parse

        Returns:
            List of parsed utterances
        """
        parsed_results = []
        for text in texts:
            parsed = get_parsed_utterance(text, self.parser_cache)
            parsed_results.append(parsed)

        return parsed_results

    def parse_single(self, text: str) -> ParsedUtterance:
        """
        Parse a single utterance into structured representation.

        Args:
            text: Single utterance to parse

        Returns:
            Parsed utterance
        """
        return get_parsed_utterance(text, self.parser_cache)


# Backward compatibility function
def parse_utterance_batch(texts: List[str], cache_dir: Optional[str] = "data/parser_cache") -> List[ParsedUtterance]:
    """
    Convenience function for parsing batches of utterances.

    Args:
        texts: List of utterances to parse
        cache_dir: Directory for parser cache

    Returns:
        List of parsed utterances
    """
    parser_interface = StructuredParserInterface(cache_dir)
    return parser_interface.parse_utterances(texts)