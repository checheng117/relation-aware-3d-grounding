"""Structured parser cache layer for 3D grounding."""
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, List
from enum import Enum
from rag3d.datasets.schemas import ParsedUtterance


class ParseStatus(Enum):
    HEURISTIC = "heuristic"
    CACHED_EXTERNAL = "cached_external"
    FAILED = "failed"
    PENDING = "pending"


class ParserCache:
    """Cache system for structured utterance parses."""

    def __init__(self, cache_dir: Path = Path("data/parser_cache")):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _generate_cache_key(self, text: str, parser_type: str = "heuristic") -> str:
        """Generate a cache key for the given text and parser type."""
        content = f"{text}_{parser_type}".encode('utf-8')
        return hashlib.sha256(content).hexdigest()

    def get_cached_parse(self, text: str, parser_type: str = "heuristic") -> Optional[ParsedUtterance]:
        """Retrieve a cached parse if available."""
        cache_key = self._generate_cache_key(text, parser_type)
        cache_file = self.cache_dir / f"{cache_key}.json"

        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)

                # Convert dict back to ParsedUtterance
                return ParsedUtterance(
                    raw_text=data.get('raw_text', ''),
                    target_head=data.get('target_head'),
                    target_modifiers=data.get('target_modifiers', []),
                    anchor_head=data.get('anchor_head'),
                    relation_types=data.get('relation_types', []),
                    parser_confidence=data.get('parser_confidence', 0.5),
                    paraphrase_set=data.get('paraphrase_set', []),
                    parse_source=data.get('parse_source', 'heuristic')
                )
            except Exception:
                # If cache file is corrupted, return None to trigger recalculation
                return None

        return None

    def save_parse_to_cache(self, text: str, parsed: ParsedUtterance, parser_type: str = "heuristic"):
        """Save a parsed utterance to cache."""
        cache_key = self._generate_cache_key(text, parser_type)
        cache_file = self.cache_dir / f"{cache_key}.json"

        data = {
            'raw_text': parsed.raw_text,
            'target_head': parsed.target_head,
            'target_modifiers': parsed.target_modifiers,
            'anchor_head': parsed.anchor_head,
            'relation_types': parsed.relation_types,
            'parser_confidence': parsed.parser_confidence,
            'paraphrase_set': parsed.paraphrase_set,
            'parse_source': parsed.parse_source
        }

        with open(cache_file, 'w') as f:
            json.dump(data, f, indent=2)


class HeuristicStructuredParser:
    """Lightweight heuristic parser for creating structured representations."""

    def __init__(self):
        # Define relation keywords
        self.relation_keywords = [
            'left', 'right', 'behind', 'front', 'in front of', 'next to',
            'beside', 'between', 'among', 'near', 'close to', 'far from',
            'above', 'below', 'on top of', 'under', 'beneath', 'adjacent to',
            'closest', 'furthest', 'biggest', 'smallest', 'largest', 'smallest'
        ]

        # Define common object categories that might serve as anchors
        self.anchor_indicators = [
            'table', 'chair', 'sofa', 'couch', 'bed', 'desk', 'lamp',
            'window', 'door', 'wall', 'floor', 'ceiling', 'shelf', 'counter'
        ]

    def parse_utterance(self, text: str) -> ParsedUtterance:
        """
        Parse an utterance into structured components using heuristics.
        """
        text_lower = text.lower()
        words = text_lower.split()

        # Find relation keywords
        found_relations = []
        for rel in self.relation_keywords:
            if rel in text_lower:
                found_relations.append(rel)

        # Identify potential target and anchor based on context
        target_word = None
        anchor_word = None

        # Find words that might be targets (usually preceded by determiners or adjectives)
        potential_targets = []
        for i, word in enumerate(words):
            if i > 0 and words[i-1] in ['the', 'a', 'an', 'this', 'that', 'these', 'those']:
                if word not in self.relation_keywords and word not in self.anchor_indicators:
                    potential_targets.append(word)

        if potential_targets:
            target_word = potential_targets[0]  # Take the first potential target

        # Find potential anchor (often near relation words or common furniture)
        for anchor_candidate in self.anchor_indicators:
            if anchor_candidate in text_lower:
                anchor_word = anchor_candidate
                break

        # If no specific anchor found, look for any noun near relation keywords
        if not anchor_word and found_relations:
            # Look for nouns near relation keywords
            for i, word in enumerate(words):
                if word in found_relations:
                    # Check nearby words for potential anchors
                    for j in range(max(0, i-3), min(len(words), i+4)):
                        if words[j] not in self.relation_keywords and words[j] not in ['the', 'a', 'an', 'is', 'are', 'was', 'were']:
                            anchor_word = words[j]
                            break
                if anchor_word:
                    break

        # Estimate confidence based on how many structural elements we found
        confidence = 0.5  # Default confidence
        if target_word:
            confidence += 0.1
        if anchor_word and found_relations:
            confidence += 0.2
        elif anchor_word or found_relations:
            confidence += 0.1

        # Cap confidence
        confidence = min(0.8, confidence)  # Even heuristic parsers shouldn't claim high confidence

        return ParsedUtterance(
            raw_text=text,
            target_head=target_word,
            target_modifiers=[],  # Heuristic doesn't extract modifiers well
            anchor_head=anchor_word,
            relation_types=found_relations if found_relations else [],
            parser_confidence=confidence,
            paraphrase_set=[],  # Heuristic doesn't generate paraphrases
            parse_source="heuristic_keyword_match"
        )


def get_parsed_utterance(text: str, parser_cache: Optional[ParserCache] = None,
                        force_fallback: bool = False) -> ParsedUtterance:
    """
    Get parsed utterance with cache support and fallback to heuristic.
    """
    if parser_cache and not force_fallback:
        # Try to get from cache first
        cached = parser_cache.get_cached_parse(text)
        if cached:
            return cached

    # Use heuristic parser as fallback
    heuristic_parser = HeuristicStructuredParser()
    parsed = heuristic_parser.parse_utterance(text)

    # Save to cache if cache is provided
    if parser_cache:
        parser_cache.save_parse_to_cache(text, parsed)

    return parsed