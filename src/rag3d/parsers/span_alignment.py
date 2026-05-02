"""Span alignment utility for mapping parsed text to utterance token spans.

This module provides utilities to align parsed components (target, anchor, relation)
to their corresponding token spans in the original utterance.

Purpose:
- Parser extracts text strings: target="chair", anchor="table", relation="next to"
- We need to find where these appear in the utterance and map to token indices
- This enables extracting BERT span embeddings for the parsed components

Approach:
- Simple substring matching in the original utterance
- Character span → token span conversion using BERT tokenizer
- Robust fallbacks when exact match is not found
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

log = logging.getLogger(__name__)


@dataclass
class SpanAlignment:
    """Result of aligning a parsed text to a span in the utterance.

    Attributes:
        text: The parsed text string (e.g., "chair")
        char_start: Character start index in utterance (None if not found)
        char_end: Character end index in utterance (None if not found)
        token_start: Token start index (None if not found or tokenizer not used)
        token_end: Token end index (None if not found or tokenizer not used)
        found: Whether the span was successfully found in the utterance
        fallback_used: Whether a fallback was used (e.g., CLS token)
        match_method: How the span was found ("exact", "fuzzy", "fallback")
    """
    text: str
    char_start: Optional[int] = None
    char_end: Optional[int] = None
    token_start: Optional[int] = None
    token_end: Optional[int] = None
    found: bool = False
    fallback_used: bool = False
    match_method: str = "not_found"


@dataclass
class UtteranceSpanAlignment:
    """Complete alignment result for all parsed components.

    Attributes:
        utterance: Original utterance text
        target: SpanAlignment for target component
        anchor: SpanAlignment for anchor component
        relation: SpanAlignment for relation component
        any_found: Whether at least one component was found
        all_found: Whether all three components were found
    """
    utterance: str
    target: SpanAlignment
    anchor: SpanAlignment
    relation: SpanAlignment

    @property
    def any_found(self) -> bool:
        return self.target.found or self.anchor.found or self.relation.found

    @property
    def all_found(self) -> bool:
        return self.target.found and self.anchor.found and self.relation.found

    @property
    def found_count(self) -> int:
        return sum([self.target.found, self.anchor.found, self.relation.found])


# Relation type normalization mapping
# Parser outputs normalized forms like "next-to", but utterances have "next to"
RELATION_VARIANTS = {
    "left-of": ["left of", "left to", "on the left of", "to the left of"],
    "right-of": ["right of", "right to", "on the right of", "to the right of"],
    "front-of": ["front of", "in front of", "in the front of"],
    "behind": ["behind", "at the back of", "back of"],
    "next-to": ["next to", "beside", "near", "next", "beside the"],
    "between": ["between"],
    "above": ["above", "on top of", "on", "over", "on the", "atop"],
    "below": ["below", "under", "underneath", "beneath"],
    "none": [],  # No relation text to search for
}


def normalize_relation_for_search(relation_text: str) -> List[str]:
    """Convert normalized relation type to search variants.

    Args:
        relation_text: Normalized relation type (e.g., "next-to")

    Returns:
        List of search patterns to try (e.g., ["next to", "beside", "near"])
    """
    if not relation_text:
        return []

    # Clean the relation text
    relation_clean = relation_text.strip().lower()

    # Check if it's a known relation type
    if relation_clean in RELATION_VARIANTS:
        return RELATION_VARIANTS[relation_clean]

    # If not found, try removing hyphen
    relation_no_hyphen = relation_clean.replace("-", " ")
    if relation_no_hyphen != relation_clean:
        return [relation_no_hyphen, relation_clean]

    # Return original text as fallback
    return [relation_clean]


def find_substring_span(
    utterance: str,
    text: str,
    case_sensitive: bool = False,
) -> Tuple[Optional[int], Optional[int]]:
    """Find character span of text in utterance.

    Args:
        utterance: Original utterance text
        text: Text to find (e.g., "chair")
        case_sensitive: Whether to match case exactly

    Returns:
        (start, end) character indices, or (None, None) if not found
    """
    if not text or not utterance:
        return None, None

    # Clean text
    text_clean = text.strip()
    if not text_clean:
        return None, None

    # Search
    search_text = utterance if case_sensitive else utterance.lower()
    search_pattern = text_clean if case_sensitive else text_clean.lower()

    # Find first occurrence
    idx = search_text.find(search_pattern)
    if idx == -1:
        return None, None

    return idx, idx + len(text_clean)


def find_token_span(
    utterance: str,
    char_start: int,
    char_end: int,
    tokenizer: Optional[object] = None,
) -> Tuple[Optional[int], Optional[int]]:
    """Convert character span to token span using tokenizer.

    Args:
        utterance: Original utterance text
        char_start: Character start index
        char_end: Character end index
        tokenizer: BERT tokenizer with encode() method (optional)

    Returns:
        (token_start, token_end) indices, or (None, None) if tokenizer not provided
    """
    if tokenizer is None:
        return None, None

    if char_start is None or char_end is None:
        return None, None

    try:
        # Use tokenizer's char_to_tokens method if available (HuggingFace tokenizers)
        encoding = tokenizer(
            utterance,
            return_offsets_mapping=True,
            add_special_tokens=True,
        )

        # Find tokens that overlap with the character span
        offsets = encoding.get("offset_mapping", None)
        if offsets is None:
            return None, None

        token_start = None
        token_end = None

        # Skip special tokens (CLS at start, SEP at end)
        # offsets[0] is (0,0) for CLS, offsets[-1] is (0,0) for SEP
        for i, (start, end) in enumerate(offsets):
            if start == end:  # Special token
                continue
            if start <= char_start and end > char_start:
                token_start = i
            if start < char_end and end >= char_end:
                token_end = i + 1  # End is exclusive
            if token_start is not None and token_end is not None:
                break

        # Handle case where span extends beyond last token
        if token_start is not None and token_end is None:
            token_end = len(offsets) - 1  # Exclude SEP token

        return token_start, token_end

    except Exception as e:
        log.warning(f"Token span conversion failed: {e}")
        return None, None


def align_single_text(
    utterance: str,
    text: str,
    tokenizer: Optional[object] = None,
    fuzzy_match: bool = True,
    is_relation: bool = False,
) -> SpanAlignment:
    """Align a single parsed text to span in utterance.

    Args:
        utterance: Original utterance text
        text: Parsed text to find (e.g., "chair")
        tokenizer: BERT tokenizer (optional, for token-level alignment)
        fuzzy_match: Whether to try fuzzy matching if exact match fails
        is_relation: If True, use relation-specific matching with variants

    Returns:
        SpanAlignment with character and token spans
    """
    if not text:
        return SpanAlignment(
            text="",
            found=False,
            fallback_used=True,
            match_method="empty_text",
        )

    # For relations, use variant matching
    if is_relation:
        search_variants = normalize_relation_for_search(text)
        for variant in search_variants:
            char_start, char_end = find_substring_span(utterance, variant, case_sensitive=False)
            if char_start is not None:
                token_start, token_end = find_token_span(
                    utterance, char_start, char_end, tokenizer
                )
                return SpanAlignment(
                    text=text,
                    char_start=char_start,
                    char_end=char_end,
                    token_start=token_start,
                    token_end=token_end,
                    found=True,
                    fallback_used=False,
                    match_method="relation_variant",
                )
        # Relation not found
        return SpanAlignment(
            text=text,
            found=False,
            fallback_used=True,
            match_method="relation_not_found",
        )

    # Try exact substring match first (for target/anchor)
    char_start, char_end = find_substring_span(utterance, text, case_sensitive=False)

    if char_start is not None:
        # Found exact match
        token_start, token_end = find_token_span(
            utterance, char_start, char_end, tokenizer
        )

        return SpanAlignment(
            text=text,
            char_start=char_start,
            char_end=char_end,
            token_start=token_start,
            token_end=token_end,
            found=True,
            fallback_used=False,
            match_method="exact",
        )

    # Try fuzzy matching
    if fuzzy_match:
        # Handle multi-word relations like "next to"
        # Try matching individual words
        words = text.split()
        if len(words) > 1:
            # Try to find the relation pattern in utterance
            pattern = " ".join(words)
            char_start, char_end = find_substring_span(utterance, pattern, case_sensitive=False)

            if char_start is not None:
                token_start, token_end = find_token_span(
                    utterance, char_start, char_end, tokenizer
                )
                return SpanAlignment(
                    text=text,
                    char_start=char_start,
                    char_end=char_end,
                    token_start=token_start,
                    token_end=token_end,
                    found=True,
                    fallback_used=False,
                    match_method="fuzzy_multiword",
                )

        # Try partial word match (first word only)
        if words:
            first_word = words[0]
            char_start, char_end = find_substring_span(utterance, first_word, case_sensitive=False)
            if char_start is not None:
                token_start, token_end = find_token_span(
                    utterance, char_start, char_end, tokenizer
                )
                return SpanAlignment(
                    text=text,
                    char_start=char_start,
                    char_end=char_end,
                    token_start=token_start,
                    token_end=token_end,
                    found=True,
                    fallback_used=True,  # Partial match is a fallback
                    match_method="fuzzy_partial",
                )

    # No match found
    return SpanAlignment(
        text=text,
        found=False,
        fallback_used=True,
        match_method="not_found",
    )


def align_parsed_utterance(
    utterance: str,
    target_text: Optional[str],
    anchor_text: Optional[str],
    relation_text: Optional[str],
    tokenizer: Optional[object] = None,
) -> UtteranceSpanAlignment:
    """Align all parsed components to spans in utterance.

    Args:
        utterance: Original utterance text
        target_text: Parsed target head (e.g., "chair")
        anchor_text: Parsed anchor head (e.g., "table")
        relation_text: Parsed relation (e.g., "next to")
        tokenizer: BERT tokenizer (optional)

    Returns:
        UtteranceSpanAlignment with all component alignments
    """
    target_alignment = align_single_text(
        utterance, target_text or "", tokenizer, fuzzy_match=True, is_relation=False
    )

    anchor_alignment = align_single_text(
        utterance, anchor_text or "", tokenizer, fuzzy_match=True, is_relation=False
    )

    # Use relation-specific matching for relation component
    relation_alignment = align_single_text(
        utterance, relation_text or "", tokenizer, fuzzy_match=True, is_relation=True
    )

    result = UtteranceSpanAlignment(
        utterance=utterance,
        target=target_alignment,
        anchor=anchor_alignment,
        relation=relation_alignment,
    )

    # Log alignment statistics
    if not result.all_found:
        log.debug(
            f"Partial alignment for '{utterance}': "
            f"target={target_alignment.found}, anchor={anchor_alignment.found}, "
            f"relation={relation_alignment.found}"
        )

    return result


def align_batch_utterances(
    utterances: List[str],
    parsed_list: List[dict],
    tokenizer: Optional[object] = None,
) -> List[UtteranceSpanAlignment]:
    """Align parsed components for a batch of utterances.

    Args:
        utterances: List of original utterance texts
        parsed_list: List of parsed outputs (each with target_head, anchor_head, relation_types)
        tokenizer: BERT tokenizer (optional)

    Returns:
        List of UtteranceSpanAlignment results
    """
    results = []

    for i, utterance in enumerate(utterances):
        parsed = parsed_list[i] if i < len(parsed_list) else None

        if parsed is None:
            # No parse available, use empty alignments
            result = UtteranceSpanAlignment(
                utterance=utterance,
                target=SpanAlignment(text="", found=False, fallback_used=True),
                anchor=SpanAlignment(text="", found=False, fallback_used=True),
                relation=SpanAlignment(text="", found=False, fallback_used=True),
            )
        else:
            # Extract text from parsed output
            target_text = parsed.target_head if hasattr(parsed, 'target_head') else parsed.get('target_head', None)
            anchor_text = parsed.anchor_head if hasattr(parsed, 'anchor_head') else parsed.get('anchor_head', None)

            # Handle relation types (list or string)
            relation_types = parsed.relation_types if hasattr(parsed, 'relation_types') else parsed.get('relation_types', [])
            if isinstance(relation_types, list):
                relation_text = " ".join(relation_types) if relation_types else None
            else:
                relation_text = relation_types

            result = align_parsed_utterance(
                utterance, target_text, anchor_text, relation_text, tokenizer
            )

        results.append(result)

    return results


def get_span_found_rates(
    alignments: List[UtteranceSpanAlignment],
) -> dict:
    """Compute statistics on span alignment success rates.

    Args:
        alignments: List of UtteranceSpanAlignment results

    Returns:
        Dictionary with found rates and statistics
    """
    total = len(alignments)

    target_found = sum(1 for a in alignments if a.target.found)
    anchor_found = sum(1 for a in alignments if a.anchor.found)
    relation_found = sum(1 for a in alignments if a.relation.found)
    all_found = sum(1 for a in alignments if a.all_found)
    any_found = sum(1 for a in alignments if a.any_found)

    return {
        "total_samples": total,
        "target_found_count": target_found,
        "target_found_rate": target_found / total if total > 0 else 0.0,
        "anchor_found_count": anchor_found,
        "anchor_found_rate": anchor_found / total if total > 0 else 0.0,
        "relation_found_count": relation_found,
        "relation_found_rate": relation_found / total if total > 0 else 0.0,
        "all_found_count": all_found,
        "all_found_rate": all_found / total if total > 0 else 0.0,
        "any_found_count": any_found,
        "any_found_rate": any_found / total if total > 0 else 0.0,
    }


# Test function for development
def test_span_alignment():
    """Test span alignment on sample utterances."""
    test_cases = [
        {
            "utterance": "the chair next to the table",
            "target": "chair",
            "anchor": "table",
            "relation": "next to",
        },
        {
            "utterance": "the lamp above the desk",
            "target": "lamp",
            "anchor": "desk",
            "relation": "above",
        },
        {
            "utterance": "the red sofa",
            "target": "sofa",
            "anchor": None,
            "relation": None,
        },
    ]

    for case in test_cases:
        result = align_parsed_utterance(
            case["utterance"],
            case["target"],
            case["anchor"],
            case["relation"],
        )
        print(f"\nUtterance: '{case['utterance']}'")
        print(f"  Target: found={result.target.found}, char={result.target.char_start}-{result.target.char_end}")
        print(f"  Anchor: found={result.anchor.found}, char={result.anchor.char_start}-{result.anchor.char_end}")
        print(f"  Relation: found={result.relation.found}, char={result.relation.char_start}-{result.relation.char_end}")
        print(f"  All found: {result.all_found}")


if __name__ == "__main__":
    test_span_alignment()