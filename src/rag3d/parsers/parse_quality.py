"""Parse quality validation for structured utterance parses.

Validates ParsedUtterance records and assigns parse_status:
- valid: all required fields present and consistent
- partial: some fields missing but usable
- invalid: critical fields missing or malformed
- missing: no parse record found
- fallback_used: parse was rejected, fallback triggered
"""

from __future__ import annotations

from rag3d.datasets.schemas import ParsedUtterance


def validate_parse_quality(parsed: ParsedUtterance) -> str:
    """
    Validate parse quality and return parse_status string.

    Rules:
    - valid: target_head present, anchor_head present if relations exist,
             confidence >= 0.5, no critical warnings
    - partial: target_head present but anchor missing when relations exist,
               or confidence < 0.5
    - invalid: target_head missing, or JSON malformed, or critical warnings
    - missing: placeholder status when no parse found (set by adapter)
    - fallback_used: set by fallback controller after decision

    Args:
        parsed: ParsedUtterance to validate

    Returns:
        parse_status string: "valid", "partial", or "invalid"
    """
    # Check critical warnings
    critical_warnings = {"parser_failure", "malformed_json", "empty_target"}
    if any(w in parsed.parse_warnings for w in critical_warnings):
        return "invalid"

    # Check target_head (required for any grounding)
    if parsed.target_head is None or parsed.target_head.strip() == "":
        return "invalid"

    # Check anchor_head when relations present
    has_relations = len(parsed.relation_types) > 0 and parsed.relation_types != ["none"]
    if has_relations and (parsed.anchor_head is None or parsed.anchor_head.strip() == ""):
        # Relations need anchor, but anchor missing -> partial
        return "partial"

    # Check confidence
    if parsed.parser_confidence < 0.5:
        return "partial"

    # All checks passed
    return "valid"


def get_parse_status(parsed: ParsedUtterance | None) -> str:
    """
    Get parse status, handling None case.

    Args:
        parsed: ParsedUtterance or None

    Returns:
        parse_status string
    """
    if parsed is None:
        return "missing"
    return validate_parse_quality(parsed)


def classify_parse_quality_batch(parsed_list: list[ParsedUtterance | None]) -> dict[str, int]:
    """
    Count parse_status categories across a batch.

    Args:
        parsed_list: List of ParsedUtterance or None

    Returns:
        Dict with counts for each status category
    """
    counts = {"valid": 0, "partial": 0, "invalid": 0, "missing": 0, "fallback_used": 0}
    for p in parsed_list:
        status = get_parse_status(p)
        counts[status] += 1
    return counts


def compute_parse_confidence_bucket(confidence: float) -> str:
    """
    Bucket parser confidence into discrete ranges.

    Args:
        confidence: float in [0, 1]

    Returns:
        Bucket string: "high", "medium", "low"
    """
    if confidence >= 0.7:
        return "high"
    elif confidence >= 0.5:
        return "medium"
    else:
        return "low"