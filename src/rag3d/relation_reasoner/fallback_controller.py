"""Fallback controller for structured grounding pipeline.

Decides whether to use structured reasoning or fall back to raw-text scoring
based on parse quality and confidence thresholds.

Modes:
- none: Always use structured reasoning (no fallback)
- hard: If parse invalid/low confidence, use raw-text only
- hybrid: Blend structured and raw-text scores based on confidence
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from rag3d.datasets.schemas import ParsedUtterance
from rag3d.parsers.parse_quality import validate_parse_quality, compute_parse_confidence_bucket

log = logging.getLogger(__name__)


@dataclass
class FallbackDecision:
    """Result of fallback decision for a single sample."""

    should_fallback: bool
    fallback_mode: str  # none, hard, hybrid
    reason: str | None
    structured_weight: float  # Weight for structured score (0.0 to 1.0)
    raw_text_weight: float    # Weight for raw-text score (0.0 to 1.0)
    parser_confidence: float
    parse_status: str


class FallbackController:
    """
    Controller for deciding fallback from structured to raw-text reasoning.

    Usage:
        controller = FallbackController(mode="hard", confidence_threshold=0.5)
        decision = controller.decide(parsed)
        if decision.should_fallback:
            # Use raw-text scoring path
        else:
            # Use structured scoring path
    """

    def __init__(
        self,
        mode: str = "none",
        confidence_threshold: float = 0.5,
        hybrid_blend_factor: float = 0.5,
    ) -> None:
        """
        Initialize fallback controller.

        Args:
            mode: Fallback mode - "none", "hard", or "hybrid"
            confidence_threshold: Minimum confidence to use structured parse
            hybrid_blend_factor: For hybrid mode, how much to weight structured
                                 when confidence at threshold (default 0.5)
        """
        if mode not in ("none", "hard", "hybrid"):
            raise ValueError(f"Invalid fallback mode: {mode}")

        self.mode = mode
        self.confidence_threshold = confidence_threshold
        self.hybrid_blend_factor = hybrid_blend_factor

    def decide(self, parsed: ParsedUtterance) -> FallbackDecision:
        """
        Make fallback decision for a single parse.

        Args:
            parsed: ParsedUtterance with parse_status and confidence

        Returns:
            FallbackDecision with weights and reason
        """
        parse_status = validate_parse_quality(parsed)
        confidence = parsed.parser_confidence

        # Mode: none - always use structured
        if self.mode == "none":
            return FallbackDecision(
                should_fallback=False,
                fallback_mode="none",
                reason=None,
                structured_weight=1.0,
                raw_text_weight=0.0,
                parser_confidence=confidence,
                parse_status=parse_status,
            )

        # Mode: hard - full fallback if below threshold or invalid
        if self.mode == "hard":
            should_fallback, reason = self._hard_fallback_check(parse_status, confidence)
            if should_fallback:
                return FallbackDecision(
                    should_fallback=True,
                    fallback_mode="hard",
                    reason=reason,
                    structured_weight=0.0,
                    raw_text_weight=1.0,
                    parser_confidence=confidence,
                    parse_status=parse_status,
                )
            else:
                return FallbackDecision(
                    should_fallback=False,
                    fallback_mode="hard",
                    reason=None,
                    structured_weight=1.0,
                    raw_text_weight=0.0,
                    parser_confidence=confidence,
                    parse_status=parse_status,
                )

        # Mode: hybrid - blend based on confidence
        if self.mode == "hybrid":
            structured_weight, raw_text_weight, reason = self._hybrid_weights(
                parse_status, confidence
            )
            should_fallback = raw_text_weight > 0.0
            return FallbackDecision(
                should_fallback=should_fallback,
                fallback_mode="hybrid",
                reason=reason,
                structured_weight=structured_weight,
                raw_text_weight=raw_text_weight,
                parser_confidence=confidence,
                parse_status=parse_status,
            )

        # Should not reach here
        raise RuntimeError(f"Unhandled mode: {self.mode}")

    def _hard_fallback_check(self, parse_status: str, confidence: float) -> tuple[bool, str | None]:
        """
        Check if hard fallback should trigger.

        Returns:
            (should_fallback, reason)
        """
        if parse_status == "invalid":
            return True, "parse_invalid"
        if parse_status == "missing":
            return True, "parse_missing"
        if parse_status == "partial" and confidence < self.confidence_threshold:
            return True, "low_confidence_partial"
        if confidence < self.confidence_threshold:
            return True, "low_confidence"
        return False, None

    def _hybrid_weights(
        self, parse_status: str, confidence: float
    ) -> tuple[float, float, str | None]:
        """
        Compute hybrid blend weights based on parse quality.

        Returns:
            (structured_weight, raw_text_weight, reason)
        """
        # Invalid/missing -> full raw-text
        if parse_status in ("invalid", "missing"):
            return 0.0, 1.0, f"parse_{parse_status}"

        # High confidence valid -> full structured
        if parse_status == "valid" and confidence >= self.confidence_threshold:
            return 1.0, 0.0, None

        # Below threshold -> blend
        # Weight structured by confidence proportion
        # Minimum structured weight for partial parses
        min_structured = 0.2 if parse_status == "partial" else 0.0

        # Linear interpolation from threshold to 1.0
        # At threshold: hybrid_blend_factor
        # At 1.0: 1.0
        if confidence >= self.confidence_threshold:
            # Above threshold: scale from blend_factor to 1.0
            range_size = 1.0 - self.confidence_threshold
            if range_size > 0:
                above_threshold = confidence - self.confidence_threshold
                structured_weight = self.hybrid_blend_factor + (
                    (1.0 - self.hybrid_blend_factor) * above_threshold / range_size
                )
            else:
                structured_weight = 1.0
        else:
            # Below threshold: scale from min_structured to blend_factor
            range_size = self.confidence_threshold
            if range_size > 0:
                below_threshold = confidence
                structured_weight = min_structured + (
                    (self.hybrid_blend_factor - min_structured) * below_threshold / range_size
                )
            else:
                structured_weight = min_structured

        structured_weight = max(min_structured, min(1.0, structured_weight))
        raw_text_weight = 1.0 - structured_weight

        reason = None if raw_text_weight == 0.0 else "hybrid_blend"

        return structured_weight, raw_text_weight, reason

    def decide_batch(self, parsed_list: list[ParsedUtterance]) -> list[FallbackDecision]:
        """
        Make fallback decisions for a batch of parses.

        Args:
            parsed_list: List of ParsedUtterance

        Returns:
            List of FallbackDecision
        """
        return [self.decide(p) for p in parsed_list]

    def get_statistics(self, decisions: list[FallbackDecision]) -> dict[str, Any]:
        """
        Compute statistics from a batch of fallback decisions.

        Args:
            decisions: List of FallbackDecision

        Returns:
            Dict with fallback statistics
        """
        n = len(decisions)
        if n == 0:
            return {"total": 0}

        fallback_count = sum(1 for d in decisions if d.should_fallback)
        avg_structured_weight = sum(d.structured_weight for d in decisions) / n
        avg_raw_weight = sum(d.raw_text_weight for d in decisions) / n
        avg_confidence = sum(d.parser_confidence for d in decisions) / n

        status_counts: dict[str, int] = {}
        for d in decisions:
            status_counts[d.parse_status] = status_counts.get(d.parse_status, 0) + 1

        reason_counts: dict[str, int] = {}
        for d in decisions:
            if d.reason:
                reason_counts[d.reason] = reason_counts.get(d.reason, 0) + 1

        return {
            "total": n,
            "fallback_rate": fallback_count / n,
            "avg_structured_weight": avg_structured_weight,
            "avg_raw_text_weight": avg_raw_weight,
            "avg_parser_confidence": avg_confidence,
            "parse_status_counts": status_counts,
            "fallback_reason_counts": reason_counts,
        }


def build_fallback_controller_from_config(config: dict[str, Any]) -> FallbackController:
    """
    Build FallbackController from config dict.

    Args:
        config: Config dict with keys:
            - fallback_mode: str (none/hard/hybrid)
            - fallback_confidence_threshold: float
            - fallback_hybrid_blend_factor: float (optional)

    Returns:
        FallbackController instance
    """
    mode = config.get("fallback_mode", "none")
    threshold = float(config.get("fallback_confidence_threshold", 0.5))
    blend = float(config.get("fallback_hybrid_blend_factor", 0.5))

    return FallbackController(mode=mode, confidence_threshold=threshold, hybrid_blend_factor=blend)