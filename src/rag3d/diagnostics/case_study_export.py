"""Case study export utilities for Phase 3 parser ablation analysis.

Exports categorized case studies for:
- VLM parser clearly better than heuristic
- Heuristic parser sufficient
- VLM parse bad, fallback saves prediction
- VLM parse bad, fallback still fails
- Raw-text baseline wins
- Structured + VLM + fallback wins over raw-text
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)


@dataclass
class CaseStudy:
    """Single case study record."""

    category: str
    utterance: str
    gt_target_id: str
    scene_id: str

    # Parser outputs
    heuristic_parse: Dict[str, Any]
    vlm_parse: Optional[Dict[str, Any]] = None

    # Parse status and confidence
    heuristic_parse_status: str = "unknown"
    heuristic_parse_confidence: float = 0.0
    vlm_parse_status: str = "unknown"
    vlm_parse_confidence: float = 0.0

    # Fallback decision
    fallback_triggered: bool = False
    fallback_reason: Optional[str] = None
    structured_weight: float = 1.0
    raw_text_weight: float = 0.0

    # Predictions
    heuristic_pred_top1: Optional[str] = None
    heuristic_pred_top5: List[str] = field(default_factory=list)
    vlm_pred_top1: Optional[str] = None
    vlm_pred_top5: List[str] = field(default_factory=list)
    fallback_pred_top1: Optional[str] = None
    fallback_pred_top5: List[str] = field(default_factory=list)
    raw_text_pred_top1: Optional[str] = None
    raw_text_pred_top5: List[str] = field(default_factory=list)

    # Accuracy outcomes
    heuristic_correct_at1: bool = False
    heuristic_correct_at5: bool = False
    vlm_correct_at1: bool = False
    vlm_correct_at5: bool = False
    fallback_correct_at1: bool = False
    fallback_correct_at5: bool = False
    raw_text_correct_at1: bool = False
    raw_text_correct_at5: bool = False

    # Anchor diagnostics
    anchor_entropy: Optional[float] = None
    top_anchor_id: Optional[str] = None
    anchor_confidence: Optional[float] = None

    # Additional metadata
    notes: str = ""


CATEGORIES = [
    "vlm_better_than_heuristic",
    "heuristic_sufficient",
    "vlm_bad_fallback_saves",
    "vlm_bad_fallback_fails",
    "raw_text_wins",
    "structured_vlm_fallback_wins",
]


def categorize_case(
    heuristic_pred: Dict[str, Any],
    vlm_pred: Optional[Dict[str, Any]],
    fallback_pred: Optional[Dict[str, Any]],
    raw_text_pred: Dict[str, Any],
    target: Dict[str, Any],
    heuristic_parse: Dict[str, Any],
    vlm_parse: Optional[Dict[str, Any]],
    fallback_decision: Optional[Dict[str, Any]],
) -> str:
    """
    Determine category for a case study.

    Returns:
        Category string
    """
    target_id = target.get("target_id", target.get("target_obj_id", ""))

    # Check correctness
    heuristic_correct = heuristic_pred.get("pred_top1") == target_id
    vlm_correct = vlm_pred is not None and vlm_pred.get("pred_top1") == target_id
    fallback_correct = fallback_pred is not None and fallback_pred.get("pred_top1") == target_id
    raw_text_correct = raw_text_pred.get("pred_top1") == target_id

    # Get parse status
    vlm_status = vlm_parse.get("parse_status", "unknown") if vlm_parse else "missing"
    fallback_triggered = fallback_decision.get("fallback_triggered", False) if fallback_decision else False

    # Category logic
    if vlm_correct and not heuristic_correct:
        return "vlm_better_than_heuristic"

    if heuristic_correct and (vlm_correct or vlm_parse is None):
        return "heuristic_sufficient"

    if vlm_status in ("invalid", "partial", "missing") and fallback_triggered:
        if fallback_correct and not vlm_correct:
            return "vlm_bad_fallback_saves"
        elif not fallback_correct:
            return "vlm_bad_fallback_fails"

    if raw_text_correct and not (heuristic_correct or vlm_correct or fallback_correct):
        return "raw_text_wins"

    if (fallback_correct or vlm_correct) and not raw_text_correct:
        return "structured_vlm_fallback_wins"

    return "other"


def export_case_studies(
    all_predictions: Dict[str, List[Dict[str, Any]]],
    all_targets: List[Dict[str, Any]],
    all_parses: Dict[str, List[Dict[str, Any]]],
    output_dir: Path,
    all_fallback_decisions: Optional[List[Dict[str, Any]]] = None,
    max_per_category: int = 10,
) -> Dict[str, int]:
    """
    Export categorized case studies.

    Args:
        all_predictions: Dict with keys 'heuristic', 'vlm', 'fallback', 'raw_text'
        all_targets: List of ground truth records
        all_parses: Dict with keys 'heuristic', 'vlm'
        all_fallback_decisions: Optional list of fallback decisions
        output_dir: Directory to export case studies
        max_per_category: Maximum cases per category

    Returns:
        Dict with category counts
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    category_cases: Dict[str, List[CaseStudy]] = {cat: [] for cat in CATEGORIES}
    other_cases: List[CaseStudy] = []

    heuristic_preds = all_predictions.get("heuristic", [])
    vlm_preds = all_predictions.get("vlm", [])
    fallback_preds = all_predictions.get("fallback", [])
    raw_text_preds = all_predictions.get("raw_text", [])
    heuristic_parses = all_parses.get("heuristic", [])
    vlm_parses = all_parses.get("vlm", [])

    n_samples = len(all_targets)

    for i in range(n_samples):
        target = all_targets[i]

        heuristic_pred = heuristic_preds[i] if i < len(heuristic_preds) else {}
        vlm_pred = vlm_preds[i] if i < len(vlm_preds) else None
        fallback_pred = fallback_preds[i] if i < len(fallback_preds) else None
        raw_text_pred = raw_text_preds[i] if i < len(raw_text_preds) else {}

        heuristic_parse = heuristic_parses[i] if i < len(heuristic_parses) else {}
        vlm_parse = vlm_parses[i] if i < len(vlm_parses) else None

        fallback_decision = (
            all_fallback_decisions[i] if all_fallback_decisions and i < len(all_fallback_decisions)
            else None
        )

        category = categorize_case(
            heuristic_pred=heuristic_pred,
            vlm_pred=vlm_pred,
            fallback_pred=fallback_pred,
            raw_text_pred=raw_text_pred,
            target=target,
            heuristic_parse=heuristic_parse,
            vlm_parse=vlm_parse,
            fallback_decision=fallback_decision,
        )

        target_id = target.get("target_id", target.get("target_obj_id", ""))

        case = CaseStudy(
            category=category,
            utterance=target.get("utterance", ""),
            gt_target_id=target_id,
            scene_id=target.get("scene_id", ""),
            heuristic_parse=heuristic_parse,
            vlm_parse=vlm_parse,
            heuristic_parse_status=heuristic_parse.get("parse_status", "unknown"),
            heuristic_parse_confidence=heuristic_parse.get("parser_confidence", 0.0),
            vlm_parse_status=vlm_parse.get("parse_status", "unknown") if vlm_parse else "missing",
            vlm_parse_confidence=vlm_parse.get("parser_confidence", 0.0) if vlm_parse else 0.0,
            fallback_triggered=fallback_decision.get("fallback_triggered", False) if fallback_decision else False,
            fallback_reason=fallback_decision.get("reason") if fallback_decision else None,
            structured_weight=fallback_decision.get("structured_weight", 1.0) if fallback_decision else 1.0,
            raw_text_weight=fallback_decision.get("raw_text_weight", 0.0) if fallback_decision else 0.0,
            heuristic_pred_top1=heuristic_pred.get("pred_top1"),
            heuristic_pred_top5=heuristic_pred.get("pred_top5", []),
            vlm_pred_top1=vlm_pred.get("pred_top1") if vlm_pred else None,
            vlm_pred_top5=vlm_pred.get("pred_top5", []) if vlm_pred else [],
            fallback_pred_top1=fallback_pred.get("pred_top1") if fallback_pred else None,
            fallback_pred_top5=fallback_pred.get("pred_top5", []) if fallback_pred else [],
            raw_text_pred_top1=raw_text_pred.get("pred_top1"),
            raw_text_pred_top5=raw_text_pred.get("pred_top5", []),
            heuristic_correct_at1=heuristic_pred.get("pred_top1") == target_id,
            heuristic_correct_at5=target_id in heuristic_pred.get("pred_top5", []),
            vlm_correct_at1=vlm_pred is not None and vlm_pred.get("pred_top1") == target_id,
            vlm_correct_at5=vlm_pred is not None and target_id in vlm_pred.get("pred_top5", []),
            fallback_correct_at1=fallback_pred is not None and fallback_pred.get("pred_top1") == target_id,
            fallback_correct_at5=fallback_pred is not None and target_id in fallback_pred.get("pred_top5", []),
            raw_text_correct_at1=raw_text_pred.get("pred_top1") == target_id,
            raw_text_correct_at5=target_id in raw_text_pred.get("pred_top5", []),
            anchor_entropy=heuristic_pred.get("anchor_entropy"),
            top_anchor_id=heuristic_pred.get("top_anchor_id"),
            anchor_confidence=heuristic_pred.get("anchor_confidence"),
        )

        if category in category_cases:
            if len(category_cases[category]) < max_per_category:
                category_cases[category].append(case)
        else:
            if len(other_cases) < max_per_category:
                other_cases.append(case)

    # Export each category
    counts: Dict[str, int] = {}
    for category, cases in category_cases.items():
        if cases:
            cat_file = output_dir / f"{category}.json"
            with cat_file.open("w") as f:
                json.dump(
                    [{"category": c.category, **c.__dict__} for c in cases],
                    f,
                    indent=2,
                )
            counts[category] = len(cases)
            log.info(f"Exported {len(cases)} cases to {cat_file}")

    if other_cases:
        other_file = output_dir / "other.json"
        with other_file.open("w") as f:
            json.dump([{"category": c.category, **c.__dict__} for c in other_cases], f, indent=2)
        counts["other"] = len(other_cases)

    # Export README
    readme_path = output_dir / "README.md"
    with readme_path.open("w") as f:
        f.write("# Phase 3 Case Studies\n\n")
        f.write("This directory contains categorized case studies for parser ablation analysis.\n\n")
        f.write("## Categories\n\n")
        for cat in CATEGORIES:
            f.write(f"- **{cat}**: {counts.get(cat, 0)} cases\n")
        f.write(f"- **other**: {counts.get('other', 0)} cases\n\n")
        f.write("## Files\n\n")
        for cat in CATEGORIES + ["other"]:
            if counts.get(cat, 0) > 0:
                f.write(f"- `{cat}.json`\n")
        f.write("\n## Case Study Fields\n\n")
        f.write("| Field | Description |\n")
        f.write("|---|---|\n")
        f.write("| `category` | Category classification |\n")
        f.write("| `utterance` | Original utterance |\n")
        f.write("| `gt_target_id` | Ground truth target ID |\n")
        f.write("| `heuristic_parse` | Heuristic parser output |\n")
        f.write("| `vlm_parse` | VLM parser output |\n")
        f.write("| `fallback_triggered` | Whether fallback was triggered |\n")
        f.write("| `heuristic_pred_top1` | Heuristic model prediction |\n")
        f.write("| `vlm_pred_top1` | VLM model prediction |\n")
        f.write("| `fallback_pred_top1` | Fallback model prediction |\n")
        f.write("| `raw_text_pred_top1` | Raw-text baseline prediction |\n")
        f.write("| `heuristic_correct_at1` | Heuristic Acc@1 outcome |\n")
        f.write("| `vlm_correct_at1` | VLM Acc@1 outcome |\n")
        f.write("| `fallback_correct_at1` | Fallback Acc@1 outcome |\n")
        f.write("| `raw_text_correct_at1` | Raw-text Acc@1 outcome |\n")

    log.info(f"Exported case study README to {readme_path}")
    return counts
