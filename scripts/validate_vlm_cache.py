#!/usr/bin/env python3
"""Validate and QA the parser cache.

Analyzes parser cache quality:
- Coverage: how many utterances have cached parses
- Parse validity: valid/partial/invalid/missing counts
- Target-head quality: missing target head rate
- Anchor quality: missing anchor rate on relation-heavy utterances
- Relation extraction quality: empty relation_types rate
- Parser confidence statistics

Exports:
- parser_cache_quality_summary.json
- parser_cache_quality_summary.md
- parser_cache_sample_audit.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag3d.datasets.schemas import ParsedUtterance
from rag3d.parsers.parse_quality import validate_parse_quality, compute_parse_confidence_bucket
from rag3d.utils.logging import setup_logging

log = logging.getLogger(__name__)


def load_manifest_utterances(manifest_path: Path) -> List[Dict[str, Any]]:
    """Load utterances from manifest file."""
    samples = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                samples.append({
                    "utterance": data.get("utterance", ""),
                    "scene_id": data.get("scene_id", ""),
                    "target_object_id": data.get("target_object_id", ""),
                })
    return samples


def utterance_cache_key(utterance: str) -> str:
    """Generate cache key for utterance."""
    return hashlib.sha256(utterance.encode("utf-8")).hexdigest()


def load_cached_parse(cache_dir: Path, utterance: str) -> Optional[ParsedUtterance]:
    """Load cached parse if exists."""
    key = utterance_cache_key(utterance)
    cache_path = cache_dir / f"{key}.json"
    if cache_path.is_file():
        try:
            with cache_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return ParsedUtterance.model_validate(data)
        except Exception as e:
            log.warning(f"Failed to load cached parse for key {key[:16]}: {e}")
    return None


def has_relation_keywords(utterance: str) -> bool:
    """Check if utterance contains relation keywords."""
    relation_keywords = [
        "left", "right", "front", "behind", "next", "near", "beside",
        "between", "above", "below", "on", "under", "behind", "in front of",
    ]
    utterance_lower = utterance.lower()
    return any(kw in utterance_lower for kw in relation_keywords)


def analyze_single_parse(parsed: ParsedUtterance, utterance: str) -> Dict[str, Any]:
    """Analyze a single parsed utterance."""
    is_relation_heavy = has_relation_keywords(utterance)

    return {
        "utterance": utterance,
        "parse_status": parsed.parse_status,
        "parser_confidence": parsed.parser_confidence,
        "confidence_bucket": compute_parse_confidence_bucket(parsed.parser_confidence),
        "target_head": parsed.target_head,
        "target_head_missing": parsed.target_head is None or parsed.target_head.strip() == "",
        "anchor_head": parsed.anchor_head,
        "anchor_head_missing": parsed.anchor_head is None or parsed.anchor_head.strip() == "",
        "relation_types": parsed.relation_types,
        "has_relations": len(parsed.relation_types) > 0 and parsed.relation_types != ["none"],
        "relation_types_empty": len(parsed.relation_types) == 0 or parsed.relation_types == ["none"],
        "is_relation_heavy_utterance": is_relation_heavy,
        "parse_warnings": parsed.parse_warnings,
        "parse_source": parsed.parse_source,
    }


def validate_cache(
    cache_dir: Path,
    manifest_path: Path,
    output_dir: Path,
    sample_size: int = 50,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Validate parser cache and generate QA reports.

    Args:
        cache_dir: Directory containing cached parses
        manifest_path: Path to manifest file
        output_dir: Directory to export reports
        sample_size: Number of samples for manual audit

    Returns:
        Dict with validation statistics
    """
    setup_logging()
    random.seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load manifest utterances
    samples = load_manifest_utterances(manifest_path)
    log.info(f"Loaded {len(samples)} samples from {manifest_path}")

    # Analyze each utterance
    analyses: List[Dict[str, Any]] = []
    missing_utterances: List[str] = []
    parse_errors: List[str] = []

    for sample in samples:
        utterance = sample["utterance"]
        parsed = load_cached_parse(cache_dir, utterance)

        if parsed is None:
            missing_utterances.append(utterance)
            analyses.append({
                "utterance": utterance,
                "parse_status": "missing",
                "parser_confidence": 0.0,
                "confidence_bucket": "low",
                "target_head_missing": True,
                "anchor_head_missing": True,
                "relation_types_empty": True,
                "is_relation_heavy_utterance": has_relation_keywords(utterance),
                "parse_source": "none",
            })
        else:
            try:
                analysis = analyze_single_parse(parsed, utterance)
                analyses.append(analysis)
            except Exception as e:
                parse_errors.append(f"{utterance[:50]}: {e}")

    log.info(f"Analyzed {len(analyses)} utterances")
    log.info(f"Missing: {len(missing_utterances)}, Errors: {len(parse_errors)}")

    # Compute statistics
    total = len(analyses)
    if total == 0:
        log.error("No analyses to report")
        return {"error": "No analyses"}

    # Parse status counts
    status_counts = Counter(a["parse_status"] for a in analyses)

    # Confidence bucket counts
    confidence_counts = Counter(a["confidence_bucket"] for a in analyses)

    # Target head quality
    target_missing_count = sum(1 for a in analyses if a.get("target_head_missing", False))

    # Anchor quality (for relation-heavy utterances)
    relation_heavy_analyses = [a for a in analyses if a.get("is_relation_heavy_utterance", False)]
    anchor_missing_in_relation_heavy = sum(
        1 for a in relation_heavy_analyses if a.get("anchor_head_missing", False)
    ) if relation_heavy_analyses else 0

    # Relation extraction quality
    empty_relations_count = sum(1 for a in analyses if a.get("relation_types_empty", False))

    # Parser source accounting
    source_counts = Counter(a.get("parse_source", "unknown") for a in analyses)

    # Compile statistics
    stats = {
        "total_utterances": total,
        "missing_utterances": len(missing_utterances),
        "parse_errors": len(parse_errors),
        "coverage_rate": (total - len(missing_utterances)) / total if total > 0 else 0,
        "parse_status_counts": dict(status_counts),
        "confidence_bucket_counts": dict(confidence_counts),
        "target_head_missing_count": target_missing_count,
        "target_head_missing_rate": target_missing_count / total if total > 0 else 0,
        "relation_heavy_utterance_count": len(relation_heavy_analyses),
        "anchor_missing_in_relation_heavy_count": anchor_missing_in_relation_heavy,
        "anchor_missing_in_relation_heavy_rate": (
            anchor_missing_in_relation_heavy / len(relation_heavy_analyses)
            if relation_heavy_analyses else 0
        ),
        "empty_relations_count": empty_relations_count,
        "empty_relations_rate": empty_relations_count / total if total > 0 else 0,
        "parse_source_counts": dict(source_counts),
        "validation_timestamp": datetime.now().isoformat(),
    }

    # Export JSON summary
    json_path = output_dir / "parser_cache_quality_summary.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    log.info(f"Exported JSON summary to {json_path}")

    # Export Markdown summary
    md_path = output_dir / "parser_cache_quality_summary.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Parser Cache Quality Summary\n\n")
        f.write(f"**Timestamp**: {stats['validation_timestamp']}\n\n")
        f.write(f"**Manifest**: `{manifest_path}`\n\n")
        f.write(f"**Cache directory**: `{cache_dir}`\n\n")

        f.write("## Coverage\n\n")
        f.write(f"- Total utterances: {total}\n")
        f.write(f"- Missing utterances: {len(missing_utterances)}\n")
        f.write(f"- Coverage rate: {stats['coverage_rate']:.2%}\n\n")

        f.write("## Parse Status Distribution\n\n")
        f.write("| Status | Count | Rate |\n")
        f.write("|---|---|---|\n")
        for status, count in sorted(status_counts.items()):
            rate = count / total if total > 0 else 0
            f.write(f"| {status} | {count} | {rate:.2%} |\n")
        f.write("\n")

        f.write("## Confidence Distribution\n\n")
        f.write("| Bucket | Count | Rate |\n")
        f.write("|---|---|---|\n")
        for bucket in ["high", "medium", "low"]:
            count = confidence_counts.get(bucket, 0)
            rate = count / total if total > 0 else 0
            f.write(f"| {bucket} | {count} | {rate:.2%} |\n")
        f.write("\n")

        f.write("## Target Head Quality\n\n")
        f.write(f"- Missing target head: {target_missing_count} ({stats['target_head_missing_rate']:.2%})\n\n")

        f.write("## Anchor Quality (Relation-Heavy Utterances)\n\n")
        f.write(f"- Relation-heavy utterances: {len(relation_heavy_analyses)}\n")
        f.write(f"- Missing anchor in relation-heavy: {anchor_missing_in_relation_heavy} ")
        f.write(f"({stats['anchor_missing_in_relation_heavy_rate']:.2%})\n\n")

        f.write("## Relation Extraction Quality\n\n")
        f.write(f"- Empty relations: {empty_relations_count} ({stats['empty_relations_rate']:.2%})\n\n")

        f.write("## Parser Source Accounting\n\n")
        for source, count in sorted(source_counts.items()):
            rate = count / total if total > 0 else 0
            f.write(f"- {source}: {count} ({rate:.2%})\n")

    log.info(f"Exported Markdown summary to {md_path}")

    # Generate sample audit
    audit_samples = []

    # Random samples
    random_samples = random.sample(analyses, min(sample_size, len(analyses)))
    audit_samples.extend([("random", a) for a in random_samples])

    # Relation-heavy samples
    relation_samples = [a for a in analyses if a.get("is_relation_heavy_utterance", False)]
    if relation_samples:
        audit_samples.extend([
            ("relation_heavy", a)
            for a in random.sample(relation_samples, min(20, len(relation_samples)))
        ])

    # Same-class clutter samples (utterances with same class mentioned multiple times)
    clutter_samples = [
        a for a in analyses
        if a.get("target_head") and a["utterance"].lower().count(a["target_head"].lower()) > 1
    ]
    if clutter_samples:
        audit_samples.extend([
            ("same_class_clutter", a)
            for a in random.sample(clutter_samples, min(20, len(clutter_samples)))
        ])

    # Export sample audit
    audit_path = output_dir / "parser_cache_sample_audit.md"
    with audit_path.open("w", encoding="utf-8") as f:
        f.write("# Parser Cache Sample Audit\n\n")
        f.write(f"**Timestamp**: {datetime.now().isoformat()}\n\n")
        f.write(f"**Total samples audited**: {len(audit_samples)}\n\n")

        current_category = None
        for category, analysis in audit_samples:
            if category != current_category:
                f.write(f"\n## {category.replace('_', ' ').title()}\n\n")
                current_category = category

            f.write(f"### Utterance\n\n")
            utterance_text = analysis["utterance"]
            f.write(f'"{utterance_text}"\n\n')
            f.write(f"**Parse Status**: {analysis['parse_status']}\n\n")
            f.write(f"**Confidence**: {analysis['parser_confidence']:.2f} ({analysis['confidence_bucket']})\n\n")
            target_head = analysis.get('target_head', 'N/A')
            f.write(f"**Target Head**: {target_head}\n\n")
            anchor_head = analysis.get('anchor_head', 'N/A')
            f.write(f"**Anchor Head**: {anchor_head}\n\n")
            relation_types = analysis.get('relation_types', [])
            f.write(f"**Relations**: {relation_types}\n\n")
            if analysis.get('parse_warnings'):
                f.write(f"**Warnings**: {analysis['parse_warnings']}\n\n")
            f.write("---\n\n")

    log.info(f"Exported sample audit to {audit_path}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Validate and QA parser cache")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT / "data/parser_cache/vlm",
        help="Directory containing cached parses",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "data/processed/val_manifest.jsonl",
        help="Path to manifest file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to export reports (default: cache-dir)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=50,
        help="Number of random samples for audit",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or args.cache_dir

    stats = validate_cache(
        cache_dir=args.cache_dir,
        manifest_path=args.manifest,
        output_dir=output_dir,
        sample_size=args.sample_size,
        seed=args.seed,
    )

    print(f"\nValidation Summary:")
    print(f"  Total utterances: {stats.get('total_utterances', 0)}")
    print(f"  Coverage rate: {stats.get('coverage_rate', 0):.2%}")
    print(f"  Parse status: {stats.get('parse_status_counts', {})}")


if __name__ == "__main__":
    main()