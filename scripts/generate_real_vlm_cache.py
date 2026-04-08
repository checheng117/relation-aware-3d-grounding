#!/usr/bin/env python3
"""Generate real parser cache from dataset utterances.

Reads utterances from manifest files and generates cached parses using
a specified parser backend (heuristic, structured_rule, or VLM API).

Usage:
    python scripts/generate_real_vlm_cache.py \\
        --manifest data/processed/val_manifest.jsonl \\
        --parser-backend heuristic \\
        --output-dir data/parser_cache/vlm

This script is resumable: it skips utterances already in cache.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag3d.datasets.schemas import ParsedUtterance
from rag3d.parsers.base import BaseParser
from rag3d.parsers.heuristic_parser import HeuristicParser
from rag3d.parsers.structured_rule_parser import StructuredRuleParser
from rag3d.parsers.parse_quality import validate_parse_quality
from rag3d.utils.logging import setup_logging

log = logging.getLogger(__name__)


def load_manifest_utterances(manifest_path: Path) -> List[Dict[str, Any]]:
    """
    Load utterances from a manifest file.

    Returns:
        List of dicts with 'utterance', 'scene_id', 'target_object_id'
    """
    samples = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                samples.append({
                    "utterance": data.get("utterance", ""),
                    "scene_id": data.get("scene_id", ""),
                    "target_object_id": data.get("target_object_id", ""),
                    "target_index": data.get("target_index", -1),
                })
    return samples


def deduplicate_utterances(samples: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Group samples by utterance for deduplication.

    Returns:
        Dict mapping utterance -> list of samples with that utterance
    """
    utterance_to_samples: Dict[str, List[Dict[str, Any]]] = {}
    for sample in samples:
        utterance = sample["utterance"]
        if utterance not in utterance_to_samples:
            utterance_to_samples[utterance] = []
        utterance_to_samples[utterance].append(sample)
    return utterance_to_samples


def utterance_cache_key(utterance: str) -> str:
    """Generate cache key for utterance."""
    return hashlib.sha256(utterance.encode("utf-8")).hexdigest()


def check_cache_exists(cache_dir: Path, utterance: str) -> bool:
    """Check if a cached parse already exists for utterance."""
    key = utterance_cache_key(utterance)
    cache_path = cache_dir / f"{key}.json"
    return cache_path.is_file()


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
            log.warning(f"Failed to load cached parse: {e}")
    return None


def save_cached_parse(cache_dir: Path, utterance: str, parsed: ParsedUtterance) -> Path:
    """Save parse to cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = utterance_cache_key(utterance)
    cache_path = cache_dir / f"{key}.json"
    with cache_path.open("w", encoding="utf-8") as f:
        f.write(parsed.model_dump_json(indent=2))
    return cache_path


def build_parser_backend(backend: str) -> BaseParser:
    """Build parser backend instance."""
    if backend == "heuristic":
        return HeuristicParser()
    elif backend == "structured_rule":
        return StructuredRuleParser()
    else:
        raise ValueError(f"Unknown parser backend: {backend}")


def generate_cache_for_utterance(
    utterance: str,
    parser: BaseParser,
    parser_source: str,
    cache_version: str,
) -> ParsedUtterance:
    """
    Generate a cached parse for a single utterance.

    Args:
        utterance: The utterance to parse
        parser: Parser backend instance
        parser_source: Source label (e.g., "heuristic", "cached_vlm")
        cache_version: Version string for cache metadata

    Returns:
        ParsedUtterance with populated fields
    """
    # Parse using backend
    parsed = parser.parse(utterance)

    # Override source label
    parsed.parse_source = parser_source

    # Validate quality
    parsed.parse_status = validate_parse_quality(parsed)

    # Add metadata
    if parsed.vlm_metadata is None:
        parsed.vlm_metadata = {}
    parsed.vlm_metadata.update({
        "cache_version": cache_version,
        "generation_timestamp": datetime.now().isoformat(),
        "parser_backend": type(parser).__name__,
    })

    return parsed


def generate_parser_cache(
    manifest_paths: List[Path],
    output_dir: Path,
    parser_backend: str = "heuristic",
    parser_source_label: str = "cached_vlm",
    cache_version: str = "1.0",
    skip_existing: bool = True,
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Generate parser cache for all utterances in manifest files.

    Args:
        manifest_paths: List of manifest file paths
        output_dir: Directory to write cache files
        parser_backend: Parser backend to use ("heuristic", "structured_rule")
        parser_source_label: Label for parse_source field
        cache_version: Version string for cache metadata
        skip_existing: Skip utterances already in cache
        max_samples: Maximum utterances to process (for testing)

    Returns:
        Dict with generation statistics
    """
    setup_logging()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build parser
    parser = build_parser_backend(parser_backend)
    log.info(f"Using parser backend: {type(parser).__name__}")

    # Load all utterances
    all_samples: List[Dict[str, Any]] = []
    for manifest_path in manifest_paths:
        if manifest_path.is_file():
            samples = load_manifest_utterances(manifest_path)
            log.info(f"Loaded {len(samples)} samples from {manifest_path}")
            all_samples.extend(samples)
        else:
            log.warning(f"Manifest not found: {manifest_path}")

    log.info(f"Total samples loaded: {len(all_samples)}")

    # Deduplicate by utterance
    utterance_to_samples = deduplicate_utterances(all_samples)
    unique_utterances = list(utterance_to_samples.keys())
    log.info(f"Unique utterances: {len(unique_utterances)}")

    if max_samples is not None:
        unique_utterances = unique_utterances[:max_samples]
        log.info(f"Limited to {max_samples} utterances")

    # Process each utterance
    stats = {
        "total_unique_utterances": len(unique_utterances),
        "already_cached": 0,
        "newly_parsed": 0,
        "parse_errors": 0,
        "parse_status_counts": {"valid": 0, "partial": 0, "invalid": 0, "missing": 0},
        "parser_backend": parser_backend,
        "cache_version": cache_version,
        "generation_timestamp": datetime.now().isoformat(),
    }

    generation_log: List[Dict[str, Any]] = []

    for i, utterance in enumerate(unique_utterances):
        if (i + 1) % 50 == 0:
            log.info(f"Processing utterance {i + 1}/{len(unique_utterances)}")

        # Check if already cached
        if skip_existing and check_cache_exists(output_dir, utterance):
            stats["already_cached"] += 1
            continue

        # Generate parse
        try:
            parsed = generate_cache_for_utterance(
                utterance=utterance,
                parser=parser,
                parser_source=parser_source_label,
                cache_version=cache_version,
            )

            # Save to cache
            cache_path = save_cached_parse(output_dir, utterance, parsed)
            stats["newly_parsed"] += 1
            stats["parse_status_counts"][parsed.parse_status] += 1

            # Log entry
            log_entry = {
                "utterance": utterance[:100],
                "cache_key": utterance_cache_key(utterance)[:16],
                "parse_status": parsed.parse_status,
                "target_head": parsed.target_head,
                "anchor_head": parsed.anchor_head,
                "relation_types": parsed.relation_types,
                "parser_confidence": parsed.parser_confidence,
            }
            generation_log.append(log_entry)

        except Exception as e:
            log.error(f"Error parsing utterance: {utterance[:50]}: {e}")
            stats["parse_errors"] += 1

            # Create error record
            error_parsed = ParsedUtterance(
                raw_text=utterance,
                parse_source=parser_source_label,
                parse_status="invalid",
                parse_warnings=[f"generation_error: {str(e)[:100]}"],
                parser_confidence=0.0,
                vlm_metadata={
                    "cache_version": cache_version,
                    "generation_timestamp": datetime.now().isoformat(),
                    "error": str(e)[:200],
                },
            )
            save_cached_parse(output_dir, utterance, error_parsed)
            stats["parse_status_counts"]["invalid"] += 1

    # Export generation log
    log_path = output_dir / "generation_log.json"
    with log_path.open("w", encoding="utf-8") as f:
        json.dump(generation_log, f, indent=2)
    log.info(f"Exported generation log to {log_path}")

    # Export statistics
    stats_path = output_dir / "generation_stats.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    log.info(f"Exported statistics to {stats_path}")

    # Print summary
    print(f"\nGeneration Summary:")
    print(f"  Total unique utterances: {stats['total_unique_utterances']}")
    print(f"  Already cached (skipped): {stats['already_cached']}")
    print(f"  Newly parsed: {stats['newly_parsed']}")
    print(f"  Parse errors: {stats['parse_errors']}")
    print(f"  Parse status distribution:")
    for status, count in stats["parse_status_counts"].items():
        print(f"    {status}: {count}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Generate real parser cache from dataset utterances"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        nargs="+",
        required=True,
        help="Path(s) to manifest file(s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data/parser_cache/vlm",
        help="Output directory for cache files",
    )
    parser.add_argument(
        "--parser-backend",
        type=str,
        choices=["heuristic", "structured_rule"],
        default="heuristic",
        help="Parser backend to use",
    )
    parser.add_argument(
        "--parser-source-label",
        type=str,
        default="cached_vlm",
        help="Label for parse_source field",
    )
    parser.add_argument(
        "--cache-version",
        type=str,
        default="1.0",
        help="Cache version string",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Regenerate even if cache exists",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum utterances to process (for testing)",
    )
    args = parser.parse_args()

    generate_parser_cache(
        manifest_paths=args.manifest,
        output_dir=args.output_dir,
        parser_backend=args.parser_backend,
        parser_source_label=args.parser_source_label,
        cache_version=args.cache_version,
        skip_existing=not args.force,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()