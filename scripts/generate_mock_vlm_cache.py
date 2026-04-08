#!/usr/bin/env python3
"""Generate mock VLM parser cache for testing.

Creates synthetic VLM parse records in data/parser_cache/vlm/
for testing the Phase 3 parser ablation pipeline.

This is for development/testing only. Real VLM parses should be
generated using a separate script with actual VLM API calls.
"""

import argparse
import hashlib
import json
import logging
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag3d.utils.logging import setup_logging

log = logging.getLogger(__name__)

# Sample utterances and mock parse templates
MOCK_PARSE_TEMPLATES = [
    {
        "template": "the {target} {relation} the {anchor}",
        "targets": ["chair", "table", "lamp", "couch", "desk", "bed", "door", "window", "shelf", "cabinet"],
        "anchors": ["table", "chair", "wall", "door", "window", "desk", "lamp", "shelf"],
        "relations": ["left of", "right of", "in front of", "behind", "next to", "near", "above", "below"],
    },
    {
        "template": "the {target} on the {anchor}",
        "targets": ["cup", "book", "lamp", "vase", "bottle", "pillow", "blanket"],
        "anchors": ["table", "desk", "shelf", "bed", "counter"],
        "relations": ["on"],
    },
    {
        "template": "the {target} between the {anchor1} and the {anchor2}",
        "targets": ["chair", "lamp", "table", "plant"],
        "anchors": ["table", "chair", "lamp", "wall", "door"],
        "relations": ["between"],
    },
]


def generate_mock_vlm_parse(utterance: str, quality: str = "good") -> dict:
    """
    Generate a mock VLM parse for an utterance.

    Args:
        utterance: The utterance to parse
        quality: "good", "partial", or "bad" - controls parse quality

    Returns:
        Dict with ParsedUtterance fields
    """
    # Determine confidence based on quality
    if quality == "good":
        confidence = random.uniform(0.75, 0.95)
    elif quality == "partial":
        confidence = random.uniform(0.45, 0.65)
    else:  # bad
        confidence = random.uniform(0.1, 0.4)

    # Parse the utterance with simple heuristics
    words = utterance.lower().split()

    # Find target (usually first noun after "the")
    target_head = None
    target_modifiers = []
    for i, w in enumerate(words):
        if w in ("the", "a", "an") and i + 1 < len(words):
            candidate = words[i + 1]
            if candidate not in ("left", "right", "front", "behind", "next", "near", "on", "above", "below", "between"):
                target_head = candidate
                # Look for modifiers before target
                for j in range(i + 1):
                    if words[j] in ("red", "blue", "green", "white", "black", "brown", "large", "small", "big", "tall", "short"):
                        target_modifiers.append(words[j])
                break

    # Find relation
    relation_types = []
    rel_keywords = {
        "left": "left-of",
        "right": "right-of",
        "front": "front-of",
        "behind": "behind",
        "next": "next-to",
        "near": "next-to",
        "beside": "next-to",
        "above": "above",
        "below": "below",
        "on": "on",
        "under": "below",
        "between": "between",
    }
    for w in words:
        if w in rel_keywords:
            relation_types.append(rel_keywords[w])

    # Find anchor (usually after relation keyword)
    anchor_head = None
    for i, w in enumerate(words):
        if w in ("of", "to", "the", "and") and i + 1 < len(words):
            candidate = words[i + 1]
            if candidate not in relation_types and candidate != target_head:
                anchor_head = candidate
                break

    # Apply quality degradation
    if quality == "partial":
        # Missing anchor but has relations
        if relation_types and anchor_head:
            if random.random() > 0.5:
                anchor_head = None
    elif quality == "bad":
        # Missing critical fields
        if random.random() > 0.3:
            target_head = None
        if random.random() > 0.3:
            anchor_head = None
        if random.random() > 0.5:
            relation_types = []
        confidence = random.uniform(0.1, 0.35)

    # Build parse record
    parse = {
        "raw_text": utterance,
        "target_head": target_head,
        "target_modifiers": target_modifiers,
        "anchor_head": anchor_head,
        "relation_types": relation_types if relation_types else ["none"],
        "parser_confidence": confidence,
        "paraphrase_set": [],
        "parse_source": "cached_vlm",
        "parse_warnings": [],
        "parse_status": "unknown",  # Will be set by validator
        "vlm_metadata": {
            "model_version": "mock_vlm_v1",
            "generated_by": "mock_vlm_cache_generator",
        },
    }

    return parse


def generate_mock_cache(
    cache_dir: Path,
    num_samples: int = 100,
    quality_distribution: tuple = (0.7, 0.2, 0.1),  # good, partial, bad
    seed: int = 42,
) -> dict:
    """
    Generate mock VLM parser cache.

    Args:
        cache_dir: Directory to write cache files
        num_samples: Number of mock samples to generate
        quality_distribution: Tuple of (good, partial, bad) proportions
        seed: Random seed

    Returns:
        Dict with statistics
    """
    random.seed(seed)
    setup_logging()

    vlm_cache_dir = cache_dir / "vlm"
    vlm_cache_dir.mkdir(parents=True, exist_ok=True)

    # Generate utterances from templates
    utterances = []
    for template in MOCK_PARSE_TEMPLATES:
        for target in template["targets"]:
            if "anchor1" in template["template"]:
                # Between case
                for anchor1 in template["anchors"]:
                    for anchor2 in template["anchors"]:
                        if anchor1 != anchor2:
                            utterance = template["template"].format(
                                target=target,
                                anchor1=anchor1,
                                anchor2=anchor2,
                                relation="between",
                            )
                            utterances.append(utterance)
            else:
                for anchor in template["anchors"]:
                    for relation in template["relations"]:
                        utterance = template["template"].format(
                            target=target,
                            anchor=anchor,
                            relation=relation,
                        )
                        utterances.append(utterance)

    # Sample and add random variations
    sampled = random.sample(utterances, min(num_samples, len(utterances)))

    # Add quality variations
    good_frac, partial_frac, bad_frac = quality_distribution
    quality_labels = (
        ["good"] * int(num_samples * good_frac)
        + ["partial"] * int(num_samples * partial_frac)
        + ["bad"] * int(num_samples * bad_frac)
    )
    random.shuffle(quality_labels)

    # Generate and cache parses
    stats = {"good": 0, "partial": 0, "bad": 0, "total": 0}
    for i, utterance in enumerate(sampled):
        quality = quality_labels[i] if i < len(quality_labels) else "good"

        parse = generate_mock_vlm_parse(utterance, quality)
        stats[quality] += 1
        stats["total"] += 1

        # Generate cache key
        key = hashlib.sha256(utterance.encode("utf-8")).hexdigest()
        cache_path = vlm_cache_dir / f"{key}.json"

        with cache_path.open("w") as f:
            json.dump(parse, f, indent=2)

    log.info(f"Generated {stats['total']} mock VLM parses in {vlm_cache_dir}")
    log.info(f"Quality distribution: good={stats['good']}, partial={stats['partial']}, bad={stats['bad']}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Generate mock VLM parser cache")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=ROOT / "data/parser_cache",
        help="Cache directory",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of mock samples to generate",
    )
    parser.add_argument(
        "--good-fraction",
        type=float,
        default=0.7,
        help="Fraction of good quality parses",
    )
    parser.add_argument(
        "--partial-fraction",
        type=float,
        default=0.2,
        help="Fraction of partial quality parses",
    )
    parser.add_argument(
        "--bad-fraction",
        type=float,
        default=0.1,
        help="Fraction of bad quality parses",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    args = parser.parse_args()

    quality_dist = (args.good_fraction, args.partial_fraction, args.bad_fraction)

    stats = generate_mock_cache(
        cache_dir=args.cache_dir,
        num_samples=args.num_samples,
        quality_distribution=quality_dist,
        seed=args.seed,
    )

    print(f"Generated mock VLM cache with {stats['total']} samples")
    print(f"  Good: {stats['good']}")
    print(f"  Partial: {stats['partial']}")
    print(f"  Bad: {stats['bad']}")


if __name__ == "__main__":
    main()