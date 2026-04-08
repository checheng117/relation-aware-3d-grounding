#!/usr/bin/env python3
"""Analyze train/val/test split distributions for distribution mismatch investigation."""

import json
import sys
from pathlib import Path
from collections import Counter, defaultdict
import statistics

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

def load_manifest(path: Path) -> list[dict]:
    """Load JSONL manifest."""
    records = []
    with path.open("r") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records

def analyze_split(records: list[dict], split_name: str) -> dict:
    """Compute distribution statistics for a split."""
    # Basic counts
    n_samples = len(records)
    scenes = set(r["scene_id"] for r in records)
    n_scenes = len(scenes)

    # Utterance lengths
    utterance_lengths = [len(r["utterance"].split()) for r in records]

    # Target class distribution
    target_classes = Counter()
    for r in records:
        target_idx = r["target_index"]
        if target_idx < len(r["objects"]):
            target_classes[r["objects"][target_idx]["class_name"]] += 1

    # Candidate set sizes
    candidate_sizes = [len(r["objects"]) for r in records]

    # Same-class clutter analysis
    same_class_clutter_count = 0
    for r in records:
        target_idx = r["target_index"]
        if target_idx < len(r["objects"]):
            target_class = r["objects"][target_idx]["class_name"]
            same_class = sum(1 for o in r["objects"] if o["class_name"] == target_class)
            if same_class > 1:
                same_class_clutter_count += 1

    # Hard case tags
    hard_tags = defaultdict(int)
    for r in records:
        for k, v in (r.get("tags") or {}).items():
            if v:
                hard_tags[k] += 1

    # Relation keywords in utterances
    relation_keywords = ["left", "right", "front", "behind", "back", "near", "far",
                         "above", "below", "top", "bottom", "between", "next", "corner"]
    relation_counts = Counter()
    for r in records:
        utt_lower = r["utterance"].lower()
        for kw in relation_keywords:
            if kw in utt_lower:
                relation_counts[kw] += 1

    # Scene overlap analysis
    scenes_with_multiple_samples = Counter()
    for r in records:
        scenes_with_multiple_samples[r["scene_id"]] += 1

    return {
        "split": split_name,
        "n_samples": n_samples,
        "n_unique_scenes": n_scenes,
        "samples_per_scene_mean": statistics.mean(scenes_with_multiple_samples.values()) if scenes_with_multiple_samples else 0,
        "samples_per_scene_max": max(scenes_with_multiple_samples.values()) if scenes_with_multiple_samples else 0,
        "utterance_length_mean": statistics.mean(utterance_lengths) if utterance_lengths else 0,
        "utterance_length_std": statistics.stdev(utterance_lengths) if len(utterance_lengths) > 1 else 0,
        "utterance_length_min": min(utterance_lengths) if utterance_lengths else 0,
        "utterance_length_max": max(utterance_lengths) if utterance_lengths else 0,
        "target_class_top10": dict(target_classes.most_common(10)),
        "target_class_unique": len(target_classes),
        "candidate_size_mean": statistics.mean(candidate_sizes) if candidate_sizes else 0,
        "candidate_size_std": statistics.stdev(candidate_sizes) if len(candidate_sizes) > 1 else 0,
        "candidate_size_min": min(candidate_sizes) if candidate_sizes else 0,
        "candidate_size_max": max(candidate_sizes) if candidate_sizes else 0,
        "same_class_clutter_samples": same_class_clutter_count,
        "same_class_clutter_rate": same_class_clutter_count / n_samples if n_samples > 0 else 0,
        "hard_tags": dict(hard_tags),
        "relation_keyword_counts": dict(relation_counts),
        "scenes_list": sorted(list(scenes)),
    }

def compare_splits(train_stats: dict, val_stats: dict, test_stats: dict) -> dict:
    """Compute comparison metrics between splits."""
    def diff_pct(a, b):
        if a == 0 or b == 0:
            return 0.0
        return abs(a - b) / max(a, b) * 100

    return {
        "utterance_length_diff_val_test_pct": diff_pct(val_stats["utterance_length_mean"], test_stats["utterance_length_mean"]),
        "candidate_size_diff_val_test_pct": diff_pct(val_stats["candidate_size_mean"], test_stats["candidate_size_mean"]),
        "same_class_clutter_diff_val_test_pct": diff_pct(val_stats["same_class_clutter_rate"], test_stats["same_class_clutter_rate"]),
        "scene_overlap_val_test": len(set(val_stats["scenes_list"]) & set(test_stats["scenes_list"])),
        "scene_overlap_train_val": len(set(train_stats["scenes_list"]) & set(val_stats["scenes_list"])),
        "scene_overlap_train_test": len(set(train_stats["scenes_list"]) & set(test_stats["scenes_list"])),
        "val_test_scenes_disjoint": len(set(val_stats["scenes_list"]) & set(test_stats["scenes_list"])) == 0,
    }

def main():
    processed_dir = ROOT / "data" / "processed"

    train = load_manifest(processed_dir / "train_manifest.jsonl")
    val = load_manifest(processed_dir / "val_manifest.jsonl")
    test = load_manifest(processed_dir / "test_manifest.jsonl")

    train_stats = analyze_split(train, "train")
    val_stats = analyze_split(val, "val")
    test_stats = analyze_split(test, "test")

    comparison = compare_splits(train_stats, val_stats, test_stats)

    # Save full stats
    full_stats = {
        "train": train_stats,
        "val": val_stats,
        "test": test_stats,
        "comparison": comparison,
        "analysis_timestamp": "2026-04-06",
    }

    # Remove verbose scene lists from saved JSON
    for s in [train_stats, val_stats, test_stats]:
        s.pop("scenes_list", None)

    output_json = ROOT / "split_distribution_comparison.json"
    with output_json.open("w") as f:
        json.dump(full_stats, f, indent=2)
    print(f"Wrote: {output_json}")

    # Print summary
    print("\n=== Split Distribution Summary ===\n")
    for split_name, stats in [("Train", train_stats), ("Val", val_stats), ("Test", test_stats)]:
        print(f"{split_name}:")
        print(f"  Samples: {stats['n_samples']}")
        print(f"  Unique scenes: {stats['n_unique_scenes']}")
        print(f"  Utterance length (words): mean={stats['utterance_length_mean']:.2f}, std={stats['utterance_length_std']:.2f}")
        print(f"  Candidate size: mean={stats['candidate_size_mean']:.2f}, std={stats['candidate_size_std']:.2f}")
        print(f"  Same-class clutter rate: {stats['same_class_clutter_rate']*100:.1f}%")
        print(f"  Hard tags: {stats['hard_tags']}")
        print()

    print("=== Comparison ===")
    print(f"  Val-Test utterance length diff: {comparison['utterance_length_diff_val_test_pct']:.1f}%")
    print(f"  Val-Test candidate size diff: {comparison['candidate_size_diff_val_test_pct']:.1f}%")
    print(f"  Val-Test same-class clutter diff: {comparison['same_class_clutter_diff_val_test_pct']:.1f}%")
    print(f"  Scene overlap val-test: {comparison['scene_overlap_val_test']} scenes")
    print(f"  Val-Test scenes disjoint: {comparison['val_test_scenes_disjoint']}")

if __name__ == "__main__":
    main()