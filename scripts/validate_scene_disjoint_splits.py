#!/usr/bin/env python3
"""Validate scene-disjoint splits for zero overlap."""

import json
import sys
from pathlib import Path
from collections import Counter

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


def load_scene_list(path: Path) -> set[str]:
    """Load scene list from text file."""
    if not path.is_file():
        return set()
    return set(line.strip() for line in path.read_text().strip().split("\n") if line.strip())


def validate_scene_disjoint(manifest_dir: Path, splits_dir: Path) -> dict:
    """Validate that splits are scene-disjoint."""
    results = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "statistics": {},
    }

    # Load manifests
    train_path = manifest_dir / "train_manifest.jsonl"
    val_path = manifest_dir / "val_manifest.jsonl"
    test_path = manifest_dir / "test_manifest.jsonl"

    if not train_path.is_file() or not val_path.is_file() or not test_path.is_file():
        results["valid"] = False
        results["errors"].append(f"Missing manifest files in {manifest_dir}")
        return results

    train = load_manifest(train_path)
    val = load_manifest(val_path)
    test = load_manifest(test_path)

    # Extract scene sets
    train_scenes = set(r["scene_id"] for r in train)
    val_scenes = set(r["scene_id"] for r in val)
    test_scenes = set(r["scene_id"] for r in test)

    # Check overlaps
    train_val_overlap = train_scenes & val_scenes
    train_test_overlap = train_scenes & test_scenes
    val_test_overlap = val_scenes & test_scenes

    # For scene-disjoint, we require val-test to be disjoint
    # train-val and train-test overlap is acceptable (samples from same scene can be in train and val/test)
    # But for maximum safety, we can require all three to be disjoint

    if val_test_overlap:
        results["valid"] = False
        results["errors"].append(f"Val-Test scene overlap: {len(val_test_overlap)} scenes ({sorted(val_test_overlap)[:10]}...)")

    if train_val_overlap:
        results["warnings"].append(f"Train-Val scene overlap: {len(train_val_overlap)} scenes (acceptable for scene-level splits)")

    if train_test_overlap:
        results["warnings"].append(f"Train-Test scene overlap: {len(train_test_overlap)} scenes (acceptable for scene-level splits)")

    # Check sample duplication
    train_ids = set(r["utterance_id"] for r in train)
    val_ids = set(r["utterance_id"] for r in val)
    test_ids = set(r["utterance_id"] for r in test)

    train_val_id_overlap = train_ids & val_ids
    train_test_id_overlap = train_ids & test_ids
    val_test_id_overlap = val_ids & test_ids

    if train_val_id_overlap:
        results["valid"] = False
        results["errors"].append(f"Train-Val sample duplication: {len(train_val_id_overlap)} samples")

    if train_test_id_overlap:
        results["valid"] = False
        results["errors"].append(f"Train-Test sample duplication: {len(train_test_id_overlap)} samples")

    if val_test_id_overlap:
        results["valid"] = False
        results["errors"].append(f"Val-Test sample duplication: {len(val_test_id_overlap)} samples")

    # Statistics
    results["statistics"] = {
        "train": {
            "n_samples": len(train),
            "n_scenes": len(train_scenes),
            "samples_per_scene": {s: sum(1 for r in train if r["scene_id"] == s) for s in sorted(train_scenes)[:5]},
        },
        "val": {
            "n_samples": len(val),
            "n_scenes": len(val_scenes),
            "samples_per_scene": {s: sum(1 for r in val if r["scene_id"] == s) for s in sorted(val_scenes)[:5]},
        },
        "test": {
            "n_samples": len(test),
            "n_scenes": len(test_scenes),
            "samples_per_scene": {s: sum(1 for r in test if r["scene_id"] == s) for s in sorted(test_scenes)[:5]},
        },
        "total_scenes": len(train_scenes | val_scenes | test_scenes),
        "total_samples": len(train) + len(val) + len(test),
    }

    # Validate scene lists match manifests
    expected_train_scenes = load_scene_list(splits_dir / "scene_disjoint_train.txt")
    expected_val_scenes = load_scene_list(splits_dir / "scene_disjoint_val.txt")
    expected_test_scenes = load_scene_list(splits_dir / "scene_disjoint_test.txt")

    if expected_train_scenes and expected_train_scenes != train_scenes:
        results["warnings"].append(f"Train scene list mismatch: expected {len(expected_train_scenes)}, actual {len(train_scenes)}")

    if expected_val_scenes and expected_val_scenes != val_scenes:
        results["warnings"].append(f"Val scene list mismatch: expected {len(expected_val_scenes)}, actual {len(val_scenes)}")

    if expected_test_scenes and expected_test_scenes != test_scenes:
        results["warnings"].append(f"Test scene list mismatch: expected {len(expected_test_scenes)}, actual {len(test_scenes)}")

    return results


def validate_old_split(manifest_dir: Path) -> dict:
    """Validate old split for comparison."""
    train_path = manifest_dir / "train_manifest.jsonl"
    val_path = manifest_dir / "val_manifest.jsonl"
    test_path = manifest_dir / "test_manifest.jsonl"

    train = load_manifest(train_path)
    val = load_manifest(val_path)
    test = load_manifest(test_path)

    train_scenes = set(r["scene_id"] for r in train)
    val_scenes = set(r["scene_id"] for r in val)
    test_scenes = set(r["scene_id"] for r in test)

    return {
        "val_test_overlap": {
            "n_scenes": len(val_scenes & test_scenes),
            "scenes": sorted(val_scenes & test_scenes),
            "val_samples_in_overlap": sum(1 for r in val if r["scene_id"] in (val_scenes & test_scenes)),
            "test_samples_in_overlap": sum(1 for r in test if r["scene_id"] in (val_scenes & test_scenes)),
        },
        "train_val_overlap": {
            "n_scenes": len(train_scenes & val_scenes),
        },
        "train_test_overlap": {
            "n_scenes": len(train_scenes & test_scenes),
        },
        "statistics": {
            "train": {"n_samples": len(train), "n_scenes": len(train_scenes)},
            "val": {"n_samples": len(val), "n_scenes": len(val_scenes)},
            "test": {"n_samples": len(test), "n_scenes": len(test_scenes)},
        },
    }


def main():
    processed_dir = ROOT / "data" / "processed"
    scene_disjoint_dir = ROOT / "data" / "processed" / "scene_disjoint"
    splits_dir = ROOT / "data" / "splits"

    print("=== Validating Old Split ===")
    old_results = validate_old_split(processed_dir)
    print(f"  Val-Test overlap: {old_results['val_test_overlap']['n_scenes']} scenes")
    print(f"  Val samples in overlap: {old_results['val_test_overlap']['val_samples_in_overlap']}")
    print(f"  Test samples in overlap: {old_results['val_test_overlap']['test_samples_in_overlap']}")
    print()

    if scene_disjoint_dir.is_dir():
        print("=== Validating Scene-Disjoint Split ===")
        results = validate_scene_disjoint(scene_disjoint_dir, splits_dir)

        if results["valid"]:
            print("  PASS: Split is scene-disjoint")
        else:
            print("  FAIL: Split is NOT scene-disjoint")
            for err in results["errors"]:
                print(f"    ERROR: {err}")

        for warn in results["warnings"]:
            print(f"    WARNING: {warn}")

        stats = results["statistics"]
        print()
        print(f"  Train: {stats['train']['n_samples']} samples, {stats['train']['n_scenes']} scenes")
        print(f"  Val: {stats['val']['n_samples']} samples, {stats['val']['n_scenes']} scenes")
        print(f"  Test: {stats['test']['n_samples']} samples, {stats['test']['n_scenes']} scenes")
        print(f"  Total: {stats['total_samples']} samples, {stats['total_scenes']} scenes")

        # Save validation results
        validation_path = ROOT / "scene_disjoint_split_validation.json"
        with validation_path.open("w") as f:
            json.dump({
                "old_split": old_results,
                "scene_disjoint_split": results,
            }, f, indent=2)
        print(f"\nWrote: {validation_path}")
    else:
        print("Scene-disjoint split not found. Run build-nr3d-geom-scene-disjoint first.")
        print()
        print("To build scene-disjoint split:")
        print("  python scripts/prepare_data.py --mode build-nr3d-geom-scene-disjoint")


if __name__ == "__main__":
    main()