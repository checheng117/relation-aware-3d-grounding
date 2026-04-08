#!/usr/bin/env python3
"""Prepare BERT language features for ReferIt3D baseline.

This script generates BERT embeddings for all utterances in the dataset.
Uses DistilBERT by default for efficiency.

Output:
- Cached text embeddings for train/val/test splits
- Feature statistics and validation
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag3d.utils.logging import setup_logging

log = logging.getLogger(__name__)

# Default model
DEFAULT_MODEL = "distilbert-base-uncased"


def load_utterances(manifest_path: Path) -> List[Dict[str, Any]]:
    """Load utterances from manifest file."""
    utterances = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                utterances.append({
                    "utterance": data.get("utterance", ""),
                    "scene_id": data.get("scene_id", ""),
                    "target_object_id": data.get("target_object_id", ""),
                })
    return utterances


def encode_utterances_bert(
    utterances: List[str],
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 32,
    device: str = "cpu",
    max_length: int = 128,
) -> np.ndarray:
    """
    Encode utterances using BERT.

    Args:
        utterances: List of text strings
        model_name: HuggingFace model name
        batch_size: Batch size for encoding
        device: "cpu" or "cuda"
        max_length: Maximum token length

    Returns:
        Embeddings array [N, hidden_dim]
    """
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError:
        raise ImportError(
            "transformers not installed. Run: pip install transformers"
        )

    log.info(f"Loading model: {model_name}")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model = model.to(device)
    model.eval()

    hidden_dim = model.config.hidden_size
    log.info(f"Hidden dimension: {hidden_dim}")

    # Encode in batches
    all_embeddings = []

    with torch.no_grad():
        for i in range(0, len(utterances), batch_size):
            batch_texts = utterances[i : i + batch_size]

            # Tokenize
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )

            # Move to device
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            # Use [CLS] token embedding (first token)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(cls_embeddings)

            if (i + batch_size) % 500 == 0:
                log.info(f"Encoded {i + len(batch_texts)}/{len(utterances)} utterances")

    return np.concatenate(all_embeddings, axis=0)


def prepare_split_features(
    manifest_path: Path,
    output_path: Path,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 32,
    device: str = "cpu",
    max_length: int = 128,
) -> Dict[str, Any]:
    """
    Prepare BERT features for a single split.

    Returns:
        Statistics dictionary
    """
    log.info(f"Processing {manifest_path.name}")

    # Load utterances
    utterances = load_utterances(manifest_path)
    if not utterances:
        log.warning(f"No utterances found in {manifest_path}")
        return {"error": "no utterances"}

    texts = [u["utterance"] for u in utterances]

    # Encode
    embeddings = encode_utterances_bert(
        texts,
        model_name=model_name,
        batch_size=batch_size,
        device=device,
        max_length=max_length,
    )

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, embeddings)

    # Also save metadata
    meta_path = output_path.with_suffix(".json")
    meta = {
        "manifest": str(manifest_path),
        "model": model_name,
        "num_samples": len(utterances),
        "embedding_dim": embeddings.shape[1],
        "max_length": max_length,
    }
    with meta_path.open("w") as f:
        json.dump(meta, f, indent=2)

    log.info(f"Saved {len(utterances)} embeddings to {output_path}")

    return {
        "split": manifest_path.stem,
        "num_samples": len(utterances),
        "embedding_dim": embeddings.shape[1],
        "output_file": str(output_path),
    }


def prepare_all_splits(
    manifest_dir: Path,
    output_dir: Path,
    model_name: str = DEFAULT_MODEL,
    batch_size: int = 32,
    device: str = "cpu",
    max_length: int = 128,
) -> Dict[str, Any]:
    """
    Prepare BERT features for all splits.

    Returns:
        Overall statistics
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = ["train", "val", "test"]
    stats = {
        "model": model_name,
        "splits": {},
    }

    for split in splits:
        manifest_path = manifest_dir / f"{split}_manifest.jsonl"
        if not manifest_path.exists():
            log.warning(f"Manifest not found: {manifest_path}")
            continue

        output_path = output_dir / f"{split}_bert_embeddings.npy"

        split_stats = prepare_split_features(
            manifest_path=manifest_path,
            output_path=output_path,
            model_name=model_name,
            batch_size=batch_size,
            device=device,
            max_length=max_length,
        )

        stats["splits"][split] = split_stats

    # Save overall statistics
    stats_path = output_dir / "bert_feature_statistics.json"
    with stats_path.open("w") as f:
        json.dump(stats, f, indent=2)

    log.info(f"Saved statistics to {stats_path}")

    return stats


def validate_features(output_dir: Path) -> Dict[str, Any]:
    """Validate generated BERT features."""
    validation = {"splits": {}, "issues": []}

    for split in ["train", "val", "test"]:
        embed_path = output_dir / f"{split}_bert_embeddings.npy"
        meta_path = output_dir / f"{split}_bert_embeddings.json"

        if not embed_path.exists():
            validation["issues"].append(f"{split}: embeddings not found")
            continue

        embeddings = np.load(embed_path)

        with meta_path.open() as f:
            meta = json.load(f)

        validation["splits"][split] = {
            "shape": list(embeddings.shape),
            "mean": float(embeddings.mean()),
            "std": float(embeddings.std()),
            "min": float(embeddings.min()),
            "max": float(embeddings.max()),
            "metadata": meta,
        }

        # Check for issues
        if embeddings.shape[0] != meta.get("num_samples", -1):
            validation["issues"].append(
                f"{split}: sample count mismatch "
                f"({embeddings.shape[0]} vs {meta.get('num_samples')})"
            )

        if np.any(np.isnan(embeddings)):
            validation["issues"].append(f"{split}: NaN values detected")

    return validation


def main():
    parser = argparse.ArgumentParser(description="Prepare BERT language features")
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=ROOT / "data/processed",
        help="Directory containing manifest files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data/text_features",
        help="Output directory for features",
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    setup_logging()

    if args.validate_only:
        stats = validate_features(args.output_dir)
        print(json.dumps(stats, indent=2))
        return

    # Prepare features
    stats = prepare_all_splits(
        manifest_dir=args.manifest_dir,
        output_dir=args.output_dir,
        model_name=args.model,
        batch_size=args.batch_size,
        device=args.device,
        max_length=args.max_length,
    )

    # Print summary
    print(f"\nBERT Feature Preparation Summary:")
    print(f"  Model: {stats['model']}")
    for split, split_stats in stats["splits"].items():
        print(f"  {split}: {split_stats['num_samples']} samples, dim={split_stats['embedding_dim']}")
    print(f"  Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()