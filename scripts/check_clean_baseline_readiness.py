#!/usr/bin/env python3
"""Check that a clean baseline export is usable for pre-method validation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch


REQUIRED_LOGIT_FIELDS = {
    "base_logits",
    "base_margin",
    "base_entropy",
    "base_top1_logit",
    "base_top2_logit",
    "base_top1_tie_count",
    "target_logit",
    "target_rank",
}


def norm_text(text: str) -> str:
    return " ".join(str(text).strip().strip("'\"").lower().split())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def softmax(values: list[float]) -> list[float]:
    max_value = max(values)
    exps = [math.exp(v - max_value) for v in values]
    total = sum(exps)
    return [v / total for v in exps]


def entropy(probs: list[float]) -> float:
    return -sum(p * math.log(max(p, 1e-12)) for p in probs)


def close(a: float | None, b: float | None, tol: float = 1e-5) -> bool:
    if a is None or b is None:
        return a is b
    return abs(float(a) - float(b)) <= tol


def check_export(
    manifest_path: Path,
    predictions_path: Path,
    results_path: Path,
    checkpoint_path: Path | None,
) -> dict[str, Any]:
    manifest = load_jsonl(manifest_path)
    predictions = json.load(predictions_path.open())
    results = json.load(results_path.open())

    issues: list[str] = []
    warnings: list[str] = []

    if len(manifest) != len(predictions):
        issues.append(f"manifest/prediction count mismatch: {len(manifest)} vs {len(predictions)}")

    total = min(len(manifest), len(predictions))
    correct_at_1 = 0
    correct_at_5 = 0
    entropy_values: list[float] = []
    margin_values: list[float] = []
    tie_counts: list[int] = []

    for idx, (sample, pred) in enumerate(zip(manifest, predictions)):
        for field in REQUIRED_LOGIT_FIELDS:
            if field not in pred:
                issues.append(f"row {idx}: missing {field}")
                continue

        key_sample = (
            str(sample["scene_id"]),
            str(sample["target_object_id"]),
            norm_text(sample["utterance"]),
        )
        key_pred = (
            str(pred.get("scene_id")),
            str(pred.get("target_id")),
            norm_text(pred.get("utterance", "")),
        )
        if key_sample != key_pred:
            issues.append(f"row {idx}: manifest/prediction identity mismatch {key_sample} != {key_pred}")
            continue

        logits = pred.get("base_logits")
        if not isinstance(logits, list) or not logits:
            issues.append(f"row {idx}: invalid base_logits")
            continue
        if len(logits) != len(sample["objects"]):
            issues.append(
                f"row {idx}: base_logits length {len(logits)} != object count {len(sample['objects'])}"
            )
            continue

        target_index = int(sample["target_index"])
        order = sorted(range(len(logits)), key=lambda i: (-float(logits[i]), i))
        top1 = order[0]
        top5 = order[: min(5, len(order))]
        if int(pred["pred_top1"]) != top1:
            issues.append(f"row {idx}: pred_top1 {pred['pred_top1']} != logit argmax {top1}")
        exported_top5 = [int(x) for x in pred["pred_top5"]]
        if len(exported_top5) != len(top5) or len(set(exported_top5)) != len(exported_top5):
            issues.append(f"row {idx}: pred_top5 has invalid length or duplicates")
        elif any(index < 0 or index >= len(logits) for index in exported_top5):
            issues.append(f"row {idx}: pred_top5 contains out-of-range indices")
        else:
            kth_logit = sorted([float(v) for v in logits], reverse=True)[len(top5) - 1]
            if any(float(logits[index]) + 1e-6 < kth_logit for index in exported_top5):
                issues.append(f"row {idx}: pred_top5 contains an index below the logit top-k threshold")

        row_correct_at_1 = top1 == target_index
        row_correct_at_5 = target_index in top5
        if bool(pred["correct_at_1"]) != row_correct_at_1:
            issues.append(f"row {idx}: correct_at_1 flag mismatch")
        if bool(pred["correct_at_5"]) != row_correct_at_5:
            issues.append(f"row {idx}: correct_at_5 flag mismatch")
        correct_at_1 += int(row_correct_at_1)
        correct_at_5 += int(row_correct_at_5)

        margin = float(logits[order[0]] - logits[order[1]]) if len(order) >= 2 else None
        if not close(pred.get("base_margin"), margin):
            issues.append(f"row {idx}: base_margin mismatch")
        if margin is not None:
            margin_values.append(margin)

        probs = softmax([float(v) for v in logits])
        ent = entropy(probs)
        entropy_values.append(ent)
        if not close(pred.get("base_entropy"), ent):
            issues.append(f"row {idx}: base_entropy mismatch")

        target_logit = float(logits[target_index])
        best_possible_rank = sum(1 for value in logits if float(value) > target_logit) + 1
        worst_possible_rank = sum(1 for value in logits if float(value) >= target_logit)
        exported_rank = int(pred.get("target_rank"))
        if not (best_possible_rank <= exported_rank <= worst_possible_rank):
            issues.append(f"row {idx}: target_rank mismatch")

        top_logit = float(logits[order[0]])
        tie_count = sum(1 for value in logits if abs(float(value) - top_logit) <= 1e-6)
        tie_counts.append(tie_count)
        if int(pred.get("base_top1_tie_count")) != tie_count:
            issues.append(f"row {idx}: base_top1_tie_count mismatch")

    acc_at_1 = correct_at_1 / max(total, 1)
    acc_at_5 = correct_at_5 / max(total, 1)
    if not close(results.get("acc_at_1"), acc_at_1, tol=1e-9):
        issues.append(f"results acc_at_1 {results.get('acc_at_1')} != recomputed {acc_at_1}")
    if not close(results.get("acc_at_5"), acc_at_5, tol=1e-9):
        issues.append(f"results acc_at_5 {results.get('acc_at_5')} != recomputed {acc_at_5}")

    checkpoint_summary: dict[str, Any] | None = None
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        class_vocab = checkpoint.get("class_vocabulary") if isinstance(checkpoint, dict) else None
        class_to_idx = checkpoint.get("class_to_idx") if isinstance(checkpoint, dict) else None
        checkpoint_summary = {
            "checkpoint": str(checkpoint_path),
            "has_class_vocabulary": bool(class_vocab),
            "class_vocabulary_size": len(class_vocab) if class_vocab else None,
            "has_class_to_idx": bool(class_to_idx),
            "class_vocabulary_ordering": checkpoint.get("class_vocabulary_ordering")
            if isinstance(checkpoint, dict)
            else None,
        }
        if class_vocab:
            if class_vocab != sorted(class_vocab):
                issues.append("checkpoint class_vocabulary is not sorted")
            manifest_vocab = sorted({
                obj["class_name"]
                for sample in manifest
                for obj in sample.get("objects", [])
            })
            # The split vocabulary can be a subset of the all-split checkpoint vocabulary.
            missing_from_checkpoint = sorted(set(manifest_vocab) - set(class_vocab))
            if missing_from_checkpoint:
                issues.append(
                    "checkpoint class_vocabulary misses manifest classes: "
                    + ", ".join(missing_from_checkpoint[:10])
                )
        else:
            warnings.append("checkpoint does not store class_vocabulary; sorted rebuild fallback is required")

    return {
        "status": "PASS" if not issues else "FAIL",
        "inputs": {
            "manifest": str(manifest_path),
            "predictions": str(predictions_path),
            "results": str(results_path),
            "checkpoint": str(checkpoint_path) if checkpoint_path else None,
        },
        "total_checked": total,
        "acc_at_1": acc_at_1,
        "acc_at_5": acc_at_5,
        "mean_base_margin": sum(margin_values) / len(margin_values) if margin_values else None,
        "mean_base_entropy": sum(entropy_values) / len(entropy_values) if entropy_values else None,
        "max_top1_tie_count": max(tie_counts) if tie_counts else None,
        "rows_with_top1_ties": sum(1 for count in tie_counts if count > 1),
        "checkpoint_summary": checkpoint_summary,
        "issues": issues,
        "warnings": warnings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check clean baseline logits export readiness")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    summary = check_export(
        manifest_path=args.manifest,
        predictions_path=args.predictions,
        results_path=args.results,
        checkpoint_path=args.checkpoint,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"clean baseline readiness: {summary['status']}")
    print(f"  checked: {summary['total_checked']}")
    print(f"  Acc@1/Acc@5: {summary['acc_at_1']:.6f}/{summary['acc_at_5']:.6f}")
    print(f"  issues: {len(summary['issues'])}")
    print(f"  warnings: {len(summary['warnings'])}")
    print(f"  output: {args.output}")
    raise SystemExit(0 if summary["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
