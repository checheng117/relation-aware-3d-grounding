#!/usr/bin/env python3
"""Dense Scorer Fundamentals Analysis.

This script analyzes the Dense-no-cal-v1 model to answer:
- Q1-fundamental: Do relation scores have discriminative power?
- Q2-ranking: Does pair ranking work?
- Q3-worth: Is dense scorer line worth continuing?

Usage:
    python scripts/analyze_dense_fundamentals.py
"""

from __future__ import annotations

import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag3d.models.cover3d_model import Cover3DModel
from rag3d.models.cover3d_dense_relation import DenseRelationModule
from rag3d.utils.seed import set_seed

import torch
import torch.nn.functional as F

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


# ============================================================================
# Data Loading
# ============================================================================

def load_predictions(path: Path) -> List[Dict[str, Any]]:
    """Load predictions JSON."""
    with path.open() as f:
        return json.load(f)


def load_embeddings(path: Path) -> List[Dict[str, Any]]:
    """Load embeddings JSON."""
    with path.open() as f:
        return json.load(f)


def load_coverage(path: Path) -> Dict[str, Dict[str, Any]]:
    """Load coverage JSONL."""
    coverage = {}
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            key = (row["scene_id"], row["target_id"], row["utterance"])
            coverage[key] = row
    return coverage


# ============================================================================
# Model Loading
# ============================================================================

def load_dense_model(device: torch.device) -> DenseRelationModule:
    """Load DenseRelationModule from checkpoint."""
    model = DenseRelationModule(
        object_dim=320,
        language_dim=256,
        geometry_dim=6,
        hidden_dim=256,
        chunk_size=16,
        use_geometry=False,
        aggregation="weighted",
        use_focal=False,
    )

    checkpoint_path = ROOT / "outputs/cover3d_round1/best_model.pt"
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        # Filter dense_relation.* keys
        dense_state = {}
        for k, v in checkpoint.items():
            if k.startswith("dense_relation."):
                dense_state[k.replace("dense_relation.", "")] = v

        if dense_state:
            model.load_state_dict(dense_state, strict=False)
            log.info(f"Loaded dense_relation weights from {checkpoint_path}")
        else:
            log.warning("No dense_relation weights found in checkpoint")
    else:
        log.warning(f"Checkpoint not found at {checkpoint_path}")

    model = model.to(device)
    model.eval()
    return model


# ============================================================================
# Relation Score Extraction
# ============================================================================

def extract_relation_scores(
    model: DenseRelationModule,
    samples: List[Dict[str, Any]],
    device: torch.device,
    batch_size: int = 8,
) -> List[Dict[str, Any]]:
    """Extract relation scores and pair weights from samples."""
    model.eval()
    results = []

    with torch.no_grad():
        for i in range(0, len(samples), batch_size):
            batch = samples[i:i+batch_size]

            # Prepare tensors
            max_n = max(len(s["object_embeddings"]) for s in batch)
            B = len(batch)

            object_embeddings = torch.zeros(B, max_n, 320, device=device)
            lang_features = torch.zeros(B, 256, device=device)
            object_mask = torch.zeros(B, max_n, dtype=torch.bool, device=device)

            for j, sample in enumerate(batch):
                n = len(sample["object_embeddings"])
                obj_emb = np.array(sample["object_embeddings"], dtype=np.float32)
                lang_feat = np.array(sample["lang_features"], dtype=np.float32)

                object_embeddings[j, :n] = torch.from_numpy(obj_emb)
                lang_features[j] = torch.from_numpy(lang_feat)
                object_mask[j, :n] = True

            # Forward pass
            result = model(
                object_embeddings=object_embeddings,
                utterance_features=lang_features,
                candidate_mask=object_mask,
            )

            relation_scores = result["relation_scores"].cpu().numpy()  # [B, N]
            pair_weights = result["pair_weights"].cpu().numpy()  # [B, N, N]
            all_pair_scores = result["all_pair_scores"].cpu().numpy()  # [B, N, N]
            diagnostics = result["diagnostics"]

            for j, sample in enumerate(batch):
                n = len(sample["object_embeddings"])
                results.append({
                    "scene_id": sample["scene_id"],
                    "utterance": sample["utterance"],
                    "target_index": sample["target_index"],
                    "relation_scores": relation_scores[j, :n].tolist(),
                    "pair_weights": pair_weights[j, :n, :n].tolist(),
                    "all_pair_scores": all_pair_scores[j, :n, :n].tolist(),
                    "diagnostics": diagnostics,
                    "anchor_ids": sample.get("anchor_ids", []),
                    "anchor_count": sample.get("anchor_count", 0),
                    "subsets": sample.get("subsets", {}),
                })

    return results


# ============================================================================
# Analysis Functions
# ============================================================================

def analyze_score_distributions(
    relation_results: List[Dict[str, Any]],
    base_predictions: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Analyze relation score distributions."""

    # Match by (scene_id, utterance)
    base_map = {}
    for pred in base_predictions:
        key = (pred["scene_id"], pred["utterance"])
        base_map[key] = pred

    # Collect scores
    correct_target_scores = []
    wrong_target_scores = []
    top1_scores = []
    top2_scores = []
    margins = []

    base_correct_scores = []
    base_wrong_scores = []

    multi_anchor_scores = []
    relative_pos_scores = []
    easy_scores = []

    for res in relation_results:
        key = (res["scene_id"], res["utterance"])
        base_pred = base_map.get(key)

        scores = np.array(res["relation_scores"])
        target_idx = res["target_index"]

        if len(scores) == 0:
            continue

        # Valid scores (not -inf)
        valid_scores = scores[scores != float("-inf")]

        if len(valid_scores) == 0:
            continue

        # Score for correct target
        if target_idx < len(scores):
            correct_target_scores.append(scores[target_idx])

        # Top-1 and Top-2 scores
        sorted_scores = np.sort(valid_scores)[::-1]
        if len(sorted_scores) >= 1:
            top1_scores.append(sorted_scores[0])
        if len(sorted_scores) >= 2:
            top2_scores.append(sorted_scores[1])
        if len(sorted_scores) >= 2:
            margins.append(sorted_scores[0] - sorted_scores[1])

        # Wrong target scores (all except target)
        if target_idx < len(scores):
            wrong_mask = np.arange(len(scores)) != target_idx
            wrong_target_scores.extend(scores[wrong_mask][scores[wrong_mask] != float("-inf")].tolist())

        # Base-correct vs base-wrong
        if base_pred:
            if base_pred["base_correct"]:
                base_correct_scores.append(np.mean(valid_scores))
            else:
                base_wrong_scores.append(np.mean(valid_scores))

        # Subset analysis
        subsets = res.get("subsets", {})
        if subsets.get("multi_anchor", False) or res.get("anchor_count", 0) > 0:
            multi_anchor_scores.append(np.mean(valid_scores))
        elif subsets.get("relative_position", False):
            relative_pos_scores.append(np.mean(valid_scores))
        else:
            easy_scores.append(np.mean(valid_scores))

    return {
        "correct_target_scores": correct_target_scores,
        "wrong_target_scores": wrong_target_scores,
        "top1_scores": top1_scores,
        "top2_scores": top2_scores,
        "margins": margins,
        "base_correct_scores": base_correct_scores,
        "base_wrong_scores": base_wrong_scores,
        "multi_anchor_scores": multi_anchor_scores,
        "relative_pos_scores": relative_pos_scores,
        "easy_scores": easy_scores,
    }


def analyze_pair_ranking(
    relation_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Analyze pair ranking usefulness."""

    hit_at_1 = []
    hit_at_3 = []
    hit_at_5 = []

    # For multi-anchor samples
    multi_anchor_hit_at_1 = []
    multi_anchor_hit_at_3 = []

    # For single-anchor samples
    single_anchor_hit_at_1 = []

    for res in relation_results:
        pair_weights = np.array(res["pair_weights"])  # [N, N]
        target_idx = res["target_index"]
        anchor_count = res.get("anchor_count", 0)
        subsets = res.get("subsets", {})

        if len(pair_weights.shape) != 2:
            continue

        N = pair_weights.shape[0]
        if N < 2:
            continue

        # For each candidate i, find which anchor j it attends to most
        # pair_weights[i, j] = weight given to anchor j when scoring candidate i

        # Check if target candidate attends to any anchor in top positions
        if target_idx < N:
            target_weights = pair_weights[target_idx]  # [N]
            top_indices = np.argsort(target_weights)[::-1]

            # Hit@K: does target attend to itself or high-scoring candidates?
            # For now, just check if any position is attended
            hit_at_1.append(1 if target_idx in top_indices[:1] else 0)
            hit_at_3.append(1 if target_idx in top_indices[:3] else 0)
            hit_at_5.append(1 if target_idx in top_indices[:min(5, N)] else 0)

        # Multi-anchor analysis
        if anchor_count > 0 or subsets.get("multi_anchor", False):
            if target_idx < N:
                target_weights = pair_weights[target_idx]
                top_indices = np.argsort(target_weights)[::-1]
                multi_anchor_hit_at_1.append(1 if target_idx in top_indices[:1] else 0)
                multi_anchor_hit_at_3.append(1 if target_idx in top_indices[:3] else 0)
        else:
            if target_idx < N:
                target_weights = pair_weights[target_idx]
                top_indices = np.argsort(target_weights)[::-1]
                single_anchor_hit_at_1.append(1 if target_idx in top_indices[:1] else 0)

    return {
        "hit_at_1": np.mean(hit_at_1) if hit_at_1 else 0.0,
        "hit_at_3": np.mean(hit_at_3) if hit_at_3 else 0.0,
        "hit_at_5": np.mean(hit_at_5) if hit_at_5 else 0.0,
        "multi_anchor_hit_at_1": np.mean(multi_anchor_hit_at_1) if multi_anchor_hit_at_1 else 0.0,
        "multi_anchor_hit_at_3": np.mean(multi_anchor_hit_at_3) if multi_anchor_hit_at_3 else 0.0,
        "single_anchor_hit_at_1": np.mean(single_anchor_hit_at_1) if single_anchor_hit_at_1 else 0.0,
        "num_samples": len(hit_at_1),
        "num_multi_anchor": len(multi_anchor_hit_at_1),
        "num_single_anchor": len(single_anchor_hit_at_1),
    }


def analyze_contribution(
    relation_results: List[Dict[str, Any]],
    base_predictions: List[Dict[str, Any]],
    dense_predictions: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Analyze dense branch contribution."""

    # Match by (scene_id, utterance)
    base_map = {(p["scene_id"], p["utterance"]): p for p in base_predictions}
    dense_map = {(p["scene_id"], p["utterance"]): p for p in dense_predictions}

    recovered = []  # base wrong -> dense correct
    harmed = []     # base correct -> dense wrong
    both_correct = []
    both_wrong = []

    for res in relation_results:
        key = (res["scene_id"], res["utterance"])
        base_pred = base_map.get(key)
        dense_pred = dense_map.get(key)

        if not base_pred or not dense_pred:
            continue

        base_correct = base_pred["base_correct"]
        dense_correct = dense_pred["fused_correct"]

        subsets = res.get("subsets", {})
        anchor_count = res.get("anchor_count", 0)

        case = {
            "scene_id": res["scene_id"],
            "utterance": res["utterance"],
            "target_index": res["target_index"],
            "subsets": subsets,
            "anchor_count": anchor_count,
            "relation_scores": res["relation_scores"],
        }

        if base_correct and not dense_correct:
            harmed.append(case)
        elif not base_correct and dense_correct:
            recovered.append(case)
        elif base_correct and dense_correct:
            both_correct.append(case)
        else:
            both_wrong.append(case)

    # Analyze harmed cases
    harmed_subsets = defaultdict(int)
    for case in harmed:
        subsets = case["subsets"]
        if subsets.get("same_class_clutter", False):
            harmed_subsets["same_class_clutter"] += 1
        if subsets.get("multi_anchor", False) or case["anchor_count"] > 0:
            harmed_subsets["multi_anchor"] += 1
        if subsets.get("relative_position", False):
            harmed_subsets["relative_position"] += 1
        if not any([
            subsets.get("same_class_clutter", False),
            subsets.get("multi_anchor", False),
            subsets.get("relative_position", False),
        ]):
            harmed_subsets["easy"] += 1

    # Analyze recovered cases
    recovered_subsets = defaultdict(int)
    for case in recovered:
        subsets = case["subsets"]
        if subsets.get("same_class_clutter", False):
            recovered_subsets["same_class_clutter"] += 1
        if subsets.get("multi_anchor", False) or case["anchor_count"] > 0:
            recovered_subsets["multi_anchor"] += 1
        if subsets.get("relative_position", False):
            recovered_subsets["relative_position"] += 1
        if not any([
            subsets.get("same_class_clutter", False),
            subsets.get("multi_anchor", False),
            subsets.get("relative_position", False),
        ]):
            recovered_subsets["easy"] += 1

    return {
        "recovered": recovered,
        "harmed": harmed,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "recovered_count": len(recovered),
        "harmed_count": len(harmed),
        "net": len(recovered) - len(harmed),
        "harmed_subsets": dict(harmed_subsets),
        "recovered_subsets": dict(recovered_subsets),
    }


# ============================================================================
# Report Generation
# ============================================================================

def generate_score_audit_report(score_analysis: Dict[str, Any]) -> str:
    """Generate relation score quality audit report."""

    def percentile(arr, p):
        if not arr:
            return 0.0
        return np.percentile(arr, p)

    report = """# Dense Fundamentals: Relation Score Quality Audit

**Date**: 2026-04-22
**Model**: Dense-no-cal-v1

---

## Executive Summary

**Q1-fundamental Answer**: {q1_answer}

Relation scores show {signal_strength} discriminative power.

---

## 1. Score Distributions

### 1.1 Correct Target Scores

| Statistic | Value |
|-----------|-------|
| Count | {correct_count} |
| Mean | {correct_mean:.4f} |
| Std | {correct_std:.4f} |
| Median | {correct_median:.4f} |
| 25th %ile | {correct_p25:.4f} |
| 75th %ile | {correct_p75:.4f} |

### 1.2 Wrong Target Scores

| Statistic | Value |
|-----------|-------|
| Count | {wrong_count} |
| Mean | {wrong_mean:.4f} |
| Std | {wrong_std:.4f} |

### 1.3 Score Gap (Correct - Wrong Mean)

**Gap**: {score_gap:.4f}

Interpretation:
- Gap > 0.5: Strong discriminative signal
- Gap 0.1-0.5: Weak but real signal
- Gap < 0.1: Mostly noisy

---

## 2. Margin Analysis

### 2.1 Top-1 vs Top-2 Margin Distribution

| Statistic | Value |
|-----------|-------|
| Count | {margin_count} |
| Mean | {margin_mean:.4f} |
| Std | {margin_std:.4f} |
| Median | {margin_median:.4f} |
| % Positive | {margin_pos_pct:.2f}% |

Interpretation:
- Margin > 0.5: Confident top-1 selection
- Margin 0.1-0.5: Moderate confidence
- Margin < 0.1: Uncertain selection

---

## 3. Base-Correct vs Base-Wrong Analysis

| Subset | Mean Relation Score | Count |
|--------|--------------------|-------|
| Base-Correct | {base_correct_mean:.4f} | {base_correct_count} |
| Base-Wrong | {base_wrong_mean:.4f} | {base_wrong_count} |

**Difference**: {base_diff:.4f}

Interpretation:
- If Base-Wrong > Base-Correct: Dense helps where Base fails
- If Base-Correct > Base-Wrong: Dense amplifies already-easy cases

---

## 4. Subset-Level Analysis

| Subset | Mean Relation Score | Count |
|--------|--------------------|-------|
| Multi-Anchor | {multi_anchor_mean:.4f} | {multi_anchor_count} |
| Relative-Position | {relative_pos_mean:.4f} | {relative_pos_count} |
| Easy | {easy_mean:.4f} | {easy_count} |

---

## 5. Diagnostic Histograms (text-based)

### Correct Target Score Distribution
{correct_histogram}

### Wrong Target Score Distribution
{wrong_histogram}

### Margin Distribution
{margin_histogram}

---

## 6. Conclusion

**Signal Quality Assessment**:

1. **Score Separation**: {score_sep_assessment}
2. **Margin Confidence**: {margin_assessment}
3. **Subset Discrimination**: {subset_assessment}

**Q1-fundamental Final Answer**: {q1_final}

Rationale: {q1_rationale}
""".format(
        q1_answer="WEAK-BUT-REAL" if score_analysis.get("score_gap", 0) > 0.1 else "MOSTLY-NOISY",
        signal_strength="weak but real" if score_analysis.get("score_gap", 0) > 0.1 else "mostly noisy",
        correct_count=len(score_analysis.get("correct_target_scores", [])),
        correct_mean=np.mean(score_analysis.get("correct_target_scores", [0])),
        correct_std=np.std(score_analysis.get("correct_target_scores", [0])),
        correct_median=np.median(score_analysis.get("correct_target_scores", [0])),
        correct_p25=percentile(score_analysis.get("correct_target_scores", [0]), 25),
        correct_p75=percentile(score_analysis.get("correct_target_scores", [0]), 75),
        wrong_count=len(score_analysis.get("wrong_target_scores", [])),
        wrong_mean=np.mean(score_analysis.get("wrong_target_scores", [0])),
        wrong_std=np.std(score_analysis.get("wrong_target_scores", [0])),
        score_gap=score_analysis.get("score_gap", 0),
        margin_count=len(score_analysis.get("margins", [])),
        margin_mean=np.mean(score_analysis.get("margins", [0])),
        margin_std=np.std(score_analysis.get("margins", [0])),
        margin_median=np.median(score_analysis.get("margins", [0])),
        margin_pos_pct=100 * len([m for m in score_analysis.get("margins", []) if m > 0]) / max(1, len(score_analysis.get("margins", []))),
        base_correct_mean=np.mean(score_analysis.get("base_correct_scores", [0])),
        base_correct_count=len(score_analysis.get("base_correct_scores", [])),
        base_wrong_mean=np.mean(score_analysis.get("base_wrong_scores", [0])),
        base_wrong_count=len(score_analysis.get("base_wrong_scores", [])),
        base_diff=np.mean(score_analysis.get("base_correct_scores", [0])) - np.mean(score_analysis.get("base_wrong_scores", [0])),
        multi_anchor_mean=np.mean(score_analysis.get("multi_anchor_scores", [0])),
        multi_anchor_count=len(score_analysis.get("multi_anchor_scores", [])),
        relative_pos_mean=np.mean(score_analysis.get("relative_pos_scores", [0])),
        relative_pos_count=len(score_analysis.get("relative_pos_scores", [])),
        easy_mean=np.mean(score_analysis.get("easy_scores", [0])),
        easy_count=len(score_analysis.get("easy_scores", [])),
        correct_histogram=make_histogram(score_analysis.get("correct_target_scores", []), bins=10, width=50),
        wrong_histogram=make_histogram(score_analysis.get("wrong_target_scores", []), bins=10, width=50),
        margin_histogram=make_histogram(score_analysis.get("margins", []), bins=10, width=50),
        score_sep_assessment="Strong" if score_analysis.get("score_gap", 0) > 0.5 else ("Moderate" if score_analysis.get("score_gap", 0) > 0.1 else "Weak"),
        margin_assessment="Confident" if score_analysis.get("margin_mean", 0) > 0.5 else ("Moderate" if score_analysis.get("margin_mean", 0) > 0.1 else "Uncertain"),
        subset_assessment="Present" if score_analysis.get("multi_anchor_mean", 0) > score_analysis.get("easy_mean", 0) else "Absent",
        q1_final="WEAK-BUT-REAL" if score_analysis.get("score_gap", 0) > 0.1 else "MOSTLY-NOISY",
        q1_rationale=f"Score gap of {score_analysis.get('score_gap', 0):.4f} indicates {'discriminative but weak signal' if score_analysis.get('score_gap', 0) > 0.1 else 'limited discriminative power'}",
    )

    return report


def make_histogram(values: List[float], bins: int = 10, width: int = 50) -> str:
    """Create ASCII histogram."""
    if not values:
        return "(no data)"

    values = [v for v in values if v != float("-inf") and v != float("inf") and not np.isnan(v)]
    if not values:
        return "(no valid data)"

    hist, bin_edges = np.histogram(values, bins=bins)
    max_count = max(hist) if hist.max() > 0 else 1

    lines = []
    for i, (count, edge) in enumerate(zip(hist, bin_edges[:-1])):
        bar_len = int(count / max_count * width)
        lines.append(f"  [{edge:6.2f}-{bin_edges[i+1]:6.2f}] |{'█' * bar_len}| ({count})")

    return "\n".join(lines)


def generate_pair_audit_report(pair_analysis: Dict[str, Any]) -> str:
    """Generate pair ranking usefulness audit report."""

    report = """# Dense Fundamentals: Pair Ranking Usefulness Audit

**Date**: 2026-04-22
**Model**: Dense-no-cal-v1

---

## Executive Summary

**Q2-ranking Answer**: {q2_answer}

Pair ranking shows {ranking_quality} effectiveness.

---

## 1. Overall Hit@K Metrics

| Metric | Value |
|--------|-------|
| Hit@1 | {hit1:.2f}% |
| Hit@3 | {hit3:.2f}% |
| Hit@5 | {hit5:.2f}% |
| Samples | {num_samples} |

Interpretation:
- Hit@1 > 50%: Strong pair ranking
- Hit@1 20-50%: Moderate pair ranking
- Hit@1 < 20%: Weak pair ranking

---

## 2. Multi-Anchor vs Single-Anchor Analysis

| Subset | Hit@1 | Hit@3 | Count |
|--------|-------|-------|-------|
| Multi-Anchor | {ma_hit1:.2f}% | {ma_hit3:.2f}% | {ma_count} |
| Single-Anchor | {sa_hit1:.2f}% | N/A | {sa_count} |

---

## 3. Pair Weight Distribution Analysis

The pair weights represent attention from candidate i to anchor j.

**Key Questions**:
1. Do correct targets attend to appropriate anchors?
2. Are high pair weights concentrated or diffuse?

---

## 4. Case Analysis

### When Pair Ranking Works
- Multi-anchor cases with clear spatial relations
- Cases with distinctive anchor objects

### When Pair Ranking Fails
- Single-anchor cases (less benefit from pair modeling)
- High-clutter scenes (many similar objects)

---

## 5. Conclusion

**Pair Ranking Assessment**:

1. **Overall Effectiveness**: {overall_effectiveness}
2. **Multi-Anchor Benefit**: {ma_benefit}
3. **Single-Anchor Limitation**: {sa_limitation}

**Q2-ranking Final Answer**: {q2_final}

Rationale: {q2_rationale}
""".format(
        q2_answer="MODERATE" if pair_analysis.get("hit_at_1", 0) > 0.2 else "WEAK",
        ranking_quality="moderate" if pair_analysis.get("hit_at_1", 0) > 0.2 else "weak",
        hit1=100 * pair_analysis.get("hit_at_1", 0),
        hit3=100 * pair_analysis.get("hit_at_3", 0),
        hit5=100 * pair_analysis.get("hit_at_5", 0),
        num_samples=pair_analysis.get("num_samples", 0),
        ma_hit1=100 * pair_analysis.get("multi_anchor_hit_at_1", 0),
        ma_hit3=100 * pair_analysis.get("multi_anchor_hit_at_3", 0),
        ma_count=pair_analysis.get("num_multi_anchor", 0),
        sa_hit1=100 * pair_analysis.get("single_anchor_hit_at_1", 0),
        sa_count=pair_analysis.get("num_single_anchor", 0),
        overall_effectiveness="Moderate" if pair_analysis.get("hit_at_1", 0) > 0.2 else "Weak",
        ma_benefit="Present" if pair_analysis.get("multi_anchor_hit_at_1", 0) > 0.1 else "Limited",
        sa_limitation="Significant" if pair_analysis.get("single_anchor_hit_at_1", 0) < 0.3 else "Modest",
        q2_final="MODERATE" if pair_analysis.get("hit_at_1", 0) > 0.2 else "WEAK",
        q2_rationale=f"Hit@1 of {100 * pair_analysis.get('hit_at_1', 0):.1f}% indicates {'moderate' if pair_analysis.get('hit_at_1', 0) > 0.2 else 'weak'} pair ranking ability",
    )

    return report


def generate_contribution_audit_report(contribution_analysis: Dict[str, Any]) -> str:
    """Generate dense branch contribution audit report."""

    report = """# Dense Fundamentals: Dense Branch Contribution Audit

**Date**: 2026-04-22
**Model**: Dense-no-cal-v1

---

## Executive Summary

**Dense Branch Net Effect**: {net} (Recovered: {recovered}, Harmed: {harmed})

---

## 1. Overall Contribution

| Category | Count | Percentage |
|----------|-------|------------|
| Recovered (base→dense) | {recovered} | {recovered_pct:.2f}% |
| Harmed (base→dense) | {harmed} | {harmed_pct:.2f}% |
| Both Correct | {both_correct} | {both_correct_pct:.2f}% |
| Both Wrong | {both_wrong} | {both_wrong_pct:.2f}% |
| Total | {total} | 100% |

---

## 2. Harmed Case Analysis

### Distribution by Subset

| Subset | Count | % of Harmed |
|--------|-------|-------------|
| Same-Class Clutter | {harmed_clutter} | {harmed_clutter_pct:.1f}% |
| Multi-Anchor | {harmed_ma} | {harmed_ma_pct:.1f}% |
| Relative-Position | {harmed_rp} | {harmed_rp_pct:.1f}% |
| Easy | {harmed_easy} | {harmed_easy_pct:.1f}% |

### Key Finding: Where Dense Branch Hurts

{harmed_finding}

---

## 3. Recovered Case Analysis

### Distribution by Subset

| Subset | Count | % of Recovered |
|--------|-------|----------------|
| Same-Class Clutter | {recovered_clutter} | {recovered_clutter_pct:.1f}% |
| Multi-Anchor | {recovered_ma} | {recovered_ma_pct:.1f}% |
| Relative-Position | {recovered_rp} | {recovered_rp_pct:.1f}% |
| Easy | {recovered_easy} | {recovered_easy_pct:.1f}% |

### Key Finding: Where Dense Branch Helps

{recovered_finding}

---

## 4. Comparison: Harmed vs Recovered Patterns

| Pattern | Harmed | Recovered | Interpretation |
|---------|--------|-----------|----------------|
| Multi-Anchor Rate | {harmed_ma_rate:.1f}% | {recovered_ma_rate:.1f}% | {ma_interp} |
| Clutter Rate | {harmed_clutter_rate:.1f}% | {recovered_clutter_rate:.1f}% | {clutter_interp} |
| Relative-Position Rate | {harmed_rp_rate:.1f}% | {recovered_rp_rate:.1f}% | {rp_interp} |

---

## 5. Case Studies

### Examples of Harmed Cases (Base-Correct → Dense-Wrong)

{harmed_examples}

### Examples of Recovered Cases (Base-Wrong → Dense-Correct)

{recovered_examples}

---

## 6. Conclusion

**Dense Branch Assessment**:

1. **Net Effect**: {net_effect}
2. **Systematic Patterns**: {systematic}
3. **Actionable Insights**: {actionable}

**Route Recommendation**: {route_recommendation}
""".format(
        net=contribution_analysis.get("net", 0),
        recovered=contribution_analysis.get("recovered_count", 0),
        harmed=contribution_analysis.get("harmed_count", 0),
        total_samples=contribution_analysis.get("recovered_count", 0) + contribution_analysis.get("harmed_count", 0) + len(contribution_analysis.get("both_correct", [])) + len(contribution_analysis.get("both_wrong", [])),
        recovered_pct=100 * contribution_analysis.get("recovered_count", 0) / max(1, contribution_analysis.get("recovered_count", 0) + contribution_analysis.get("harmed_count", 0) + len(contribution_analysis.get("both_correct", [])) + len(contribution_analysis.get("both_wrong", []))),
        harmed_pct=100 * contribution_analysis.get("harmed_count", 0) / max(1, contribution_analysis.get("recovered_count", 0) + contribution_analysis.get("harmed_count", 0) + len(contribution_analysis.get("both_correct", [])) + len(contribution_analysis.get("both_wrong", []))),
        both_correct=len(contribution_analysis.get("both_correct", [])),
        both_wrong=len(contribution_analysis.get("both_wrong", [])),
        total=contribution_analysis.get("recovered_count", 0) + contribution_analysis.get("harmed_count", 0) + len(contribution_analysis.get("both_correct", [])) + len(contribution_analysis.get("both_wrong", [])),
        both_correct_pct=100 * len(contribution_analysis.get("both_correct", [])) / max(1, contribution_analysis.get("recovered_count", 0) + contribution_analysis.get("harmed_count", 0) + len(contribution_analysis.get("both_correct", [])) + len(contribution_analysis.get("both_wrong", []))),
        both_wrong_pct=100 * len(contribution_analysis.get("both_wrong", [])) / max(1, contribution_analysis.get("recovered_count", 0) + contribution_analysis.get("harmed_count", 0) + len(contribution_analysis.get("both_correct", [])) + len(contribution_analysis.get("both_wrong", []))),
        harmed_clutter=contribution_analysis.get("harmed_subsets", {}).get("same_class_clutter", 0),
        harmed_clutter_pct=100 * contribution_analysis.get("harmed_subsets", {}).get("same_class_clutter", 0) / max(1, contribution_analysis.get("harmed_count", 0)),
        harmed_ma=contribution_analysis.get("harmed_subsets", {}).get("multi_anchor", 0),
        harmed_ma_pct=100 * contribution_analysis.get("harmed_subsets", {}).get("multi_anchor", 0) / max(1, contribution_analysis.get("harmed_count", 0)),
        harmed_rp=contribution_analysis.get("harmed_subsets", {}).get("relative_position", 0),
        harmed_rp_pct=100 * contribution_analysis.get("harmed_subsets", {}).get("relative_position", 0) / max(1, contribution_analysis.get("harmed_count", 0)),
        harmed_easy=contribution_analysis.get("harmed_subsets", {}).get("easy", 0),
        harmed_easy_pct=100 * contribution_analysis.get("harmed_subsets", {}).get("easy", 0) / max(1, contribution_analysis.get("harmed_count", 0)),
        harmed_finding="Dense branch primarily harms easy cases (no hard subsets)" if contribution_analysis.get("harmed_subsets", {}).get("easy", 0) > contribution_analysis.get("harmed_subsets", {}).get("multi_anchor", 0) else "Dense branch harms multi-anchor/clutter cases",
        recovered_clutter=contribution_analysis.get("recovered_subsets", {}).get("same_class_clutter", 0),
        recovered_clutter_pct=100 * contribution_analysis.get("recovered_subsets", {}).get("same_class_clutter", 0) / max(1, contribution_analysis.get("recovered_count", 0)),
        recovered_ma=contribution_analysis.get("recovered_subsets", {}).get("multi_anchor", 0),
        recovered_ma_pct=100 * contribution_analysis.get("recovered_subsets", {}).get("multi_anchor", 0) / max(1, contribution_analysis.get("recovered_count", 0)),
        recovered_rp=contribution_analysis.get("recovered_subsets", {}).get("relative_position", 0),
        recovered_rp_pct=100 * contribution_analysis.get("recovered_subsets", {}).get("relative_position", 0) / max(1, contribution_analysis.get("recovered_count", 0)),
        recovered_easy=contribution_analysis.get("recovered_subsets", {}).get("easy", 0),
        recovered_easy_pct=100 * contribution_analysis.get("recovered_subsets", {}).get("easy", 0) / max(1, contribution_analysis.get("recovered_count", 0)),
        recovered_finding="Dense branch primarily helps multi-anchor/clutter cases" if contribution_analysis.get("recovered_subsets", {}).get("multi_anchor", 0) > contribution_analysis.get("recovered_subsets", {}).get("easy", 0) else "Dense branch helps easy cases",
        harmed_ma_rate=100 * contribution_analysis.get("harmed_subsets", {}).get("multi_anchor", 0) / max(1, contribution_analysis.get("harmed_count", 0)),
        recovered_ma_rate=100 * contribution_analysis.get("recovered_subsets", {}).get("multi_anchor", 0) / max(1, contribution_analysis.get("recovered_count", 0)),
        harmed_clutter_rate=100 * contribution_analysis.get("harmed_subsets", {}).get("same_class_clutter", 0) / max(1, contribution_analysis.get("harmed_count", 0)),
        recovered_clutter_rate=100 * contribution_analysis.get("recovered_subsets", {}).get("same_class_clutter", 0) / max(1, contribution_analysis.get("recovered_count", 0)),
        harmed_rp_rate=100 * contribution_analysis.get("harmed_subsets", {}).get("relative_position", 0) / max(1, contribution_analysis.get("harmed_count", 0)),
        recovered_rp_rate=100 * contribution_analysis.get("recovered_subsets", {}).get("relative_position", 0) / max(1, contribution_analysis.get("recovered_count", 0)),
        ma_interp="Dense helps multi-anchor more than it hurts" if contribution_analysis.get("recovered_subsets", {}).get("multi_anchor", 0) > contribution_analysis.get("harmed_subsets", {}).get("multi_anchor", 0) else "Dense hurts multi-anchor more than it helps",
        clutter_interp="Dense helps clutter more than it hurts" if contribution_analysis.get("recovered_subsets", {}).get("same_class_clutter", 0) > contribution_analysis.get("harmed_subsets", {}).get("same_class_clutter", 0) else "Dense hurts clutter more than it helps",
        rp_interp="Dense helps relative-position more than it hurts" if contribution_analysis.get("recovered_subsets", {}).get("relative_position", 0) > contribution_analysis.get("harmed_subsets", {}).get("relative_position", 0) else "Dense hurts relative-position more than it helps",
        harmed_examples=generate_examples(contribution_analysis.get("harmed", [])[:5]),
        recovered_examples=generate_examples(contribution_analysis.get("recovered", [])[:5]),
        net_effect="Positive but modest (+{})".format(contribution_analysis.get("net", 0)) if contribution_analysis.get("net", 0) > 0 else "Negative ({})".format(contribution_analysis.get("net", 0)),
        systematic="Yes - multi-anchor recovery, easy-case harm" if contribution_analysis.get("recovered_subsets", {}).get("multi_anchor", 0) > contribution_analysis.get("harmed_subsets", {}).get("multi_anchor", 0) else "No clear pattern",
        actionable="Focus on multi-anchor cases, protect easy cases from over-correction" if contribution_analysis.get("recovered_subsets", {}).get("multi_anchor", 0) > contribution_analysis.get("harmed_subsets", {}).get("multi_anchor", 0) else "Unclear how to improve",
        route_recommendation="Route B-fundamentals: Continue with redesign based on findings" if contribution_analysis.get("net", 0) > 0 and contribution_analysis.get("recovered_subsets", {}).get("multi_anchor", 0) > 0 else "Route C-hard-stop: Dense scorer line not worth continuing",
    )

    return report


def generate_examples(cases: List[Dict[str, Any]]) -> str:
    """Generate example cases for report."""
    if not cases:
        return "(no examples)"

    lines = []
    for case in cases[:5]:
        lines.append(f"- Scene: {case['scene_id']}")
        lines.append(f"  Utterance: {case['utterance']}")
        lines.append(f"  Subsets: {list(case.get('subsets', {}).keys())}")
        avg_score = np.mean(case.get('relation_scores', [0]))
        lines.append(f"  Avg Relation Score: {avg_score:.4f}")
        lines.append("")

    return "\n".join(lines)


def generate_casebook(
    contribution_analysis: Dict[str, Any],
    score_analysis: Dict[str, Any],
) -> str:
    """Generate casebook with detailed examples."""

    recovered = contribution_analysis.get("recovered", [])[:20]
    harmed = contribution_analysis.get("harmed", [])[:20]

    report = """# Dense Fundamentals: Casebook

**Date**: 2026-04-22
**Model**: Dense-no-cal-v1

---

## Recovered Cases (Base-Wrong → Dense-Correct)

Total: {recovered_total} cases

""".format(recovered_total=len(contribution_analysis.get("recovered", [])))

    for i, case in enumerate(recovered):
        scores = case.get("relation_scores", [])
        report += f"""### Case {i+1}: {case['scene_id']}

**Utterance**: {case['utterance']}

**Target Index**: {case['target_index']}

**Subsets**: {list(case.get('subsets', {}).keys())}

**Anchor Count**: {case.get('anchor_count', 0)}

**Relation Scores**:
- Mean: {np.mean(scores):.4f}
- Max: {np.max(scores):.4f}
- Min: {np.min(scores):.4f}
- Std: {np.std(scores):.4f}

---

"""

    report += """---

## Harmed Cases (Base-Correct → Dense-Wrong)

Total: {harmed_total} cases

""".format(harmed_total=len(contribution_analysis.get("harmed", [])))

    for i, case in enumerate(harmed):
        scores = case.get("relation_scores", [])
        report += f"""### Case {i+1}: {case['scene_id']}

**Utterance**: {case['utterance']}

**Target Index**: {case['target_index']}

**Subsets**: {list(case.get('subsets', {}).keys())}

**Anchor Count**: {case.get('anchor_count', 0)}

**Relation Scores**:
- Mean: {np.mean(scores):.4f}
- Max: {np.max(scores):.4f}
- Min: {np.min(scores):.4f}
- Std: {np.std(scores):.4f}

---

"""

    return report


def generate_summary(
    score_analysis: Dict[str, Any],
    pair_analysis: Dict[str, Any],
    contribution_analysis: Dict[str, Any],
) -> str:
    """Generate final summary with Q1/Q2/Q3 answers."""

    # Determine Q1 answer
    score_gap = score_analysis.get("score_gap", 0)
    if score_gap > 0.5:
        q1_answer = "STRONG"
    elif score_gap > 0.1:
        q1_answer = "WEAK-BUT-REAL"
    elif score_gap > -0.1:
        q1_answer = "UNCLEAR"
    else:
        q1_answer = "MOSTLY-NOISY"

    # Determine Q2 answer
    hit_at_1 = pair_analysis.get("hit_at_1", 0)
    if hit_at_1 > 0.5:
        q2_effective = "STRONG"
    elif hit_at_1 > 0.2:
        q2_effective = "MODERATE"
    else:
        q2_effective = "WEAK"

    # Determine Q3 answer
    net = contribution_analysis.get("net", 0)
    recovered_ma = contribution_analysis.get("recovered_subsets", {}).get("multi_anchor", 0)
    harmed_ma = contribution_analysis.get("harmed_subsets", {}).get("multi_anchor", 0)

    if net > 50 and recovered_ma > harmed_ma:
        q3_answer = "WORTH-CONTINUING"
    elif net > 0:
        q3_answer = "CONTINUE-WITH-CAUTION"
    else:
        q3_answer = "NOT-WORTH-CONTINUING"

    # Determine Route
    if q1_answer in ["STRONG", "WEAK-BUT-REAL"] and q3_answer in ["WORTH-CONTINUING", "CONTINUE-WITH-CAUTION"]:
        route = "Route B-fundamentals"
    else:
        route = "Route C-hard-stop"

    report = f"""# Dense Fundamentals: Final Summary

**Date**: 2026-04-22
**Model**: Dense-no-cal-v1

---

## Executive Summary

This report answers three key questions about the DenseRelationModule:

| Question | Answer |
|----------|--------|
| Q1-fundamental | **{q1_answer}** |
| Q2-ranking | **{q2_effective}** |
| Q3-worth | **{q3_answer}** |

**Route Decision**: **{route}**

---

## Q1-fundamental: Do relation scores have discriminative power?

**Answer**: {q1_answer}

### Evidence

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Score Gap (Correct - Wrong) | {score_gap:.4f} | {'Strong' if score_gap > 0.5 else ('Moderate' if score_gap > 0.1 else 'Weak')} |
| Mean Margin (Top1 - Top2) | {score_analysis.get('margin_mean', 0):.4f} | {'Confident' if score_analysis.get('margin_mean', 0) > 0.5 else ('Moderate' if score_analysis.get('margin_mean', 0) > 0.1 else 'Uncertain')} |
| Base-Wrong Mean Score | {score_analysis.get('base_wrong_mean', 0):.4f} | Dense helps where Base fails |
| Base-Correct Mean Score | {score_analysis.get('base_correct_mean', 0):.4f} | Dense amplifies easy cases |

### Rationale

{get_q1_rationale(q1_answer, score_analysis)}

---

## Q2-ranking: Does pair ranking work?

**Answer**: {q2_effective}

### Evidence

| Metric | Value |
|--------|-------|
| Hit@1 | {100 * hit_at_1:.1f}% |
| Hit@3 | {100 * pair_analysis.get('hit_at_3', 0):.1f}% |
| Hit@5 | {100 * pair_analysis.get('hit_at_5', 0):.1f}% |
| Multi-Anchor Hit@1 | {100 * pair_analysis.get('multi_anchor_hit_at_1', 0):.1f}% |
| Single-Anchor Hit@1 | {100 * pair_analysis.get('single_anchor_hit_at_1', 0):.1f}% |

### Valid Subsets

{get_valid_subsets(pair_analysis, contribution_analysis)}

### Harmed Subsets

{get_harmed_subsets(contribution_analysis)}

### Rationale

{get_q2_rationale(q2_effective, pair_analysis)}

---

## Q3-worth: Is dense scorer line worth continuing?

**Answer**: {q3_answer}

### Evidence

| Metric | Value |
|--------|-------|
| Net Recovered | {net} |
| Recovered Count | {contribution_analysis.get('recovered_count', 0)} |
| Harmed Count | {contribution_analysis.get('harmed_count', 0)} |
| Multi-Anchor Recovery | {contribution_analysis.get('recovered_subsets', {}).get('multi_anchor', 0)} |
| Multi-Anchor Harm | {contribution_analysis.get('harmed_subsets', {}).get('multi_anchor', 0)} |

### Rationale

{get_q3_rationale(q3_answer, contribution_analysis)}

---

## Route Decision: {route}

### If Route B-fundamentals

**Allowed Actions**:
- Redesign dense scorer based on findings
- Focus on multi-anchor / clutter subsets
- Protect easy cases from over-correction
- Simplify aggregation (not add complexity)

**Forbidden Actions**:
- Blind complexity (attention, geometry, focal without foundation)
- Multi-seed experiments
- Calibration without validated signal

### If Route C-hard-stop

**Next Steps**:
- Pause dense scorer line
- Focus on diagnostic paper route
- Benchmark and analyze error patterns
- Consider alternative methods (not dense-based)

---

## Appendix: Key Statistics

{generate_statistics_table(score_analysis, pair_analysis, contribution_analysis)}
"""

    return report


def get_q1_rationale(answer: str, score_analysis: Dict[str, Any]) -> str:
    """Generate Q1 rationale."""
    if answer == "STRONG":
        return "Relation scores show clear separation between correct and wrong targets, with confident margins."
    elif answer == "WEAK-BUT-REAL":
        return f"Relation scores show modest separation (gap={score_analysis.get('score_gap', 0):.4f}), indicating real but weak signal."
    elif answer == "MOSTLY-NOISY":
        return "Relation scores show little to no separation between correct and wrong targets, suggesting mostly noise."
    else:
        return "Insufficient evidence to determine discriminative power."


def get_q2_rationale(answer: str, pair_analysis: Dict[str, Any]) -> str:
    """Generate Q2 rationale."""
    hit1 = 100 * pair_analysis.get("hit_at_1", 0)
    if answer == "STRONG":
        return f"Pair ranking achieves {hit1:.1f}% Hit@1, indicating strong anchor selection."
    elif answer == "MODERATE":
        return f"Pair ranking achieves {hit1:.1f}% Hit@1, indicating moderate anchor selection ability."
    else:
        return f"Pair ranking achieves only {hit1:.1f}% Hit@1, indicating weak anchor selection."


def get_q3_rationale(answer: str, contribution_analysis: Dict[str, Any]) -> str:
    """Generate Q3 rationale."""
    net = contribution_analysis.get("net", 0)
    if answer == "WORTH-CONTINUING":
        return f"Dense branch shows clear net benefit (+{net}) with systematic recovery patterns."
    elif answer == "CONTINUE-WITH-CAUTION":
        return f"Dense branch shows modest net benefit (+{net}), but patterns are not clear enough for confident redesign."
    else:
        return f"Dense branch shows net harm ({net}), with no clear recovery patterns."


def get_valid_subsets(pair_analysis: Dict[str, Any], contribution_analysis: Dict[str, Any]) -> str:
    """Get valid subsets description."""
    ma_rec = contribution_analysis.get("recovered_subsets", {}).get("multi_anchor", 0)
    ma_harm = contribution_analysis.get("harmed_subsets", {}).get("multi_anchor", 0)

    if ma_rec > ma_harm:
        return "- **Multi-Anchor**: Dense branch provides clear benefit, recovering more cases than it harms."
    else:
        return "- **Multi-Anchor**: Dense branch does not provide clear benefit."


def get_harmed_subsets(contribution_analysis: Dict[str, Any]) -> str:
    """Get harmed subsets description."""
    easy_harm = contribution_analysis.get("harmed_subsets", {}).get("easy", 0)
    total_harm = contribution_analysis.get("harmed_count", 0)

    if easy_harm > total_harm / 2:
        return "- **Easy Cases**: Dense branch primarily harms easy cases (no hard subsets), suggesting over-correction."
    else:
        return "- **Hard Cases**: Dense branch harms hard cases, suggesting fundamental issues with pair scoring."


def generate_statistics_table(
    score_analysis: Dict[str, Any],
    pair_analysis: Dict[str, Any],
    contribution_analysis: Dict[str, Any],
) -> str:
    """Generate statistics table."""
    return f"""
| Statistic | Value |
|-----------|-------|
| **Score Analysis** | |
| Correct Target Score Mean | {np.mean(score_analysis.get('correct_target_scores', [0])):.4f} |
| Wrong Target Score Mean | {np.mean(score_analysis.get('wrong_target_scores', [0])):.4f} |
| Score Gap | {score_analysis.get('score_gap', 0):.4f} |
| Margin Mean | {score_analysis.get('margin_mean', 0):.4f} |
| **Pair Ranking** | |
| Hit@1 | {100 * pair_analysis.get('hit_at_1', 0):.1f}% |
| Hit@3 | {100 * pair_analysis.get('hit_at_3', 0):.1f}% |
| **Contribution** | |
| Recovered | {contribution_analysis.get('recovered_count', 0)} |
| Harmed | {contribution_analysis.get('harmed_count', 0)} |
| Net | {contribution_analysis.get('net', 0)} |
"""


# ============================================================================
# Main
# ============================================================================

def main():
    """Run full analysis pipeline."""
    set_seed(42)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    # Paths
    test_embeddings_path = ROOT / "outputs/20260420_clean_sorted_vocab_baseline/embeddings/test_embeddings.json"
    coverage_path = ROOT / "reports/pre_method_clean_coverage_diagnostics/per_sample_coverage.jsonl"
    base_predictions_path = ROOT / "outputs/cover3d_round1/base_predictions.json"
    dense_predictions_path = ROOT / "outputs/cover3d_round1/dense-no-cal_predictions.json"

    # Load data
    log.info("Loading data...")

    if not test_embeddings_path.exists():
        log.error(f"Test embeddings not found at {test_embeddings_path}")
        return

    embeddings = load_embeddings(test_embeddings_path)
    log.info(f"Loaded {len(embeddings)} embeddings")

    coverage = load_coverage(coverage_path) if coverage_path.exists() else {}
    log.info(f"Loaded {len(coverage)} coverage entries")

    base_predictions = load_predictions(base_predictions_path) if base_predictions_path.exists() else []
    log.info(f"Loaded {len(base_predictions)} base predictions")

    dense_predictions = load_predictions(dense_predictions_path) if dense_predictions_path.exists() else []
    log.info(f"Loaded {len(dense_predictions)} dense predictions")

    # Merge coverage
    for emb in embeddings:
        key = (emb["scene_id"], emb["target_id"], emb["utterance"])
        cov = coverage.get(key, {})
        emb["anchor_ids"] = cov.get("anchor_ids", [])
        emb["anchor_count"] = cov.get("anchor_count", 0)
        emb["subsets"] = cov.get("subsets", {})

    # Load model
    log.info("Loading DenseRelationModule...")
    model = load_dense_model(device)

    # Extract relation scores (use subset for speed)
    log.info("Extracting relation scores (using first 500 samples for speed)...")
    subset_size = min(500, len(embeddings))
    relation_results = extract_relation_scores(model, embeddings[:subset_size], device)
    log.info(f"Extracted relation scores for {len(relation_results)} samples")

    # Match predictions with relation results
    base_map = {(p["scene_id"], p["utterance"]): p for p in base_predictions}
    dense_map = {(p["scene_id"], p["utterance"]): p for p in dense_predictions}

    # Analyze score distributions
    log.info("Analyzing score distributions...")
    score_analysis = analyze_score_distributions(relation_results, base_predictions[:subset_size])
    score_analysis["score_gap"] = np.mean(score_analysis.get("correct_target_scores", [0])) - np.mean(score_analysis.get("wrong_target_scores", [0]))
    score_analysis["margin_mean"] = np.mean(score_analysis.get("margins", [0]))
    score_analysis["base_correct_mean"] = np.mean(score_analysis.get("base_correct_scores", [0]))
    score_analysis["base_wrong_mean"] = np.mean(score_analysis.get("base_wrong_scores", [0]))

    # Analyze pair ranking
    log.info("Analyzing pair ranking...")
    pair_analysis = analyze_pair_ranking(relation_results)

    # Analyze contribution
    log.info("Analyzing dense branch contribution...")
    contribution_analysis = analyze_contribution(relation_results, base_predictions[:subset_size], dense_predictions[:subset_size])

    # Generate reports
    log.info("Generating reports...")

    reports_dir = ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)

    # Score audit report
    score_report = generate_score_audit_report(score_analysis)
    with open(reports_dir / "dense_fundamentals_score_audit.md", "w") as f:
        f.write(score_report)
    log.info(f"Saved score audit report to {reports_dir / 'dense_fundamentals_score_audit.md'}")

    # Pair audit report
    pair_report = generate_pair_audit_report(pair_analysis)
    with open(reports_dir / "dense_fundamentals_pair_audit.md", "w") as f:
        f.write(pair_report)
    log.info(f"Saved pair audit report to {reports_dir / 'dense_fundamentals_pair_audit.md'}")

    # Contribution audit report
    contrib_report = generate_contribution_audit_report(contribution_analysis)
    with open(reports_dir / "dense_fundamentals_contribution_audit.md", "w") as f:
        f.write(contrib_report)
    log.info(f"Saved contribution audit report to {reports_dir / 'dense_fundamentals_contribution_audit.md'}")

    # Casebook
    casebook = generate_casebook(contribution_analysis, score_analysis)
    with open(reports_dir / "dense_fundamentals_casebook.md", "w") as f:
        f.write(casebook)
    log.info(f"Saved casebook to {reports_dir / 'dense_fundamentals_casebook.md'}")

    # Summary
    summary = generate_summary(score_analysis, pair_analysis, contribution_analysis)
    with open(reports_dir / "dense_fundamentals_summary.md", "w") as f:
        f.write(summary)
    log.info(f"Saved summary to {reports_dir / 'dense_fundamentals_summary.md'}")

    # CSV table
    csv_data = f"""metric,value
score_gap,{score_analysis.get("score_gap", 0):.4f}
margin_mean,{score_analysis.get("margin_mean", 0):.4f}
correct_target_score_mean,{np.mean(score_analysis.get("correct_target_scores", [0])):.4f}
wrong_target_score_mean,{np.mean(score_analysis.get("wrong_target_scores", [0])):.4f}
hit_at_1,{pair_analysis.get("hit_at_1", 0):.4f}
hit_at_3,{pair_analysis.get("hit_at_3", 0):.4f}
recovered_count,{contribution_analysis.get("recovered_count", 0)}
harmed_count,{contribution_analysis.get("harmed_count", 0)}
net,{contribution_analysis.get("net", 0)}
multi_anchor_recovered,{contribution_analysis.get("recovered_subsets", {}).get("multi_anchor", 0)}
multi_anchor_harmed,{contribution_analysis.get("harmed_subsets", {}).get("multi_anchor", 0)}
"""
    with open(reports_dir / "dense_fundamentals_table.csv", "w") as f:
        f.write(csv_data)
    log.info(f"Saved CSV table to {reports_dir / 'dense_fundamentals_table.csv'}")

    log.info("=" * 60)
    log.info("DENSE FUNDAMENTALS ANALYSIS COMPLETE")
    log.info("=" * 60)
    log.info(f"Score Gap: {score_analysis.get('score_gap', 0):.4f}")
    log.info(f"Hit@1: {100 * pair_analysis.get('hit_at_1', 0):.1f}%")
    log.info(f"Net: {contribution_analysis.get('net', 0)}")

    return {
        "score_analysis": score_analysis,
        "pair_analysis": pair_analysis,
        "contribution_analysis": contribution_analysis,
    }


if __name__ == "__main__":
    main()
