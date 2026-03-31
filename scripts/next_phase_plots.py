#!/usr/bin/env python3
"""Headless figures for next-phase experiment bundle (matplotlib Agg)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _setup_agg():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_relation_stratified_flat(metrics: dict[str, Any], out_png: Path, title: str = "") -> bool:
    plt = _setup_agg()
    rels = {k.replace("acc@1_rel::", ""): float(v) for k, v in metrics.items() if k.startswith("acc@1_rel::")}
    if not rels:
        return False
    names = list(rels.keys())[:20]
    vals = [rels[k] for k in names]
    fig, ax = plt.subplots(figsize=(max(6, len(names) * 0.35), 4))
    ax.bar(range(len(names)), vals, color="steelblue")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Acc@1")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return True


def plot_hard_case_bars_flat(metrics: dict[str, Any], out_png: Path) -> bool:
    plt = _setup_agg()
    sub = {
        k.replace("acc@1_subset::", "").replace("acc@1_slice::", "slice::"): float(v)
        for k, v in metrics.items()
        if "subset::" in k or ("slice::" in k and "geometry" in k)
    }
    if not sub:
        return False
    names = list(sub.keys())
    vals = list(sub.values())
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(names, vals, color="coral")
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Acc@1")
    ax.set_title("Hard-case subsets (higher is better)")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return True


def plot_shortlist_curve(diag: dict[str, Any], out_png: Path) -> bool:
    plt = _setup_agg()
    coarse = diag.get("coarse_recall_at_k") or {}
    if not coarse:
        return False
    ks = sorted(int(k.replace("recall@", "")) for k in coarse if k.startswith("recall@"))
    ys = [float(coarse[f"recall@{k}"]) for k in ks]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(ks, ys, "o-", label="Stage-1 recall@K")
    if "final_acc@1" in diag:
        ax.axhline(float(diag["final_acc@1"]), color="green", linestyle="--", label="Final Acc@1 (if set)")
    if diag.get("oracle_upper_bound_perfect_rerank") is not None:
        ax.axhline(float(diag["oracle_upper_bound_perfect_rerank"]), color="gray", linestyle=":", label="Oracle (in-K)")
    ax.set_xlabel("K")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Rate")
    ax.legend(fontsize=8)
    ax.set_title("Shortlist bottleneck")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return True


def plot_geometry_bars_flat(metrics: dict[str, Any], out_png: Path) -> bool:
    plt = _setup_agg()
    keys = [
        ("High geometry fallback", "acc@1_subset::geometry_high_fallback"),
        ("Low fallback (≤half)", "acc@1_slice::geometry_fallback_le_half"),
        ("High fallback (>half)", "acc@1_slice::geometry_fallback_gt_half"),
    ]
    labels = []
    vals = []
    for lab, k in keys:
        if k in metrics:
            labels.append(lab)
            vals.append(float(metrics[k]))
    if not labels:
        return False
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(labels, vals, color="seagreen")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Acc@1")
    ax.set_title("Geometry-quality slices")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return True


def plot_paraphrase_bars(para_path: Path, out_png: Path) -> bool:
    plt = _setup_agg()
    if not para_path.is_file():
        return False
    d = json.loads(para_path.read_text(encoding="utf-8"))
    pairs = [
        ("target_agreement", "mean_paraphrase_target_agreement"),
        ("mean_acc@1", "mean_paraphrase_mean_acc@1"),
        ("anchor_drift", "mean_anchor_distribution_drift"),
    ]
    keys = [lab for lab, k in pairs if d.get(k) is not None]
    vals = [float(d[k]) for lab, k in pairs if d.get(k) is not None]
    if not keys:
        return False
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(keys, vals)
    ax.set_ylim(0, 1.05)
    ax.set_title("Paraphrase consistency")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return True


def plot_failure_taxonomy(failure_json: Path, out_png: Path) -> bool:
    plt = _setup_agg()
    if not failure_json.is_file():
        return False
    d = json.loads(failure_json.read_text(encoding="utf-8"))
    counts = d.get("failure_tag_counts") or {}
    if not counts:
        return False
    names = list(counts.keys())[:15]
    vals = [int(counts[k]) for k in names]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(names, vals, color="slategray")
    ax.set_xlabel("Count")
    ax.set_title("Failure tag taxonomy (sampled batches)")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return True


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: next_phase_plots.py <experiment_root>")
        return 1
    root = Path(sys.argv[1])
    rel = root / "_plot_relation_metrics.json"
    if rel.is_file():
        plot_relation_stratified_flat(
            json.loads(rel.read_text(encoding="utf-8")),
            root / "relation_stratified_plot.png",
            "Relation-stratified (C structured, controlled)",
        )
    hc = root / "_plot_hard_case_metrics.json"
    if hc.is_file():
        plot_hard_case_bars_flat(json.loads(hc.read_text(encoding="utf-8")), root / "hard_case_plot.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
