#!/usr/bin/env python3
"""
COVER-3D Phase 1: Figure Generation

Generates paper-quality figures for the coverage diagnostics analysis.
Uses matplotlib with academic styling.
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend before importing pyplot

import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
from collections import defaultdict
import argparse


# ============================================================================
# Configuration
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_STATS = PROJECT_ROOT / "reports/cover3d_phase1_diagnostics_stats.json"
DEFAULT_SUBSETS = PROJECT_ROOT / "reports/cover3d_phase1_subset_performance.json"
FIGURE_DIR = PROJECT_ROOT / "reports/figures/cover3d_phase1"


# ============================================================================
# Academic Style Setup
# ============================================================================

def setup_academic_style():
    """Configure matplotlib for paper-quality figures."""
    plt.style.use('seaborn-v0_8-whitegrid')

    # Font settings
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
    })


# ============================================================================
# Figure 1: Subset Performance Bar Chart
# ============================================================================

def plot_subset_accuracy_bars(subset_data: dict, output_path: Path):
    """Figure 1: Hard subset accuracy comparison."""

    setup_academic_style()

    # Filter to meaningful subsets
    exclude = ['correct', 'incorrect', 'no_anchor', 'hard_combined']
    subsets = [s for s in subset_data['subsets']
               if s['subset'] not in exclude and s['count'] > 50]

    # Sort by accuracy gap from baseline
    baseline = subset_data['aggregate']['overall_acc_at_1']
    subsets_sorted = sorted(subsets, key=lambda x: x['acc_at_1'])

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    names = [s['subset'].replace('_', ' ').title() for s in subsets_sorted]
    accuracies = [s['acc_at_1'] for s in subsets_sorted]
    counts = [s['count'] for s in subsets_sorted]

    # Color by difficulty (red for hard, green for easy)
    colors = ['#d62728' if acc < baseline else '#2ca02c' for acc in accuracies]

    bars = ax.barh(names, accuracies, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)

    # Add baseline reference line
    ax.axvline(baseline, color='#1f77b4', linestyle='--', linewidth=2, label=f'Overall ({baseline}%')

    # Add count annotations
    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f'n={count}', va='center', ha='left', fontsize=8)

    ax.set_xlabel('Accuracy@1 (%)')
    ax.set_title('Baseline Accuracy by Subset\n(Red = Below Overall, Green = Above Overall)')
    ax.set_xlim(0, 55)
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(output_path, format='png', dpi=300)
    plt.savefig(output_path.with_suffix('.pdf'), format='pdf')
    plt.close()

    print(f"Saved: {output_path}")


# ============================================================================
# Figure 2: Same-Class Clutter Impact
# ============================================================================

def plot_clutter_impact(stats_data: dict, output_path: Path):
    """Figure 2: Accuracy vs same-class count."""

    setup_academic_style()

    per_sample = stats_data['per_sample']

    # Compute accuracy by same_class_count bins
    bins = defaultdict(list)
    for s in per_sample:
        count = s['same_class_count']
        bins[count].append(s['correct_at_1'])

    # Aggregate
    clutter_values = sorted(bins.keys())
    accuracies = [sum(bins[v]) / len(bins[v]) * 100 for v in clutter_values]
    counts = [len(bins[v]) for v in clutter_values]

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 5))

    # Bar plot
    bars = ax.bar(clutter_values, accuracies,
                   color='#d62728', alpha=0.7, edgecolor='black', linewidth=0.5)

    # Add overall reference
    baseline = stats_data['aggregate']['overall_acc_at_1']
    ax.axhline(baseline, color='#1f77b4', linestyle='--', linewidth=2,
               label=f'Overall ({baseline}%)')

    # Annotate counts
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'n={count}', ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Same-Class Object Count')
    ax.set_ylabel('Accuracy@1 (%)')
    ax.set_title('Baseline Accuracy vs Same-Class Clutter\n(Higher Clutter → Lower Accuracy)')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 50)

    plt.tight_layout()
    plt.savefig(output_path, format='png', dpi=300)
    plt.savefig(output_path.with_suffix('.pdf'), format='pdf')
    plt.close()

    print(f"Saved: {output_path}")


# ============================================================================
# Figure 3: Scene Size Impact
# ============================================================================

def plot_scene_size_impact(stats_data: dict, output_path: Path):
    """Figure 3: Accuracy vs scene size."""

    setup_academic_style()

    per_sample = stats_data['per_sample']

    # Bin by n_objects
    bins = defaultdict(list)
    for s in per_sample:
        n = s['n_objects']
        if n < 20:
            bin_label = '<20'
        elif n < 30:
            bin_label = '20-29'
        elif n < 40:
            bin_label = '30-39'
        elif n < 50:
            bin_label = '40-49'
        elif n < 60:
            bin_label = '50-59'
        else:
            bin_label = '60+'
        bins[bin_label].append(s['correct_at_1'])

    # Order bins
    bin_order = ['<20', '20-29', '30-39', '40-49', '50-59', '60+']
    accuracies = [sum(bins[b]) / len(bins[b]) * 100 if bins[b] else 0 for b in bin_order]
    counts = [len(bins[b]) for b in bin_order]

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(range(len(bin_order)), accuracies,
                   color='#ff7f0e', alpha=0.7, edgecolor='black', linewidth=0.5)

    # Add overall reference
    baseline = stats_data['aggregate']['overall_acc_at_1']
    ax.axhline(baseline, color='#1f77b4', linestyle='--', linewidth=2,
               label=f'Overall ({baseline}%)')

    ax.set_xticks(range(len(bin_order)))
    ax.set_xticklabels(bin_order)

    # Annotate counts
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'n={count}', ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Scene Object Count')
    ax.set_ylabel('Accuracy@1 (%)')
    ax.set_title('Baseline Accuracy vs Scene Size')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 45)

    plt.tight_layout()
    plt.savefig(output_path, format='png', dpi=300)
    plt.savefig(output_path.with_suffix('.pdf'), format='pdf')
    plt.close()

    print(f"Saved: {output_path}")


# ============================================================================
# Figure 4: Relation Type Distribution
# ============================================================================

def plot_relation_type_distribution(stats_data: dict, output_path: Path):
    """Figure 4: Accuracy by relation type."""

    setup_academic_style()

    per_sample = stats_data['per_sample']

    # Compute accuracy by relation_type
    bins = defaultdict(list)
    for s in per_sample:
        bins[s['relation_type']].append(s['correct_at_1'])

    # Order by count
    types_sorted = sorted(bins.keys(), key=lambda x: -len(bins[x]))
    accuracies = [sum(bins[t]) / len(bins[t]) * 100 for t in types_sorted]
    counts = [len(bins[t]) for t in types_sorted]

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 5))

    # Color by is_relational
    colors = ['#9467bd' if t in ['between', 'directional', 'relative', 'support', 'container']
              else '#8c564b' for t in types_sorted]

    bars = ax.bar(range(len(types_sorted)), accuracies, color=colors,
                   alpha=0.7, edgecolor='black', linewidth=0.5)

    # Add overall reference
    baseline = stats_data['aggregate']['overall_acc_at_1']
    ax.axhline(baseline, color='#1f77b4', linestyle='--', linewidth=2,
               label=f'Overall ({baseline}%)')

    ax.set_xticks(range(len(types_sorted)))
    ax.set_xticklabels([t.title() for t in types_sorted], rotation=30, ha='right')

    # Annotate counts
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'n={count}', ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Relation Type')
    ax.set_ylabel('Accuracy@1 (%)')
    ax.set_title('Baseline Accuracy by Relation Type\n(Purple = Relational, Brown = Non-Relational)')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 55)

    plt.tight_layout()
    plt.savefig(output_path, format='png', dpi=300)
    plt.savefig(output_path.with_suffix('.pdf'), format='pdf')
    plt.close()

    print(f"Saved: {output_path}")


# ============================================================================
# Figure 5: Failure Taxonomy Pie Chart
# ============================================================================

def plot_failure_taxonomy(stats_data: dict, subset_data: dict, output_path: Path):
    """Figure 5: Failure case taxonomy."""

    setup_academic_style()

    per_sample = stats_data['per_sample']

    # Categorize incorrect predictions
    incorrect = [s for s in per_sample if not s['correct_at_1']]

    # Taxonomy
    categories = defaultdict(int)
    for s in incorrect:
        if s['same_class_count'] >= 5:
            categories['High Clutter'] += 1
        elif s['same_class_count'] >= 3:
            categories['Same-Class Clutter'] += 1
        elif s['is_multi_anchor']:
            categories['Multi-Anchor'] += 1
        elif s['anchor_count'] == 1:
            categories['Single-Anchor'] += 1
        elif s['is_relational']:
            categories['Other Relational'] += 1
        elif s['n_objects'] >= 50:
            categories['Dense Scene'] += 1
        else:
            categories['Other'] += 1

    # Create pie chart
    fig, ax = plt.subplots(figsize=(8, 6))

    labels = list(categories.keys())
    sizes = list(categories.values())
    colors = ['#d62728', '#ff7f0e', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22']

    # Only show categories with significant counts
    total = sum(sizes)
    threshold = 0.02 * total

    filtered_labels = []
    filtered_sizes = []
    filtered_colors = []
    other_sum = 0

    for label, size, color in zip(labels, sizes, colors):
        if size >= threshold:
            filtered_labels.append(label)
            filtered_sizes.append(size)
            filtered_colors.append(color)
        else:
            other_sum += size

    if other_sum > 0:
        filtered_labels.append('Other')
        filtered_sizes.append(other_sum)
        filtered_colors.append('#17becf')

    wedges, texts, autotexts = ax.pie(
        filtered_sizes, labels=filtered_labels, colors=filtered_colors,
        autopct=lambda pct: f'{pct:.1f}%\n({int(pct*total/100)})',
        startangle=90, pctdistance=0.75,
        wedgeprops=dict(linewidth=1, edgecolor='white')
    )

    ax.set_title(f'Failure Case Taxonomy\n(Total Failures: {total} / {len(per_sample)} samples)')

    plt.tight_layout()
    plt.savefig(output_path, format='png', dpi=300)
    plt.savefig(output_path.with_suffix('.pdf'), format='pdf')
    plt.close()

    print(f"Saved: {output_path}")


# ============================================================================
# Figure 6: Combined Hard Subset Comparison
# ============================================================================

def plot_hard_vs_easy(stats_data: dict, subset_data: dict, output_path: Path):
    """Figure 6: Hard vs Easy subset comparison."""

    setup_academic_style()

    # Select key subsets
    key_subsets = ['same_class_clutter', 'same_class_high_clutter', 'multi_anchor',
                   'single_anchor', 'relative', 'dense_scene']

    baseline = subset_data['aggregate']['overall_acc_at_1']

    # Get subset data
    subset_lookup = {s['subset']: s for s in subset_data['subsets']}
    selected = [subset_lookup.get(k) for k in key_subsets if subset_lookup.get(k)]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    names = [s['subset'].replace('_', '\n').title() for s in selected]
    accuracies = [s['acc_at_1'] for s in selected]
    deltas = [acc - baseline for acc in accuracies]

    # Bar colors by delta
    colors = ['#d62728' if d < -3 else '#ff7f0e' if d < 0 else '#2ca02c' for d in deltas]

    bars = ax.bar(range(len(names)), accuracies, color=colors,
                   alpha=0.7, edgecolor='black', linewidth=0.5)

    # Add baseline
    ax.axhline(baseline, color='#1f77b4', linestyle='--', linewidth=2,
               label=f'Overall ({baseline}%)')

    # Annotate deltas
    for i, (bar, delta) in enumerate(zip(bars, deltas)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{delta:+.1f}%', ha='center', va='bottom', fontsize=9,
                color='red' if delta < -3 else 'black')

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=0, ha='center')
    ax.set_xlabel('Hard Subset')
    ax.set_ylabel('Accuracy@1 (%)')
    ax.set_title('Hard Subset Accuracy Gaps\n(Negative Gap = Harder than Overall)')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 40)

    plt.tight_layout()
    plt.savefig(output_path, format='png', dpi=300)
    plt.savefig(output_path.with_suffix('.pdf'), format='pdf')
    plt.close()

    print(f"Saved: {output_path}")


# ============================================================================
# Main
# ============================================================================

def generate_all_figures(stats_path: Path, subset_path: Path, output_dir: Path):
    """Generate all figures."""

    print("COVER-3D Phase 1: Figure Generation")
    print("=" * 60)

    # Load data
    stats_data = json.load(open(stats_path))
    subset_data = json.load(open(subset_path))

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate figures
    print("\n[1] Subset accuracy bars...")
    plot_subset_accuracy_bars(subset_data, output_dir / "fig1_subset_accuracy.png")

    print("\n[2] Clutter impact...")
    plot_clutter_impact(stats_data, output_dir / "fig2_clutter_impact.png")

    print("\n[3] Scene size impact...")
    plot_scene_size_impact(stats_data, output_dir / "fig3_scene_size.png")

    print("\n[4] Relation type distribution...")
    plot_relation_type_distribution(stats_data, output_dir / "fig4_relation_type.png")

    print("\n[5] Failure taxonomy...")
    plot_failure_taxonomy(stats_data, subset_data, output_dir / "fig5_failure_taxonomy.png")

    print("\n[6] Hard vs Easy comparison...")
    plot_hard_vs_easy(stats_data, subset_data, output_dir / "fig6_hard_vs_easy.png")

    print("\n" + "=" * 60)
    print("FIGURE GENERATION COMPLETE")
    print("=" * 60)
    print(f"\nOutput directory: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="COVER-3D Phase 1 Figure Generation")
    parser.add_argument("--stats", type=Path, default=DEFAULT_STATS)
    parser.add_argument("--subsets", type=Path, default=DEFAULT_SUBSETS)
    parser.add_argument("--output", type=Path, default=FIGURE_DIR)

    args = parser.parse_args()

    generate_all_figures(args.stats, args.subsets, args.output)


if __name__ == "__main__":
    main()