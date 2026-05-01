#!/usr/bin/env python3
"""Generate the course-report pipeline figure for PDF and README use."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parent


COLORS = {
    "input_edge": "#2F5597",
    "input_fill": "#EEF3FB",
    "process_edge": "#2F7D45",
    "process_fill": "#EEF8F1",
    "output_edge": "#C55A11",
    "output_fill": "#FFF3E8",
    "diagnostic_edge": "#6B7280",
    "diagnostic_fill": "#F5F6F8",
    "arrow": "#555555",
    "group": "#B7BCC5",
}


def add_box(ax, xy, size, label, edge, fill, fontsize=11):
    x, y = xy
    w, h = size
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.035,rounding_size=0.035",
        linewidth=1.25,
        edgecolor=edge,
        facecolor=fill,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2,
        y + h / 2,
        label,
        ha="center",
        va="center",
        fontsize=fontsize,
        family="DejaVu Sans",
        color="#111111",
        linespacing=1.12,
    )
    return box


def arrow(ax, start, end, rad=0.0):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.2,
            color=COLORS["arrow"],
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=2,
            shrinkB=2,
        )
    )


def main():
    fig, ax = plt.subplots(figsize=(9.8, 3.65))
    ax.set_xlim(0, 9.75)
    ax.set_ylim(0, 3.65)
    ax.axis("off")

    w, h = 1.32, 0.68
    y = 1.78
    boxes = {}
    boxes["scene"] = add_box(
        ax,
        (0.20, 2.73),
        (w, h),
        "3D scene\nobjects",
        COLORS["input_edge"],
        COLORS["input_fill"],
    )
    boxes["text"] = add_box(
        ax,
        (1.68, 2.73),
        (w, h),
        "Language\nexpression",
        COLORS["input_edge"],
        COLORS["input_fill"],
    )
    boxes["features"] = add_box(
        ax,
        (0.94, y),
        (w, h),
        "Object/text\nfeatures",
        COLORS["process_edge"],
        COLORS["process_fill"],
    )
    boxes["pairs"] = add_box(
        ax,
        (2.55, y),
        (1.62, h),
        "Candidate-anchor\nrelation scores",
        COLORS["process_edge"],
        COLORS["process_fill"],
        fontsize=10.2,
    )
    boxes["modes"] = add_box(
        ax,
        (4.50, y),
        (1.45, h),
        "Conditioned\nrelation modes",
        COLORS["process_edge"],
        COLORS["process_fill"],
    )
    boxes["fusion"] = add_box(
        ax,
        (6.28, y),
        (1.42, h),
        "Logit fusion\nand ranking",
        COLORS["output_edge"],
        COLORS["output_fill"],
    )
    boxes["prediction"] = add_box(
        ax,
        (8.03, y),
        (1.50, h),
        "Prediction\nand diagnostics",
        COLORS["output_edge"],
        COLORS["output_fill"],
    )
    boxes["diagnostics"] = add_box(
        ax,
        (7.89, 0.52),
        (1.78, h),
        "Hard-subset analysis\nCoverage analysis",
        COLORS["diagnostic_edge"],
        COLORS["diagnostic_fill"],
        fontsize=10.2,
    )

    # Relation-scoring module outline.
    module = FancyBboxPatch(
        (2.38, 1.55),
        3.77,
        1.08,
        boxstyle="round,pad=0.08,rounding_size=0.045",
        linewidth=0.9,
        edgecolor=COLORS["group"],
        facecolor="none",
    )
    ax.add_patch(module)
    ax.text(
        4.26,
        2.72,
        "relation-scoring module",
        ha="center",
        va="bottom",
        fontsize=10,
        family="DejaVu Sans",
        color="#555555",
    )

    arrow(ax, (0.86, 2.73), (1.36, 2.46))
    arrow(ax, (2.34, 2.73), (1.72, 2.46))
    arrow(ax, (2.26, 2.12), (2.55, 2.12))
    arrow(ax, (4.17, 2.12), (4.50, 2.12))
    arrow(ax, (5.95, 2.12), (6.28, 2.12))
    arrow(ax, (7.70, 2.12), (8.03, 2.12))
    arrow(ax, (8.78, 1.78), (8.78, 1.20))

    fig.tight_layout(pad=0.15)
    for ext in ("png", "svg", "pdf"):
        output = ROOT / f"figure1_pipeline.{ext}"
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.03}
        if ext == "png":
            kwargs["dpi"] = 220
        fig.savefig(output, **kwargs)
    plt.close(fig)


if __name__ == "__main__":
    main()
