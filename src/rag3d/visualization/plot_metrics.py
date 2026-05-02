from __future__ import annotations

from pathlib import Path
from typing import Sequence


def plot_bar_metrics(
    names: Sequence[str],
    values: Sequence[float],
    out_path: Path,
    title: str = "Metrics",
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError("matplotlib required for plot_bar_metrics") from e
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(list(names), list(values))
    ax.set_ylabel("value")
    ax.set_title(title)
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
