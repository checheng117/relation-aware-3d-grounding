from __future__ import annotations

from pathlib import Path

import torch


def save_anchor_bar_data(
    object_ids: list[str],
    probs: list[float],
    out_csv: Path,
) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8") as f:
        f.write("object_id,prob\n")
        for oid, p in zip(object_ids, probs, strict=True):
            f.write(f"{oid},{p:.6f}\n")


def anchor_probs_from_tensor(
    object_ids: list[str],
    dist_row: torch.Tensor,
    mask_row: torch.Tensor,
) -> tuple[list[str], list[float]]:
    ids = [object_ids[i] for i in range(len(object_ids)) if bool(mask_row[i].item())]
    vals = [float(dist_row[i].item()) for i in range(len(object_ids)) if bool(mask_row[i].item())]
    return ids, vals
