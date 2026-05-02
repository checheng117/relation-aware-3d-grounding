from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from rag3d.datasets.transforms import normalize_class_name


def grounding_cross_entropy(
    logits: torch.Tensor,
    target_index: torch.Tensor,
    mask: torch.Tensor,
    *,
    reduction: str = "mean",
) -> torch.Tensor:
    """logits [B,N], target_index [B], mask [B,N]. ``reduction`` is ``mean`` or ``none`` (per-sample [B])."""
    logits = logits.masked_fill(~mask, float("-inf"))
    return F.cross_entropy(logits, target_index, reduction=reduction)  # type: ignore[arg-type]


def candidate_load_weights(mask: torch.Tensor, meta: list[Any] | None) -> torch.Tensor:
    """Per-sample weights from valid object count, normalized to mean 1."""
    b = mask.size(0)
    device = mask.device
    counts = mask.sum(dim=1).float().clamp_min(1.0)
    if meta is not None and len(meta) == b:
        for i in range(b):
            n_obj = meta[i].get("n_objects")
            if n_obj is not None:
                try:
                    counts[i] = max(float(n_obj), 1.0)
                except (TypeError, ValueError):
                    pass
    w = torch.log1p(counts)
    return w / w.mean().clamp_min(1e-6)


def hardest_negative_margin_loss(
    logits: torch.Tensor,
    target_index: torch.Tensor,
    mask: torch.Tensor,
    margin: float,
    valid_rows: torch.Tensor | None = None,
) -> torch.Tensor:
    """Hinge: max(0, margin + max_{j!=t} z_j - z_t) averaged over valid batch rows."""
    b, n = logits.shape
    device = logits.device
    total = torch.zeros((), device=device)
    count = 0
    for bi in range(b):
        if valid_rows is not None and not bool(valid_rows[bi].item()):
            continue
        t = int(target_index[bi].item())
        if t < 0 or t >= n or not mask[bi, t]:
            continue
        lt = logits[bi, t]
        neg_mask = mask[bi].clone()
        neg_mask[t] = False
        if not neg_mask.any():
            continue
        max_neg = logits[bi].masked_fill(~neg_mask, float("-inf")).max()
        if torch.isfinite(max_neg):
            total = total + F.relu(margin + max_neg - lt)
            count += 1
    if count == 0:
        return torch.zeros((), device=device)
    return total / count


def spatial_nearby_hinge_loss(
    logits: torch.Tensor,
    target_index: torch.Tensor,
    mask: torch.Tensor,
    samples: list[Any],
    margin: float,
    max_neighbors: int = 4,
    valid_rows: torch.Tensor | None = None,
) -> torch.Tensor:
    """Push gold above the highest-logit spatial neighbor (by 3D center distance)."""
    b, n = logits.shape
    device = logits.device
    total = torch.zeros((), device=device)
    count = 0
    k_nn = max(1, int(max_neighbors))
    for bi in range(b):
        if valid_rows is not None and not bool(valid_rows[bi].item()):
            continue
        t = int(target_index[bi].item())
        if bi >= len(samples) or t < 0 or t >= n or not mask[bi, t]:
            continue
        objs = samples[bi].objects
        if t >= len(objs):
            continue
        cx, cy, cz = objs[t].center
        dists: list[tuple[float, int]] = []
        for j in range(min(n, len(objs))):
            if j == t or not mask[bi, j]:
                continue
            ox, oy, oz = objs[j].center
            d = (ox - cx) ** 2 + (oy - cy) ** 2 + (oz - cz) ** 2
            dists.append((d, j))
        dists.sort(key=lambda x: x[0])
        neigh = [j for _, j in dists[:k_nn]]
        if not neigh:
            continue
        lt = logits[bi, t]
        idx_t = torch.tensor(neigh, device=device, dtype=torch.long)
        mx = logits[bi, idx_t].max()
        total = total + F.relu(margin + mx - lt)
        count += 1
    if count == 0:
        return torch.zeros((), device=device)
    return total / count


def same_class_hinge_loss(
    logits: torch.Tensor,
    target_index: torch.Tensor,
    mask: torch.Tensor,
    samples: list[Any],
    margin: float,
    valid_rows: torch.Tensor | None = None,
) -> torch.Tensor:
    """Push target logit above same-class negatives by ``margin`` (mean over batch).

    ``samples`` must be ``GroundingSample``-like with ``.objects[j].class_name``.
    """
    b, n = logits.shape
    device = logits.device
    total = torch.zeros((), device=device)
    count = 0
    for bi in range(b):
        if valid_rows is not None and not bool(valid_rows[bi].item()):
            continue
        t = int(target_index[bi].item())
        if t < 0 or t >= n or not mask[bi, t]:
            continue
        if bi >= len(samples):
            continue
        objs = samples[bi].objects
        if t >= len(objs):
            continue
        gold_cls = normalize_class_name(objs[t].class_name)
        lt = logits[bi, t]
        max_neg = torch.tensor(float("-inf"), device=device)
        for j in range(n):
            if j == t or not mask[bi, j] or j >= len(objs):
                continue
            if normalize_class_name(objs[j].class_name) != gold_cls:
                continue
            max_neg = torch.maximum(max_neg, logits[bi, j])
        if torch.isfinite(max_neg):
            total = total + F.relu(margin + max_neg - lt)
            count += 1
    if count == 0:
        return torch.zeros((), device=device)
    return total / count


def compute_batch_training_loss(
    logits: torch.Tensor,
    target_index: torch.Tensor,
    mask: torch.Tensor,
    loss_cfg: dict[str, Any] | None,
    samples: list[Any] | None,
    meta: list[Any] | None = None,
    valid_rows: torch.Tensor | None = None,
) -> torch.Tensor:
    """CE (optional load weighting) + optional ranking / spatial / same-class hinges."""
    loss_cfg = loss_cfg or {}

    ce_vec = grounding_cross_entropy(logits, target_index, mask, reduction="none")
    if valid_rows is not None:
        valid_rows = valid_rows.to(device=logits.device, dtype=torch.bool)
        ce_valid = ce_vec[valid_rows]
    else:
        ce_valid = ce_vec

    if ce_valid.numel() == 0:
        loss = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0).sum() * 0.0
    elif (loss_cfg.get("candidate_load_weight") or {}).get("enabled"):
        w = candidate_load_weights(mask, meta)
        if valid_rows is not None:
            w = w[valid_rows]
            ce_vec = ce_vec[valid_rows]
        loss = (ce_vec * w).mean()
    else:
        loss = ce_valid.mean()

    rm = loss_cfg.get("ranking_margin") or {}
    if rm.get("enabled"):
        loss = loss + float(rm.get("lambda", 0.15)) * hardest_negative_margin_loss(
            logits,
            target_index,
            mask,
            float(rm.get("margin", 0.2)),
            valid_rows=valid_rows,
        )

    sh = loss_cfg.get("spatial_nearby_hinge") or {}
    if sh.get("enabled") and samples:
        loss = loss + float(sh.get("lambda", 0.15)) * spatial_nearby_hinge_loss(
            logits,
            target_index,
            mask,
            samples,
            float(sh.get("margin", 0.2)),
            int(sh.get("max_neighbors", 4)),
            valid_rows=valid_rows,
        )

    hn = loss_cfg.get("hard_negative") or {}
    if hn.get("enabled") and samples:
        margin = float(hn.get("margin", 0.25))
        lam = float(hn.get("lambda_hinge", 0.5))
        loss = loss + lam * same_class_hinge_loss(logits, target_index, mask, samples, margin, valid_rows=valid_rows)

    return loss
