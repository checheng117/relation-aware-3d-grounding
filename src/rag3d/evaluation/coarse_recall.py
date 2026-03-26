"""Stage-1 (coarse) retrieval metrics: recall@K, gold rank, margins, stratified slices."""

from __future__ import annotations

from typing import Any

import copy

import torch
from torch.utils.data import DataLoader

from rag3d.evaluation.metrics import per_sample_correct_at5
from rag3d.evaluation.stratified_eval import (
    augment_meta_geometry_fallback_tags,
    augment_meta_with_model_margins,
    stratified_accuracy_from_lists,
)
from rag3d.evaluation.two_stage_eval import coarse_forward, to_dev_batch


def _masked_logits_row(logits_row: torch.Tensor, mask_row: torch.Tensor) -> torch.Tensor:
    return logits_row.masked_fill(~mask_row, float("-inf"))


def gold_rank_in_scene(
    logits: torch.Tensor,
    target_index: torch.Tensor,
    mask: torch.Tensor,
) -> list[int | None]:
    """0-based rank by descending logit among valid objects; None if target invalid."""
    out: list[int | None] = []
    for i in range(logits.size(0)):
        t = int(target_index[i].item())
        if t < 0 or t >= logits.size(1) or not mask[i, t]:
            out.append(None)
            continue
        row = _masked_logits_row(logits[i], mask[i])
        vals, order = torch.sort(row, descending=True)
        # order[j] is object index
        pos = (order == t).nonzero(as_tuple=True)[0]
        if pos.numel() == 0:
            out.append(None)
        else:
            out.append(int(pos[0].item()))
    return out


def per_sample_recall_at_k(
    logits: torch.Tensor,
    target_index: torch.Tensor,
    mask: torch.Tensor,
    k: int,
) -> list[bool]:
    """Whether gold is in top-k by coarse logits (masked)."""
    row = logits.masked_fill(~mask, float("-inf"))
    out: list[bool] = []
    for i in range(logits.size(0)):
        t = int(target_index[i].item())
        ni = min(k, int(mask[i].sum().item()))
        if ni < 1 or t < 0 or t >= logits.size(1) or not mask[i, t]:
            out.append(False)
            continue
        _, topi = torch.topk(row[i], k=ni)
        out.append(bool((topi == t).any().item()))
    return out


def coarse_logit_margin(
    logits: torch.Tensor,
    target_index: torch.Tensor,
    mask: torch.Tensor,
) -> list[float | None]:
    """target_logit - max_{j!=t, mask} logit_j; None if undefined."""
    out: list[float | None] = []
    b, n = logits.shape
    for i in range(b):
        t = int(target_index[i].item())
        if t < 0 or t >= n or not mask[i, t]:
            out.append(None)
            continue
        lt = logits[i, t]
        neg_mask = mask[i].clone()
        neg_mask[t] = False
        if not neg_mask.any():
            out.append(None)
            continue
        neg_max = logits[i].masked_fill(~neg_mask, float("-inf")).max()
        if not torch.isfinite(neg_max):
            out.append(None)
        else:
            out.append(float((lt - neg_max).item()))
    return out


def aggregate_recall_at_ks(
    logits: torch.Tensor,
    target_index: torch.Tensor,
    mask: torch.Tensor,
    ks: tuple[int, ...] = (1, 5, 10, 20),
) -> dict[str, float]:
    out: dict[str, float] = {}
    n = logits.size(0)
    if n == 0:
        for k in ks:
            out[f"recall@{k}"] = 0.0
        return out
    for k in ks:
        hits = per_sample_recall_at_k(logits, target_index, mask, k)
        out[f"recall@{k}"] = sum(hits) / len(hits)
    return out


def recall_bucket(rank: int | None, ks: tuple[int, ...] = (1, 5, 10, 20)) -> str:
    if rank is None:
        return "invalid_target"
    for k in ks:
        if rank < k:
            return f"top{k}"
    return "beyond_20"


def topk_coverage_stats(
    logits: torch.Tensor,
    mask: torch.Tensor,
    k: int,
) -> dict[str, float]:
    """Mean effective K (min(k, n_objects)) and fraction of scenes with at least k valid objects."""
    eff: list[int] = []
    full = 0
    for i in range(logits.size(0)):
        nobj = int(mask[i].sum().item())
        eff.append(min(k, max(nobj, 1)))
        if nobj >= k:
            full += 1
    return {
        "mean_effective_k": float(sum(eff) / max(len(eff), 1)),
        "fraction_scenes_with_at_least_k_objects": full / max(logits.size(0), 1),
    }


def _meta_tag_bool(m: dict[str, Any], key: str) -> bool:
    return bool((m.get("tags") or {}).get(key, False))


def _meta_candidate_load_bucket(m: dict[str, Any]) -> str:
    cl = m.get("candidate_load")
    if isinstance(cl, str) and cl:
        return cl
    n = m.get("n_objects")
    if n is None:
        return "unknown"
    try:
        ni = int(n)
    except (TypeError, ValueError):
        return "unknown"
    if ni <= 8:
        return "low"
    if ni <= 24:
        return "medium"
    return "high"


def stratified_recall_from_lists(
    recall_hits: dict[int, list[bool]],
    meta: list[dict[str, Any]],
    ks: tuple[int, ...] = (1, 5, 10, 20),
) -> dict[str, float]:
    """Slice recall@K by geometry fallback, candidate load, same-class clutter."""
    n = len(meta)
    out: dict[str, float] = {}
    if n == 0:
        return out

    def recall_where(k: int, pred: Any) -> float | None:
        hits = recall_hits.get(k)
        if not hits or len(hits) != n:
            return None
        idx = [i for i in range(n) if pred(meta[i])]
        if not idx:
            return None
        return sum(hits[i] for i in idx) / len(idx)

    for k in ks:
        for name, pred in [
            ("geometry_high_fallback", lambda m: _meta_tag_bool(m, "geometry_high_fallback")),
            ("geometry_low_fallback", lambda m: not _meta_tag_bool(m, "geometry_high_fallback")),
            ("same_class_clutter", lambda m: _meta_tag_bool(m, "same_class_clutter")),
            ("candidate_load::low", lambda m: _meta_candidate_load_bucket(m) == "low"),
            ("candidate_load::medium", lambda m: _meta_candidate_load_bucket(m) == "medium"),
            ("candidate_load::high", lambda m: _meta_candidate_load_bucket(m) == "high"),
        ]:
            v = recall_where(k, pred)
            if v is not None:
                out[f"recall@{k}_slice::{name}"] = float(v)

    return out


def gold_rank_summary(ranks: list[int | None]) -> dict[str, float]:
    valid = [r for r in ranks if r is not None]
    if not valid:
        return {"gold_rank_mean": float("nan"), "gold_rank_median": float("nan")}
    valid_sorted = sorted(valid)
    mid = len(valid_sorted) // 2
    med = valid_sorted[mid] if len(valid_sorted) % 2 else (valid_sorted[mid - 1] + valid_sorted[mid]) / 2
    return {
        "gold_rank_mean": sum(valid) / len(valid),
        "gold_rank_median": float(med),
        "gold_rank_valid_fraction": len(valid) / max(len(ranks), 1),
    }


def margin_summary(margins: list[float | None]) -> dict[str, float]:
    vals = [m for m in margins if m is not None]
    if not vals:
        return {"coarse_margin_mean": float("nan")}
    return {
        "coarse_margin_mean": sum(vals) / len(vals),
        "coarse_margin_median": float(sorted(vals)[len(vals) // 2]),
    }


def coarse_accuracy_at1_from_logits(
    logits: torch.Tensor,
    target_index: torch.Tensor,
    mask: torch.Tensor,
) -> list[bool]:
    row = logits.masked_fill(~mask, float("-inf"))
    pred = row.argmax(dim=-1)
    return [bool(pred[i].item() == target_index[i].item()) for i in range(logits.size(0))]


def build_coarse_eval_bundle(
    logits: torch.Tensor,
    target_index: torch.Tensor,
    mask: torch.Tensor,
    meta: list[dict[str, Any]],
    topk_for_coverage: int = 10,
    ks: tuple[int, ...] = (1, 5, 10, 20),
) -> dict[str, Any]:
    """Single dict for JSON metrics (coarse-only pass)."""
    ranks = gold_rank_in_scene(logits, target_index, mask)
    margins = coarse_logit_margin(logits, target_index, mask)
    recall_hits = {k: per_sample_recall_at_k(logits, target_index, mask, k) for k in ks}
    c1 = coarse_accuracy_at1_from_logits(logits, target_index, mask)
    strat_c1 = stratified_accuracy_from_lists(c1, meta)

    bundle: dict[str, Any] = {
        "n": int(logits.size(0)),
        **aggregate_recall_at_ks(logits, target_index, mask, ks),
        **gold_rank_summary(ranks),
        **margin_summary(margins),
        "topk_coverage_k{}".format(topk_for_coverage): topk_coverage_stats(logits, mask, topk_for_coverage),
        "stratified_acc@1_coarse": strat_c1,
        "stratified_recall_slices": stratified_recall_from_lists(recall_hits, meta, ks),
    }
    return bundle


def eval_coarse_stage1_metrics(
    coarse: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    margin_thresh: float,
    ks: tuple[int, ...] = (1, 5, 10, 20),
    topk_coverage: int = 10,
) -> dict[str, Any]:
    """Full-scene coarse diagnostics for stage-1 recall pass (single forward per batch)."""
    recall_hits: dict[int, list[bool]] = {k: [] for k in ks}
    ranks: list[int | None] = []
    margins: list[float | None] = []
    c1: list[bool] = []
    c5: list[bool] = []
    meta_flat: list[dict[str, Any]] = []
    eff_k_sum = 0
    cov_count = 0
    ge_k = 0
    coarse.eval()
    for batch in loader:
        b = to_dev_batch(batch, device)
        samples = b["samples_ref"]
        with torch.no_grad():
            logits = coarse_forward(coarse, b)
        mask = b["object_mask"]
        target = b["target_index"]
        meta = copy.deepcopy(b["meta"])
        # Match order in eval_attribute_full_scene (two_stage_eval) for identical meta tags per batch.
        augment_meta_with_model_margins(logits.detach().cpu(), mask.cpu(), meta, margin_thresh=margin_thresh)
        augment_meta_geometry_fallback_tags(meta, samples)
        for k in ks:
            recall_hits[k].extend(per_sample_recall_at_k(logits, target, mask, k))
        ranks.extend(gold_rank_in_scene(logits, target, mask))
        margins.extend(coarse_logit_margin(logits, target, mask))
        c1.extend(coarse_accuracy_at1_from_logits(logits, target, mask))
        c5.extend(per_sample_correct_at5(logits, target, mask))
        meta_flat.extend(meta)
        for i in range(mask.size(0)):
            nobj = int(mask[i].sum().item())
            eff_k_sum += min(topk_coverage, max(nobj, 1))
            if nobj >= topk_coverage:
                ge_k += 1
            cov_count += 1
    n = len(c1)
    out: dict[str, Any] = {
        "n": n,
        "acc@1": sum(c1) / max(n, 1),
        "acc@5": sum(c5) / max(n, 1),
        **{f"recall@{k}": sum(recall_hits[k]) / max(len(recall_hits[k]), 1) for k in ks},
        **gold_rank_summary(ranks),
        **margin_summary(margins),
        f"topk_coverage_k{topk_coverage}": {
            "mean_effective_k": eff_k_sum / max(cov_count, 1),
            "fraction_scenes_with_at_least_k_objects": ge_k / max(cov_count, 1),
        },
        "stratified_acc@1_coarse": stratified_accuracy_from_lists(c1, meta_flat),
        "stratified_recall_slices": stratified_recall_from_lists(recall_hits, meta_flat, ks),
    }
    return out
