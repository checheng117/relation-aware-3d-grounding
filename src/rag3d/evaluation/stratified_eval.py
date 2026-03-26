from __future__ import annotations

from typing import Any

import torch

from rag3d.evaluation.metrics import accuracy_at_k


def stratified_accuracy(
    logits: torch.Tensor,
    target_index: torch.Tensor,
    mask: torch.Tensor,
    meta: list[dict[str, Any]],
) -> dict[str, float]:
    """meta[i] may include relation_type, tags: {same_class_clutter, ...}."""
    out: dict[str, float] = {}
    if not meta or len(meta) != logits.size(0):
        return out

    def _subset(pred: callable) -> torch.Tensor:
        return torch.tensor(
            [bool(pred(meta[i])) for i in range(len(meta))],
            device=logits.device,
            dtype=torch.bool,
        )

    rel_keys = {m.get("relation_type_gold") or m.get("relation_type") for m in meta}
    for rk in rel_keys:
        if rk is None:
            continue
        sel = _subset(lambda m, r=rk: (m.get("relation_type_gold") or m.get("relation_type")) == r)
        if sel.any():
            key = f"acc@1_rel::{rk}"
            out[key] = accuracy_at_k(logits[sel], target_index[sel], mask[sel], 1)

    sel_clutter = _subset(lambda m: (m.get("tags") or {}).get("same_class_clutter", False))
    if sel_clutter.any():
        out["acc@1_subset::same_class_clutter"] = accuracy_at_k(
            logits[sel_clutter], target_index[sel_clutter], mask[sel_clutter], 1
        )

    sel_occ = _subset(lambda m: (m.get("tags") or {}).get("occlusion_heavy", False))
    if sel_occ.any():
        out["acc@1_subset::occlusion_heavy"] = accuracy_at_k(
            logits[sel_occ], target_index[sel_occ], mask[sel_occ], 1
        )

    sel_anchor = _subset(lambda m: (m.get("tags") or {}).get("anchor_confusion", False))
    if sel_anchor.any():
        out["acc@1_subset::anchor_confusion"] = accuracy_at_k(
            logits[sel_anchor], target_index[sel_anchor], mask[sel_anchor], 1
        )

    sel_pf = _subset(lambda m: (m.get("tags") or {}).get("parser_failure", False))
    if sel_pf.any():
        out["acc@1_subset::parser_failure"] = accuracy_at_k(logits[sel_pf], target_index[sel_pf], mask[sel_pf], 1)

    sel_lm = _subset(lambda m: (m.get("tags") or {}).get("low_model_margin", False))
    if sel_lm.any():
        out["acc@1_subset::low_model_margin"] = accuracy_at_k(
            logits[sel_lm], target_index[sel_lm], mask[sel_lm], 1
        )

    return out


def augment_meta_with_model_margins(
    logits: torch.Tensor,
    mask: torch.Tensor,
    meta: list[dict[str, Any]],
    margin_thresh: float = 0.15,
) -> None:
    """In-place: add tags['low_model_margin'] from predicted logit margin."""
    from rag3d.evaluation.metrics import logit_top12_margin

    for i in range(logits.size(0)):
        marg = logit_top12_margin(logits[i], mask[i])
        m = dict(meta[i])
        tags = dict(m.get("tags") or {})
        tags["low_model_margin"] = marg < margin_thresh
        m["tags"] = tags
        meta[i] = m


def stratified_accuracy_from_lists(correct: list[bool], meta: list[dict[str, Any]]) -> dict[str, float]:
    """Dataset-wide accuracy on named strata (per-sample lists)."""
    n = len(correct)
    out: dict[str, float] = {}
    if n == 0 or len(meta) != n:
        return out

    def acc_where(pred: Any) -> float | None:
        idx = [i for i in range(n) if pred(meta[i])]
        if not idx:
            return None
        return sum(correct[i] for i in idx) / len(idx)

    rel_keys = {m.get("relation_type_gold") or m.get("relation_type") for m in meta}
    for rk in rel_keys:
        if rk is None:
            continue
        v = acc_where(lambda m, r=rk: (m.get("relation_type_gold") or m.get("relation_type")) == r)
        if v is not None:
            out[f"acc@1_rel::{rk}"] = float(v)

    for key, tag in [
        ("acc@1_subset::same_class_clutter", "same_class_clutter"),
        ("acc@1_subset::occlusion_heavy", "occlusion_heavy"),
        ("acc@1_subset::anchor_confusion", "anchor_confusion"),
        ("acc@1_subset::parser_failure", "parser_failure"),
        ("acc@1_subset::low_model_margin", "low_model_margin"),
        ("acc@1_subset::geometry_high_fallback", "geometry_high_fallback"),
        ("acc@1_subset::real_box_heavy", "real_box_heavy"),
        ("acc@1_subset::weak_feature_source", "weak_feature_source"),
    ]:
        v = acc_where(lambda m, t=tag: bool((m.get("tags") or {}).get(t, False)))
        if v is not None:
            out[key] = float(v)

    gfb = acc_where(
        lambda m: float(m.get("geometry_fallback_fraction") or (m.get("tags") or {}).get("geometry_fallback_fraction") or 0.0)
        > 0.5
    )
    if gfb is not None:
        out["acc@1_slice::geometry_fallback_gt_half"] = float(gfb)
    gfb_lo = acc_where(
        lambda m: float(m.get("geometry_fallback_fraction") or (m.get("tags") or {}).get("geometry_fallback_fraction") or 0.0)
        <= 0.5
    )
    if gfb_lo is not None:
        out["acc@1_slice::geometry_fallback_le_half"] = float(gfb_lo)

    return out


def augment_meta_geometry_fallback_tags(meta: list[dict[str, Any]], samples: list[Any]) -> None:
    """In-place: tags for blueprint geometry slices (uses ``SceneObject.geometry_quality`` when present)."""
    for i, m in enumerate(meta):
        if i >= len(samples):
            continue
        s = samples[i]
        objs = getattr(s, "objects", None) or []
        if not objs:
            continue
        n = len(objs)
        fb = sum(1 for o in objs if getattr(o, "geometry_quality", None) == "fallback_centroid") / max(n, 1)
        real = sum(1 for o in objs if getattr(o, "geometry_quality", None) == "obb_aabb") / max(n, 1)
        syn = sum(1 for o in objs if getattr(o, "feature_source", None) == "synthetic_collate") / max(n, 1)
        mm = dict(m)
        tags = dict(mm.get("tags") or {})
        tags["geometry_high_fallback"] = fb > 0.5
        tags["real_box_heavy"] = real >= 0.5
        tags["weak_feature_source"] = syn > 0.8
        mm["geometry_fallback_fraction"] = fb
        mm["tags"] = tags
        meta[i] = mm
