"""Build the downstream-facing :class:`BridgeModuleOutput` from logits and optional parse state."""

from __future__ import annotations

from typing import Any

import torch

from rag3d.datasets.schemas import BridgeModuleOutput, FailureTag, GroundingSample, ModelPrediction, ParsedUtterance
from rag3d.diagnostics.confidence import anchor_entropy, logits_to_confidence_masked, target_margin
from rag3d.diagnostics.failure_tags import infer_failure_tags
from rag3d.evaluation.coarse_recall import recall_bucket


def rerank_extras_for_sample(
    coarse_logits_row: torch.Tensor,
    mask_row: torch.Tensor,
    rerank_idx_row: torch.Tensor,
    target_index: int,
    k_eff: int,
) -> dict[str, Any]:
    """Serializable rerank diagnostics for :func:`build_bridge_module_output`."""
    idx = rerank_idx_row.detach().long()
    scores = coarse_logits_row[idx].detach().cpu().tolist()
    ids = idx.cpu().tolist()
    ti = int(target_index)
    gold_in = ti in set(ids)
    row = coarse_logits_row.masked_fill(~mask_row, float("-inf"))
    coarse_pred_idx = int(row.argmax().item()) if mask_row.any() else -1
    gold_rank: int | None = None
    coarse_margin: float | None = None
    if 0 <= ti < coarse_logits_row.numel() and bool(mask_row[ti].item()):
        _, order = torch.sort(row, descending=True)
        hit = (order == ti).nonzero(as_tuple=True)[0]
        gold_rank = int(hit[0].item()) if hit.numel() else None
        lt = coarse_logits_row[ti]
        neg_m = mask_row.clone()
        neg_m[ti] = False
        if neg_m.any():
            mx = coarse_logits_row.masked_fill(~neg_m, float("-inf")).max()
            coarse_margin = float((lt - mx).item()) if torch.isfinite(mx) else None
    bucket = recall_bucket(gold_rank) if gold_rank is not None else "invalid_target"
    # softmax confidence over shortlisted coarse logits (diagnostic only)
    sub = coarse_logits_row[idx].float()
    sub = sub - sub.max()
    exp = sub.exp()
    stage_conf = float((exp / exp.sum().clamp_min(1e-8)).max().item()) if exp.numel() else None
    return {
        "rerank_applied": True,
        "coarse_topk_ids": ids,
        "coarse_topk_scores": scores,
        "coarse_target_in_topk": gold_in,
        "coarse_gold_rank": gold_rank,
        "coarse_margin": coarse_margin,
        "coarse_recall_bucket": bucket,
        "coarse_pred_idx": coarse_pred_idx,
        "coarse_stage_confidence": stage_conf,
        "topk_recall_success": gold_in,
        "rerank_k": int(k_eff),
    }


def candidate_summary_from_sample(sample: GroundingSample) -> dict[str, Any]:
    tags = dict(sample.tags) if sample.tags else {}
    n = len(sample.objects)
    fb = tags.get("geometry_fallback_fraction")
    if fb is None:
        fb = sum(1 for o in sample.objects if o.geometry_quality == "fallback_centroid") / max(n, 1)
    return {
        "n_objects": n,
        "scene_id": sample.scene_id,
        "candidate_load": tags.get("candidate_load", "unknown"),
        "geometry_fallback_fraction": float(fb),
        "same_class_clutter": bool(tags.get("same_class_clutter", False)),
        "anchor_confusion_tag": bool(tags.get("anchor_confusion", False)),
    }


def build_bridge_module_output(
    logits_row: torch.Tensor,
    mask_row: torch.Tensor,
    target_index: int,
    sample: GroundingSample,
    pred_idx: int,
    anchor_dist_row: torch.Tensor | None,
    parser_confidence: float,
    parsed: ParsedUtterance | None,
    rerank_extras: dict[str, Any] | None = None,
) -> BridgeModuleOutput:
    margin = target_margin(logits_row, mask_row)
    ent = anchor_entropy(anchor_dist_row, mask_row) if anchor_dist_row is not None else 0.0
    rel_types = list(parsed.relation_types) if parsed else []
    rex = rerank_extras or {}
    rerank_applied = bool(rex.get("rerank_applied", False))
    coarse_in = rex.get("coarse_target_in_topk")
    coarse_in_b = coarse_in if isinstance(coarse_in, bool) else None
    cgr = rex.get("coarse_gold_rank")
    coarse_gold_rank = int(cgr) if isinstance(cgr, int) else None
    crb = rex.get("coarse_recall_bucket")
    coarse_recall_bucket = str(crb) if isinstance(crb, str) else None
    csc = rex.get("coarse_stage_confidence")
    coarse_stage_confidence = float(csc) if isinstance(csc, (int, float)) else None
    trs = rex.get("topk_recall_success")
    topk_recall_success = trs if isinstance(trs, bool) else coarse_in_b
    cpred = rex.get("coarse_pred_idx")
    rescued: bool | None = None
    if rerank_applied and coarse_in_b is True and isinstance(cpred, int):
        rescued = cpred != target_index and pred_idx == target_index
    elif rerank_applied and coarse_in_b is False:
        rescued = False
    stage1_reason: str | None = None
    if rerank_applied and coarse_in_b is False:
        stage1_reason = "coarse_target_not_in_topk"
    elif rerank_applied and coarse_in_b is True and isinstance(cpred, int) and cpred != target_index and pred_idx != target_index:
        stage1_reason = "coarse_wrong_but_in_shortlist_rerank_failed"
    ftags = infer_failure_tags(
        pred_idx,
        target_index,
        parser_confidence,
        ent,
        margin,
        sample,
        relation_types_parsed=rel_types,
        coarse_target_in_topk=coarse_in_b,
        rerank_applied=rerank_applied,
    )
    conf = logits_to_confidence_masked(logits_row, mask_row, pred_idx) if mask_row.any() else 0.0
    scores = logits_row[mask_row].detach().cpu().tolist()
    adist = (
        anchor_dist_row[mask_row].detach().cpu().tolist()
        if anchor_dist_row is not None and mask_row.any()
        else []
    )
    tid = None
    if 0 <= pred_idx < len(sample.objects):
        tid = str(sample.objects[pred_idx].object_id)
    rationale = "structured_relation_soft_anchor" if anchor_dist_row is not None else "attribute_or_raw_text"
    if rerank_applied:
        rationale = "two_stage_coarse_topk_relation_rerank"
    mp_tags = [t for t in ftags if t != FailureTag.OK]
    cand = candidate_summary_from_sample(sample)
    cload = cand.get("candidate_load")
    if isinstance(cload, str):
        cload_out: str | None = cload
    else:
        cload_out = sample.tags.get("candidate_load") if isinstance(sample.tags.get("candidate_load"), str) else None
    return BridgeModuleOutput(
        target_id=tid,
        target_index_pred=pred_idx,
        final_target_id=tid,
        target_scores=scores,
        anchor_distribution=adist,
        relation_rationale=rationale,
        confidence=float(conf),
        failure_tags=[t.value for t in mp_tags],
        target_margin=float(margin),
        anchor_entropy=float(ent),
        candidate_summary=cand,
        candidate_load=cload_out,
        coarse_topk_ids=list(rex.get("coarse_topk_ids", [])),
        coarse_topk_scores=list(rex.get("coarse_topk_scores", [])),
        coarse_target_in_topk=coarse_in_b,
        coarse_gold_rank=coarse_gold_rank,
        coarse_recall_bucket=coarse_recall_bucket,
        coarse_stage_confidence=coarse_stage_confidence,
        topk_recall_success=topk_recall_success,
        rerank_rescued_from_coarse_shortlist=rescued,
        stage1_failure_reason=stage1_reason,
        rerank_applied=rerank_applied,
        rerank_k=rex.get("rerank_k") if rex.get("rerank_k") is not None else None,
        parse_source=parsed.parse_source if parsed else None,
        parse_warnings=list(parsed.parse_warnings) if parsed else [],
    )


def model_prediction_from_bridge(
    bridge: BridgeModuleOutput,
) -> ModelPrediction:
    """Legacy :class:`ModelPrediction` view (enum failure tags)."""
    fenum = []
    for s in bridge.failure_tags:
        try:
            fenum.append(FailureTag(s))
        except ValueError:
            continue
    return ModelPrediction(
        target_id_pred=bridge.target_id,
        target_index_pred=bridge.target_index_pred,
        target_scores=bridge.target_scores,
        anchor_distribution=bridge.anchor_distribution,
        relation_rationale=bridge.relation_rationale,
        confidence=bridge.confidence,
        failure_tags=fenum,
    )
