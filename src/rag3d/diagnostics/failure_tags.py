from __future__ import annotations

from rag3d.datasets.schemas import FailureTag, GroundingSample


def _norm_relation_token(s: str) -> str:
    return s.replace("_", "-").replace(" ", "-").strip().lower()


def infer_failure_tags(
    pred_idx: int,
    gold_idx: int,
    parser_confidence: float,
    anchor_entropy: float,
    target_margin: float,
    sample: GroundingSample | None = None,
    relation_types_parsed: list[str] | None = None,
    coarse_target_in_topk: bool | None = None,
    rerank_applied: bool = False,
) -> list[FailureTag]:
    tags: list[FailureTag] = []
    if parser_confidence < 0.35:
        tags.append(FailureTag.PARSER_FAILURE)
    if anchor_entropy > 1.5:
        tags.append(FailureTag.AMBIGUOUS_ANCHOR)
    if target_margin < 0.1:
        tags.append(FailureTag.LOW_CONFIDENCE)
    if pred_idx != gold_idx and sample is not None:
        gold_cls = sample.objects[gold_idx].class_name
        pred_cls = sample.objects[pred_idx].class_name
        if gold_cls == pred_cls:
            tags.append(FailureTag.SAME_CLASS_CONFUSION)
    if sample is not None:
        vo = sample.objects[pred_idx].visibility_occlusion_proxy
        if vo is not None and vo < 0.3:
            tags.append(FailureTag.OCCLUSION_RISK)
        if sample.tags.get("candidate_load") == "high":
            tags.append(FailureTag.HIGH_CANDIDATE_LOAD)
    if (
        sample is not None
        and sample.relation_type_gold
        and relation_types_parsed
        and str(sample.relation_type_gold).lower() not in {"", "none"}
    ):
        gold_n = _norm_relation_token(str(sample.relation_type_gold))
        parsed_n = {_norm_relation_token(p) for p in relation_types_parsed if p and str(p).lower() != "none"}
        if parsed_n and gold_n not in parsed_n and not any(gold_n in p or p in gold_n for p in parsed_n):
            tags.append(FailureTag.RELATION_MISMATCH)
    if rerank_applied and coarse_target_in_topk is False:
        tags.append(FailureTag.COARSE_TARGET_NOT_IN_TOPK)
    if (
        rerank_applied
        and coarse_target_in_topk is True
        and pred_idx != gold_idx
        and sample is not None
    ):
        tags.append(FailureTag.RERANK_STAGE_FAILURE)
    if rerank_applied and sample is not None and sample.objects:
        nobj = len(sample.objects)
        fb = sum(1 for o in sample.objects if o.geometry_quality == "fallback_centroid") / max(nobj, 1)
        if fb > 0.5:
            tags.append(FailureTag.WEAK_GEOMETRY_CONTEXT)
        syn = sum(1 for o in sample.objects if o.feature_source == "synthetic_collate") / max(nobj, 1)
        if syn > 0.8:
            tags.append(FailureTag.WEAK_FEATURE_SOURCE)
    if not tags:
        tags.append(FailureTag.OK)
    return tags
