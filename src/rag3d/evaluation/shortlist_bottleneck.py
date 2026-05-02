"""Shortlist recall curve + two-stage bottleneck (oracle vs rerank)."""

from __future__ import annotations

import copy
from typing import Any

import torch
from torch.utils.data import DataLoader

from rag3d.evaluation.coarse_recall import eval_coarse_stage1_metrics
from rag3d.evaluation.two_stage_eval import to_dev_batch
from rag3d.relation_reasoner.two_stage_rerank import TwoStageCoarseRerankModel


def coarse_recall_curve(
    coarse: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    margin_thresh: float,
    ks: tuple[int, ...] = (1, 5, 10, 20, 40),
) -> dict[str, Any]:
    """Wrapper with K=40 default for bottleneck plots."""
    return eval_coarse_stage1_metrics(coarse, loader, device, margin_thresh, ks=ks)


def eval_two_stage_bottleneck(
    model: TwoStageCoarseRerankModel,
    loader: DataLoader,
    device: torch.device,
    parser: Any,
    margin_thresh: float,
) -> dict[str, Any]:
    """Oracle upper bound = P(target in shortlist) under eval (no train-time target injection).

    Conditional rerank accuracy = Acc@1 among samples with target in shortlist.
    """
    from rag3d.evaluation.metrics import per_sample_correct_at1, per_sample_correct_at5
    from rag3d.evaluation.stratified_eval import augment_meta_with_model_margins, stratified_accuracy_from_lists

    c1: list[bool] = []
    c5: list[bool] = []
    in_shortlist: list[bool] = []
    meta_flat: list[dict] = []
    rerank_k = int(model.rerank_k)

    for batch in loader:
        b = to_dev_batch(batch, device)
        samples = b["samples_ref"]
        parsed_list = [parser.parse(s.utterance) for s in samples]
        with torch.no_grad():
            logits, _, aux = model(
                {k: b[k] for k in ("object_features", "object_mask", "raw_texts", "samples_ref")},
                parsed_list=parsed_list,
                target_index=None,
            )
        idx = aux["rerank_idx"]
        mask = b["object_mask"]
        target = b["target_index"]
        for bi in range(target.size(0)):
            t = int(target[bi].item())
            gold_in = bool((idx[bi] == t).any().item())
            in_shortlist.append(gold_in)
        meta = copy.deepcopy(b["meta"])
        augment_meta_with_model_margins(logits.detach().cpu(), mask.cpu(), meta, margin_thresh=margin_thresh)
        c1.extend(per_sample_correct_at1(logits, target, mask))
        c5.extend(per_sample_correct_at5(logits, target, mask))
        meta_flat.extend(meta)

    n = len(c1)
    in_k = sum(in_shortlist)
    cond_correct = sum(c1[i] for i in range(n) if in_shortlist[i])
    oracle_acc = in_k / max(n, 1)
    rerank_given_in = cond_correct / max(in_k, 1) if in_k else float("nan")

    strat = stratified_accuracy_from_lists(c1, meta_flat)
    return {
        "n": n,
        "rerank_k": rerank_k,
        "acc@1": sum(c1) / max(n, 1),
        "acc@5": sum(c5) / max(n, 1),
        "shortlist_recall": oracle_acc,
        "oracle_upper_bound_perfect_rerank": oracle_acc,
        "n_target_in_shortlist": in_k,
        "rerank_acc_given_target_in_shortlist": rerank_given_in,
        "stratified": strat,
    }
