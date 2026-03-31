"""Extended two-stage metrics: MRR, oracle vs natural shortlist, conditional rerank accuracy."""

from __future__ import annotations

import copy
from typing import Any

import torch
from torch.utils.data import DataLoader

from rag3d.evaluation.metrics import per_sample_correct_at1, per_sample_correct_at5
from rag3d.datasets.transforms import compute_stratification_tags
from rag3d.evaluation.stratified_eval import augment_meta_geometry_fallback_tags, augment_meta_with_model_margins
from rag3d.evaluation.two_stage_eval import to_dev_batch
from rag3d.relation_reasoner.two_stage_rerank import TwoStageCoarseRerankModel


def _per_sample_mrr(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> list[float]:
    """Reciprocal rank of gold among valid positions (1-indexed rank -> 1/rank)."""
    out: list[float] = []
    for i in range(logits.size(0)):
        row = logits[i].masked_fill(~mask[i], float("-inf"))
        t = int(target[i].item())
        if t < 0 or t >= row.numel() or not mask[i, t]:
            out.append(0.0)
            continue
        order = torch.argsort(row, descending=True)
        pos = (order == t).nonzero(as_tuple=True)[0]
        if pos.numel() == 0:
            out.append(0.0)
        else:
            rank = int(pos[0].item()) + 1
            out.append(1.0 / float(rank))
    return out


def eval_two_stage_inject_mode(
    model: TwoStageCoarseRerankModel,
    loader: DataLoader,
    device: torch.device,
    parser: Any,
    margin_thresh: float,
    inject_gold_in_shortlist: bool,
) -> dict[str, Any]:
    """Evaluate pipeline with fixed shortlist construction: natural (False) or oracle (True)."""
    model.eval()
    c1: list[bool] = []
    c5: list[bool] = []
    mrr: list[float] = []
    in_sl: list[bool] = []
    cond_ok: list[bool] = []
    meta_flat: list[dict] = []
    k_eff_seen: list[int] = []

    for batch in loader:
        b = to_dev_batch(batch, device)
        samples = b["samples_ref"]
        parsed_list = [parser.parse(s.utterance) for s in samples]
        with torch.no_grad():
            logits, _, aux = model(
                {k: b[k] for k in ("object_features", "object_mask", "raw_texts", "samples_ref")},
                parsed_list=parsed_list,
                target_index=b["target_index"],
                inject_gold_in_shortlist=inject_gold_in_shortlist,
            )
        idx = aux["rerank_idx"]
        mask = b["object_mask"]
        target = b["target_index"]
        k_eff_seen.append(int(aux.get("k_eff", model.rerank_k)))
        for bi in range(target.size(0)):
            t = int(target[bi].item())
            gold_in = bool((idx[bi] == t).any().item())
            in_sl.append(gold_in)
        meta = copy.deepcopy(b["meta"])
        augment_meta_with_model_margins(logits.detach().cpu(), mask.cpu(), meta, margin_thresh=margin_thresh)
        augment_meta_geometry_fallback_tags(meta, samples)
        c1.extend(per_sample_correct_at1(logits, target, mask))
        c5.extend(per_sample_correct_at5(logits, target, mask))
        mrr.extend(_per_sample_mrr(logits, target, mask))
        meta_flat.extend(meta)

    n = len(c1)
    for i in range(n):
        if in_sl[i]:
            cond_ok.append(c1[i])

    in_count = sum(in_sl)
    cond_acc = sum(cond_ok) / max(len(cond_ok), 1) if cond_ok else float("nan")

    return {
        "n": n,
        "inject_gold_in_shortlist": inject_gold_in_shortlist,
        "acc@1": sum(c1) / max(n, 1),
        "acc@5": sum(c5) / max(n, 1),
        "mrr": sum(mrr) / max(n, 1),
        "shortlist_recall": in_count / max(n, 1),
        "rerank_acc_given_gold_in_shortlist": cond_acc,
        "mean_k_eff": sum(k_eff_seen) / max(len(k_eff_seen), 1),
    }


def eval_by_candidate_load_bucket(
    model: TwoStageCoarseRerankModel,
    loader: DataLoader,
    device: torch.device,
    parser: Any,
    margin_thresh: float,
    inject_gold_in_shortlist: bool,
    bucket_key: str = "candidate_load",
) -> dict[str, dict[str, float]]:
    """Stratify acc@1 by meta bucket (e.g. low ~ controlled, high ~ full-scene proxy)."""
    model.eval()
    per_bucket: dict[str, list[bool]] = {}
    for batch in loader:
        b = to_dev_batch(batch, device)
        samples = b["samples_ref"]
        parsed_list = [parser.parse(s.utterance) for s in samples]
        with torch.no_grad():
            logits, _, _ = model(
                {k: b[k] for k in ("object_features", "object_mask", "raw_texts", "samples_ref")},
                parsed_list=parsed_list,
                target_index=b["target_index"],
                inject_gold_in_shortlist=inject_gold_in_shortlist,
            )
        mask = b["object_mask"]
        target = b["target_index"]
        pred = logits.masked_fill(~mask, float("-inf")).argmax(dim=-1)
        for i in range(target.size(0)):
            s = samples[i]
            st = compute_stratification_tags(s)
            cl = st.get("candidate_load") or "unknown"
            bk = str(cl) if bucket_key == "candidate_load" else "all"
            per_bucket.setdefault(bk, []).append(bool(pred[i].item() == target[i].item()))
    out: dict[str, dict[str, float]] = {}
    for bk, hits in per_bucket.items():
        out[bk] = {"acc@1": sum(hits) / max(len(hits), 1), "n": float(len(hits))}
    return out
