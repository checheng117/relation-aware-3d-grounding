"""Shared eval helpers for coarse-only, coarse@topK, and two-stage rerank (blueprint scripts)."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from rag3d.evaluation.metrics import per_sample_correct_at1, per_sample_correct_at5
from rag3d.evaluation.stratified_eval import (
    augment_meta_geometry_fallback_tags,
    augment_meta_with_model_margins,
    stratified_accuracy_from_lists,
)
from rag3d.parsers.cached_parser import CachedParser
from rag3d.relation_reasoner.model import AttributeOnlyModel, CoarseGeomAttributeModel
from rag3d.relation_reasoner.two_stage_rerank import (
    RelationAwareGeomModel,
    TwoStageCoarseRerankModel,
    _effective_topk,
    _topk_union_target,
)


def coarse_forward(coarse: nn.Module, b: dict[str, Any]) -> torch.Tensor:
    sub: dict[str, Any] = {k: b[k] for k in ("object_features", "object_mask", "raw_texts")}
    if getattr(coarse, "uses_geometry_context", False):
        sub["samples_ref"] = b["samples_ref"]
    return coarse(sub)


def to_dev_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in batch.items():
        if k in ("meta", "samples_ref"):
            out[k] = v
        elif isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def load_coarse_model(
    mcfg: dict[str, Any],
    ckpt: Path,
    device: torch.device,
    coarse_kind: str = "attribute_only",
) -> nn.Module:
    kind = str(coarse_kind).lower()
    if kind == "coarse_geom":
        m: nn.Module = CoarseGeomAttributeModel(
            int(mcfg["object_dim"]),
            int(mcfg["language_dim"]),
            int(mcfg["hidden_dim"]),
            dropout=float(mcfg.get("dropout", 0.1)),
        )
    else:
        m = AttributeOnlyModel(
            int(mcfg["object_dim"]),
            int(mcfg["language_dim"]),
            int(mcfg["hidden_dim"]),
            dropout=float(mcfg.get("dropout", 0.1)),
        )
    m = m.to(device)
    try:
        data = torch.load(ckpt, map_location=device, weights_only=False)
    except TypeError:
        data = torch.load(ckpt, map_location=device)
    sd = data["model"] if isinstance(data, dict) and "model" in data else data
    m.load_state_dict(sd, strict=True)
    m.eval()
    return m


def load_two_stage_model(
    mcfg: dict[str, Any],
    coarse_ckpt: Path,
    fine_ckpt: Path | None,
    rerank_k: int,
    device: torch.device,
    coarse_kind: str = "attribute_only",
) -> TwoStageCoarseRerankModel:
    coarse = load_coarse_model(mcfg, coarse_ckpt, device, coarse_kind)
    for p in coarse.parameters():
        p.requires_grad = False
    fine = RelationAwareGeomModel(
        int(mcfg["object_dim"]),
        int(mcfg["language_dim"]),
        int(mcfg["hidden_dim"]),
        int(mcfg["relation_dim"]),
        anchor_temperature=float(mcfg.get("anchor_temperature", 1.0)),
        dropout=float(mcfg.get("dropout", 0.1)),
    )
    model = TwoStageCoarseRerankModel(coarse, fine, rerank_k=rerank_k).to(device)
    if fine_ckpt is not None and fine_ckpt.is_file():
        try:
            data = torch.load(fine_ckpt, map_location=device, weights_only=False)
        except TypeError:
            data = torch.load(fine_ckpt, map_location=device)
        sd = data["model"] if isinstance(data, dict) and "model" in data else data
        model.load_state_dict(sd, strict=True)
    model.eval()
    return model


def eval_two_stage(
    model: TwoStageCoarseRerankModel,
    loader: DataLoader,
    device: torch.device,
    parser: CachedParser,
    margin_thresh: float,
) -> dict[str, Any]:
    c1: list[bool] = []
    c5: list[bool] = []
    coarse_hit: list[bool] = []
    rescue_hit: list[bool] = []
    meta_flat: list[dict] = []
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
        coarse_logits = aux["coarse_logits"]
        mask = b["object_mask"]
        target = b["target_index"]
        for bi in range(b["target_index"].size(0)):
            t = int(b["target_index"][bi].item())
            gold_in = bool((idx[bi] == t).any().item())
            coarse_hit.append(gold_in)
            row_c = coarse_logits[bi].masked_fill(~mask[bi], float("-inf"))
            row_f = logits[bi].masked_fill(~mask[bi], float("-inf"))
            cpred = int(row_c.argmax().item())
            fpred = int(row_f.argmax().item())
            rescue_hit.append(gold_in and cpred != t and fpred == t)
        meta = copy.deepcopy(b["meta"])
        augment_meta_with_model_margins(logits.detach().cpu(), mask.cpu(), meta, margin_thresh=margin_thresh)
        augment_meta_geometry_fallback_tags(meta, samples)
        c1.extend(per_sample_correct_at1(logits, target, mask))
        c5.extend(per_sample_correct_at5(logits, target, mask))
        meta_flat.extend(meta)
    n = len(c1)
    strat = stratified_accuracy_from_lists(c1, meta_flat)
    return {
        "acc@1": sum(c1) / max(n, 1),
        "acc@5": sum(c5) / max(n, 1),
        "n": n,
        "coarse_target_in_topk_rate": sum(coarse_hit) / max(len(coarse_hit), 1),
        "topk_recall_success_rate": sum(coarse_hit) / max(len(coarse_hit), 1),
        "rerank_rescue_rate": sum(rescue_hit) / max(len(rescue_hit), 1),
        "stratified": strat,
    }


def eval_coarse_topk_attribute(
    coarse: nn.Module,
    loader: DataLoader,
    device: torch.device,
    rerank_k: int,
    margin_thresh: float,
) -> dict[str, Any]:
    """Argmax of coarse logits restricted to top-``rerank_k`` coarse candidates (no target oracle)."""
    c1: list[bool] = []
    c5: list[bool] = []
    coarse_hit: list[bool] = []
    meta_flat: list[dict] = []
    for batch in loader:
        b = to_dev_batch(batch, device)
        samples = b["samples_ref"]
        with torch.no_grad():
            coarse_logits = coarse_forward(coarse, b)
        k_eff = _effective_topk(b["object_mask"], rerank_k)
        idx = _topk_union_target(coarse_logits, b["object_mask"], None, k_eff, training=False)
        bsz, nobj = coarse_logits.shape
        for bi in range(bsz):
            row_idx = idx[bi]
            t = int(b["target_index"][bi].item())
            coarse_hit.append(bool((row_idx == t).any().item()))
            mask_row = torch.zeros(nobj, dtype=torch.bool, device=device)
            mask_row[row_idx] = True
            mask_row = mask_row & b["object_mask"][bi]
            vals = coarse_logits[bi].masked_fill(~mask_row, float("-inf"))
            pred = int(vals.argmax().item())
            gold = t
            c1.append(pred == gold)
            k5 = min(5, int(mask_row.sum().item()))
            if k5 > 0:
                _, top5 = torch.topk(vals, k=k5)
                c5.append(gold in top5.tolist())
            else:
                c5.append(False)
        meta = copy.deepcopy(b["meta"])
        augment_meta_geometry_fallback_tags(meta, samples)
        full_logits = coarse_logits.masked_fill(~b["object_mask"], float("-inf"))
        augment_meta_with_model_margins(full_logits.detach().cpu(), b["object_mask"].cpu(), meta, margin_thresh=margin_thresh)
        meta_flat.extend(meta)
    n = len(c1)
    return {
        "acc@1": sum(c1) / max(n, 1),
        "acc@5": sum(c5) / max(n, 1),
        "n": n,
        "coarse_target_in_topk_rate": sum(coarse_hit) / max(len(coarse_hit), 1),
        "stratified": stratified_accuracy_from_lists(c1, meta_flat),
    }


def eval_full_scene_relation_aware(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    parser: CachedParser,
    margin_thresh: float,
) -> dict[str, Any]:
    from rag3d.relation_reasoner.model import RelationAwareModel

    assert isinstance(model, RelationAwareModel)
    c1: list[bool] = []
    c5: list[bool] = []
    meta_flat: list[dict] = []
    for batch in loader:
        b = to_dev_batch(batch, device)
        samples = b["samples_ref"]
        parsed_list = [parser.parse(s.utterance) for s in samples]
        sub = {k: b[k] for k in ("object_features", "object_mask", "raw_texts")}
        with torch.no_grad():
            logits, _ = model(sub, parsed_list=parsed_list)
        mask = b["object_mask"]
        target = b["target_index"]
        meta = copy.deepcopy(b["meta"])
        augment_meta_with_model_margins(logits.detach().cpu(), mask.cpu(), meta, margin_thresh=margin_thresh)
        augment_meta_geometry_fallback_tags(meta, samples)
        c1.extend(per_sample_correct_at1(logits, target, mask))
        c5.extend(per_sample_correct_at5(logits, target, mask))
        meta_flat.extend(meta)
    n = len(c1)
    return {
        "acc@1": sum(c1) / max(n, 1),
        "acc@5": sum(c5) / max(n, 1),
        "n": n,
        "stratified": stratified_accuracy_from_lists(c1, meta_flat),
    }


def eval_attribute_full_scene(
    coarse: nn.Module,
    loader: DataLoader,
    device: torch.device,
    margin_thresh: float,
) -> dict[str, Any]:
    c1: list[bool] = []
    c5: list[bool] = []
    meta_flat: list[dict] = []
    for batch in loader:
        b = to_dev_batch(batch, device)
        samples = b["samples_ref"]
        with torch.no_grad():
            logits = coarse_forward(coarse, b)
        mask = b["object_mask"]
        target = b["target_index"]
        meta = copy.deepcopy(b["meta"])
        augment_meta_with_model_margins(logits.detach().cpu(), mask.cpu(), meta, margin_thresh=margin_thresh)
        augment_meta_geometry_fallback_tags(meta, samples)
        c1.extend(per_sample_correct_at1(logits, target, mask))
        c5.extend(per_sample_correct_at5(logits, target, mask))
        meta_flat.extend(meta)
    n = len(c1)
    return {
        "acc@1": sum(c1) / max(n, 1),
        "acc@5": sum(c5) / max(n, 1),
        "n": n,
        "stratified": stratified_accuracy_from_lists(c1, meta_flat),
    }
