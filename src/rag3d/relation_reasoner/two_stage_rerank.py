"""Two-stage bridge: frozen coarse (attribute) scorer → top-K → relation-aware rerank with geometry context."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from rag3d.datasets.schemas import ParsedUtterance
from rag3d.encoders.object_encoder import ObjectMLPEncoderWithGeomContext
from rag3d.relation_reasoner.anchor_selector import soft_anchor_distribution
from rag3d.relation_reasoner.attribute_scorer import AttributeScorer
from rag3d.relation_reasoner.geom_context import batch_geom_context_tensor8
from rag3d.relation_reasoner.relation_scorer import PairwiseRelationScorer
from rag3d.relation_reasoner.text_encoding import StructuredTextEncoder


GEOM_DIM = 8


class RelationAwareGeomModel(nn.Module):
    """Structured relation path with quality-aware object MLP (box + flags, no fabricated points)."""

    def __init__(
        self,
        object_dim: int,
        language_dim: int,
        hidden_dim: int,
        relation_dim: int,
        geom_dim: int = GEOM_DIM,
        anchor_temperature: float = 1.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.geom_dim = geom_dim
        self.object_enc = ObjectMLPEncoderWithGeomContext(object_dim, geom_dim, hidden_dim, dropout)
        self.struct_text = StructuredTextEncoder(dim=language_dim)
        self.attr = AttributeScorer(hidden_dim, language_dim)
        self.rel = PairwiseRelationScorer(hidden_dim, relation_dim, language_dim)
        self.anchor_temperature = anchor_temperature

    def forward(
        self,
        object_features: torch.Tensor,
        object_geom: torch.Tensor,
        mask: torch.Tensor,
        parsed_list: list[ParsedUtterance],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns logits [B,N], anchor_dist [B,N] on the (possibly K-sized) candidate set."""
        h_o = self.object_enc(object_features, object_geom)
        b = object_features.size(0)
        th, ah, rh = [], [], []
        for i in range(b):
            if i < len(parsed_list) and parsed_list[i] is not None:
                p = parsed_list[i]
                th.append(p.target_head or "object")
                ah.append(p.anchor_head or "object")
                rh.append(" ".join(p.relation_types))
            else:
                th.append("object")
                ah.append("object")
                rh.append("none")
        q_t, q_a, q_r = self.struct_text.forward_batch_from_parsed(th, ah, rh)
        s_attr = self.attr(h_o, q_t)
        p_anchor = soft_anchor_distribution(h_o, q_a, mask, temperature=self.anchor_temperature)
        r_ij = self.rel(h_o, q_r)
        s_rel = torch.einsum("bj,bij->bi", p_anchor, r_ij)
        logits = s_attr + s_rel
        logits = logits.masked_fill(~mask, float("-inf"))
        return logits, p_anchor


def _effective_topk(mask: torch.Tensor, rerank_k: int) -> int:
    min_n = int(mask.sum(dim=1).min().item())
    return max(1, min(rerank_k, min_n))


def _topk_union_target(
    coarse_logits: torch.Tensor,
    mask: torch.Tensor,
    target_index: torch.Tensor | None,
    k_eff: int,
    training: bool,
) -> torch.Tensor:
    """Indices [B, k_eff] into full object axis; when training, ensure gold index is included if valid."""
    neg = coarse_logits.masked_fill(~mask, float("-inf"))
    _, idx = torch.topk(neg, k=k_eff, dim=1)
    if training and target_index is not None:
        for b in range(idx.size(0)):
            t = int(target_index[b].item())
            if t < 0 or t >= coarse_logits.size(1) or not mask[b, t]:
                continue
            row = idx[b]
            if (row == t).any():
                continue
            # replace the coarse-lowest slot in the current top-k
            worst = int(coarse_logits[b, row].argmin().item())
            idx[b, worst] = t
    return idx


class TwoStageCoarseRerankModel(nn.Module):
    """Coarse attribute scorer (typically frozen) + top-K relation-aware rerank with geometry context."""

    def __init__(
        self,
        coarse: nn.Module,
        fine: RelationAwareGeomModel,
        rerank_k: int = 10,
        *,
        freeze_coarse: bool = True,
    ) -> None:
        super().__init__()
        self.coarse = coarse
        self.fine = fine
        self.rerank_k = int(rerank_k)
        if freeze_coarse:
            for p in self.coarse.parameters():
                p.requires_grad = False

    def forward(
        self,
        batch: dict[str, Any],
        parsed_list: list[ParsedUtterance] | None = None,
        target_index: torch.Tensor | None = None,
        inject_gold_in_shortlist: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, dict[str, Any]]:
        """Full-scene logits [B,N]; anchor on full scene (non-candidates zero). ``aux`` is for bridge / eval.

        If ``inject_gold_in_shortlist`` is None, gold is injected into the top-K shortlist iff ``self.training``
        (legacy behavior). If True/False, overrides training mode for shortlist construction only.
        """
        obj = batch["object_features"]
        mask = batch["object_mask"]
        samples = batch.get("samples_ref")
        if samples is None:
            raise KeyError("TwoStageCoarseRerankModel requires batch['samples_ref']")
        b, n, d = obj.shape
        device, dtype = obj.device, obj.dtype
        geom = batch_geom_context_tensor8(samples, n, device, dtype)

        sub: dict[str, Any] = {k: batch[k] for k in ("object_features", "object_mask", "raw_texts")}
        if getattr(self.coarse, "uses_geometry_context", False):
            sub["samples_ref"] = batch["samples_ref"]
        with torch.no_grad():
            coarse_logits = self.coarse(sub)

        ti = target_index if target_index is not None else batch.get("target_index")
        if inject_gold_in_shortlist is None:
            inject = self.training
        else:
            inject = inject_gold_in_shortlist
        k_eff = _effective_topk(mask, self.rerank_k)
        idx = _topk_union_target(
            coarse_logits,
            mask,
            ti if inject else None,
            k_eff,
            training=inject,
        )

        exp = idx.unsqueeze(-1).expand(-1, -1, d)
        gexp = idx.unsqueeze(-1).expand(-1, -1, GEOM_DIM)
        sub_obj = torch.gather(obj, 1, exp)
        sub_geom = torch.gather(geom, 1, gexp)
        sub_mask = torch.ones(b, k_eff, dtype=torch.bool, device=device)

        if parsed_list is None:
            parsed_list = []
        fine_logits, p_anchor = self.fine(sub_obj, sub_geom, sub_mask, parsed_list)

        full_logits = torch.full((b, n), float("-inf"), device=device, dtype=dtype)
        full_logits.scatter_(1, idx, fine_logits)
        full_logits = full_logits.masked_fill(~mask, float("-inf"))

        full_anchor = torch.zeros(b, n, device=device, dtype=p_anchor.dtype)
        full_anchor.scatter_(1, idx, p_anchor)
        full_anchor = full_anchor.masked_fill(~mask, 0.0)

        aux: dict[str, Any] = {
            "coarse_logits": coarse_logits,
            "rerank_idx": idx,
            "k_eff": k_eff,
        }
        return full_logits, full_anchor, aux


def forward_two_stage_rerank(
    model: TwoStageCoarseRerankModel,
    batch: dict[str, Any],
    parser: Any,
    *,
    return_aux: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
    """Training / val forward: returns [B,N] logits for CE + hinge."""
    samples = batch["samples_ref"]
    parsed_list = [parser.parse(s.utterance) for s in samples]
    inject: bool | None = None
    if model.training and not getattr(model, "shortlist_train_inject_gold", True):
        inject = False
    logits, _, aux = model(
        {k: batch[k] for k in ("object_features", "object_mask", "raw_texts", "samples_ref")},
        parsed_list=parsed_list,
        target_index=batch["target_index"],
        inject_gold_in_shortlist=inject,
    )
    if return_aux:
        return logits, aux
    return logits
