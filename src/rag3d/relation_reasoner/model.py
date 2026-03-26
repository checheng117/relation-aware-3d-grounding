from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from rag3d.datasets.schemas import ParsedUtterance
from rag3d.encoders.object_encoder import ObjectMLPEncoder, ObjectMLPEncoderWithGeomContext
from rag3d.relation_reasoner.anchor_selector import soft_anchor_distribution
from rag3d.relation_reasoner.geom_context import batch_geom_context_tensor8
from rag3d.relation_reasoner.attribute_scorer import AttributeScorer
from rag3d.relation_reasoner.relation_scorer import PairwiseRelationScorer
from rag3d.relation_reasoner.text_encoding import StructuredTextEncoder, TextHashEncoder


class AttributeOnlyModel(nn.Module):
    """Baseline A: object features + target-side language only."""

    uses_geometry_context: bool = False

    def __init__(self, object_dim: int, language_dim: int, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.object_enc = ObjectMLPEncoder(object_dim, hidden_dim, dropout)
        self.text_enc = TextHashEncoder(dim=language_dim)
        self.attr = AttributeScorer(hidden_dim, language_dim)

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        obj = batch["object_features"]
        mask = batch["object_mask"]
        texts = batch["raw_texts"]
        h_o = self.object_enc(obj)
        h_t = self.text_enc(texts)
        logits = self.attr(h_o, h_t)
        logits = logits.masked_fill(~mask, float("-inf"))
        return logits


GEOM_CTX_DIM = 8


class CoarseGeomAttributeModel(nn.Module):
    """Stage-1 coarse scorer: object features + 8-D geometry/provenance context (aligned with fine trunk)."""

    uses_geometry_context: bool = True

    def __init__(self, object_dim: int, language_dim: int, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.object_enc = ObjectMLPEncoderWithGeomContext(object_dim, GEOM_CTX_DIM, hidden_dim, dropout)
        self.text_enc = TextHashEncoder(dim=language_dim)
        self.attr = AttributeScorer(hidden_dim, language_dim)

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        obj = batch["object_features"]
        mask = batch["object_mask"]
        texts = batch["raw_texts"]
        b, n, _ = obj.shape
        device, dtype = obj.device, obj.dtype
        samples = batch.get("samples_ref")
        if samples is not None:
            geom = batch_geom_context_tensor8(samples, n, device, dtype)
        else:
            geom = torch.zeros(b, n, GEOM_CTX_DIM, device=device, dtype=dtype)
        h_o = self.object_enc(obj, geom)
        h_t = self.text_enc(texts)
        logits = self.attr(h_o, h_t)
        logits = logits.masked_fill(~mask, float("-inf"))
        return logits


class RawTextRelationModel(nn.Module):
    """Baseline B: relation-style scoring but language is a single raw-text embedding."""

    def __init__(
        self,
        object_dim: int,
        language_dim: int,
        hidden_dim: int,
        relation_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.object_enc = ObjectMLPEncoder(object_dim, hidden_dim, dropout)
        self.text_enc = TextHashEncoder(dim=language_dim)
        self.attr = AttributeScorer(hidden_dim, language_dim)
        self.rel = PairwiseRelationScorer(hidden_dim, relation_dim, language_dim)
        self.gate = nn.Linear(language_dim, 1)

    def forward(self, batch: dict[str, Any]) -> torch.Tensor:
        obj = batch["object_features"]
        mask = batch["object_mask"]
        texts = batch["raw_texts"]
        h_o = self.object_enc(obj)
        h_t = self.text_enc(texts)
        s_attr = self.attr(h_o, h_t)
        uniform = mask.float() / (mask.float().sum(dim=1, keepdim=True).clamp_min(1.0))
        r_ij = self.rel(h_o, h_t)
        s_rel = torch.einsum("bj,bij->bi", uniform, r_ij)
        g = torch.sigmoid(self.gate(h_t))
        logits = s_attr + g * s_rel
        logits = logits.masked_fill(~mask, float("-inf"))
        return logits


class RelationAwareModel(nn.Module):
    """Structured model: soft anchor selection + attribute + relation paths."""

    def __init__(
        self,
        object_dim: int,
        language_dim: int,
        hidden_dim: int,
        relation_dim: int,
        anchor_temperature: float = 1.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.object_enc = ObjectMLPEncoder(object_dim, hidden_dim, dropout)
        self.struct_text = StructuredTextEncoder(dim=language_dim)
        self.attr = AttributeScorer(hidden_dim, language_dim)
        self.rel = PairwiseRelationScorer(hidden_dim, relation_dim, language_dim)
        self.anchor_temperature = anchor_temperature

    def forward(
        self,
        batch: dict[str, Any],
        parsed_list: list[ParsedUtterance] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (logits [B,N], anchor_dist [B,N])."""
        obj = batch["object_features"]
        mask = batch["object_mask"]
        h_o = self.object_enc(obj)
        if parsed_list is None:
            parsed_list = []
        b = obj.size(0)
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
