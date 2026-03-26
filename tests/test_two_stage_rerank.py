"""Two-stage coarse → top-K → relation rerank (training union target, eval scatter logits)."""

from __future__ import annotations

import torch

from rag3d.datasets.schemas import GroundingSample, ParsedUtterance, SceneObject
from rag3d.diagnostics.bridge_output import build_bridge_module_output, rerank_extras_for_sample
from rag3d.parsers.heuristic_parser import HeuristicParser
from rag3d.relation_reasoner.model import AttributeOnlyModel
from rag3d.relation_reasoner.two_stage_rerank import (
    RelationAwareGeomModel,
    TwoStageCoarseRerankModel,
    _topk_union_target,
)


def _tiny_batch(feat_dim: int = 8) -> dict:
    objs = [
        SceneObject(object_id=str(i), class_name="o", center=(0, 0, 0), size=(1, 1, 1)) for i in range(5)
    ]
    s = GroundingSample(
        scene_id="s",
        utterance="the object left of the other",
        target_object_id="2",
        target_index=2,
        objects=objs,
    )
    b = 1
    n = 5
    feats = torch.randn(b, n, feat_dim)
    mask = torch.ones(b, n, dtype=torch.bool)
    return {
        "object_features": feats,
        "object_mask": mask,
        "raw_texts": [s.utterance],
        "target_index": torch.tensor([2]),
        "samples_ref": [s],
    }


def test_topk_union_includes_target_when_training() -> None:
    coarse_logits = torch.tensor([[0.0, 1.0, 0.5, 0.1, 0.2]])
    mask = torch.ones(1, 5, dtype=torch.bool)
    target = torch.tensor([4])
    idx = _topk_union_target(coarse_logits, mask, target, k_eff=3, training=True)
    assert 4 in idx[0].tolist()


def test_two_stage_forward_scatters_to_full_scene() -> None:
    feat_dim = 16
    batch = _tiny_batch(feat_dim)
    coarse = AttributeOnlyModel(feat_dim, 32, 32, dropout=0.0)
    fine = RelationAwareGeomModel(feat_dim, 32, 32, 16, dropout=0.0)
    m = TwoStageCoarseRerankModel(coarse, fine, rerank_k=3)
    m.train()
    p = HeuristicParser().parse(batch["samples_ref"][0].utterance)
    logits, anchor, aux = m(batch, parsed_list=[p], target_index=batch["target_index"])
    assert logits.shape == (1, 5)
    assert torch.isfinite(logits[0, batch["object_mask"][0]]).any()
    assert aux["k_eff"] == 3


def test_bridge_rerank_extras_serializable() -> None:
    batch = _tiny_batch(8)
    coarse = AttributeOnlyModel(8, 32, 32, dropout=0.0)
    fine = RelationAwareGeomModel(8, 32, 32, 16, dropout=0.0)
    m = TwoStageCoarseRerankModel(coarse, fine, rerank_k=3)
    m.eval()
    p = ParsedUtterance(raw_text="x", target_head="object", anchor_head="object", relation_types=["left"])
    with torch.no_grad():
        coarse_logits = m.coarse({k: batch[k] for k in ("object_features", "object_mask", "raw_texts")})
        logits, anchor, aux = m(batch, parsed_list=[p], target_index=None)
    idx = aux["rerank_idx"][0]
    rex = rerank_extras_for_sample(
        coarse_logits[0],
        batch["object_mask"][0],
        idx,
        target_index=2,
        k_eff=int(aux["k_eff"]),
    )
    bridge = build_bridge_module_output(
        logits[0],
        batch["object_mask"][0],
        2,
        batch["samples_ref"][0],
        int(logits[0].argmax().item()),
        anchor[0],
        0.9,
        p,
        rerank_extras=rex,
    )
    d = bridge.model_dump()
    assert d["rerank_applied"] is True
    assert d["coarse_topk_ids"] == idx.tolist()
    assert d["final_target_id"] == d["target_id"]
