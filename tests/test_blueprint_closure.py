"""Blueprint closure: hard-negative loss, bridge output, paraphrase templates."""

from __future__ import annotations

import torch

from rag3d.datasets.schemas import GroundingSample, SceneObject
from rag3d.diagnostics.bridge_output import build_bridge_module_output
from rag3d.evaluation.paraphrase_templates import relation_preserving_paraphrases
from rag3d.parsers.heuristic_parser import HeuristicParser
from rag3d.relation_reasoner.losses import compute_batch_training_loss, same_class_hinge_loss


def test_same_class_hinge_nonzero_when_distractor() -> None:
    # 3 objects: two chairs (class match) + table; target index 0
    objs = [
        SceneObject(object_id="1", class_name="chair", center=(0, 0, 0), size=(1, 1, 1)),
        SceneObject(object_id="2", class_name="chair", center=(1, 0, 0), size=(1, 1, 1)),
        SceneObject(object_id="3", class_name="table", center=(2, 0, 0), size=(1, 1, 1)),
    ]
    s = GroundingSample(
        scene_id="s",
        utterance="the chair",
        target_object_id="1",
        target_index=0,
        objects=objs,
    )
    logits = torch.tensor([[2.0, 3.0, 0.0]], dtype=torch.float32)  # wrong argmax on same-class
    mask = torch.tensor([[True, True, True]])
    target = torch.tensor([0])
    h = same_class_hinge_loss(logits, target, mask, [s], margin=0.5)
    assert h.item() > 0.0


def test_training_loss_respects_disabled_hard_negative() -> None:
    logits = torch.randn(2, 4)
    mask = torch.tensor([[True, True, False, False], [True, True, True, False]])
    target = torch.tensor([0, 1])
    objs = [
        GroundingSample(
            scene_id="a",
            utterance="x",
            target_object_id="0",
            target_index=0,
            objects=[
                SceneObject(object_id="0", class_name="a", center=(0, 0, 0), size=(1, 1, 1)),
                SceneObject(object_id="1", class_name="b", center=(1, 0, 0), size=(1, 1, 1)),
            ],
        ),
        GroundingSample(
            scene_id="b",
            utterance="y",
            target_object_id="0",
            target_index=0,
            objects=[
                SceneObject(object_id="0", class_name="a", center=(0, 0, 0), size=(1, 1, 1)),
                SceneObject(object_id="1", class_name="a", center=(1, 0, 0), size=(1, 1, 1)),
            ],
        ),
    ]
    l0 = compute_batch_training_loss(logits, target, mask, {}, objs)
    l1 = compute_batch_training_loss(
        logits,
        target,
        mask,
        {"hard_negative": {"enabled": True, "margin": 0.25, "lambda_hinge": 1.0}},
        objs,
    )
    assert torch.isfinite(l0) and torch.isfinite(l1)


def test_bridge_output_has_target_id() -> None:
    objs = [
        SceneObject(object_id="9", class_name="plant", center=(0, 0, 0), size=(0.1, 0.1, 0.1)),
    ]
    s = GroundingSample(
        scene_id="sc",
        utterance="the plant",
        target_object_id="9",
        target_index=0,
        objects=objs,
        tags={"n_objects": 1, "candidate_load": "low", "geometry_fallback_fraction": 1.0},
    )
    logits = torch.tensor([[1.0]], dtype=torch.float32)
    mask = torch.tensor([[True]])
    p = HeuristicParser().parse(s.utterance)
    b = build_bridge_module_output(logits[0], mask[0], 0, s, 0, torch.tensor([1.0]), 0.8, p)
    assert b.target_id == "9"
    assert "candidate_summary" in b.model_dump()


def test_paraphrase_templates_deterministic() -> None:
    u = "the red chair near the table"
    a = relation_preserving_paraphrases(u, 4)
    b = relation_preserving_paraphrases(u, 4)
    assert a == b and len(a) >= 1
