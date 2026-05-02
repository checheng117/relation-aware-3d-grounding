from __future__ import annotations

from typing import Any

import torch

from rag3d.datasets.schemas import GroundingSample, ParsedUtterance
from rag3d.diagnostics.bridge_output import build_bridge_module_output, model_prediction_from_bridge
from rag3d.diagnostics.confidence import anchor_entropy, target_margin


def summarize_batch_predictions(
    logits: torch.Tensor,
    mask: torch.Tensor,
    target_index: torch.Tensor,
    samples: list[GroundingSample],
    anchor_dist: torch.Tensor | None,
    parser_confidences: list[float] | None = None,
    parsed_list: list[ParsedUtterance] | None = None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    pred = logits.argmax(dim=-1)
    b = logits.size(0)
    for i in range(b):
        pi = int(pred[i].item())
        gi = int(target_index[i].item())
        pc = parser_confidences[i] if parser_confidences else 0.5
        ent = anchor_entropy(anchor_dist[i], mask[i]) if anchor_dist is not None else 0.0
        margin = target_margin(logits[i], mask[i])
        parsed_i = parsed_list[i] if parsed_list and i < len(parsed_list) else None
        bridge = build_bridge_module_output(
            logits[i],
            mask[i],
            gi,
            samples[i],
            pi,
            anchor_dist[i] if anchor_dist is not None else None,
            pc,
            parsed_i,
        )
        mp = model_prediction_from_bridge(bridge)
        out.append(
            {
                "scene_id": samples[i].scene_id,
                "pred": pi,
                "gold": gi,
                "correct": pi == gi,
                "parser_confidence": pc,
                "anchor_entropy": ent,
                "target_margin": margin,
                "failure_tags": [t.value for t in mp.failure_tags],
                "model_prediction": mp.model_dump(),
                "bridge_module_output": bridge.model_dump(),
            }
        )
    return out
