from __future__ import annotations

from rag3d.datasets.schemas import GroundingSample


def scene_text_summary(sample: GroundingSample) -> str:
    lines = [f"scene={sample.scene_id}", f"utterance={sample.utterance}", f"target_idx={sample.target_index}"]
    for j, o in enumerate(sample.objects):
        lines.append(f"  [{j}] {o.object_id} cls={o.class_name} center={o.center}")
    return "\n".join(lines)
