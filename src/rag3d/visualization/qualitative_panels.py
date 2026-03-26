from __future__ import annotations

from pathlib import Path

from rag3d.datasets.schemas import GroundingSample


def write_side_by_side_case(
    sample: GroundingSample,
    pred_idx: int,
    out_md: Path,
) -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)
    gold = sample.target_index
    body = "\n".join(
        [
            f"# Case {sample.scene_id}",
            "",
            f"- utterance: {sample.utterance}",
            f"- gold index: {gold}",
            f"- pred index: {pred_idx}",
            "",
            "## Objects",
            *[f"- {j}: {o.class_name} ({o.object_id})" for j, o in enumerate(sample.objects)],
        ]
    )
    out_md.write_text(body, encoding="utf-8")
