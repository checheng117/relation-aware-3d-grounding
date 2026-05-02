"""Transforms and automatic stratification tags for grounding samples."""

from __future__ import annotations

from rag3d.datasets.schemas import GroundingSample, SceneObject
from rag3d.parsers.heuristic_parser import HeuristicParser


def normalize_class_name(name: str) -> str:
    return name.strip().lower().replace("_", " ")


def attach_synthetic_features(sample: GroundingSample, dim: int) -> GroundingSample:
    """Fill missing feature_vector with deterministic pseudo-random vectors for tests / debug."""
    import numpy as np

    rng = np.random.default_rng(hash(sample.scene_id) % (2**32))
    new_objs: list[SceneObject] = []
    for o in sample.objects:
        if o.feature_vector is None or len(o.feature_vector) != dim:
            v = rng.standard_normal(dim).astype(float)
            v = (v / (np.linalg.norm(v) + 1e-8)).tolist()
            new_objs.append(
                o.model_copy(update={"feature_vector": v, "feature_source": "synthetic_collate"})
            )
        else:
            new_objs.append(o)
    return sample.model_copy(update={"objects": new_objs})


def compute_stratification_tags(sample: GroundingSample, parser_confidence: float | None = None) -> dict[str, bool]:
    """Derive subset flags for evaluation (does not require gold anchor)."""
    tags = dict(sample.tags) if sample.tags else {}
    if not sample.objects or sample.target_index >= len(sample.objects):
        return tags
    tgt = sample.objects[sample.target_index]
    tcls = normalize_class_name(tgt.class_name)
    same_class = sum(1 for o in sample.objects if normalize_class_name(o.class_name) == tcls)
    tags["same_class_clutter"] = same_class >= 2

    nobj = len(sample.objects)
    tags["n_objects"] = nobj
    if nobj >= 48:
        tags["candidate_load"] = "high"
    elif nobj >= 24:
        tags["candidate_load"] = "medium"
    else:
        tags["candidate_load"] = "low"
    fb = sum(1 for o in sample.objects if getattr(o, "geometry_quality", "unknown") == "fallback_centroid")
    tags["geometry_fallback_fraction"] = float(fb) / float(max(nobj, 1))

    occ_vals = [o.visibility_occlusion_proxy for o in sample.objects if o.visibility_occlusion_proxy is not None]
    if occ_vals:
        tags["occlusion_heavy"] = float(min(occ_vals)) < 0.35
    else:
        tags.setdefault("occlusion_heavy", False)

    hp = HeuristicParser().parse(sample.utterance)
    ah = (hp.anchor_head or "").strip().lower()
    if ah and ah != "object":
        ac = sum(1 for o in sample.objects if ah in normalize_class_name(o.class_name))
        tags["anchor_confusion"] = ac >= 2
    else:
        tags.setdefault("anchor_confusion", False)

    if parser_confidence is not None:
        tags["parser_failure"] = parser_confidence < 0.4
    elif hp.parser_confidence < 0.4:
        tags["parser_failure"] = True
    else:
        tags.setdefault("parser_failure", False)

    return tags


def apply_tags_to_sample(sample: GroundingSample) -> GroundingSample:
    merged = compute_stratification_tags(sample)
    return sample.model_copy(update={"tags": merged})
