"""Structured schemas for objects, language parse, and model outputs."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

import numpy as np
import torch
from pydantic import BaseModel, Field


class FailureTag(str, Enum):
    OK = "ok"
    PARSER_FAILURE = "parser_failure"
    RELATION_MISMATCH = "relation_mismatch"
    SAME_CLASS_CONFUSION = "same_class_confusion"
    AMBIGUOUS_ANCHOR = "ambiguous_anchor"
    LOW_CONFIDENCE = "low_confidence"
    OCCLUSION_RISK = "occlusion_risk"
    HIGH_CANDIDATE_LOAD = "high_candidate_load"
    COARSE_TARGET_NOT_IN_TOPK = "coarse_target_not_in_topk"
    RERANK_STAGE_FAILURE = "rerank_stage_failure"
    WEAK_GEOMETRY_CONTEXT = "weak_geometry_context"
    WEAK_FEATURE_SOURCE = "weak_feature_source"


GeometryQuality = Literal["obb_aabb", "fallback_centroid", "unknown"]
FeatureSource = Literal["synthetic_collate", "aggregated_file", "user_provided", "unknown"]


class SceneObject(BaseModel):
    object_id: str
    class_name: str
    center: tuple[float, float, float]
    size: tuple[float, float, float]
    bbox: tuple[float, float, float, float, float, float] | None = None
    point_indices: list[int] | None = None
    sampled_points: np.ndarray | None = Field(default=None, exclude=True)
    visibility_occlusion_proxy: float | None = None
    feature_vector: list[float] | None = None
    geometry_quality: GeometryQuality = "unknown"
    feature_source: FeatureSource = "unknown"

    model_config = {"arbitrary_types_allowed": True}


class ParsedUtterance(BaseModel):
    raw_text: str
    target_head: str | None = None
    target_modifiers: list[str] = Field(default_factory=list)
    anchor_head: str | None = None
    relation_types: list[str] = Field(default_factory=list)
    parser_confidence: float = 0.5
    paraphrase_set: list[str] = Field(default_factory=list)
    parse_source: str = "unknown"
    parse_warnings: list[str] = Field(default_factory=list)


class ModelPrediction(BaseModel):
    target_id_pred: str | None = None
    target_index_pred: int | None = None
    target_scores: list[float] = Field(default_factory=list)
    anchor_distribution: list[float] = Field(default_factory=list)
    relation_rationale: str = ""
    confidence: float = 0.0
    failure_tags: list[FailureTag] = Field(default_factory=list)


class BridgeModuleOutput(BaseModel):
    """Downstream-facing contract: serializable summary for planners / scene memory (per sample)."""

    target_id: str | None = None
    target_index_pred: int | None = None
    final_target_id: str | None = None
    target_scores: list[float] = Field(default_factory=list)
    anchor_distribution: list[float] = Field(default_factory=list)
    relation_rationale: str = ""
    confidence: float = 0.0
    failure_tags: list[str] = Field(default_factory=list)
    target_margin: float = 0.0
    anchor_entropy: float = 0.0
    candidate_summary: dict[str, Any] = Field(default_factory=dict)
    candidate_load: str | None = None
    coarse_topk_ids: list[int] = Field(default_factory=list)
    coarse_topk_scores: list[float] = Field(default_factory=list)
    coarse_target_in_topk: bool | None = None
    coarse_gold_rank: int | None = None
    coarse_recall_bucket: str | None = None
    coarse_stage_confidence: float | None = None
    topk_recall_success: bool | None = None
    rerank_rescued_from_coarse_shortlist: bool | None = None
    stage1_failure_reason: str | None = None
    rerank_applied: bool = False
    rerank_k: int | None = None
    parse_source: str | None = None
    parse_warnings: list[str] = Field(default_factory=list)


class GroundingSample(BaseModel):
    scene_id: str
    utterance: str
    target_object_id: str
    target_index: int
    objects: list[SceneObject]
    utterance_id: str | None = None
    parsed: ParsedUtterance | None = None
    relation_type_gold: str | None = None
    tags: dict[str, Any] = Field(default_factory=dict)

    model_config = {"arbitrary_types_allowed": True}


class GroundingBatch(BaseModel):
    samples: list[GroundingSample]

    def to_tensors(
        self,
        feat_dim: int,
        device: torch.device,
    ) -> dict[str, Any]:
        """Stack per-object features into padded tensors (uses feature_vector or zeros)."""
        b = len(self.samples)
        max_n = max(len(s.objects) for s in self.samples)
        feats = torch.zeros(b, max_n, feat_dim, device=device)
        mask = torch.zeros(b, max_n, dtype=torch.bool, device=device)
        target_idx = torch.zeros(b, dtype=torch.long, device=device)
        texts: list[str] = []
        class_names: list[list[str]] = []
        for i, s in enumerate(self.samples):
            texts.append(s.utterance)
            class_names.append([o.class_name for o in s.objects])
            for j, o in enumerate(s.objects):
                if o.feature_vector is not None and len(o.feature_vector) == feat_dim:
                    feats[i, j] = torch.tensor(o.feature_vector, device=device)
                mask[i, j] = True
            target_idx[i] = s.target_index
        return {
            "object_features": feats,
            "object_mask": mask,
            "raw_texts": texts,
            "target_index": target_idx,
            "class_names": class_names,
        }
