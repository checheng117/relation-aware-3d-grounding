"""Per-object geometric / provenance channels for quality-aware encoding (no fake point clouds)."""

from __future__ import annotations

import math

import torch

from rag3d.datasets.schemas import GroundingSample


def object_geom_context_tensor8(
    sample: GroundingSample,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Shape ``[N, 8]``: center3, size3, fallback_geom (0/1), synthetic_feat (0/1)."""
    n = len(sample.objects)
    out = torch.zeros(n, 8, device=device, dtype=dtype)
    for j, o in enumerate(sample.objects):
        cx, cy, cz = o.center
        sx, sy, sz = o.size
        out[j, 0] = math.tanh(cx / 5.0)
        out[j, 1] = math.tanh(cy / 5.0)
        out[j, 2] = math.tanh(cz / 5.0)
        out[j, 3] = math.tanh(sx / 2.0)
        out[j, 4] = math.tanh(sy / 2.0)
        out[j, 5] = math.tanh(sz / 2.0)
        out[j, 6] = 1.0 if o.geometry_quality == "fallback_centroid" else 0.0
        out[j, 7] = 1.0 if o.feature_source == "synthetic_collate" else 0.0
    return out


def batch_geom_context_tensor8(
    samples: list[GroundingSample],
    max_n: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """``[B, max_n, 8]`` padded with zeros."""
    b = len(samples)
    g = torch.zeros(b, max_n, 8, device=device, dtype=dtype)
    for i, s in enumerate(samples):
        t = object_geom_context_tensor8(s, device, dtype)
        n = t.size(0)
        g[i, :n] = t
    return g
