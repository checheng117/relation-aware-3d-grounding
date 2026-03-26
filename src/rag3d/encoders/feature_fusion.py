from __future__ import annotations

import torch


def fuse_object_language(obj_h: torch.Tensor, lang_h: torch.Tensor) -> torch.Tensor:
    """Concat fusion broadcast: obj_h [B,N,H], lang_h [B,H] -> [B,N,2H]."""
    b, n, h = obj_h.shape
    lang = lang_h.unsqueeze(1).expand(-1, n, -1)
    return torch.cat([obj_h, lang], dim=-1)
