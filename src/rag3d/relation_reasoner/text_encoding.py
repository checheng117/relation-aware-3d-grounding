from __future__ import annotations

import re
import torch
import torch.nn as nn


def tokenize_simple(text: str) -> list[int]:
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return [hash(w) % 8192 for w in words] or [0]


class TextHashEncoder(nn.Module):
    """Bag-of-hashed-tokens embedding (no pretrained LM; lightweight baseline)."""

    def __init__(self, vocab_size: int = 8192, dim: int = 256) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim)
        self.dim = dim

    def forward(self, texts: list[str]) -> torch.Tensor:
        device = self.emb.weight.device
        out = torch.zeros(len(texts), self.dim, device=device)
        for i, t in enumerate(texts):
            ids = tokenize_simple(t)
            e = self.emb(torch.tensor(ids, device=device, dtype=torch.long))
            out[i] = e.mean(dim=0)
        return out


class StructuredTextEncoder(nn.Module):
    """Encode target/anchor/relation string hints (concat projections)."""

    def __init__(self, dim: int = 256) -> None:
        super().__init__()
        self.base = TextHashEncoder(dim=dim)
        self.proj_anchor = nn.Linear(dim, dim)
        self.proj_rel = nn.Linear(dim, dim)
        self.tanh = nn.Tanh()

    def forward(
        self,
        target_hint: str,
        anchor_hint: str,
        rel_hint: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q_t = self.base([target_hint])
        q_a = self.proj_anchor(self.base([anchor_hint]))
        q_r = self.proj_rel(self.base([rel_hint]))
        return self.tanh(q_t), self.tanh(q_a), self.tanh(q_r)

    def forward_batch_from_parsed(
        self,
        target_heads: list[str],
        anchor_heads: list[str],
        rel_types: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rel_strs = [" ".join(rt) if isinstance(rt, list) else str(rt) for rt in rel_types]
        bt = self.base([h or "object" for h in target_heads])
        ba = self.proj_anchor(self.base([h or "object" for h in anchor_heads]))
        br = self.proj_rel(self.base(rel_strs))
        return self.tanh(bt), self.tanh(ba), self.tanh(br)
