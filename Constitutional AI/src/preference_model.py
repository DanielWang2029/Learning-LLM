"""The RLAIF preference model, trained on AI-generated feedback.

The second half of Constitutional AI (RLAIF): the revised responses are labeled
as preferred over the originals — **AI feedback**, no humans — and a preference
model is trained on those comparisons. A well-trained preference model then
scores compliant responses above non-compliant ones and can drive RL.

The preference model is a tiny Transformer encoder (like the paper's LM-based
reward model): a `[CLS]` summary token attends over the response so it can see
word *order* and *length* — necessary to judge the "no repeats" and "be
concise" principles, which a bag-of-words model cannot. It is trained with the
Bradley-Terry pairwise loss, exactly as an RLHF reward model but on AI labels.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        self.h = num_heads
        self.dk = d_model // num_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, key_mask: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(b, t, self.h, self.dk).transpose(1, 2)
        k = k.view(b, t, self.h, self.dk).transpose(1, 2)
        v = v.view(b, t, self.h, self.dk).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.dk)
        # key_mask: (b, t) True where padding -> disallow attending there.
        scores = scores.masked_fill(key_mask[:, None, None, :], float("-inf"))
        attn = F.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(b, t, d)
        return self.proj(out)


class Block(nn.Module):
    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        self.n1 = nn.LayerNorm(d_model)
        self.attn = SelfAttention(d_model, num_heads)
        self.n2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, 2 * d_model), nn.GELU(),
                                nn.Linear(2 * d_model, d_model))

    def forward(self, x, key_mask):
        x = x + self.attn(self.n1(x), key_mask)
        x = x + self.ff(self.n2(x))
        return x


class PreferenceModel(nn.Module):
    def __init__(
        self, vocab_size: int, pad_id: int, max_len: int,
        d_model: int = 32, num_layers: int = 2, num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.pad_id = pad_id
        self.tok = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos = nn.Embedding(max_len + 1, d_model)  # +1 for the [CLS] slot
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.blocks = nn.ModuleList([Block(d_model, num_heads) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """Scalar preference score per response (b,). Reads the [CLS] summary."""
        b, t = ids.shape
        pad = ids == self.pad_id
        x = self.tok(ids)
        cls = self.cls.expand(b, 1, -1)
        x = torch.cat([cls, x], dim=1)  # prepend CLS
        pos = torch.arange(t + 1, device=ids.device).unsqueeze(0)
        x = x + self.pos(pos)
        key_mask = torch.cat([torch.zeros(b, 1, dtype=torch.bool, device=ids.device), pad], dim=1)
        for blk in self.blocks:
            x = blk(x, key_mask)
        return self.head(self.norm(x[:, 0])).squeeze(-1)  # CLS -> scalar


def preference_loss(
    score_pref: torch.Tensor, score_disp: torch.Tensor
) -> torch.Tensor:
    """-log σ(score_preferred - score_dispreferred), averaged (Bradley-Terry)."""
    return -F.logsigmoid(score_pref - score_disp).mean()
