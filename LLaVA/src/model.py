"""The learned PROJECTION + a tiny language model = LLaVA at small scale.

LLaVA's key architectural idea is minimal: keep the vision tower and the LM
essentially fixed, and learn a single **projection** W that maps visual
features into the LM's word-embedding space, so image "tokens" and text tokens
live in the same space. The LM then treats projected patches like extra input
tokens and answers instructions about the image.

Here the projection is a small MLP (LLaVA-1.5 also uses an MLP). Ablate it —
feed the LM zeros instead of projected visual tokens — and the model is blind:
accuracy collapses to chance. That is the whole point of the paper.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Projector(nn.Module):
    """Vision feature dim -> LM embedding dim (the learned bridge)."""

    def __init__(self, in_dim: int, d_model: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _Block(nn.Module):
    def __init__(self, d_model: int, n_head: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x


class TinyLM(nn.Module):
    """A very small bidirectional Transformer that answers from the last token."""

    def __init__(self, vocab_size: int, d_model: int = 48, n_head: int = 4,
                 n_layer: int = 2, max_len: int = 32) -> None:
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([_Block(d_model, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self.d_model = d_model

    def embed_tokens(self, ids: torch.Tensor) -> torch.Tensor:
        return self.tok_emb(ids)

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        B, T, _ = inputs_embeds.shape
        pos = torch.arange(T, device=inputs_embeds.device)
        x = inputs_embeds + self.pos_emb(pos)[None]
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln_f(x))  # (B, T, vocab)


class LlavaTiny(nn.Module):
    """vision(frozen) -> projector(learned) -> [visual tokens | instruction] -> LM."""

    def __init__(self, vision, vocab_size: int, d_model: int = 48) -> None:
        super().__init__()
        self.vision = vision                              # frozen
        self.projector = Projector(vision.feat_dim, d_model)
        self.lm = TinyLM(vocab_size, d_model=d_model)

    def forward(self, images: torch.Tensor, instr_ids: torch.Tensor,
                ablate_projection: bool = False) -> torch.Tensor:
        """Return logits over the vocab at the answer position (B, vocab)."""
        patches = self.vision(images)                      # (B, P, feat_dim)
        vis_tokens = self.projector(patches)               # (B, P, d_model)
        if ablate_projection:
            vis_tokens = torch.zeros_like(vis_tokens)      # blind the model
        instr_emb = self.lm.embed_tokens(instr_ids)[:, None, :]  # (B,1,d_model)
        seq = torch.cat([vis_tokens, instr_emb], dim=1)    # instruction is last
        logits = self.lm(seq)
        return logits[:, -1, :]                            # answer from last position
