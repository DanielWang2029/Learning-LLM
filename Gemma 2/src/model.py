"""The tiny Gemma 2 language model (paper: Gemma 2, arXiv 2408.00118).

Puts the pieces together:

* token embeddings scaled by ``sqrt(d_model)`` (Gemma normalizer),
* a stack of decoder blocks that **alternate** local (sliding-window) and global
  attention layers,
* RMSNorm applied both *before* and *after* each sub-block (pre+post norm),
* a GeGLU feed-forward network,
* final **logit soft-capping** on the vocabulary logits.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import GemmaAttention, _rope_cache
from .config import Gemma2Config
from .norm import RMSNorm


def soft_cap(x: torch.Tensor, cap: float) -> torch.Tensor:
    """Logit soft-capping: ``cap * tanh(x / cap)`` bounds x to (-cap, cap)."""
    return cap * torch.tanh(x / cap)


class GeGLU(nn.Module):
    """Gated feed-forward network with GELU gate (Gemma uses GeGLU MLPs)."""

    def __init__(self, cfg: Gemma2Config) -> None:
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.up = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.down = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.gelu(self.gate(x)) * self.up(x))


class DecoderBlock(nn.Module):
    """Pre-norm + post-norm block wrapping attention and the MLP (paper §2)."""

    def __init__(self, cfg: Gemma2Config, is_local: bool) -> None:
        super().__init__()
        self.is_local = is_local
        self.pre_attn_norm = RMSNorm(cfg.d_model)
        self.attn = GemmaAttention(cfg, is_local=is_local)
        self.post_attn_norm = RMSNorm(cfg.d_model)

        self.pre_mlp_norm = RMSNorm(cfg.d_model)
        self.mlp = GeGLU(cfg)
        self.post_mlp_norm = RMSNorm(cfg.d_model)

    def forward(self, x, cos, sin):
        # Attention sub-block: norm -> attn -> norm, added to the residual.
        h = self.post_attn_norm(self.attn(self.pre_attn_norm(x), cos, sin))
        x = x + h
        # MLP sub-block: norm -> MLP -> norm, added to the residual.
        h = self.post_mlp_norm(self.mlp(self.pre_mlp_norm(x)))
        x = x + h
        return x


class Gemma2Model(nn.Module):
    def __init__(self, cfg: Gemma2Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)

        # Alternate local / global: layer 0 local, layer 1 global, ...  (paper §2).
        self.layer_is_local = [i % 2 == 0 for i in range(cfg.n_layers)]
        self.blocks = nn.ModuleList(
            DecoderBlock(cfg, is_local=loc) for loc in self.layer_is_local
        )
        self.final_norm = RMSNorm(cfg.d_model)
        # Gemma ties the output projection to the input embedding table.
        self.embed_scale = cfg.d_model ** 0.5

        cos, sin = _rope_cache(cfg.max_seq_len, cfg.head_dim)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.last_final_max_logit = 0.0

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        b, t = tokens.shape
        cos = self.rope_cos[:t]
        sin = self.rope_sin[:t]

        x = self.embed(tokens) * self.embed_scale
        for block in self.blocks:
            x = block(x, cos, sin)
        x = self.final_norm(x)

        # Tied embeddings: logits = x @ E^T.
        logits = F.linear(x, self.embed.weight)

        # Track the raw logit magnitude BEFORE capping (for the demo comparison).
        self.last_final_max_logit = float(logits.abs().max().item())

        # Final logit soft-capping (paper §2 / Table 1: cap = 30).
        if self.cfg.use_soft_cap:
            logits = soft_cap(logits, self.cfg.final_logit_softcap)
        return logits

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
