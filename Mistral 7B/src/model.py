"""A tiny Mistral-style decoder LM: SWA + GQA blocks, with a cached decode path.

Kept intentionally minimal (learned absolute positions rather than RoPE) so the
sliding-window attention, grouped-query attention, and rolling-buffer KV cache
are the clear focus. See ``attention.py`` for the three headline mechanisms.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .attention import RollingKVCache, SlidingWindowAttention


@dataclass
class MistralConfig:
    vocab_size: int
    block_size: int
    n_layer: int = 3
    n_head: int = 4
    n_kv_head: int = 2       # GQA: fewer KV heads than query heads
    n_embd: int = 48
    window: int = 4          # sliding-window size W


class Block(nn.Module):
    def __init__(self, cfg: MistralConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = SlidingWindowAttention(
            cfg.n_embd, cfg.n_head, cfg.n_kv_head, cfg.window
        )
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd), nn.SiLU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

    def step(self, x_t: torch.Tensor, cache: RollingKVCache) -> torch.Tensor:
        x_t = x_t + self.attn.step(self.ln1(x_t), cache)
        x_t = x_t + self.mlp(self.ln2(x_t))
        return x_t


class MistralLM(nn.Module):
    def __init__(self, cfg: MistralConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)[None]
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln_f(x))

    @torch.no_grad()
    def forward_cached(self, idx: torch.Tensor) -> torch.Tensor:
        """Decode the sequence token-by-token with per-layer rolling KV caches.

        Returns per-position logits identical (up to fp error) to ``forward``,
        demonstrating the rolling buffer is a correct, memory-bounded equivalent.
        """
        B, T = idx.shape
        caches = [RollingKVCache(self.cfg.window) for _ in self.blocks]
        outs = []
        for t in range(T):
            pos = torch.full((B, 1), t, dtype=torch.long, device=idx.device)
            x_t = self.tok_emb(idx[:, t:t + 1]) + self.pos_emb(pos)
            for blk, cache in zip(self.blocks, caches):
                x_t = blk.step(x_t, cache)
            outs.append(self.head(self.ln_f(x_t)))
        return torch.cat(outs, dim=1)
