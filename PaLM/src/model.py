"""The full PaLM decoder-only language model (PaLM paper, Section 2).

This ties together every distinctive PaLM ingredient:

- token embeddings shared with the output projection (weight tying),
- a stack of `PaLMBlock`s (parallel attention + SwiGLU MLP),
- multi-query attention with RoPE inside each block,
- LayerNorm without biases, and no biases anywhere,
- a final norm and a linear head to vocabulary logits.

The configuration here is intentionally tiny so the demo trains on CPU in
seconds; the same code scales to the 540B-parameter model in the paper.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .block import PaLMBlock
from .normalization import LayerNormNoBias
from .rope import RotaryEmbedding


@dataclass
class PaLMConfig:
    vocab_size: int = 32
    dim: int = 128
    n_layers: int = 4
    n_heads: int = 4
    ffn_hidden: int = 256  # SwiGLU hidden size
    rope_base: float = 10000.0


class PaLM(nn.Module):
    def __init__(self, config: PaLMConfig) -> None:
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.dim)

        head_dim = config.dim // config.n_heads
        # A single RoPE table is shared by every block (positions are global).
        self.rope = RotaryEmbedding(head_dim, base=config.rope_base)

        self.blocks = nn.ModuleList(
            PaLMBlock(config.dim, config.n_heads, config.ffn_hidden, self.rope)
            for _ in range(config.n_layers)
        )
        self.norm_f = LayerNormNoBias(config.dim)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)

        # Weight tying between input embedding and output projection.
        self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        """idx: (batch, seq) token ids. Returns (logits, loss)."""
        x = self.tok_emb(idx)
        for block in self.blocks:
            x = block(x)
        x = self.norm_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-1,
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """Greedy autoregressive decoding for the demo."""
        self.eval()
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            idx = torch.cat((idx, next_token), dim=1)
        return idx
