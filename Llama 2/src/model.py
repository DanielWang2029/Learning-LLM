"""The full Llama 2 decoder-only language model (Llama 2 paper, Section 2.2).

Same as the LLaMA recipe (RMSNorm pre-norm, RoPE, SwiGLU) plus grouped-query
attention. The ``n_kv_heads`` field selects the attention regime:

    n_kv_heads == n_heads      -> MHA
    1 < n_kv_heads < n_heads   -> GQA  (Llama 2's choice for its large models)
    n_kv_heads == 1            -> MQA

The config is tiny so the demo trains on CPU in seconds; the same code
describes the 7B/13B/70B models in the paper (the 70B model uses GQA with
n_kv_heads=8 for n_heads=64).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import GroupedQueryAttention
from .block import TransformerBlock
from .feedforward import SwiGLU, swiglu_hidden_dim
from .normalization import RMSNorm
from .rope import RotaryEmbedding


@dataclass
class Llama2Config:
    vocab_size: int = 16
    dim: int = 64
    n_layers: int = 3
    n_heads: int = 8
    n_kv_heads: int = 2  # GQA: 2 K/V heads shared across 8 query heads (4 per group)
    rope_base: float = 10000.0


class Llama2(nn.Module):
    def __init__(self, config: Llama2Config) -> None:
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.dim)

        head_dim = config.dim // config.n_heads
        self.rope = RotaryEmbedding(head_dim, base=config.rope_base)

        hidden = swiglu_hidden_dim(config.dim)
        blocks = []
        for _ in range(config.n_layers):
            attn = GroupedQueryAttention(
                config.dim, config.n_heads, config.n_kv_heads, self.rope
            )
            ffn = SwiGLU(config.dim, hidden)
            blocks.append(TransformerBlock(config.dim, attn, ffn))
        self.blocks = nn.ModuleList(blocks)

        self.norm_f = RMSNorm(config.dim)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
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
        self.eval()
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            idx = torch.cat((idx, next_token), dim=1)
        return idx
