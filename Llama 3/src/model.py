"""The Llama 3 decoder-only language model (The Llama 3 Herd of Models, §3.1).

Llama 3 is a dense decoder-only Transformer. Relative to the original Transformer
it swaps in four now-canonical ingredients, all implemented in this package:

- RMSNorm pre-normalization                 (normalization.py)
- Rotary position embeddings, base 500,000  (rope.py)
- Grouped-Query Attention (8 K/V heads)     (attention.py)
- SwiGLU feed-forward                        (feedforward.py)

Plus a much larger byte-level BPE tokenizer (128K vocab, up from Llama 2's 32K)
implemented in tokenizer.py. The config here is intentionally tiny so the demo
trains on CPU in seconds; the same code describes the 8B/70B/405B models.
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
class Llama3Config:
    vocab_size: int = 128
    dim: int = 64
    n_layers: int = 3
    n_heads: int = 4
    n_kv_heads: int = 2          # Grouped-Query Attention: fewer K/V heads than Q heads
    rope_base: float = 500000.0  # Llama 3 raises RoPE base from 10k -> 500k


class Llama3(nn.Module):
    def __init__(self, config: Llama3Config) -> None:
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
        if isinstance(module, (nn.Linear, nn.Embedding)):
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
