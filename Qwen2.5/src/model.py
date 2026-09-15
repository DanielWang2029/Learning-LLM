"""The Qwen2.5 decoder-only language model (Qwen2.5 Technical Report, §Architecture).

Qwen2.5 is a dense decoder-only Transformer whose *distinctive* choices — the
ones that set it apart from the otherwise-identical Llama-style recipe — are:

1. **QKV bias** on the attention query/key/value projections  (attention.py)
2. **Untied input/output embeddings** — the token embedding and the LM head are
   SEPARATE weight matrices (many small models tie them; Qwen2.5 does not for
   most sizes, spending the extra parameters on capacity).

It also keeps the standard modern ingredients: RMSNorm pre-norm, RoPE,
Grouped-Query Attention and SwiGLU. Config toggles (``qkv_bias``, ``tie_embeddings``)
let the demo verify each distinctive feature is actually active.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import QwenAttention
from .layers import RMSNorm, RotaryEmbedding, SwiGLU


@dataclass
class Qwen25Config:
    vocab_size: int = 32
    dim: int = 64
    n_layers: int = 3
    n_heads: int = 4
    n_kv_heads: int = 2
    ffn_hidden: int = 176
    rope_base: float = 1000000.0
    qkv_bias: bool = True        # distinctive: bias on Q/K/V
    tie_embeddings: bool = False  # distinctive: untied input/output embeddings


class QwenBlock(nn.Module):
    def __init__(self, cfg: Qwen25Config, rope: RotaryEmbedding) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.dim)
        self.self_attn = QwenAttention(
            cfg.dim, cfg.n_heads, cfg.n_kv_heads, rope, qkv_bias=cfg.qkv_bias
        )
        self.post_attention_layernorm = RMSNorm(cfg.dim)
        self.mlp = SwiGLU(cfg.dim, cfg.ffn_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x))
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class Qwen25(nn.Module):
    def __init__(self, cfg: Qwen25Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.dim)
        head_dim = cfg.dim // cfg.n_heads
        self.rope = RotaryEmbedding(head_dim, base=cfg.rope_base)
        self.layers = nn.ModuleList([QwenBlock(cfg, self.rope) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.dim)

        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed_tokens.weight  # shared matrix
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def embeddings_are_tied(self) -> bool:
        return self.lm_head.weight is self.embed_tokens.weight

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        x = self.embed_tokens(idx)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-1
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            idx = torch.cat((idx, logits[:, -1].argmax(-1, keepdim=True)), dim=1)
        return idx
