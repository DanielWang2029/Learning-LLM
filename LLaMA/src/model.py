"""The full LLaMA decoder-only language model (LLaMA paper, Section 2.2).

The configuration is intentionally tiny so the demo trains on CPU in seconds;
the same code describes the 7B–65B models in the paper. The config exposes
ablation switches (``norm``, ``use_rope``, ``ffn``) so the demo can swap out one
LLaMA ingredient at a time and measure the effect.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import Attention
from .block import TransformerBlock
from .feedforward import ReLUFFN, SwiGLU, swiglu_hidden_dim
from .normalization import RMSNorm
from .rope import RotaryEmbedding


@dataclass
class LLaMAConfig:
    vocab_size: int = 16
    dim: int = 64
    n_layers: int = 3
    n_heads: int = 4
    rope_base: float = 10000.0
    # Ablation switches (defaults = the real LLaMA recipe).
    norm: str = "rmsnorm"   # "rmsnorm" | "layernorm"
    use_rope: bool = True   # RoPE vs no position information
    ffn: str = "swiglu"     # "swiglu" | "relu"


def _norm_cls(name: str):
    if name == "rmsnorm":
        return RMSNorm
    if name == "layernorm":
        return lambda dim: nn.LayerNorm(dim)
    raise ValueError(f"unknown norm: {name}")


class LLaMA(nn.Module):
    def __init__(self, config: LLaMAConfig) -> None:
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.dim)

        head_dim = config.dim // config.n_heads
        self.rope = RotaryEmbedding(head_dim, base=config.rope_base)
        norm_cls = _norm_cls(config.norm)

        hidden = swiglu_hidden_dim(config.dim)
        blocks = []
        for _ in range(config.n_layers):
            attn = Attention(
                config.dim, config.n_heads, self.rope, use_rope=config.use_rope
            )
            if config.ffn == "swiglu":
                ffn = SwiGLU(config.dim, hidden)
            elif config.ffn == "relu":
                # Match SwiGLU's parameter budget roughly (3 vs 2 matrices).
                ffn = ReLUFFN(config.dim, hidden * 3 // 2)
            else:
                raise ValueError(f"unknown ffn: {config.ffn}")
            blocks.append(TransformerBlock(config.dim, attn, ffn, norm_cls))
        self.blocks = nn.ModuleList(blocks)

        self.norm_f = norm_cls(config.dim)
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
