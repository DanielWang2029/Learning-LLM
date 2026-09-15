"""A small decoder-only GPT-2 language model.

Radford et al., "Language Models are Unsupervised Multitask Learners"
(GPT-2, 2019). Architecturally GPT-2 is the GPT decoder with pre-LayerNorm
blocks and a final LayerNorm. Its headline claim is behavioural rather than
architectural: a language model trained only to predict the next token learns
to perform *many tasks zero-shot*, when those tasks appear naturally in the
text (e.g. ``reverse: abc = cba``). No task-specific heads are added — the same
LM head produces every answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import CausalSelfAttention


@dataclass
class GPTConfig:
    vocab_size: int
    d_model: int = 128
    num_layers: int = 4
    num_heads: int = 4
    d_ff: int = 256
    max_len: int = 128
    dropout: float = 0.1


class Block(nn.Module):
    """Pre-LN Transformer decoder block (GPT-2 moved LayerNorm to the input)."""

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.num_heads, cfg.dropout)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Linear(cfg.d_ff, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    """Decoder-only causal language model."""

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.num_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def hidden_states(self, idx: torch.Tensor) -> torch.Tensor:
        b, t = idx.shape
        pos = torch.arange(t, device=idx.device).unsqueeze(0)
        x = self.drop(self.token_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x)
        return self.ln_f(x)

    def forward(
        self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        logits = self.lm_head(self.hidden_states(idx))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        stop_token: Optional[int] = None,
        greedy: bool = True,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Autoregressively extend ``idx``; stop early if ``stop_token`` appears."""
        self.eval()
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -self.cfg.max_len :])
            logits = logits[:, -1] / temperature
            if greedy:
                nxt = logits.argmax(dim=-1, keepdim=True)
            else:
                nxt = torch.multinomial(F.softmax(logits, dim=-1), 1)
            idx = torch.cat([idx, nxt], dim=1)
            if stop_token is not None and idx.size(0) == 1 and nxt.item() == stop_token:
                break
        return idx
