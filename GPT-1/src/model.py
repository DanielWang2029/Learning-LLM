"""A small decoder-only causal Transformer language model (GPT).

Implements the architecture from Radford et al., "Improving Language
Understanding by Generative Pre-Training" (GPT-1, 2018): a stack of
Transformer decoder blocks with masked self-attention, learned token and
position embeddings, and a language-model head tied to the token embedding.

The paper's central idea is a two-stage recipe:

1. **Generative pre-training** — train the LM to predict the next token on a
   large unlabeled corpus (``forward`` / ``lm_loss``).
2. **Discriminative fine-tuning** — attach a small linear head to the final
   token's representation and fine-tune on a labeled downstream task
   (``GPTClassifier``).

Pre-LayerNorm blocks are used for training stability on CPU-sized models.
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
    """Pre-LN Transformer decoder block: attention then feed-forward."""

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
        self.lm_head.weight = self.token_emb.weight  # weight tying
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
        """Return (logits, loss). ``targets`` enables the next-token LM loss."""
        logits = self.lm_head(self.hidden_states(idx))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100,
            )
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        greedy: bool = True,
    ) -> torch.Tensor:
        """Autoregressively extend ``idx`` by ``max_new_tokens`` tokens."""
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.max_len :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1] / temperature
            if greedy:
                nxt = logits.argmax(dim=-1, keepdim=True)
            else:
                probs = F.softmax(logits, dim=-1)
                nxt = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx


class GPTClassifier(nn.Module):
    """GPT backbone + linear head on the last token (GPT-1 fine-tuning, §3.3)."""

    def __init__(self, gpt: GPT, num_classes: int) -> None:
        super().__init__()
        self.gpt = gpt
        self.head = nn.Linear(gpt.cfg.d_model, num_classes)

    def forward(
        self, idx: torch.Tensor, lengths: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        h = self.gpt.hidden_states(idx)  # (b, t, d)
        last = h[torch.arange(h.size(0)), lengths - 1]  # final real token per row
        logits = self.head(last)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
        return logits, loss
