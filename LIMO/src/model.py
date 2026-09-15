"""A tiny decoder-only Transformer language model (the "reasoner").

This is the smallest thing that can be *supervised fine-tuned* (SFT) to emit a
chain of reasoning followed by an answer. LIMO's claim is not about the model
architecture — it is about the **data** the model is fine-tuned on — so we keep
the model deliberately small and standard and vary only the training set.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TinyGPT(nn.Module):
    """A small causal (decoder-only) Transformer trained with next-token loss."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
        d_ff: int = 256,
        max_len: int = 48,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self.max_len = max_len
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        b, t = idx.shape
        pos = torch.arange(t, device=idx.device).unsqueeze(0).expand(b, t)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        causal = torch.triu(
            torch.ones(t, t, device=idx.device, dtype=torch.bool), diagonal=1
        )
        x = self.blocks(x, mask=causal)
        return self.head(self.norm(x))

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new: int, eos_id: int) -> torch.Tensor:
        """Greedy decode until ``eos_id`` or ``max_new`` tokens are produced."""
        self.eval()
        for _ in range(max_new):
            logits = self(idx[:, -self.max_len:])
            nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            idx = torch.cat([idx, nxt], dim=1)
            if (nxt == eos_id).all():
                break
        return idx
