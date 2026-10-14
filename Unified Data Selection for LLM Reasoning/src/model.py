"""A tiny causal Transformer LM.

Used two ways in the demo, exactly as in the paper's pipeline:
1. as the REFERENCE model whose per-token entropies feed the HES metric, and
2. as the SFT STUDENT trained on an HES-selected data subset.

Deliberately small so everything runs on CPU in well under a minute.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 64, n_layers: int = 2,
                 n_heads: int = 4, max_len: int = 64, dropout: float = 0.0) -> None:
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=4 * d_model, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self.max_len = max_len

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """idx: (B, T) -> logits (B, T, vocab)."""
        b, t = idx.shape
        pos = torch.arange(t, device=idx.device).unsqueeze(0)
        x = self.tok(idx) + self.pos(pos)
        mask = torch.triu(torch.ones(t, t, device=idx.device, dtype=torch.bool), 1)
        x = self.blocks(x, mask=mask)
        return self.head(self.norm(x))

    @torch.no_grad()
    def greedy_generate(self, prefix: torch.Tensor, max_new: int, eos_id: int) -> torch.Tensor:
        """Greedy decode starting from a 1-D prefix tensor; stop at eos_id."""
        self.eval()
        ids = prefix.tolist()
        for _ in range(max_new):
            ctx = torch.tensor([ids[-self.max_len:]], device=prefix.device)
            nxt = int(self.forward(ctx)[0, -1].argmax().item())
            ids.append(nxt)
            if nxt == eos_id:
                break
        return torch.tensor(ids, device=prefix.device)
