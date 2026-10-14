"""A minimal causal decoder used to demonstrate FIM training."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class TinyDecoder(nn.Module):
    """Decoder-only Transformer with learned positions (paper §3.2 objective)."""

    def __init__(self, vocab_size: int, width: int = 48, layers: int = 2) -> None:
        super().__init__()
        self.token = nn.Embedding(vocab_size, width)
        self.position = nn.Embedding(32, width)
        block = nn.TransformerEncoderLayer(
            d_model=width, nhead=4, dim_feedforward=width * 2,
            dropout=0.0, batch_first=True, activation="gelu",
        )
        self.decoder = nn.TransformerEncoder(block, num_layers=layers)
        self.head = nn.Linear(width, vocab_size)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        length = tokens.size(1)
        positions = torch.arange(length, device=tokens.device)
        x = self.token(tokens) * math.sqrt(self.token.embedding_dim) + self.position(positions)
        # A decoder can only use tokens to its left.  FIM moves suffix left of middle.
        causal = torch.triu(
            torch.ones(length, length, device=tokens.device, dtype=torch.bool), diagonal=1
        )
        return self.head(self.decoder(x, mask=causal))


def greedy_middle(
    model: TinyDecoder, prompt: list[int], middle_marker: int, end_marker: int, max_tokens: int = 4
) -> list[int]:
    """Generate tokens after <MID>, stopping at <END>."""
    ids = list(prompt)
    model.eval()
    with torch.no_grad():
        for _ in range(max_tokens):
            x = torch.tensor([ids], dtype=torch.long)
            next_id = int(model(x)[0, -1].argmax())
            ids.append(next_id)
            if next_id == end_marker:
                break
    marker_at = ids.index(middle_marker)
    return [token for token in ids[marker_at + 1 :] if token != end_marker]
