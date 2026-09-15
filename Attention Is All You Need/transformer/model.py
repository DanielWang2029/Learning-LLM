"""Full encoder-decoder Transformer model (paper Section 3)."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from .layers import Decoder, DecoderLayer, Encoder, EncoderLayer
from .positional import PositionalEncoding


class Embeddings(nn.Module):
    """Token embedding scaled by sqrt(d_model), as described in Section 3.4."""

    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embed(x) * math.sqrt(self.d_model)


def subsequent_mask(size: int, device: Optional[torch.device] = None) -> torch.Tensor:
    """Lower-triangular mask preventing positions from attending to the future."""
    ones = torch.ones(size, size, device=device, dtype=torch.bool)
    return torch.tril(ones).unsqueeze(0)  # (1, size, size)


class Transformer(nn.Module):
    """The base Transformer for sequence-to-sequence tasks.

    Defaults mirror the "base" model from the paper (Table 3): ``d_model=512``,
    ``num_layers=6``, ``num_heads=8``, ``d_ff=2048``. The demo shipped with this
    repository uses a much smaller configuration so it trains quickly on CPU.
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_len: int = 5000,
    ) -> None:
        super().__init__()
        self.src_embed = Embeddings(src_vocab_size, d_model)
        self.tgt_embed = Embeddings(tgt_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_len)

        self.encoder = Encoder(
            EncoderLayer(d_model, num_heads, d_ff, dropout), num_layers
        )
        self.decoder = Decoder(
            DecoderLayer(d_model, num_heads, d_ff, dropout), num_layers
        )
        self.generator = nn.Linear(d_model, tgt_vocab_size)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(
        self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x = self.pos_encoding(self.src_embed(src))
        return self.encoder(x, src_mask)

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.pos_encoding(self.tgt_embed(tgt))
        return self.decoder(x, memory, src_mask, tgt_mask)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return unnormalized next-token logits of shape (batch, tgt_len, vocab)."""
        memory = self.encode(src, src_mask)
        output = self.decode(tgt, memory, src_mask, tgt_mask)
        return self.generator(output)

    @torch.no_grad()
    def greedy_decode(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor],
        max_len: int,
        start_symbol: int,
        end_symbol: Optional[int] = None,
    ) -> torch.Tensor:
        """Autoregressively decode a single sequence with greedy search."""
        self.eval()
        memory = self.encode(src, src_mask)
        ys = torch.full(
            (src.size(0), 1), start_symbol, dtype=torch.long, device=src.device
        )
        for _ in range(max_len - 1):
            tgt_mask = subsequent_mask(ys.size(1), device=src.device)
            out = self.decode(ys, memory, src_mask, tgt_mask)
            logits = self.generator(out[:, -1])
            next_token = logits.argmax(dim=-1, keepdim=True)
            ys = torch.cat([ys, next_token], dim=1)
            if end_symbol is not None and (next_token == end_symbol).all():
                break
        return ys
