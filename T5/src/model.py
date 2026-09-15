"""A small T5-style encoder–decoder Transformer.

Raffel et al., "Exploring the Limits of Transfer Learning with a Unified
Text-to-Text Transformer" (T5, 2020). T5 casts *every* problem as
text-to-text: the encoder reads an input string and the decoder generates an
output string. It is pre-trained with a *span-corruption* objective — random
spans of the input are replaced by sentinel tokens and the decoder must
reconstruct them — and then the same architecture is used for many downstream
tasks distinguished only by a text prefix (here: ``copy`` / ``reverse`` /
``sort``).

This implementation keeps the standard Transformer encoder–decoder (sinusoidal
positional encodings) for clarity; T5's relative position biases are omitted as
a simplification.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from .layers import Decoder, DecoderLayer, Encoder, EncoderLayer

PAD, BOS, EOS = 0, 1, 2


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


def subsequent_mask(size: int, device=None) -> torch.Tensor:
    return torch.tril(torch.ones(size, size, device=device, dtype=torch.bool)).unsqueeze(0)


class T5(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 96,
        num_layers: int = 3,
        num_heads: int = 4,
        d_ff: int = 192,
        dropout: float = 0.1,
        max_len: int = 64,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        self.pos = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)
        self.encoder = Encoder(EncoderLayer(d_model, num_heads, d_ff, dropout), num_layers)
        self.decoder = Decoder(DecoderLayer(d_model, num_heads, d_ff, dropout), num_layers)
        self.generator = nn.Linear(d_model, vocab_size)
        self.d_model = d_model
        self._reset()

    def _reset(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _embed(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.pos(self.embed(tokens) * math.sqrt(self.d_model)))

    @staticmethod
    def src_pad_mask(src: torch.Tensor) -> torch.Tensor:
        return (src != PAD).unsqueeze(1)  # (b, 1, src_len)

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        return self.encoder(self._embed(src), src_mask)

    def decode(self, tgt, memory, src_mask, tgt_mask):
        return self.decoder(self._embed(tgt), memory, src_mask, tgt_mask)

    def forward(self, src, tgt_in, src_mask=None, tgt_mask=None):
        if src_mask is None:
            src_mask = self.src_pad_mask(src)
        if tgt_mask is None:
            tgt_mask = subsequent_mask(tgt_in.size(1), device=tgt_in.device)
        memory = self.encode(src, src_mask)
        out = self.decode(tgt_in, memory, src_mask, tgt_mask)
        return self.generator(out)

    @torch.no_grad()
    def greedy_decode(self, src, max_len: int, device=None) -> torch.Tensor:
        self.eval()
        device = device or src.device
        src_mask = self.src_pad_mask(src)
        memory = self.encode(src, src_mask)
        ys = torch.full((src.size(0), 1), BOS, dtype=torch.long, device=device)
        finished = torch.zeros(src.size(0), dtype=torch.bool, device=device)
        for _ in range(max_len - 1):
            tgt_mask = subsequent_mask(ys.size(1), device=device)
            out = self.decode(ys, memory, src_mask, tgt_mask)
            nxt = self.generator(out[:, -1]).argmax(-1, keepdim=True)
            ys = torch.cat([ys, nxt], dim=1)
            finished |= nxt.squeeze(1) == EOS
            if finished.all():
                break
        return ys
