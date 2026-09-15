"""A small BERT-style bidirectional encoder with a masked-LM head.

BERT = Bidirectional Encoder Representations from Transformers
(Devlin et al., 2018). This is the encoder stack from "Attention Is All You
Need" plus:

* learned token + position (+ segment) embeddings          (paper §3)
* a masked-language-model (MLM) head that predicts the      (paper §3.1)
  original identity of tokens replaced by ``[MASK]``.

The special tokens follow the paper's convention: ``[CLS]`` is prepended to
every sequence (its final representation is used for classification when the
model is fine-tuned) and ``[MASK]`` marks positions the MLM head must recover.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .layers import Encoder, EncoderLayer

# Reserved token ids shared across the demos.
PAD, CLS, MASK, SEP = 0, 1, 2, 3
NUM_SPECIAL = 4


class BertEmbeddings(nn.Module):
    """Sum of token, learned-position and segment embeddings (BERT §3)."""

    def __init__(self, vocab_size: int, d_model: int, max_len: int, dropout: float) -> None:
        super().__init__()
        self.token = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        self.position = nn.Embedding(max_len, d_model)
        self.segment = nn.Embedding(2, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor, segments: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, t = tokens.shape
        pos = torch.arange(t, device=tokens.device).unsqueeze(0).expand(b, t)
        if segments is None:
            segments = torch.zeros_like(tokens)
        x = self.token(tokens) + self.position(pos) + self.segment(segments)
        return self.dropout(self.norm(x))


class MaskedLMHead(nn.Module):
    """Transform + output projection tied to the token embedding (BERT §3.1)."""

    def __init__(self, d_model: int, embedding_weight: torch.Tensor) -> None:
        super().__init__()
        self.dense = nn.Linear(d_model, d_model)
        self.act = nn.GELU()
        self.norm = nn.LayerNorm(d_model)
        vocab_size, _ = embedding_weight.shape
        self.decoder = nn.Linear(d_model, vocab_size, bias=True)
        self.decoder.weight = embedding_weight  # weight tying

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.norm(self.act(self.dense(x))))


class BertModel(nn.Module):
    """A tiny BERT: embeddings → bidirectional encoder → MLM / pooler heads."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        d_ff: int = 256,
        max_len: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embeddings = BertEmbeddings(vocab_size, d_model, max_len, dropout)
        self.encoder = Encoder(EncoderLayer(d_model, num_heads, d_ff, dropout), num_layers)
        self.mlm_head = MaskedLMHead(d_model, self.embeddings.token.weight)
        # Pooler over the [CLS] token — used by the fine-tuning path.
        self.pooler = nn.Sequential(nn.Linear(d_model, d_model), nn.Tanh())
        self._init_weights()

    def _init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.normal_(p, mean=0.0, std=0.02)

    @staticmethod
    def padding_mask(tokens: torch.Tensor) -> torch.Tensor:
        """(batch, 1, seq) mask that is False on PAD positions."""
        return (tokens != PAD).unsqueeze(1)

    def encode(self, tokens: torch.Tensor, segments: Optional[torch.Tensor] = None) -> torch.Tensor:
        mask = self.padding_mask(tokens)
        return self.encoder(self.embeddings(tokens, segments), mask)

    def forward(self, tokens: torch.Tensor, segments: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return MLM logits of shape (batch, seq, vocab)."""
        return self.mlm_head(self.encode(tokens, segments))

    def pooled_cls(self, tokens: torch.Tensor, segments: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Pooled representation of the [CLS] token (for fine-tuning)."""
        return self.pooler(self.encode(tokens, segments)[:, 0])
