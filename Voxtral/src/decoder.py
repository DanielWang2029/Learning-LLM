"""Tiny causal language decoder (Voxtral §2, the "multimodal LLM decoder").

Voxtral feeds the down-sampled audio embeddings into a language model as a
prefix of *soft* tokens, then autoregressively predicts text tokens conditioned
on that audio prefix. This is a minimal decoder-only Transformer that does
exactly that: it consumes a sequence of input embeddings (audio prefix followed
by text-token embeddings) and predicts the next text token at each position.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .encoder import sinusoidal_positions


class LMDecoder(nn.Module):
    """Decoder-only Transformer over pre-embedded inputs."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        d_ff: int = 128,
        dropout: float = 0.0,
        max_len: int = 512,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.token_embed = nn.Embedding(vocab_size, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.max_len = max_len

    def embed_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.token_embed(tokens)

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        """Run the causal Transformer over a sequence of input embeddings."""
        seq = inputs_embeds.size(1)
        pos = sinusoidal_positions(seq, self.d_model, inputs_embeds.device)
        x = inputs_embeds + pos.unsqueeze(0)
        causal = torch.triu(
            torch.ones(seq, seq, device=x.device, dtype=torch.bool), diagonal=1
        )
        x = self.transformer(x, mask=causal)
        return self.lm_head(self.norm(x))
