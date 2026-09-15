"""Bidirectional Transformer used as LLaDA's mask predictor (paper §2.2).

Unlike an autoregressive LM, this network has **no causal mask**: every
position may attend to every other position. Its only job is to predict the
original token at each *masked* position given the (partially masked) context.
The same network is reused at every step of the reverse diffusion process.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MaskPredictor(nn.Module):
    """A small bidirectional Transformer encoder with a token-prediction head.

    Args:
        vocab_size: number of token ids *including* the reserved ``[MASK]`` id.
        d_model:    embedding / hidden width.
        num_layers: number of Transformer encoder blocks.
        num_heads:  attention heads per block.
        d_ff:       inner width of the position-wise feed-forward network.
        max_len:    maximum sequence length (for the learned position table).
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 96,
        num_layers: int = 3,
        num_heads: int = 4,
        d_ff: int = 192,
        max_len: int = 32,
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
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self.d_model = d_model

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Map a (batch, seq) batch of token ids to (batch, seq, vocab) logits."""
        b, t = tokens.shape
        pos = torch.arange(t, device=tokens.device).unsqueeze(0).expand(b, t)
        # Learned token + position embeddings are added at the same scale, so
        # the position signal is strong enough for the model to reason about
        # *where* each token sits (essential for a bidirectional predictor).
        x = self.tok_emb(tokens) + self.pos_emb(pos)
        # No attention mask: the encoder is fully bidirectional.
        x = self.encoder(x)
        return self.head(self.norm(x))
