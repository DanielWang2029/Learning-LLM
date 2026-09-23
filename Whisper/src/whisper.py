"""Whisper-style encoder-decoder, from scratch (paper Section 2.2).

Architecture (Figure 1 of the paper):

    log-Mel (80 bins)
      │  Conv1d(k=3) + GELU
      │  Conv1d(k=3, stride 2) + GELU     ← halves the frame rate (to ~50 Hz)
      ▼
    + sinusoidal positional encoding
      ▼
    Transformer encoder  ──────────────┐
                                        │ cross-attention
    tokens → embedding + sinusoids      ▼
      ▼                          Transformer decoder (causal)
    Transformer decoder ──► linear ──► next-token logits

Whisper processes a fixed 30 s chunk (non-streaming) and frames the task as a
single sequence-to-sequence problem with special "multitask" tokens
(<|startoftranscript|>, task/language tokens, ...). This minimal version keeps
the encoder conv stem, sinusoidal positions, and the encoder-decoder Transformer,
and uses a small multitask-style prompt token.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def sinusoids(length: int, channels: int, max_timescale: int = 10000) -> torch.Tensor:
    """Fixed sinusoidal positional encodings (paper uses these for both stacks)."""
    assert channels % 2 == 0
    log_ts = np.log(max_timescale) / (channels // 2 - 1)
    inv = torch.exp(-log_ts * torch.arange(channels // 2))
    scaled = torch.arange(length)[:, None] * inv[None, :]
    return torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=1)


class AudioEncoder(nn.Module):
    """Conv stem (2nd layer stride-2) + Transformer encoder (Section 2.2)."""

    def __init__(self, n_mels: int, d_model: int, n_layers: int, n_heads: int, ff: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(n_mels, d_model, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1)
        self.gelu = nn.GELU()
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, ff, dropout=0.0, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.ln_post = nn.LayerNorm(d_model)
        self.d_model = d_model

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """mel (B, n_mels, T) -> encoded (B, T//2, d_model)."""
        x = self.gelu(self.conv1(mel))
        x = self.gelu(self.conv2(x))          # stride-2 downsample
        x = x.transpose(1, 2)                 # (B, T', d_model)
        x = x + sinusoids(x.shape[1], self.d_model).to(x.dtype)
        return self.ln_post(self.blocks(x))


class TextDecoder(nn.Module):
    """Token embedding + sinusoidal positions + Transformer decoder (Section 2.2)."""

    def __init__(self, vocab: int, d_model: int, n_layers: int, n_heads: int, ff: int, max_len: int = 64) -> None:
        super().__init__()
        self.token_emb = nn.Embedding(vocab, d_model)
        self.register_buffer("pos", sinusoids(max_len, d_model), persistent=False)
        layer = nn.TransformerDecoderLayer(
            d_model, n_heads, ff, dropout=0.0, activation="gelu",
            batch_first=True, norm_first=True,
        )
        self.blocks = nn.TransformerDecoder(layer, n_layers)
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)

    def forward(self, tokens: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        T = tokens.shape[1]
        x = self.token_emb(tokens) + self.pos[:T].to(tokens.device)
        causal = torch.triu(torch.full((T, T), float("-inf"), device=tokens.device), diagonal=1)
        x = self.blocks(x, memory, tgt_mask=causal)
        return self.head(self.ln(x))


class Whisper(nn.Module):
    def __init__(self, n_mels=80, vocab=32, d_model=128, n_layers=2, n_heads=4, ff=256) -> None:
        super().__init__()
        self.encoder = AudioEncoder(n_mels, d_model, n_layers, n_heads, ff)
        self.decoder = TextDecoder(vocab, d_model, n_layers, n_heads, ff)

    def forward(self, mel: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        return self.decoder(tokens, self.encoder(mel))

    @torch.no_grad()
    def transcribe(self, mel: torch.Tensor, sot: int, eot: int, max_len: int = 32) -> list:
        """Greedy autoregressive decoding from the start-of-transcript token."""
        self.eval()
        memory = self.encoder(mel)
        B = mel.shape[0]
        tokens = torch.full((B, 1), sot, dtype=torch.long, device=mel.device)
        finished = torch.zeros(B, dtype=torch.bool)
        for _ in range(max_len):
            logits = self.decoder(tokens, memory)
            nxt = logits[:, -1].argmax(-1, keepdim=True)
            tokens = torch.cat([tokens, nxt], dim=1)
            finished |= nxt.squeeze(1) == eot
            if finished.all():
                break
        return tokens.tolist()
