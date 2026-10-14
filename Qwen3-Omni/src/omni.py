"""Tiny AuT-inspired audio encoder with window attention, MoE, and multi-codebook MTP."""
import math
import torch
from torch import nn


def window_attention(x: torch.Tensor, window: int | None) -> torch.Tensor:
    """Parameter-free locality-biased attention; window=None is the full-attention reference."""
    n, d = x.shape
    scores = (x @ x.T) / math.sqrt(d)
    distance = torch.arange(n)[:, None] - torch.arange(n)[None, :]
    scores = scores - distance.abs().float() * 0.8  # audio attention is strongly local
    if window is not None:
        scores = scores.masked_fill(distance.abs() > window, -1e9)
    return torch.softmax(scores, dim=-1) @ x


class TinyMoE(nn.Module):
    """Top-1 routing exposes the sparse expert count used by the toy encoder."""
    def __init__(self, width: int, experts: int = 4):
        super().__init__()
        self.router = nn.Linear(width, experts)
        self.experts = nn.ModuleList([nn.Sequential(nn.Linear(width, width * 2), nn.GELU(),
                                                   nn.Linear(width * 2, width))
                                      for _ in range(experts)])

    def forward(self, x):
        choice = self.router(x).argmax(-1)
        out = torch.empty_like(x)
        for expert, ffn in enumerate(self.experts):
            use = choice == expert
            if use.any():
                out[use] = ffn(x[use])
        return out, choice


class AudioTransformerToy(nn.Module):
    def __init__(self, mel_bins: int = 16, width: int = 24, codebooks: int = 3, vocab: int = 16):
        super().__init__()
        self.project = nn.Linear(mel_bins, width)
        self.moe = TinyMoE(width)
        self.codebook_heads = nn.ModuleList([nn.Linear(width, vocab) for _ in range(codebooks)])

    def encode(self, mels, window):
        hidden = self.project(mels)
        attended = window_attention(hidden, window)
        output, route = self.moe(attended)
        return output, route

    def predict_codebooks(self, state):
        return torch.stack([head(state) for head in self.codebook_heads], dim=1)
