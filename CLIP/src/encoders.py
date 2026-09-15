"""The two CLIP encoders (paper Section 2.4, "Choosing and Scaling a Model").

CLIP is a *dual-encoder*: an image encoder and a text encoder that map their
inputs into a shared representation space. The originals are a ResNet/ViT and a
Transformer; here we use a tiny CNN and a token-embedding + pooling text encoder
so the whole thing trains in seconds on CPU while keeping the same interface:

    image  --f_image-->  h_img  --W_img-->  e_img  (L2-normalized)
    text   --f_text -->  h_txt  --W_txt-->  e_txt  (L2-normalized)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tokenizer import PAD_ID, VOCAB_SIZE


class ImageEncoder(nn.Module):
    """A small convolutional image encoder + linear projection to embed_dim."""

    def __init__(self, embed_dim: int = 64, width: int = 32) -> None:
        super().__init__()
        # Two conv blocks with stride-2 downsampling: 24x24 -> 12x12 -> 6x6.
        self.features = nn.Sequential(
            nn.Conv2d(3, width, kernel_size=3, stride=2, padding=1),   # 24 -> 12
            nn.ReLU(inplace=True),
            nn.Conv2d(width, width * 2, kernel_size=3, stride=2, padding=1),  # 12 -> 6
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),  # global average pool -> (B, width*2, 1, 1)
        )
        self.proj = nn.Linear(width * 2, embed_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        h = self.features(images).flatten(1)  # (B, width*2)
        return self.proj(h)                   # (B, embed_dim), pre-normalization


class TextEncoder(nn.Module):
    """Token embedding + mean pooling + linear projection to embed_dim.

    A faithful-but-minimal stand-in for CLIP's text Transformer: it still learns
    per-token representations that get pooled into a single caption embedding.
    """

    def __init__(self, embed_dim: int = 64, token_dim: int = 48) -> None:
        super().__init__()
        self.token_embed = nn.Embedding(VOCAB_SIZE, token_dim, padding_idx=PAD_ID)
        self.encoder = nn.Sequential(
            nn.Linear(token_dim, token_dim),
            nn.ReLU(inplace=True),
        )
        self.proj = nn.Linear(token_dim, embed_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: (B, L) int64
        emb = self.token_embed(tokens)              # (B, L, token_dim)
        emb = self.encoder(emb)
        mask = (tokens != PAD_ID).unsqueeze(-1).float()  # ignore padding in the mean
        pooled = (emb * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        return self.proj(pooled)                    # (B, embed_dim), pre-normalization
