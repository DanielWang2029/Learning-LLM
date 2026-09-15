"""A tiny FROZEN vision encoder (the CLIP-ViT stand-in).

In LLaVA the vision tower (a pretrained CLIP ViT) is *frozen* and only produces
features; a learned projection is what maps those features into the language
model's embedding space. We mirror that split exactly: this encoder is a small
CNN whose weights are randomly initialized and then **frozen** — it plays the
role of a fixed, pretrained feature extractor. Each spatial location of its
output feature map becomes one "patch token", just like ViT patches.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class VisionEncoder(nn.Module):
    """Frozen CNN feature extractor -> a grid of patch tokens."""

    def __init__(self, feat_dim: int = 16, seed: int = 0) -> None:
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.net = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, feat_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        # Deterministic init, then freeze (never trained).
        for p in self.net.parameters():
            if p.dim() > 1:
                nn.init.normal_(p, 0.0, 0.3, generator=gen)
            else:
                nn.init.zeros_(p)
            p.requires_grad_(False)
        self.feat_dim = feat_dim

    @torch.no_grad()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """images (B,3,S,S) -> patch tokens (B, num_patches, feat_dim)."""
        fmap = self.net(images)                    # (B, feat_dim, H', W')
        B, C, H, W = fmap.shape
        return fmap.view(B, C, H * W).transpose(1, 2)  # (B, H'*W', C)
