"""Vision encoder (paper Section 2.1, "Visual processing and the Perceiver Resampler").

In Flamingo the vision encoder is a large pretrained, *frozen* contrastive model
(NFNet). We use a tiny CNN instead, but keep the important property: it turns an
image into a *set of spatial feature tokens* (one per remaining grid cell), which
the Perceiver Resampler then compresses into a fixed number of visual tokens.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class VisionEncoder(nn.Module):
    """Tiny CNN that maps an image to a grid of feature tokens."""

    def __init__(self, d_vis: int = 64, width: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, width, kernel_size=3, stride=2, padding=1),      # 24 -> 12
            nn.GELU(),
            nn.Conv2d(width, d_vis, kernel_size=3, stride=2, padding=1),   # 12 -> 6
            nn.GELU(),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """images: (B, 3, H, W) -> visual features (B, num_tokens, d_vis)."""
        feat = self.net(images)                 # (B, d_vis, h, w)
        b, c, h, w = feat.shape
        return feat.flatten(2).transpose(1, 2)  # (B, h*w, d_vis)
