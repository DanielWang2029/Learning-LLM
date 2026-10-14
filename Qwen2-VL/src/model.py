"""Tiny dynamic-resolution vision encoder with 3-axis rotary attention.

Paper-inspired mechanism: Qwen2-VL represents an image by however many patch
tokens its native size produces, and M-RoPE rotates attention queries/keys by
temporal, height, and width coordinates independently.
"""
from __future__ import annotations

import math
import torch
from torch import nn


def mrope(x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """Apply separable (t, h, w) RoPE to three equal channel groups.

    x is [batch, heads, tokens, channels]; coords is [batch, tokens, 3].
    This compact implementation uses t=0 for still images but preserves the
    temporal axis used for videos in the paper.
    """
    b, heads, tokens, dim = x.shape
    assert dim % 6 == 0, "head dimension must split into 3 even groups"
    group = dim // 3
    freq = torch.exp(
        -math.log(10_000) * torch.arange(0, group, 2, device=x.device) / group
    )
    pieces = []
    for axis in range(3):
        z = x[..., axis * group : (axis + 1) * group]
        angle = coords[..., axis].unsqueeze(1).unsqueeze(-1) * freq
        even, odd = z[..., 0::2], z[..., 1::2]
        pieces.append(torch.stack(
            (even * angle.cos() - odd * angle.sin(),
             even * angle.sin() + odd * angle.cos()), dim=-1
        ).flatten(-2))
    return torch.cat(pieces, dim=-1)


class RotaryAttention(nn.Module):
    def __init__(self, dim: int, heads: int, use_mrope: bool) -> None:
        super().__init__()
        self.heads, self.head_dim, self.use_mrope = heads, dim // heads, use_mrope
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(b, n, self.heads, self.head_dim).transpose(1, 2)
        k = k.view(b, n, self.heads, self.head_dim).transpose(1, 2)
        v = v.view(b, n, self.heads, self.head_dim).transpose(1, 2)
        if self.use_mrope:
            q, k = mrope(q, coords), mrope(k, coords)
        weights = (q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)).softmax(-1)
        return self.out((weights @ v).transpose(1, 2).reshape(b, n, d))


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, use_mrope: bool) -> None:
        super().__init__()
        self.norm1, self.attn = nn.LayerNorm(dim), RotaryAttention(dim, heads, use_mrope)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))

    def forward(self, x: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), coords)
        return x + self.mlp(self.norm2(x))


class DynamicResolutionClassifier(nn.Module):
    """Patchify every native resolution, then classify a variable token sequence."""
    def __init__(self, dim: int = 48, use_mrope: bool = True) -> None:
        super().__init__()
        self.patch = nn.Conv2d(3, dim, kernel_size=4, stride=4)
        # Two 24-D heads keep each head divisible into three even RoPE groups.
        self.blocks = nn.ModuleList([Block(dim, 2, use_mrope) for _ in range(2)])
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 4))

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, int]:
        # Each batch is one native resolution; no image is resized.
        x = self.patch(image).flatten(2).transpose(1, 2)
        h, w = image.shape[-2] // 4, image.shape[-1] // 4
        yy, xx = torch.meshgrid(torch.arange(h, device=x.device),
                                torch.arange(w, device=x.device), indexing="ij")
        coords = torch.stack((torch.zeros_like(yy), yy, xx), -1).reshape(1, h * w, 3)
        coords = coords.expand(x.size(0), -1, -1).float()
        for block in self.blocks:
            x = block(x, coords)
        return self.head(x.mean(1)), x.size(1)
