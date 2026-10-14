"""Tiny native-grid ViT with local window and occasional full attention."""
from __future__ import annotations

import math
import torch
from torch import nn


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int = 2) -> None:
        super().__init__()
        self.heads, self.dh = heads, dim // heads
        self.qkv, self.out = nn.Linear(dim, 3 * dim), nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        q, k, v = self.qkv(x).chunk(3, -1)
        q = q.reshape(b, n, self.heads, self.dh).transpose(1, 2)
        k = k.reshape(b, n, self.heads, self.dh).transpose(1, 2)
        v = v.reshape(b, n, self.heads, self.dh).transpose(1, 2)
        a = (q @ k.transpose(-2, -1) / math.sqrt(self.dh)).softmax(-1)
        return self.out((a @ v).transpose(1, 2).reshape(b, n, d))


class Block(nn.Module):
    def __init__(self, dim: int, full_attention: bool) -> None:
        super().__init__()
        self.full_attention, self.norm1 = full_attention, nn.LayerNorm(dim)
        self.attn, self.norm2 = Attention(dim), nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))

    def forward(self, x: torch.Tensor, grid: int) -> torch.Tensor:
        if self.full_attention:
            update = self.attn(self.norm1(x))
        else:
            # 8×8 tokens become sixteen independent 2×2 token windows.
            b, _, d = x.shape
            z = self.norm1(x).reshape(b, grid, grid, d)
            z = z.reshape(b, grid // 2, 2, grid // 2, 2, d).permute(0, 1, 3, 2, 4, 5)
            z = z.reshape(-1, 4, d)
            update = self.attn(z).reshape(b, grid // 2, grid // 2, 2, 2, d)
            update = update.permute(0, 1, 3, 2, 4, 5).reshape(b, grid * grid, d)
        x = x + update
        return x + self.mlp(self.norm2(x))


class WindowViT(nn.Module):
    """Two local blocks and one global block, mirroring the paper's pattern."""
    def __init__(self, dim: int = 40) -> None:
        super().__init__()
        # Explicit native-grid coordinates make the grounding head spatially aware.
        self.embed = nn.Linear(5, dim)
        self.blocks = nn.ModuleList([Block(dim, False), Block(dim, False), Block(dim, True)])
        self.class_head = nn.Linear(dim, 4)
        self.box_head = nn.Linear(dim, 4)  # normalized x0,y0,x1,y1

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, _, h, w = images.shape
        rgb = images.permute(0, 2, 3, 1).reshape(b, h * w, 3)
        yy, xx = torch.meshgrid(torch.linspace(0, 1, h, device=images.device),
                                torch.linspace(0, 1, w, device=images.device), indexing="ij")
        coords = torch.stack((xx, yy), -1).reshape(1, h * w, 2).expand(b, -1, -1)
        x = torch.cat((rgb, coords), -1)
        x = self.embed(x)
        for block in self.blocks:
            x = block(x, h)
        # Detection/grounding predicts one global box; colors classify the object.
        pooled = x.mean(1)
        return self.class_head(pooled), self.box_head(pooled).sigmoid()


def attention_pair_cost(grid: int = 8) -> tuple[int, int]:
    """Attention score entries across three layers: hybrid vs all global."""
    n = grid * grid
    hybrid = 2 * ((grid // 2) ** 2 * 2 ** 4) + n ** 2
    full = 3 * n ** 2
    return hybrid, full
