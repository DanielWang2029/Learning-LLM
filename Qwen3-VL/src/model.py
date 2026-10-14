"""A compact DeepStack visual-to-language decoder demonstration."""
from __future__ import annotations
import math
import torch
from torch import nn

class CrossAttention(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.q, self.k, self.v, self.o = (nn.Linear(dim, dim) for _ in range(4))
    def forward(self, query: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        score = self.q(query) @ self.k(memory).transpose(-2, -1) / math.sqrt(query.size(-1))
        return self.o(score.softmax(-1) @ self.v(memory))

class VisionBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim); self.attn = nn.MultiheadAttention(dim, 2, batch_first=True)
        self.ff = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 2*dim), nn.GELU(), nn.Linear(2*dim, dim))
    def forward(self, x):
        x = x + self.attn(self.norm(x), self.norm(x), self.norm(x), need_weights=False)[0]
        return x + self.ff(x)

class TinyViT(nn.Module):
    """Expose low/mid token sequences; a deliberately pooled final level loses texture."""
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.embed = nn.Linear(3, dim)
        self.pos = nn.Parameter(torch.randn(1, 64, dim) * .02)
        self.first, self.second = VisionBlock(dim), VisionBlock(dim)
    def forward(self, image):
        b = image.size(0)
        low = self.embed(image.permute(0,2,3,1).reshape(b,64,3)) + self.pos
        mid = self.second(self.first(low))
        # A pooled high level deliberately contains global image statistics only.
        high = image.mean((-1,-2)).unsqueeze(1).expand(-1, 1, 3)
        return low, mid, high

class DeepStackClassifier(nn.Module):
    """A three-layer language-side stack with visual injection at chosen depths."""
    def __init__(self, deepstack: bool, dim: int = 36) -> None:
        super().__init__()
        self.deepstack = deepstack
        self.vit = TinyViT(dim)
        self.high_project = nn.Linear(3, dim)
        self.query = nn.Parameter(torch.zeros(1, 1, dim))
        self.cross = nn.ModuleList([CrossAttention(dim) for _ in range(3)])
        self.ff = nn.ModuleList([nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU()) for _ in range(3)])
        self.head = nn.Linear(dim, 2)
    def forward(self, image):
        low, mid, high_rgb = self.vit(image)
        high = self.high_project(high_rgb)
        memories = (low, mid, high) if self.deepstack else (high, high, high)
        state = self.query.expand(image.size(0), -1, -1)
        for cross, ff, memory in zip(self.cross, self.ff, memories):
            state = state + cross(state, memory)
            state = state + ff(state)
        return self.head(state[:, 0])
