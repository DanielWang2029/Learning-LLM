"""Toy classification tasks that share structure — the LoRA setting.

A fixed random *teacher* maps inputs to hidden features; two different linear
heads on top define a **base task** and a **new task**.  Because both tasks are
built on the *same* features, a model pretrained on the base task already holds
most of what the new task needs — so adapting it with a small, low-rank update
(LoRA) is enough, which is exactly the regime LoRA is designed for.
"""

from __future__ import annotations

import torch

IN_DIM = 32
HIDDEN = 768
NUM_CLASSES = 5
# The teacher's intrinsic feature dimension is small, so the *difference*
# between the two tasks has low rank — precisely the regime the LoRA paper
# argues real weight updates live in (Section 7.2, "intrinsic rank").
_TEACHER_HIDDEN = 16


def _teacher(seed: int):
    """Build a fixed random teacher: shared feature map + a task head."""
    g = torch.Generator().manual_seed(seed)
    W_feat = torch.randn(_TEACHER_HIDDEN, IN_DIM, generator=g) / (IN_DIM ** 0.5)
    return W_feat, g


# Shared feature extractor (identical for both tasks).
_W_FEAT, _ = _teacher(seed=1234)


def _make_head(seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(NUM_CLASSES, _TEACHER_HIDDEN, generator=g)


_HEAD_BASE = _make_head(seed=1)
_HEAD_NEW = _make_head(seed=2)


def _features(x: torch.Tensor) -> torch.Tensor:
    return torch.relu(x @ _W_FEAT.T)


def make_task(which: str, n: int, seed: int):
    """Return ``(x, y)`` for ``which`` in {"base", "new"}."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, IN_DIM, generator=g)
    feats = _features(x)
    head = _HEAD_BASE if which == "base" else _HEAD_NEW
    y = (feats @ head.T).argmax(dim=-1)
    return x, y
