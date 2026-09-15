"""Toy classification tasks that share structure — the parameter-efficient
fine-tuning setting.

A fixed random *teacher* maps inputs to hidden features; two linear heads on top
define a **base task** (used for pretraining) and a related **new task** (used
for adaptation).  Because both tasks share the same features, a model pretrained
on the base task already holds most of what the new task needs — so adapting it
with a small update is enough, which is exactly the regime LoRA / DoRA target.
"""

from __future__ import annotations

import torch

IN_DIM = 32
HIDDEN = 256
NUM_CLASSES = 6
_TEACHER_HIDDEN = 16       # small intrinsic dimension -> low-rank task differences


_g = torch.Generator().manual_seed(1234)
_W_FEAT = torch.randn(_TEACHER_HIDDEN, IN_DIM, generator=_g) / (IN_DIM ** 0.5)


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
