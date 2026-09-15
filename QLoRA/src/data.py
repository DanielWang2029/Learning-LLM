"""Toy classification tasks that share structure (same setup as the LoRA folder).

A fixed random teacher maps inputs to low-dimensional features; two different
heads define a base task (for pretraining) and a new task (for QLoRA
fine-tuning). Because the tasks share features, adapting the pretrained model
needs only a small low-rank update — which QLoRA supplies in fp32 on top of a
4-bit-quantized frozen backbone.
"""

from __future__ import annotations

import torch

IN_DIM = 32
HIDDEN = 768
NUM_CLASSES = 5
_TEACHER_HIDDEN = 16


def _teacher(seed: int):
    g = torch.Generator().manual_seed(seed)
    W_feat = torch.randn(_TEACHER_HIDDEN, IN_DIM, generator=g) / (IN_DIM ** 0.5)
    return W_feat


_W_FEAT = _teacher(seed=1234)


def _make_head(seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.randn(NUM_CLASSES, _TEACHER_HIDDEN, generator=g)


_HEAD_BASE = _make_head(seed=1)
_HEAD_NEW = _make_head(seed=2)


def _features(x: torch.Tensor) -> torch.Tensor:
    return torch.relu(x @ _W_FEAT.T)


def make_task(which: str, n: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, IN_DIM, generator=g)
    feats = _features(x)
    head = _HEAD_BASE if which == "base" else _HEAD_NEW
    y = (feats @ head.T).argmax(dim=-1)
    return x, y
