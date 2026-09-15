"""The symmetric contrastive loss (paper Section 2.2, Figure 3).

Given a batch of N matched (image, text) pairs, CLIP computes an N x N matrix of
cosine similarities scaled by a learned temperature. The correct pairs are on the
diagonal. The loss is a symmetric cross-entropy: each image should pick its own
caption (rows), and each caption should pick its own image (columns).

This is exactly InfoNCE applied in both directions:

    L = (CE(logits, labels) + CE(logitsᵀ, labels)) / 2 ,  labels = [0, 1, ..., N-1]
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def clip_contrastive_loss(logits_per_image: torch.Tensor) -> torch.Tensor:
    """Symmetric InfoNCE over an (N x N) scaled-similarity matrix."""
    n = logits_per_image.size(0)
    targets = torch.arange(n, device=logits_per_image.device)
    loss_i = F.cross_entropy(logits_per_image, targets)          # image -> text
    loss_t = F.cross_entropy(logits_per_image.t(), targets)      # text  -> image
    return (loss_i + loss_t) / 2
