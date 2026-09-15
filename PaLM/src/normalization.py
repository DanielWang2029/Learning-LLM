"""Layer normalization without bias terms.

PaLM removes all bias parameters: "No biases were used in any of the dense
kernels or layer norms" (PaLM paper, Section 2, "No Biases"). Dropping biases
was found to increase training stability for large models. This is a standard
LayerNorm with a learned scale but no learned shift.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNormNoBias(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, bias=None, eps=self.eps)
