"""Low-Rank Adaptation of a Linear layer (paper Section 4, "Our Method").

LoRA freezes a pretrained weight ``W0`` and represents the task-specific
update as a low-rank product:

    h = W0 x + b   +   (alpha / r) * (B @ A) x

with ``A`` of shape (r, in) and ``B`` of shape (out, r).  Only ``A`` and ``B``
train; ``W0`` and ``b`` are frozen.  ``r`` is the rank and ``alpha`` a constant
scaling factor, exactly as in Eq. (3) / Section 4.1 of the paper.  ``A`` is
initialized from a small Gaussian and ``B`` from zero, so at the start of
training ``B @ A = 0`` and the adapted model equals the pretrained model.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """A ``nn.Linear`` whose base weight is frozen and adapted by ``ΔW = B·A``.

    Wraps an existing (already-trained) ``nn.Linear`` so that the original
    weight becomes the frozen ``W0`` and only the rank-``r`` factors update.
    """

    def __init__(self, base: nn.Linear, r: int = 4, alpha: int = 8) -> None:
        super().__init__()
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        # Frozen pretrained weight W0 (and bias). These do not receive grads.
        self.weight = nn.Parameter(base.weight.detach().clone(), requires_grad=False)
        if base.bias is not None:
            self.bias = nn.Parameter(base.bias.detach().clone(), requires_grad=False)
        else:
            self.register_parameter("bias", None)

        # Trainable low-rank factors: A (r, in) ~ N(0, .), B (out, r) = 0.
        self.lora_A = nn.Parameter(torch.empty(r, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r))
        nn.init.normal_(self.lora_A, std=1.0 / r)

        # When True, ΔW has been folded into ``weight`` (deployment mode).
        self.merged = False

    def delta_w(self) -> torch.Tensor:
        """The effective weight update ΔW = (alpha/r) · B·A  (shape out×in)."""
        return self.scaling * (self.lora_B @ self.lora_A)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # If merged, ΔW already lives inside ``weight`` — a single matmul.
        if self.merged:
            return F.linear(x, self.weight, self.bias)
        base = F.linear(x, self.weight, self.bias)
        update = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
        return base + update

    @torch.no_grad()
    def merge(self) -> None:
        """Fold ΔW into the base weight → zero extra inference cost (Sec. 3)."""
        if not self.merged:
            self.weight += self.delta_w()
            self.merged = True

    @torch.no_grad()
    def unmerge(self) -> None:
        """Recover the original frozen W0 (undo :meth:`merge`)."""
        if self.merged:
            self.weight -= self.delta_w()
            self.merged = False


def inject_lora(model: nn.Module, r: int = 4, alpha: int = 8) -> list[str]:
    """Replace every ``nn.Linear`` in ``model`` with a :class:`LoRALinear`.

    Returns the list of module paths that were wrapped.  This mirrors how LoRA
    is applied post-hoc to a frozen pretrained network.
    """
    wrapped: list[str] = []
    for name, child in list(model.named_children()):
        if isinstance(child, nn.Linear):
            setattr(model, name, LoRALinear(child, r=r, alpha=alpha))
            wrapped.append(name)
        else:
            wrapped.extend(f"{name}.{sub}" for sub in inject_lora(child, r, alpha))
    return wrapped


def lora_parameters(model: nn.Module):
    """Yield only the trainable LoRA factors (A and B) across the model."""
    for module in model.modules():
        if isinstance(module, LoRALinear):
            yield module.lora_A
            yield module.lora_B


def merge_all(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.merge()


def unmerge_all(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.unmerge()
