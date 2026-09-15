"""LoRA and DoRA adapters for a frozen ``nn.Linear`` (paper: DoRA, §4).

**LoRA** freezes ``W0`` and adds a low-rank update:

    W = W0 + (alpha/r) · B·A

**DoRA** first *decomposes* the pretrained weight into a magnitude and a
direction, and adapts them separately (paper Eq. 3):

    W0 = m0 · V0 / ||V0||          (m0 = per-output norm of W0, V0 = W0)
    W  = m  · (V0 + ΔV) / ||V0 + ΔV||        with ΔV = (alpha/r) · B·A

Here the magnitude vector ``m`` (one scalar per output unit) is trained
*directly*, while the **direction** gets the low-rank LoRA update.  Decoupling
"how big" from "which way" gives DoRA a learning behavior closer to full
fine-tuning than LoRA — the paper's central result.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 4, alpha: int = 8) -> None:
        super().__init__()
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.r = r
        self.scaling = alpha / r

        self.weight = nn.Parameter(base.weight.detach().clone(), requires_grad=False)
        if base.bias is not None:
            self.bias = nn.Parameter(base.bias.detach().clone(), requires_grad=False)
        else:
            self.register_parameter("bias", None)

        self.lora_A = nn.Parameter(torch.empty(r, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r))
        nn.init.normal_(self.lora_A, std=1.0 / r)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = self.scaling * (self.lora_B @ self.lora_A)
        return F.linear(x, self.weight + delta, self.bias)

    def adapter_params(self):
        return [self.lora_A, self.lora_B]


class DoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int = 4, alpha: int = 8) -> None:
        super().__init__()
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.r = r
        self.scaling = alpha / r

        # Frozen pretrained weight = the (also frozen) DIRECTION matrix V0.
        self.weight = nn.Parameter(base.weight.detach().clone(), requires_grad=False)
        if base.bias is not None:
            self.bias = nn.Parameter(base.bias.detach().clone(), requires_grad=False)
        else:
            self.register_parameter("bias", None)

        # Trainable MAGNITUDE m, initialized to the per-output norm of W0 so that
        # the adapted weight starts exactly equal to W0 (paper §4).
        m0 = base.weight.detach().norm(dim=1)             # (out,)
        self.magnitude = nn.Parameter(m0.clone())

        # Low-rank update applied to the DIRECTION only. B=0 at init -> ΔV=0.
        self.lora_A = nn.Parameter(torch.empty(r, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r))
        nn.init.normal_(self.lora_A, std=1.0 / r)

    def effective_weight(self) -> torch.Tensor:
        delta_v = self.scaling * (self.lora_B @ self.lora_A)   # ΔV  (out,in)
        v = self.weight + delta_v                              # V0 + ΔV
        norm = v.norm(dim=1, keepdim=True)                     # ||·|| per output
        return self.magnitude.unsqueeze(1) * v / norm          # m · V/||V||

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.effective_weight(), self.bias)

    def adapter_params(self):
        return [self.magnitude, self.lora_A, self.lora_B]


def _inject(model: nn.Module, cls, r: int, alpha: int) -> list[str]:
    wrapped: list[str] = []
    for name, child in list(model.named_children()):
        if isinstance(child, nn.Linear):
            setattr(model, name, cls(child, r=r, alpha=alpha))
            wrapped.append(name)
        else:
            wrapped.extend(f"{name}.{s}" for s in _inject(child, cls, r, alpha))
    return wrapped


def inject_lora(model: nn.Module, r: int = 4, alpha: int = 8) -> list[str]:
    return _inject(model, LoRALinear, r, alpha)


def inject_dora(model: nn.Module, r: int = 4, alpha: int = 8) -> list[str]:
    return _inject(model, DoRALinear, r, alpha)


def adapter_parameters(model: nn.Module):
    """Yield only the trainable adapter params (LoRA factors, plus m for DoRA)."""
    for module in model.modules():
        if isinstance(module, (LoRALinear, DoRALinear)):
            yield from module.adapter_params()
