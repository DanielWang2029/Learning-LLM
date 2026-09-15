"""QLoRA linear layer: 4-bit frozen base + full-precision LoRA (paper §3).

The base weight ``W0`` is stored in 4-bit NF4 and never trains.  On the forward
pass it is dequantized to compute ``W0·x`` (this is QLoRA's "dequantize for the
matmul" step).  A small full-precision LoRA update ``(alpha/r)·B·A`` is added on
top and is the *only* thing that receives gradients:

    h = dequant(W0_4bit)·x + b  +  (alpha/r) · (B · A) · x

So the large weight lives in 4 bits while the trainable adapter stays in fp32 —
the combination that lets QLoRA fine-tune large models in little memory.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .quant import Quantized, dequantize_nf4, quantize_nf4


class QLoRALinear(nn.Module):
    """A Linear whose base weight is 4-bit-quantized & frozen, adapted by B·A."""

    def __init__(self, base: nn.Linear, r: int = 4, alpha: int = 8,
                 block_size: int = 64) -> None:
        super().__init__()
        self.in_features = base.in_features
        self.out_features = base.out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.block_size = block_size

        # Quantize the (already trained) base weight to 4-bit NF4 and freeze it.
        q = quantize_nf4(base.weight.data, block_size)
        # Register the quantized payload as non-trainable buffers.
        self.register_buffer("q_codes", q.codes)
        self.register_buffer("q_absmax", q.absmax)
        self.register_buffer("q_codebook", q.codebook)
        self._q_shape = q.shape
        self._q_numel = q.numel

        if base.bias is not None:
            self.bias = nn.Parameter(base.bias.detach().clone(), requires_grad=False)
        else:
            self.register_parameter("bias", None)

        # Full-precision LoRA factors (the only trainable parameters).
        self.lora_A = nn.Parameter(torch.empty(r, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, r))
        nn.init.normal_(self.lora_A, std=1.0 / r)

    def _dequant_weight(self) -> torch.Tensor:
        q = Quantized(self.q_codes, self.q_absmax, self.q_codebook,
                      self._q_shape, self.block_size, self._q_numel)
        return dequantize_nf4(q)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self._dequant_weight()                       # 4-bit -> fp for matmul
        base = F.linear(x, w, self.bias)
        update = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
        return base + update


def inject_qlora(model: nn.Module, r: int = 4, alpha: int = 8,
                 block_size: int = 64) -> list[str]:
    """Replace every ``nn.Linear`` with a 4-bit :class:`QLoRALinear`."""
    wrapped: list[str] = []
    for name, child in list(model.named_children()):
        if isinstance(child, nn.Linear):
            setattr(model, name,
                    QLoRALinear(child, r=r, alpha=alpha, block_size=block_size))
            wrapped.append(name)
        else:
            wrapped.extend(
                f"{name}.{s}" for s in inject_qlora(child, r, alpha, block_size))
    return wrapped


def qlora_parameters(model: nn.Module):
    for module in model.modules():
        if isinstance(module, QLoRALinear):
            yield module.lora_A
            yield module.lora_B
