"""4-bit blockwise quantization (paper Section 3, "4-bit NormalFloat").

QLoRA stores the frozen base weights in 4 bits.  Two pieces make this work:

  * **Blockwise quantization** — the tensor is split into contiguous blocks of
    ``block_size`` values; each block is normalized by its own absolute maximum
    (``absmax``) before quantization.  This keeps a single outlier from ruining
    the whole tensor's resolution and is what the paper calls "block-wise
    k-bit quantization" (Section 3).

  * **NF4 codebook** — the 16 quantization levels are the quantiles of a
    standard normal distribution, so they are *information-theoretically optimal
    for normally-distributed weights* (NormalFloat, Section 3).  Values are
    mapped to the nearest of these 16 levels; only the 4-bit index is stored.

Everything here is plain numpy/torch — no bitsandbytes, no CUDA.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

# NF4 code values: the 16 levels of the NormalFloat-4 data type (paper §3),
# the same quantiles used by the reference QLoRA implementation. Symmetric-ish
# around 0, with a dedicated exact zero.
NF4_CODEBOOK = torch.tensor([
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634,
    0.33791524171829224, 0.44070982933044434, 0.5626170039176941,
    0.7229568362236023, 1.0,
], dtype=torch.float32)


@dataclass
class Quantized:
    """A 4-bit blockwise-quantized tensor plus everything to reconstruct it."""

    codes: torch.Tensor      # uint8 indices into the codebook (0..15), flat
    absmax: torch.Tensor     # fp32 scale per block
    codebook: torch.Tensor   # the 16 level values
    shape: torch.Size        # original tensor shape
    block_size: int
    numel: int               # original number of elements (pre-padding)


def quantize_nf4(w: torch.Tensor, block_size: int = 64) -> Quantized:
    """Quantize ``w`` to 4 bits with per-block absmax scaling (NF4)."""
    flat = w.detach().reshape(-1).float()
    n = flat.numel()
    pad = (-n) % block_size
    if pad:
        flat = torch.cat([flat, torch.zeros(pad, dtype=flat.dtype)])
    blocks = flat.reshape(-1, block_size)

    absmax = blocks.abs().amax(dim=1, keepdim=True)         # (num_blocks, 1)
    absmax = torch.where(absmax == 0, torch.ones_like(absmax), absmax)
    normed = blocks / absmax                                # into [-1, 1]

    # Nearest codebook level for every value (vectorized argmin over 16 levels).
    dist = (normed.unsqueeze(-1) - NF4_CODEBOOK.view(1, 1, -1)).abs()
    codes = dist.argmin(dim=-1).to(torch.uint8)            # (num_blocks, block)

    return Quantized(
        codes=codes.reshape(-1),
        absmax=absmax.reshape(-1),
        codebook=NF4_CODEBOOK.clone(),
        shape=w.shape,
        block_size=block_size,
        numel=n,
    )


def dequantize_nf4(q: Quantized) -> torch.Tensor:
    """Reconstruct the full-precision tensor from its 4-bit form."""
    levels = q.codebook[q.codes.long()]                    # look up level values
    blocks = levels.reshape(-1, q.block_size) * q.absmax.unsqueeze(1)
    flat = blocks.reshape(-1)[: q.numel]
    return flat.reshape(q.shape)


def quantization_error(w: torch.Tensor, block_size: int = 64) -> float:
    """Relative reconstruction error ‖W − dequant(quant(W))‖ / ‖W‖."""
    q = quantize_nf4(w, block_size)
    recon = dequantize_nf4(q)
    return (w - recon).norm().item() / (w.norm().item() + 1e-12)


def bytes_4bit(numel: int, block_size: int = 64) -> float:
    """Storage for a 4-bit NF4 tensor: 0.5 byte/value + one fp32 absmax/block."""
    return numel * 0.5 + (numel / block_size) * 4.0
