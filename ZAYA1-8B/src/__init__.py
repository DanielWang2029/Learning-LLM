"""Minimal from-scratch reproduction of ZAYA1-8B's Compressed Convolutional Attention.

Washbourne et al. (Zyphra), *"ZAYA1-8B Technical Report"* (2026),
arXiv:2605.05365. The documented attention mechanism is **Compressed
Convolutional Attention (CCA)**: a lightweight convolution compresses keys and
values into a shorter latent sequence before attention, cutting attention
compute and the KV-cache while preserving quality.

Public API:
- ``FullAttention``               standard (baseline) multi-head attention.
- ``CompressedConvAttention``     conv-downsample K/V by a factor, then attend.
- ``Classifier``                  a tiny model that uses either attention.
- ``kv_cache_size``               KV-cache accounting for the comparison.
"""

from .cca import (
    FullAttention,
    CompressedConvAttention,
    Classifier,
    kv_cache_size,
)

__all__ = [
    "FullAttention",
    "CompressedConvAttention",
    "Classifier",
    "kv_cache_size",
]
