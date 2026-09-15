"""DoRA: Weight-Decomposed Low-Rank Adaptation (NVIDIA 2024, arXiv 2402.09353).

Exposes the base MLP, the LoRA and DoRA adapters (for a head-to-head), and the
toy task generators.
"""

from .model import MLPClassifier, accuracy, count_trainable, count_total
from .adapters import (
    LoRALinear,
    DoRALinear,
    inject_lora,
    inject_dora,
    adapter_parameters,
)
from .data import make_task, IN_DIM, HIDDEN, NUM_CLASSES

__all__ = [
    "MLPClassifier",
    "accuracy",
    "count_trainable",
    "count_total",
    "LoRALinear",
    "DoRALinear",
    "inject_lora",
    "inject_dora",
    "adapter_parameters",
    "make_task",
    "IN_DIM",
    "HIDDEN",
    "NUM_CLASSES",
]
