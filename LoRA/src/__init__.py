"""LoRA: Low-Rank Adaptation of Large Language Models (Hu et al., 2021).

A minimal, CPU-friendly, from-scratch reproduction of the LoRA adapter.
"""

from .lora import (
    LoRALinear,
    inject_lora,
    lora_parameters,
    merge_all,
    unmerge_all,
)
from .model import MLPClassifier, accuracy, count_total, count_trainable

__all__ = [
    "LoRALinear",
    "inject_lora",
    "lora_parameters",
    "merge_all",
    "unmerge_all",
    "MLPClassifier",
    "accuracy",
    "count_total",
    "count_trainable",
]
