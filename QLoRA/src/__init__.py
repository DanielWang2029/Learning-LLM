"""QLoRA: Efficient Finetuning of Quantized LLMs (Dettmers et al., 2023).

A minimal, CPU-friendly, from-scratch reproduction: 4-bit NF4 blockwise
quantization of the frozen base weights + full-precision LoRA adapters on top.
"""

from .quant import (
    NF4_CODEBOOK,
    Quantized,
    bytes_4bit,
    dequantize_nf4,
    quantization_error,
    quantize_nf4,
)
from .qlora import QLoRALinear, inject_qlora, qlora_parameters
from .model import MLPClassifier, accuracy, count_trainable, linear_weight_numel

__all__ = [
    "NF4_CODEBOOK", "Quantized", "quantize_nf4", "dequantize_nf4",
    "quantization_error", "bytes_4bit",
    "QLoRALinear", "inject_qlora", "qlora_parameters",
    "MLPClassifier", "accuracy", "count_trainable", "linear_weight_numel",
]
