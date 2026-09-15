"""LLaVA at small scale: a frozen vision encoder connected to a tiny LM via a
learned projection, instruction-tuned on synthetic image+instruction+answer
triples (Liu et al., 2023, "Visual Instruction Tuning")."""

from .model import LlavaTiny, Projector, TinyLM
from .vision import VisionEncoder

__all__ = ["LlavaTiny", "Projector", "TinyLM", "VisionEncoder"]
