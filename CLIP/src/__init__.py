"""From-scratch CLIP: a dual-encoder trained with a symmetric contrastive loss.

Radford et al., "Learning Transferable Visual Models From Natural Language
Supervision" (2021), arXiv:2103.00020.
"""

from .encoders import ImageEncoder, TextEncoder
from .loss import clip_contrastive_loss
from .model import CLIP

__all__ = ["CLIP", "ImageEncoder", "TextEncoder", "clip_contrastive_loss"]
