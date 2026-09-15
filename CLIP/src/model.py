"""The full CLIP model: two encoders + a shared space + a learned temperature.

Paper reference: Section 2 ("Approach"), Figure 3. The model L2-normalizes both
embeddings, then scores every image against every text with a temperature-scaled
cosine similarity. `logit_scale` is learned in log-space and clamped, exactly as
in the paper/official code, to keep the temperature numerically stable.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import ImageEncoder, TextEncoder


class CLIP(nn.Module):
    def __init__(self, embed_dim: int = 64) -> None:
        super().__init__()
        self.image_encoder = ImageEncoder(embed_dim)
        self.text_encoder = TextEncoder(embed_dim)
        # Learned temperature, stored as log(1/T). Init to CLIP's 0.07 temperature.
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.image_encoder(images), dim=-1)

    def encode_text(self, tokens: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.text_encoder(tokens), dim=-1)

    def forward(self, images: torch.Tensor, tokens: torch.Tensor):
        """Return (logits_per_image, image_embeds, text_embeds).

        `logits_per_image[i, j]` = temperature * cos(image_i, text_j).
        """
        image_embeds = self.encode_image(images)
        text_embeds = self.encode_text(tokens)
        scale = self.logit_scale.clamp(max=math.log(100)).exp()
        logits_per_image = scale * image_embeds @ text_embeds.t()
        return logits_per_image, image_embeds, text_embeds
