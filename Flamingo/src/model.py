"""The full Flamingo visual language model (paper Section 2, Figure 3).

Pipeline:

    image --VisionEncoder--> features --PerceiverResampler--> R visual tokens
                                                                     |
    text tokens --> [ GatedXAttn -> frozen LM block ] x N --> LM head -> logits

The frozen pretrained LM provides language ability; the interleaved gated
cross-attention layers (initialized closed) inject vision. Only the vision
encoder, resampler, and gated layers are trained — the LM stays frozen — yet the
model learns to answer questions about the image.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .gated_xattn import GatedCrossAttentionBlock
from .lm import DecoderLM
from .resampler import PerceiverResampler
from .vision import VisionEncoder


class Flamingo(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        num_heads: int = 4,
        d_ff: int = 128,
        num_layers: int = 2,
        num_latents: int = 8,
        max_len: int = 16,
    ) -> None:
        super().__init__()
        self.vision = VisionEncoder(d_vis=d_model)
        self.resampler = PerceiverResampler(
            d_model=d_model, num_latents=num_latents, num_heads=num_heads, d_ff=d_ff
        )
        self.lm = DecoderLM(
            vocab_size, d_model, num_heads, d_ff, num_layers, max_len=max_len
        )
        # One gated cross-attention block before each frozen LM block.
        self.gated_layers = nn.ModuleList(
            [GatedCrossAttentionBlock(d_model, num_heads, d_ff) for _ in range(num_layers)]
        )

    def freeze_lm(self) -> None:
        """Freeze the language model, exactly as in the paper (Section 2.2)."""
        for p in self.lm.parameters():
            p.requires_grad = False

    def visual_tokens(self, images: torch.Tensor) -> torch.Tensor:
        return self.resampler(self.vision(images))

    def forward(self, tokens: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        """Return next-token logits (B, T, vocab) conditioned on the image."""
        vis = self.visual_tokens(images)               # (B, R, d)
        x = self.lm.embed(tokens)
        mask = self.lm.causal_mask(tokens.size(1), tokens.device)
        for gated, block in zip(self.gated_layers, self.lm.blocks):
            x = gated(x, vis)                           # inject vision (gated)
            x = block(x, mask)                          # frozen LM block
        return self.lm.head(self.lm.norm_f(x))

    def gate_values(self):
        """Current tanh-gate values for every gated layer (for logging/viz)."""
        return [g.gate_values() for g in self.gated_layers]
