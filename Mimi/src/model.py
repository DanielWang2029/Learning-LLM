"""The full tiny Mimi codec: encoder -> split-RVQ -> decoder (+ distillation).

This ties the SEANet-style causal autoencoder (``codec.py``) to the split
residual vector quantizer (``quantization.py``) and adds the semantic
distillation head described in the Moshi tech report (§3.3.2):

    "Mimi uses distillation to transfer non-causal, high-level semantic
    information [WavLM] into the tokens produced by [the] causal [encoder]."

On CPU we have no WavLM, so the demo distills the semantic VQ toward a
*transparent proxy* target (a fixed random embedding of the ground-truth class);
see ``demo/run_demo.py`` and the README. The mechanism — a cosine distillation
loss that makes the first codebook semantic — is identical to the paper's.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .codec import MimiEncoder, MimiDecoder
from .quantization import SplitRVQ


class MimiCodec(nn.Module):
    def __init__(
        self,
        strides: List[int] = (2, 4, 4, 6),
        channels: int = 32,
        latent_dim: int = 64,
        rvq_dim: int = 32,
        num_acoustic: int = 7,
        codebook_size: int = 256,
        semantic_target_dim: int = 16,
    ) -> None:
        super().__init__()
        self.encoder = MimiEncoder(list(strides), channels, latent_dim)
        self.decoder = MimiDecoder(list(strides), channels, latent_dim)
        self.quantizer = SplitRVQ(latent_dim, rvq_dim, num_acoustic, codebook_size)
        # Distillation head: project semantic code vectors into target space.
        self.semantic_proj = nn.Conv1d(rvq_dim, semantic_target_dim, 1)
        self.total_stride = self.encoder.total_stride
        self.num_acoustic = num_acoustic
        self.codebook_size = codebook_size

    def encode(self, wav: torch.Tensor) -> torch.Tensor:
        """Waveform (B, 1, L) -> latent (B, D, S)."""
        return self.encoder(wav)

    def forward(self, wav: torch.Tensor, n_acoustic: Optional[int] = None) -> dict:
        latent = self.encoder(wav)
        q = self.quantizer(latent, n_acoustic=n_acoustic)
        recon = self.decoder(q["z_q"])
        # Semantic projection for the cosine distillation loss.
        q["semantic_pred"] = self.semantic_proj(q["semantic_embed"])
        q["recon"] = recon
        q["latent"] = latent
        return q

    @staticmethod
    def distillation_loss(
        semantic_pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Cosine-distance distillation (paper uses cosine to WavLM embeds).

        ``semantic_pred`` (B, C, T), ``target`` (B, C, T) -> scalar in [0, 2].
        """
        pred = F.normalize(semantic_pred, dim=1)
        tgt = F.normalize(target, dim=1)
        return (1.0 - (pred * tgt).sum(1)).mean()

    # ---- streaming helpers (used by the streaming==offline check) ----
    def reset_stream(self, batch: int, device: torch.device) -> None:
        self.encoder.reset_stream(batch, device)
        self.decoder.reset_stream(batch, device)

    def encode_stream_step(self, wav_chunk: torch.Tensor) -> torch.Tensor:
        return self.encoder.step(wav_chunk)

    def decode_stream_step(self, latent_chunk: torch.Tensor) -> torch.Tensor:
        return self.decoder.step(latent_chunk)
