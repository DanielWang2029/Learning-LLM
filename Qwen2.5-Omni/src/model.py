"""Audio encoder + per-frame recognition head (Qwen2.5-Omni §2.1).

The paper decouples *perception* (the audio encoder) from *long-sequence
modeling* (the LLM). To measure how well block-wise perception matches full
perception, we attach a tiny per-frame classification head to the encoder: it
labels each ~40 ms frame with the token it belongs to. Frame accuracy under
full vs block-wise attention then quantifies the perception gap directly.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .encoder import BlockwiseAudioEncoder


class AudioTagger(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_mels: int = 128,
        d_model: int = 96,
        num_layers: int = 3,
        num_heads: int = 4,
        d_ff: int = 192,
    ) -> None:
        super().__init__()
        self.encoder = BlockwiseAudioEncoder(
            n_mels=n_mels, d_model=d_model, num_layers=num_layers,
            num_heads=num_heads, d_ff=d_ff,
        )
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, mel: torch.Tensor, block_frames: int | None = None) -> torch.Tensor:
        return self.head(self.encoder(mel, block_frames=block_frames))

    @torch.no_grad()
    def tag_streaming(self, mel: torch.Tensor, block_frames: int):
        emb, counts = self.encoder.streaming_encode(mel, block_frames)
        return self.head(emb), counts
