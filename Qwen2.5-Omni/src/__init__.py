"""Qwen2.5-Omni reproduction: block-wise streaming audio encoder."""

from .encoder import (
    FRAME_SECONDS,
    BlockwiseAudioEncoder,
    block_diagonal_mask,
    frames_per_seconds,
)
from .features import log_mel_spectrogram
from .model import AudioTagger

__all__ = [
    "BlockwiseAudioEncoder",
    "AudioTagger",
    "block_diagonal_mask",
    "frames_per_seconds",
    "FRAME_SECONDS",
    "log_mel_spectrogram",
]
