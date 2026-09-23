"""A minimal, from-scratch Whisper-style encoder-decoder for speech-to-text.

Based on Radford et al., "Robust Speech Recognition via Large-Scale Weak
Supervision" (2022), included here as ``whisper.pdf``.

Public API mirrors the paper (Section 2.2, "Model"):

- ``MelFrontend``        hand-rolled log-Mel front end (80 bins, 25 ms / 10 ms)
- ``AudioEncoder``       conv stem (2 layers, 2nd stride-2) + Transformer encoder
- ``TextDecoder``        Transformer decoder with cross-attention
- ``Whisper``           the full sequence-to-sequence model
- ``sinusoids``          fixed sinusoidal positional encodings
"""

from .features import MelFrontend, log_mel_spectrogram
from .whisper import AudioEncoder, TextDecoder, Whisper, sinusoids

__all__ = [
    "MelFrontend",
    "log_mel_spectrogram",
    "AudioEncoder",
    "TextDecoder",
    "Whisper",
    "sinusoids",
]
