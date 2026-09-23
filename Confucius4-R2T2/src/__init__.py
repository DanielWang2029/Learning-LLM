"""Confucius4-R2T2 reproduction: the Longest Stable Prefix (LSP) append-only
streaming decoding paradigm, at small scale, on CPU.

- `audio`      : numpy speech-like synthesizer + from-scratch STFT / log-mel.
- `recognizer` : a transparent, training-free matched-filter phone recognizer
                 that produces per-chunk streaming hypotheses.
- `lsp`        : the LSP append-only decoder and a naive revising baseline.
"""

from .audio import (
    FRAMES_PER_PHONE,
    HOP,
    NUM_PHONES,
    PHONE_NAMES,
    SAMPLE_RATE,
    log_mel,
    synth_utterance,
)
from .lsp import LSPDecoder, NaiveDecoder, longest_common_prefix
from .recognizer import MatchedFilterRecognizer, tokens_to_text

__all__ = [
    "FRAMES_PER_PHONE",
    "HOP",
    "NUM_PHONES",
    "PHONE_NAMES",
    "SAMPLE_RATE",
    "log_mel",
    "synth_utterance",
    "LSPDecoder",
    "NaiveDecoder",
    "longest_common_prefix",
    "MatchedFilterRecognizer",
    "tokens_to_text",
]
