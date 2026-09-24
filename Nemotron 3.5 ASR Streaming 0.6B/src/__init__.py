"""Nemotron 3.5 ASR reproduction: a cache-aware FastConformer-RNNT streaming
model with configurable chunk sizes and language-ID prompt conditioning, at small
scale, on CPU.

- `audio`     : numpy synthesizer + from-scratch STFT / log-mel.
- `data_gen`  : two synthetic "languages" that share acoustics (prompt matters).
- `model`     : FastConformer encoder + language fusion + RNN-T + transducer loss.
- `streaming` : cache-aware streaming vs full-context verification + WER.
"""

from .audio import ENC_FRAME_MS, ENC_FRAMES_PER_PHONE, N_MELS, NUM_PHONES, SUBSAMPLE
from .data_gen import LANG_NAMES, NUM_LANGS, VOCAB, make_dataset, token_str
from .model import FastConformerEncoder, LangFusion, RNNT, greedy_decode, transducer_loss
from .streaming import latency_decode, streaming_encode, wer

__all__ = [
    "ENC_FRAME_MS", "ENC_FRAMES_PER_PHONE", "N_MELS", "NUM_PHONES", "SUBSAMPLE",
    "LANG_NAMES", "NUM_LANGS", "VOCAB", "make_dataset", "token_str",
    "FastConformerEncoder", "LangFusion", "RNNT", "greedy_decode",
    "transducer_loss", "streaming_encode", "latency_decode", "wer",
]
