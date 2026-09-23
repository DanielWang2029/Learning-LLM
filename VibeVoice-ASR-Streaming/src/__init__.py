"""VibeVoice-ASR-Streaming reproduction: LLM-based streaming speaker-attributed
ASR, at small scale, on CPU.

- `audio`     : numpy two-speaker synthesizer + from-scratch STFT / log-mel.
- `data_gen`  : conversation generation + interleaved [speech, text] sequences.
- `model`     : a tiny causal Transformer over mixed audio + text tokens.
- `streaming` : chunk-by-chunk streaming decode + speaker-attributed metrics.
"""

from .audio import FRAME_MS, FRAMES_PER_PHONE, NUM_PHONES, NUM_SPEAKERS, N_MELS, features, synth_conversation
from .data_gen import VOCAB, build_plan, gen_conversation, make_dataset, plan_to_tensors, token_str
from .model import StreamingSAASR
from .streaming import score, stream_decode

__all__ = [
    "FRAME_MS", "FRAMES_PER_PHONE", "NUM_PHONES", "NUM_SPEAKERS", "N_MELS",
    "features", "synth_conversation", "VOCAB", "build_plan", "gen_conversation",
    "make_dataset", "plan_to_tensors", "token_str", "StreamingSAASR", "score",
    "stream_decode",
]
