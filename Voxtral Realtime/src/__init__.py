"""A faithful, minimal reproduction of Voxtral Realtime (arXiv:2602.11298).

Voxtral Realtime (Mistral, 2026) is a *natively streaming* ASR model built on
Delayed Streams Modeling (DSM). The three distinctive pieces reproduced here:

- `CausalSlidingWindowEncoder`  a fully causal audio encoder with RMSNorm /
                                SwiGLU / RoPE and sliding-window attention,
                                streamable chunk-by-chunk (vs Whisper's
                                bidirectional encoder).
- `AdaRMSNorm` / `DelayEmbedding`  Adaptive RMSNorm that conditions the decoder
                                FFN on the target delay tau, so a *single* model
                                runs at any delay (latency <-> accuracy).
- `VoxtralRealtime`             the DSM stream-synchronous decoder: one token per
                                80 ms frame, fusing the current audio embedding
                                with the previous text-token embedding, emitting
                                `[P]` (wait) until the delay elapses, then `[W]`
                                and the word.
"""

from .encoder import CausalSlidingWindowEncoder, RMSNorm, rope
from .adarmsnorm import AdaRMSNorm, DelayEmbedding
from .model import VoxtralRealtime
from . import data

__all__ = [
    "CausalSlidingWindowEncoder",
    "RMSNorm",
    "rope",
    "AdaRMSNorm",
    "DelayEmbedding",
    "VoxtralRealtime",
    "data",
]
