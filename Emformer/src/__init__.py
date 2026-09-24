"""A minimal, from-scratch Emformer streaming attention block.

Based on Shi et al., "Emformer: Efficient Memory Transformer Based Acoustic
Model for Low Latency Streaming Speech Recognition" (2021), included here as
``emformer.pdf``.

Public API mirrors the paper (Section 2):

- ``EmformerBlock``  a single Emformer layer supporting two equivalent modes:
    * ``forward_parallel`` — the training-time pass over the whole utterance with
      a block-diagonal + memory mask (Section 2.2.2), and
    * ``forward_stream``   — inference block by block with cached key/values and an
      augmented memory bank (Section 2.1), reusing past computation.
- ``full_attention`` — a plain bidirectional attention baseline for comparison.
"""

from .emformer import EmformerBlock, full_attention

__all__ = ["EmformerBlock", "full_attention"]
