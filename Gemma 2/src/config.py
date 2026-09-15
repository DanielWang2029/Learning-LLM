"""Configuration for the tiny Gemma 2 model (paper: Gemma 2, Google 2024).

Every field maps to a documented Gemma 2 design choice; the defaults here are
shrunk to laptop-CPU scale while keeping the *structure* identical to the
paper (arXiv 2408.00118).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Gemma2Config:
    vocab_size: int = 32
    d_model: int = 96
    n_layers: int = 4              # alternates local / global (paper §2, Table 1)
    n_heads: int = 6               # query heads
    n_kv_heads: int = 2            # GQA: fewer key/value heads (paper §2)
    head_dim: int = 16
    d_ff: int = 256
    max_seq_len: int = 64

    # Sliding-window size for *local* attention layers (paper uses 4096; tiny here).
    sliding_window: int = 6

    # Logit soft-capping caps (paper §2 / Table 1: attn=50.0, final=30.0).
    attn_logit_softcap: float = 50.0
    final_logit_softcap: float = 30.0

    # Toggle to compare "with vs without" soft-capping in the demo.
    use_soft_cap: bool = True

    dropout: float = 0.0

    @property
    def n_rep(self) -> int:
        """How many query heads share each key/value head (GQA grouping)."""
        return self.n_heads // self.n_kv_heads
