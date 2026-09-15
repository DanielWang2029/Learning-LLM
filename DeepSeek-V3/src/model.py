"""A tiny DeepSeek-V3 model: MLA + DeepSeekMoE + Multi-Token Prediction.

Based on the "DeepSeek-V3 Technical Report" (2024, arXiv:2412.19437), §2. This
stitches the three signature components together into a minimal decoder-only LM:

* attention is **Multi-Head Latent Attention** (``mla.py``) — swappable for plain
  MHA so the demo can compare quality and KV-cache size,
* the feed-forward is a **DeepSeekMoE** layer (``moe.py``) — fine-grained routed
  experts plus always-on shared experts,
* a **Multi-Token Prediction** head (``mtp.py``) predicts the token two steps
  ahead in addition to the usual next-token head.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mla import MLAConfig, MultiHeadLatentAttention, StandardMHA
from .moe import DeepSeekMoE
from .mtp import MTPModule


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


@dataclass
class DeepSeekV3Config:
    vocab_size: int = 32
    dim: int = 64
    n_layers: int = 3
    attn: str = "mla"                 # "mla" | "mha"
    mla: MLAConfig = field(default_factory=MLAConfig)
    expert_hidden: int = 48
    n_routed: int = 8
    n_shared: int = 1
    top_k: int = 2
    use_mtp: bool = True


class Block(nn.Module):
    def __init__(self, cfg: DeepSeekV3Config) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.dim)
        self.attn = (MultiHeadLatentAttention(cfg.mla) if cfg.attn == "mla"
                     else StandardMHA(cfg.mla))
        self.moe_norm = RMSNorm(cfg.dim)
        self.moe = DeepSeekMoE(cfg.dim, cfg.expert_hidden, cfg.n_routed,
                               cfg.n_shared, cfg.top_k)

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        moe_out, info = self.moe(self.moe_norm(x))
        return x + moe_out, info


class DeepSeekV3(nn.Module):
    def __init__(self, cfg: DeepSeekV3Config) -> None:
        super().__init__()
        # keep the MLA sub-config's dim in sync with the model dim
        cfg.mla.dim = cfg.dim
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm_f = RMSNorm(cfg.dim)
        self.head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.mtp = MTPModule(cfg.dim) if cfg.use_mtp else None
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, idx, next_idx=None):
        """Returns (main_logits, mtp_logits, moe_infos).

        ``main_logits`` predict token t+1 at each position. If MTP is on and
        ``next_idx`` (the true next tokens, teacher forced) is given, ``mtp_logits``
        predict token t+2.
        """
        x = self.tok(idx)
        infos = []
        for blk in self.blocks:
            x, info = blk(x)
            infos.append(info)
        trunk = self.norm_f(x)
        main_logits = self.head(trunk)

        mtp_logits = None
        if self.mtp is not None and next_idx is not None:
            nxt_emb = self.tok(next_idx)                      # embedding of the known t+1 token
            mtp_hidden = self.mtp(x, nxt_emb)                 # hidden state for t+2
            mtp_logits = self.head(self.norm_f(mtp_hidden))   # shares the main head
        return main_logits, mtp_logits, infos
