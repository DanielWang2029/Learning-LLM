"""A tiny autoregressive Transformer policy for GRPO (DeepSeekMath §4.1).

GRPO optimizes a *policy* language model. Here that policy is a small decoder-only
Transformer that, given the prompt tokens ``a + b =``, generates answer tokens
one at a time. Nothing about the architecture is special — the point of this
folder is the GRPO *algorithm*, so the model is kept minimal and CPU-fast.

The class exposes exactly what the GRPO update needs:

* :meth:`sample_group`  — sample G completions per prompt (the "group"),
* :meth:`sequence_logprobs` — the summed log-prob of given completions under the
  current parameters (recomputed each step so we can form the policy ratio),
* per-token log-probs for the KL-to-reference penalty.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PolicyConfig:
    vocab_size: int = 15
    dim: int = 48
    n_heads: int = 4
    n_layers: int = 2
    max_len: int = 16


class _Block(nn.Module):
    def __init__(self, dim: int, n_heads: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        h = self.norm1(x)
        q, k, v = self.qkv(h).split(d, dim=2)
        q = q.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.triu(torch.ones(t, t, device=x.device, dtype=torch.bool), 1)
        att = att.masked_fill(mask, float("-inf")).softmax(-1)
        y = (att @ v).transpose(1, 2).contiguous().view(b, t, d)
        x = x + self.proj(y)
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerPolicy(nn.Module):
    def __init__(self, cfg: PolicyConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.pos = nn.Embedding(cfg.max_len, cfg.dim)
        self.blocks = nn.ModuleList([_Block(cfg.dim, cfg.n_heads) for _ in range(cfg.n_layers)])
        self.norm_f = nn.LayerNorm(cfg.dim)
        self.head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        b, t = tokens.shape
        pos = torch.arange(t, device=tokens.device)
        x = self.tok(tokens) + self.pos(pos)[None]
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.norm_f(x))

    @torch.no_grad()
    def sample_group(self, prompt_ids: List[int], group_size: int, gen_len: int,
                     eos_id: int, generator: torch.Generator, temperature: float = 1.0):
        """Sample ``group_size`` completions of the same prompt.

        Returns the full token tensor (prompt + generated) of shape
        (group_size, prompt_len + gen_len) and the length of the prompt.
        """
        device = self.tok.weight.device
        prompt = torch.tensor(prompt_ids, device=device)
        seq = prompt[None].repeat(group_size, 1)
        plen = seq.size(1)
        finished = torch.zeros(group_size, dtype=torch.bool, device=device)
        for _ in range(gen_len):
            logits = self(seq)[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1, generator=generator).squeeze(1)
            nxt = torch.where(finished, torch.full_like(nxt, eos_id), nxt)
            seq = torch.cat([seq, nxt[:, None]], dim=1)
            finished = finished | (nxt == eos_id)
        return seq, plen

    @torch.no_grad()
    def sample_batch(self, prompts: torch.Tensor, gen_len: int, eos_id: int,
                     generator: torch.Generator, temperature: float = 1.0):
        """Sample one completion for every prompt row in ``prompts`` (B, plen).

        Returns the full token tensor (B, plen + gen_len). Once a row emits EOS,
        it is padded with EOS so the answer region is well-defined.
        """
        seq = prompts
        plen = seq.size(1)
        finished = torch.zeros(seq.size(0), dtype=torch.bool, device=seq.device)
        for _ in range(gen_len):
            logits = self(seq)[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, 1, generator=generator).squeeze(1)
            nxt = torch.where(finished, torch.full_like(nxt, eos_id), nxt)
            seq = torch.cat([seq, nxt[:, None]], dim=1)
            finished = finished | (nxt == eos_id)
        return seq, plen

    @torch.no_grad()
    def greedy(self, prompt_ids: List[int], gen_len: int, eos_id: int):
        device = self.tok.weight.device
        seq = torch.tensor(prompt_ids, device=device)[None]
        plen = seq.size(1)
        for _ in range(gen_len):
            nxt = self(seq)[:, -1, :].argmax(-1, keepdim=True)
            seq = torch.cat([seq, nxt], dim=1)
            if int(nxt) == eos_id:
                break
        return seq[0, plen:].tolist()

    def token_logprobs(self, seq: torch.Tensor, plen: int) -> torch.Tensor:
        """Log-prob of each GENERATED token under current params.

        seq: (B, plen + gen_len). Returns (B, gen_len) log-probs for the tokens
        that were actually produced after the prompt.
        """
        logits = self(seq[:, :-1])          # predict token t from tokens < t
        logp = F.log_softmax(logits, dim=-1)
        targets = seq[:, 1:]                 # the realized next tokens
        gathered = logp.gather(-1, targets[..., None]).squeeze(-1)
        return gathered[:, plen - 1:]        # keep only the generated region
