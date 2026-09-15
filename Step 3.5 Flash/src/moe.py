"""Sparse Mixture-of-Experts: high total capacity, few active parameters.

Step 3.5 Flash pairs a 196B-parameter foundation with only ~11B *active*
parameters per token (paper Abstract, §2.2 "Sparse MoE Backbone"). The trick is
sparse routing: a lightweight router scores many experts and each token is
processed by only the top-k of them, so total parameter capacity grows without
growing per-token compute.

This module implements exactly that mechanism at tiny scale:

    router:   logits = x · W_r                          (one score per expert)
    top-k:    keep the k highest-scoring experts        (sparse activation)
    combine:  y = Σ_{i∈top-k} softmax(logits)_i · Expert_i(x)

Only k of N experts run per token, so the *active* expert parameters are k/N of
the *total* expert parameters. The demo measures that ratio and shows quality
is preserved and that experts specialize by topic.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    """A small position-wise MLP (one MoE expert)."""

    def __init__(self, d_model: int, d_hidden: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.silu(self.fc1(x)))


class SparseMoE(nn.Module):
    """Top-k sparse MoE layer (paper §2.2).

    Args:
        d_model:   feature dimension.
        n_experts: total number of experts (the "total capacity").
        top_k:     experts activated per token (the "active" fraction).
        d_hidden:  hidden width of each expert MLP.
    """

    def __init__(self, d_model: int, n_experts: int, top_k: int, d_hidden: int) -> None:
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.router = nn.Linear(d_model, n_experts)
        self.experts = nn.ModuleList(
            [Expert(d_model, d_hidden) for _ in range(n_experts)]
        )

    def forward(self, x: torch.Tensor, return_routing: bool = False):
        """x: (..., d_model). Returns y with the same shape.

        Sparse compute: only the selected experts are evaluated per token.
        """
        flat = x.reshape(-1, x.shape[-1])          # (T, d_model)
        logits = self.router(flat)                 # (T, n_experts)
        probs = F.softmax(logits, dim=-1)
        top_val, top_idx = probs.topk(self.top_k, dim=-1)   # (T, k)
        top_val = top_val / top_val.sum(dim=-1, keepdim=True)  # renormalize

        y = torch.zeros_like(flat)
        # Evaluate each expert only on the tokens routed to it (true sparsity).
        for e in range(self.n_experts):
            mask = (top_idx == e)                  # (T, k) where expert e chosen
            if not mask.any():
                continue
            token_sel = mask.any(dim=-1)           # (T,) tokens using expert e
            weight = (top_val * mask).sum(dim=-1)[token_sel]  # gate weight per token
            out = self.experts[e](flat[token_sel])
            y[token_sel] += weight.unsqueeze(-1) * out

        y = y.reshape_as(x)
        if return_routing:
            return y, {"top_idx": top_idx, "probs": probs}
        return y


class MoEClassifier(nn.Module):
    """symbol embedding (+ topic embedding) -> SparseMoE -> readout.

    The demo task: apply a *topic-specific* permutation to a shared symbol. The
    symbol embedding is shared across topics, so a single small network must
    cram every topic's rule into one set of weights; an MoE instead lets experts
    specialize per topic while keeping per-token compute small.
    """

    def __init__(
        self,
        n_symbols: int,
        n_topics: int,
        d_model: int = 32,
        n_experts: int = 8,
        top_k: int = 1,
        d_hidden: int = 48,
        freeze_embed: bool = True,
    ) -> None:
        super().__init__()
        self.sym_embed = nn.Embedding(n_symbols, d_model)
        self.topic_embed = nn.Embedding(n_topics, d_model)
        if freeze_embed:
            # Fixed random inputs: the network must COMPUTE the mapping rather
            # than trivially memorize it by arranging learnable embeddings.
            self.sym_embed.weight.requires_grad_(False)
            self.topic_embed.weight.requires_grad_(False)
        self.norm = nn.LayerNorm(d_model)
        self.moe = SparseMoE(d_model, n_experts, top_k, d_hidden)
        self.readout = nn.Linear(d_model, n_symbols)

    def forward(self, symbols, topics, return_routing: bool = False):
        x = self.sym_embed(symbols) + self.topic_embed(topics)
        h = self.norm(x)
        if return_routing:
            y, routing = self.moe(h, return_routing=True)
            return self.readout(x + y), routing
        return self.readout(x + self.moe(h))


class DenseClassifier(nn.Module):
    """Same shell as MoEClassifier but a single dense MLP (all params active)."""

    def __init__(
        self,
        n_symbols: int,
        n_topics: int,
        d_model: int = 32,
        d_hidden: int = 48,
        freeze_embed: bool = True,
    ) -> None:
        super().__init__()
        self.sym_embed = nn.Embedding(n_symbols, d_model)
        self.topic_embed = nn.Embedding(n_topics, d_model)
        if freeze_embed:
            self.sym_embed.weight.requires_grad_(False)
            self.topic_embed.weight.requires_grad_(False)
        self.norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_model)
        self.readout = nn.Linear(d_model, n_symbols)

    def forward(self, symbols, topics):
        x = self.sym_embed(symbols) + self.topic_embed(topics)
        h = self.norm(x)
        y = self.fc2(F.silu(self.fc1(h)))
        return self.readout(x + y)


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def active_params(model: MoEClassifier) -> int:
    """Parameters actually used to process a single token under top-k routing."""
    shared = (
        count_params(model.sym_embed)
        + count_params(model.topic_embed)
        + count_params(model.norm)
        + count_params(model.moe.router)
        + count_params(model.readout)
    )
    one_expert = count_params(model.moe.experts[0])
    return shared + model.moe.top_k * one_expert
