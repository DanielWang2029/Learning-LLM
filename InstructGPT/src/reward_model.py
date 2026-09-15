"""Stage 2: the reward model (RM) and its pairwise preference loss.

The RM takes a (prompt, response) pair and returns a scalar "how much a human
would approve" score. InstructGPT trains it on comparisons: given a prompt and
two responses, the human labels which is better. The loss is the Bradley-Terry
/ logistic loss on the score difference (paper §2.3):

    loss = -log σ( r(prompt, chosen) - r(prompt, rejected) )

Here the RM is a tiny Transformer-free encoder: embed tokens, mean-pool, MLP to
a scalar. That is plenty to recover the "sortedness" rule from comparisons.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RewardModel(nn.Module):
    """A small scalar reward model over a full [BOS,prompt,SEP,response,EOS] seq."""

    def __init__(self, vocab_size: int, max_len: int, d_model: int = 48) -> None:
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.body = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model), nn.GELU()
        )
        self.value = nn.Linear(d_model, 1)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """Return a scalar reward per sequence (b,)."""
        b, t = seq.shape
        pos = torch.arange(t, device=seq.device).unsqueeze(0)
        x = self.tok_emb(seq) + self.pos_emb(pos)
        x = self.body(x)
        pooled = x.mean(dim=1)  # order-aware via positional embeddings
        return self.value(pooled).squeeze(-1)


def bradley_terry_loss(
    chosen_reward: torch.Tensor, rejected_reward: torch.Tensor
) -> torch.Tensor:
    """-log σ(r_chosen - r_rejected), averaged over the batch (paper §2.3)."""
    return -F.logsigmoid(chosen_reward - rejected_reward).mean()
