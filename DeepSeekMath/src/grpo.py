"""Group Relative Policy Optimization (GRPO) — introduced in DeepSeekMath (§4.1).

GRPO is a critic-free policy-gradient method. PPO trains a separate value network
to estimate a baseline; GRPO removes it entirely and instead uses the *group* of
sampled outputs as its own baseline. For each prompt it samples a group of G
completions, scores them, and standardizes the rewards within the group:

    A_i = (R_i - mean(R_1..R_G)) / (std(R_1..R_G) + eps)          (Eq. 20)

Above-average completions get a positive advantage, below-average ones negative —
no value network required. The objective is the PPO-style clipped surrogate with
a KL penalty pulling the policy toward a frozen reference model (Eq. 21):

    J = 1/G Σ_i 1/|o_i| Σ_t [ min( r_{i,t} A_i, clip(r_{i,t}, 1±ε) A_i )
                              - β · D_KL( π_θ || π_ref ) ]

    r_{i,t} = π_θ(o_{i,t}) / π_θ_old(o_{i,t})        (importance ratio)

The KL term uses the low-variance unbiased estimator from Schulman's blog
(the "k3" estimator), which DeepSeekMath adopts:

    D_KL ≈ π_ref/π_θ − log(π_ref/π_θ) − 1  ≥ 0

We implement all of this from scratch below.
"""

from __future__ import annotations

import torch


def group_advantages(rewards: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    """Standardize rewards within each group (DeepSeekMath Eq. 20).

    ``rewards``: (num_prompts, group_size). Returns advantages of the same shape.
    This is the entire GRPO baseline — the group mean replaces PPO's value net.
    """
    mean = rewards.mean(dim=1, keepdim=True)
    std = rewards.std(dim=1, keepdim=True)
    return (rewards - mean) / (std + eps)


def kl_to_reference(logp_policy: torch.Tensor, logp_ref: torch.Tensor) -> torch.Tensor:
    """Per-token unbiased KL estimate D_KL(π_θ || π_ref) >= 0 (the k3 estimator).

    Given per-token log-probs of the *sampled* tokens under the policy and the
    reference, ``log_ratio = logp_ref - logp_policy`` and
    ``kl = exp(log_ratio) - log_ratio - 1``.
    """
    log_ratio = logp_ref - logp_policy
    return torch.exp(log_ratio) - log_ratio - 1.0


def grpo_loss(
    logp_new: torch.Tensor,   # (B, L) current-policy log-probs of sampled tokens
    logp_old: torch.Tensor,   # (B, L) behavior-policy log-probs (detached)
    logp_ref: torch.Tensor,   # (B, L) reference-policy log-probs (detached)
    advantages: torch.Tensor, # (B,)   one group-relative advantage per sequence
    mask: torch.Tensor,       # (B, L) 1 for real generated tokens, 0 for padding
    beta: float = 0.04,
    clip_eps: float = 0.2,
):
    """The full GRPO loss (Eq. 21). Returns (loss, diagnostics)."""
    adv = advantages[:, None]                       # broadcast over tokens
    ratio = torch.exp(logp_new - logp_old)          # importance ratio r_{i,t}
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv
    surrogate = torch.minimum(unclipped, clipped)   # PPO clipped objective

    kl = kl_to_reference(logp_new, logp_ref)        # >= 0
    per_token = surrogate - beta * kl

    # Average over valid tokens per sequence, then over sequences (Eq. 21).
    tok = mask.sum(dim=1).clamp(min=1)
    per_seq = (per_token * mask).sum(dim=1) / tok
    loss = -per_seq.mean()

    diag = {
        "kl": float(((kl * mask).sum() / mask.sum()).item()),
        "ratio": float(((ratio * mask).sum() / mask.sum()).item()),
        "clip_frac": float(((unclipped != clipped).float() * mask).sum().item()
                           / float(mask.sum().item())),
    }
    return loss, diag
