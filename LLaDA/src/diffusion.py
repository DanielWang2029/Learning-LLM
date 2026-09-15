"""The masked-diffusion forward and reverse processes (LLaDA, paper §2).

LLaDA replaces autoregressive next-token prediction with a *masked diffusion*:

* **Forward process** (data -> mask): pick a masking ratio ``t ~ U(0, 1]`` and
  independently replace each token with ``[MASK]`` with probability ``t``. At
  ``t = 1`` the whole sequence is masked; at ``t -> 0`` it is untouched.

* **Training objective** (paper Eq. 3): a single bidirectional network predicts
  the original token at every masked position. The per-example loss is the
  cross-entropy on masked positions, scaled by ``1/t`` — a Monte-Carlo estimate
  of an upper bound on the negative log-likelihood.

* **Reverse process** (mask -> data): start from an all-``[MASK]`` sequence and
  iteratively denoise over ``T`` steps. At each step the model predicts every
  masked token; the most confident predictions are *committed* and the rest are
  re-masked ("low-confidence remasking", paper §2.4). This is **non-autoregressive**:
  tokens are filled in confidence order, not left to right.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

# The reserved mask id is always the last token id (vocab_size - 1).
MASK = -1  # sentinel; the concrete id is supplied by the caller as mask_id.


def forward_mask(
    tokens: torch.Tensor,
    mask_id: int,
    t: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
):
    """Apply the forward masking process at ratio ``t`` (per example).

    Returns ``(masked_tokens, mask, t)`` where ``mask`` is a boolean tensor that
    is ``True`` at positions that were replaced by ``[MASK]``.
    """
    b, seq = tokens.shape
    if t is None:
        # Sample a masking ratio per sequence in (0, 1].
        t = torch.rand(b, 1, generator=generator, device=tokens.device).clamp_min(0.05)
    probs = t.expand(b, seq)
    noise = torch.rand(b, seq, generator=generator, device=tokens.device)
    mask = noise < probs
    # Guarantee at least one masked position so the loss is always defined.
    no_mask = ~mask.any(dim=1)
    if no_mask.any():
        forced = torch.randint(0, seq, (int(no_mask.sum()),), device=tokens.device)
        mask[no_mask, forced] = True

    masked = tokens.masked_fill(mask, mask_id)
    return masked, mask, t


def diffusion_loss(
    model,
    tokens: torch.Tensor,
    mask_id: int,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Cross-entropy on masked positions, ``1/t`` weighted (paper Eq. 3)."""
    masked, mask, t = forward_mask(tokens, mask_id, generator=generator)
    logits = model(masked)  # (b, seq, vocab)

    b, seq, vocab = logits.shape
    ce = F.cross_entropy(
        logits.reshape(-1, vocab), tokens.reshape(-1), reduction="none"
    ).view(b, seq)
    # Paper Eq. 3: (1/t) * sum over masked positions, averaged over the batch.
    # We normalise the inner sum by the sequence length (not the mask count) so
    # the 1/t weight retains its meaning while keeping the scale stable; t is
    # clamped away from 0 to bound the estimator's variance for this tiny demo.
    t_clamped = t.squeeze(1).clamp_min(0.1)
    per_seq = (ce * mask).sum(dim=1) / seq / t_clamped
    return per_seq.mean()


@torch.no_grad()
def reconstruct(model, tokens: torch.Tensor, mask: torch.Tensor, mask_id: int):
    """One-shot fill: predict every masked position from the visible context."""
    masked = tokens.masked_fill(mask, mask_id)
    logits = model(masked)
    pred = logits.argmax(dim=-1)
    out = tokens.clone()
    out[mask] = pred[mask]
    return out


@torch.no_grad()
def generate(
    model,
    seq_len: int,
    mask_id: int,
    steps: int = 8,
    batch_size: int = 1,
    device: Optional[torch.device] = None,
    generator: Optional[torch.Generator] = None,
    temperature: float = 0.0,
    record_trace: bool = False,
):
    """Reverse diffusion: all-``[MASK]`` -> a full sequence over ``steps`` steps.

    Uses low-confidence remasking (paper §2.4): at every step we predict all
    masked tokens but only *commit* the highest-confidence ones, leaving the
    rest masked for later steps.

    Returns the generated tokens, and (if ``record_trace``) a list of per-step
    snapshots for visualization.
    """
    device = device or next(model.parameters()).device
    x = torch.full((batch_size, seq_len), mask_id, dtype=torch.long, device=device)
    trace = []

    for step in range(steps):
        # Fraction of positions that should remain masked after this step.
        s = (steps - step - 1) / steps
        keep_masked = int(round(s * seq_len))

        logits = model(x)
        probs = F.softmax(logits, dim=-1)
        if temperature > 0:
            sampled = torch.distributions.Categorical(
                logits=logits / temperature
            ).sample()
        else:
            sampled = logits.argmax(dim=-1)
        conf = probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)

        is_mask = x == mask_id
        # Confidence only matters at currently-masked positions.
        conf = conf.masked_fill(~is_mask, -1.0)

        candidate = torch.where(is_mask, sampled, x)
        new_x = x.clone()
        for b in range(batch_size):
            masked_idx = is_mask[b].nonzero(as_tuple=True)[0]
            if masked_idx.numel() == 0:
                continue
            num_to_commit = masked_idx.numel() - keep_masked
            num_to_commit = max(num_to_commit, 0)
            if num_to_commit == 0:
                continue
            order = torch.argsort(conf[b], descending=True)
            commit = order[:num_to_commit]
            new_x[b, commit] = candidate[b, commit]
        x = new_x

        if record_trace:
            trace.append(
                {
                    "step": step + 1,
                    "tokens": x[0].tolist(),
                    "is_mask": (x[0] == mask_id).tolist(),
                    "kept_masked": int((x[0] == mask_id).sum().item()),
                }
            )

    return (x, trace) if record_trace else x
