"""The minimal LLM lifecycle the survey describes, in runnable form.

Stages (Zhao et al., 2023):
  1. Pre-training      — next-token prediction on a raw corpus.
  2. Adaptation (SFT)  — supervised fine-tuning on instruction→answer pairs.
  3. Alignment (DPO)   — direct preference optimization toward preferred answers.

All three operate on ONE tiny GPT over a shared 13-token vocabulary, so a single
model visibly progresses through the whole pipeline.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

# --- shared vocabulary ------------------------------------------------------
# 0..9 digits, then markers and two instruction tokens.
EQ, INSTR_SORT, INSTR_MAX, PAD = 10, 11, 12, 13
VOCAB_SIZE = 14
N_DIGITS = 4
BLOCK = 1 + N_DIGITS + 1 + N_DIGITS  # [INSTR] x4 [EQ] y4  = 10


# --- Stage 1: pre-training data (a structured "corpus") ---------------------
def _markov_table(seed: int = 0):
    rng = np.random.default_rng(seed)
    return rng.dirichlet(np.full(10, 0.4), size=10)  # digit -> next-digit dist


_TABLE = _markov_table()


def pretrain_batch(n: int, seed: int):
    """Length-BLOCK digit sequences from a fixed order-1 Markov chain."""
    rng = np.random.default_rng(seed)
    seqs = np.zeros((n, BLOCK), dtype=np.int64)
    seqs[:, 0] = rng.integers(0, 10, size=n)
    cdf = np.cumsum(_TABLE, axis=1)
    for t in range(1, BLOCK):
        u = rng.random(n)[:, None]
        seqs[:, t] = (u > cdf[seqs[:, t - 1]]).sum(axis=1)
    mask = np.ones_like(seqs)  # predict every next token
    return seqs, mask


# --- Stage 2/3: the instruction task = "sort four digits ascending" ---------
def _sort_example(rng):
    x = rng.integers(0, 10, size=N_DIGITS)
    y = np.sort(x)
    return x, y


def _pack(instr: int, x, y):
    seq = np.array([instr, *x, EQ, *y], dtype=np.int64)
    mask = np.zeros(BLOCK, dtype=np.int64)
    mask[1 + N_DIGITS + 1:] = 1  # answer region only
    return seq, mask


def sft_batch(n: int, seed: int):
    """Supervised fine-tuning data for the SORT instruction only."""
    rng = np.random.default_rng(seed)
    seqs, masks = [], []
    for _ in range(n):
        x, y = _sort_example(rng)
        s, m = _pack(INSTR_SORT, x, y)
        seqs.append(s); masks.append(m)
    return np.stack(seqs), np.stack(masks)


def preference_batch(n: int, seed: int):
    """Alignment data for a NEW behavior the model was never SFT'd on: MAX.

    RLHF/DPO can teach behavior from comparisons alone. Here the model never saw
    the MAX instruction during SFT, so before alignment it prefers the "chosen"
    answer only at chance. Chosen = the max digit (repeated); rejected = another
    input digit that is not the max. DPO must learn the concept from preferences.
    """
    rng = np.random.default_rng(seed)
    chosen, rejected = [], []
    for _ in range(n):
        x = rng.integers(0, 10, size=N_DIGITS)
        mx = int(x.max())
        y_good = [mx] * N_DIGITS
        others = [int(v) for v in x if int(v) != mx]
        bad = others[int(rng.integers(len(others)))] if others else (mx + 1) % 10
        y_bad = [bad] * N_DIGITS
        chosen.append(_pack(INSTR_MAX, x, y_good))
        rejected.append(_pack(INSTR_MAX, x, y_bad))
    cs, cm = map(np.stack, zip(*chosen))
    rs, rm = map(np.stack, zip(*rejected))
    return cs, cm, rs, rm


def max_eval_batch(n: int, seed: int):
    """Held-out MAX examples for exact-match scoring (answer = max digit x4)."""
    rng = np.random.default_rng(seed)
    seqs, masks = [], []
    for _ in range(n):
        x = rng.integers(0, 10, size=N_DIGITS)
        s, m = _pack(INSTR_MAX, x, [int(x.max())] * N_DIGITS)
        seqs.append(s); masks.append(m)
    return np.stack(seqs), np.stack(masks)


# --- training / scoring helpers --------------------------------------------
def lm_loss(model, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    logits = model(x[:, :-1])
    tgt = x[:, 1:]
    m = mask[:, 1:].bool()
    return F.cross_entropy(
        logits.reshape(-1, VOCAB_SIZE)[m.reshape(-1)], tgt.reshape(-1)[m.reshape(-1)]
    )


def train_lm(model, batch_fn, steps: int, lr: float, seed: int):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    first = last = None
    for step in range(steps):
        seqs, masks = batch_fn(seed + step)
        x = torch.tensor(seqs, dtype=torch.long)
        m = torch.tensor(masks, dtype=torch.long)
        loss = lm_loss(model, x, m)
        opt.zero_grad(); loss.backward(); opt.step()
        last = float(loss.detach())
        if first is None:
            first = last
    return first, last


@torch.no_grad()
def exact_match(model, seqs: np.ndarray, masks: np.ndarray) -> float:
    model.eval()
    x = torch.tensor(seqs, dtype=torch.long)
    pred = model(x[:, :-1]).argmax(-1)
    tgt = x[:, 1:]
    m = torch.tensor(masks[:, 1:], dtype=torch.bool)
    ok = ((pred == tgt) | ~m).all(dim=1)
    return ok.float().mean().item()


def seq_logprob(model, seqs: np.ndarray, masks: np.ndarray) -> torch.Tensor:
    x = torch.tensor(seqs, dtype=torch.long)
    logp = torch.log_softmax(model(x[:, :-1]), dim=-1)
    tgt = x[:, 1:]
    tok_lp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    m = torch.tensor(masks[:, 1:], dtype=torch.float32)
    return (tok_lp * m).sum(dim=1)  # summed logprob over answer region


def dpo_loss(policy, ref, cs, cm, rs, rm, beta: float) -> torch.Tensor:
    lp_c = seq_logprob(policy, cs, cm)
    lp_r = seq_logprob(policy, rs, rm)
    with torch.no_grad():
        lr_c = seq_logprob(ref, cs, cm)
        lr_r = seq_logprob(ref, rs, rm)
    logits = beta * ((lp_c - lr_c) - (lp_r - lr_r))
    return -F.logsigmoid(logits).mean()


@torch.no_grad()
def preference_accuracy(model, cs, cm, rs, rm) -> float:
    """Fraction of pairs where the model prefers the chosen answer."""
    return (seq_logprob(model, cs, cm) > seq_logprob(model, rs, rm)).float().mean().item()
