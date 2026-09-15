"""GRPO-style policy gradient with Kimi k1.5's length-penalty reward (paper §2.5).

The RL loop is minimal but faithful to the two ideas the paper highlights for
reasoning RL:

* **Group-relative advantage (GRPO-style).** For each problem we sample a *group*
  of completions and use the group's mean reward as the baseline; the advantage
  is the reward standardized within the group. No learned value network.

* **Length penalty.** Kimi adds a length reward that, *within each group*, gives
  higher reward to shorter correct answers and never rewards length on incorrect
  ones. With ``min_len``/``max_len`` the group's shortest/longest completion:

      lambda_i = 0.5 - (len_i - min_len) / (max_len - min_len)
      len_reward_i = lambda_i           if the answer is correct
                   = min(0, lambda_i)   otherwise

  This discourages "overthinking" (longer, not-better chains) while preserving
  accuracy. Turning it off (weight 0) leaves the thinking budget untouched,
  because the correctness reward is independent of the chain length.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _mixture_probs(model, a_logits, length):
    """Answer distribution after ``length`` think steps: a reveal-weighted blend
    of the head's prediction and a uniform guess (see ``model.reveal_weight``)."""
    w = model.reveal_weight(length).unsqueeze(-1)          # (n,1)
    base_p = a_logits.softmax(dim=-1)
    uniform = torch.full_like(base_p, 1.0 / model.num_values)
    return w * base_p + (1.0 - w) * uniform


def rollout(model, problems, group_size, device, temperature=1.0, force_length=None):
    """Sample ``group_size`` (answer, length) completions per problem.

    ``force_length`` fixes the chain length (used by the SFT-style warm-start so
    the answer head learns cleanly at full reveal before the length policy moves).
    """
    a = torch.tensor([p[0] for p in problems], device=device).repeat_interleave(group_size)
    b = torch.tensor([p[1] for p in problems], device=device).repeat_interleave(group_size)
    tgt = torch.tensor([p[2] for p in problems], device=device).repeat_interleave(group_size)
    n = a.size(0)

    with torch.no_grad():
        if force_length is None:
            l_logits = model.length_dist(n) / temperature
            length = torch.distributions.Categorical(logits=l_logits).sample()
        else:
            length = torch.full((n,), int(force_length), device=device)
        a_logits = model.answer_logits(a, b) / temperature
        # The emitted answer is revealed only in proportion to how long we thought.
        mix_p = _mixture_probs(model, a_logits, length)
        ans = torch.distributions.Categorical(probs=mix_p).sample()

    correct = (ans == tgt).float()
    group_ids = torch.arange(len(problems), device=device).repeat_interleave(group_size)
    return {"a": a, "b": b, "tgt": tgt, "ans": ans, "length": length,
            "correct": correct, "group_ids": group_ids}


def compute_rewards(roll, use_length_penalty: bool, weight: float):
    """Correctness reward plus (optionally) Kimi's group length reward."""
    correct = roll["correct"]
    reward = correct.clone()
    if use_length_penalty:
        length = roll["length"].float()
        gids = roll["group_ids"]
        for g in gids.unique():
            m = (gids == g)
            gl = length[m]
            lo, hi = gl.min(), gl.max()
            lam = 0.5 - (gl - lo) / (hi - lo) if hi > lo else torch.zeros_like(gl)
            gc = correct[m].bool()
            reward[m] = reward[m] + weight * torch.where(gc, lam, torch.clamp(lam, max=0.0))
    return reward


def grpo_step(model, opt, roll, rewards, device, entropy_coef=0.0, update_length=True):
    """One GRPO-style update using group-standardized advantages.

    ``update_length=False`` freezes the length policy (used during warm-start, so
    only the answer head is trained)."""
    gids = roll["group_ids"]
    adv = torch.zeros_like(rewards)
    for g in gids.unique():
        m = (gids == g)
        r = rewards[m]
        adv[m] = (r - r.mean()) / (r.std() + 1e-6)

    a_logits = model.answer_logits(roll["a"], roll["b"])
    # Log-prob of the sampled answer under the same reveal-weighted mixture used
    # to sample it, so the answer head is credited through the thinking budget.
    mix_p = _mixture_probs(model, a_logits, roll["length"])
    a_logp = mix_p.clamp_min(1e-9).log().gather(-1, roll["ans"].unsqueeze(-1)).squeeze(-1)
    logp = a_logp
    if update_length:
        l_logits = model.length_dist(a_logits.size(0))
        l_logp = F.log_softmax(l_logits, dim=-1).gather(-1, roll["length"].unsqueeze(-1)).squeeze(-1)
        logp = logp + l_logp

    loss = -(adv * logp).mean()
    if entropy_coef > 0:
        a_probs = a_logits.softmax(-1)
        ent = -(a_probs * F.log_softmax(a_logits, -1)).sum(-1).mean()
        loss = loss - entropy_coef * ent

    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return float(loss.item())


@torch.no_grad()
def evaluate(model, problems, device):
    """Greedy accuracy and the *mean chain length* under the length policy.

    Accuracy uses the greedy chain length, so a chain that is too short to reveal
    the answer is scored under the same reveal-weighted mixture used in training —
    i.e. shortening the chain below ``think_need`` actually costs accuracy."""
    a = torch.tensor([p[0] for p in problems], device=device)
    b = torch.tensor([p[1] for p in problems], device=device)
    tgt = torch.tensor([p[2] for p in problems], device=device)

    greedy_len = int(model.length_logits.argmax())
    length = torch.full((a.size(0),), greedy_len, device=device)
    mix_p = _mixture_probs(model, model.answer_logits(a, b), length)
    ans = mix_p.argmax(dim=-1)
    w = model.reveal_weight(length)
    # Expected correctness of the emitted (mixture) answer.
    head_correct = (ans == tgt).float()
    acc = float((w * head_correct + (1.0 - w) / model.num_values).mean().item())

    # Expected chain length under the (soft) length distribution.
    p_len = model.length_logits.softmax(-1)
    idx = torch.arange(model.max_think + 1, device=device).float()
    mean_len = float((p_len * idx).sum().item())
    return acc, mean_len
