"""Stage 3: optimize the policy against the reward model.

Two of the paper's ingredients, in miniature:

* **best-of-n** (rejection sampling): draw n responses from the policy and keep
  the one the RM scores highest. A cheap, RL-free way to raise reward.
* **REINFORCE** policy gradient: the simplest instance of the PPO-style RL step
  in the paper (§2.3). We sample a response, score it with the RM, and push up
  the log-probability of high-reward responses:

        ∇J = E[ (R - b) · ∇ log π(response | prompt) ]

  with a running baseline `b` to reduce variance and a KL-style penalty toward
  the SFT policy to prevent it from drifting / reward-hacking.
"""

from __future__ import annotations

from typing import List

import torch

from . import data
from .model import TinyLM
from .reward_model import RewardModel


def sample_responses(
    policy: TinyLM,
    prompt: List[int],
    n: int,
    rng: torch.Generator,
    temperature: float = 1.0,
) -> List[List[int]]:
    """Sample `n` responses of length RESP_LEN from the policy for one prompt."""
    prefix = torch.tensor(
        [data.prefix_tokens(prompt)] * n, dtype=torch.long
    )
    torch.manual_seed(int(torch.randint(0, 2**31 - 1, (1,), generator=rng)))
    out = policy.generate(prefix, data.RESP_LEN, temperature=temperature)
    resp = out[:, len(data.prefix_tokens(prompt)) :]
    return resp.tolist()


def best_of_n(
    policy: TinyLM,
    reward_model: RewardModel,
    prompt: List[int],
    n: int,
    rng: torch.Generator,
    temperature: float = 1.0,
) -> List[int]:
    """Return the RM-highest-scoring of `n` sampled responses (rejection sampling)."""
    candidates = sample_responses(policy, prompt, n, rng, temperature)
    seqs = data.encode_batch([prompt] * n, candidates)
    with torch.no_grad():
        scores = reward_model(seqs)
    best = int(scores.argmax())
    return candidates[best]


def rule_reward(prompt: List[int], response: List[int]) -> float:
    """Ground-truth reward (the hidden 'human' rule): sortedness of the response."""
    return data.sortedness(response)
