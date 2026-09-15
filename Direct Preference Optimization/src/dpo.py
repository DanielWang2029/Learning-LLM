"""The Direct Preference Optimization loss (Rafailov et al., 2023).

DPO skips the reward model entirely. Its key insight: the optimal RLHF policy
has a closed form, which means the *language model is secretly its own reward
model*. The implicit reward of a response y to a prompt x is

    r(x, y) = β · [ log π_θ(y|x) − log π_ref(y|x) ]   (+ a constant that cancels)

Plugging this into the Bradley-Terry preference model and taking the negative
log-likelihood of the preferences gives a simple supervised loss (paper Eq. 7):

    L_DPO = − log σ( β · [ (logπ_θ(y_w|x) − logπ_ref(y_w|x))
                         − (logπ_θ(y_l|x) − logπ_ref(y_l|x)) ] )

where y_w is the chosen (winning) response and y_l the rejected one. No reward
model, no RL sampling loop — just gradient descent on this loss.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F


def dpo_loss(
    pol_logp_chosen: torch.Tensor,
    pol_logp_rejected: torch.Tensor,
    ref_logp_chosen: torch.Tensor,
    ref_logp_rejected: torch.Tensor,
    beta: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (loss, reward_chosen, reward_rejected, margin).

    `reward_*` are the implicit DPO rewards β·(logπ_θ − logπ_ref); `margin` is
    reward_chosen − reward_rejected (the logit inside the sigmoid).
    """
    reward_chosen = beta * (pol_logp_chosen - ref_logp_chosen)
    reward_rejected = beta * (pol_logp_rejected - ref_logp_rejected)
    margin = reward_chosen - reward_rejected
    loss = -F.logsigmoid(margin).mean()
    return loss, reward_chosen, reward_rejected, margin
