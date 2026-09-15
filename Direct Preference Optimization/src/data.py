"""The toy preference task and its tokenizer.

DPO optimizes a policy directly from preference pairs (chosen ≻ rejected). To
make that fully reproducible on CPU we define the preference with a *known
rule*: the preferred response to a prompt is the prompt's content tokens sorted
in ascending order.

    prompt:   3 1 2 0       (a small bag of tokens)
    chosen:   0 1 2 3       (sorted   — the preferred response)
    rejected: 3 1 2 0       (unsorted — the dispreferred response)

Every sequence has the layout

    [BOS] <prompt tokens> [SEP] <response tokens> [EOS]

so the model is autoregressive and we can score just the response region.
`sortedness` is our stand-in for "human approval".
"""

from __future__ import annotations

from typing import List, Tuple

import torch

PAD, BOS, SEP, EOS = 0, 1, 2, 3
NUM_SPECIAL = 4
NUM_CONTENT = 6
VOCAB_SIZE = NUM_SPECIAL + NUM_CONTENT
PROMPT_LEN = 4
RESP_LEN = PROMPT_LEN
SEQ_LEN = 1 + PROMPT_LEN + 1 + RESP_LEN + 1


def random_prompt(rng: torch.Generator) -> List[int]:
    ids = torch.randint(
        NUM_SPECIAL, NUM_SPECIAL + NUM_CONTENT, (PROMPT_LEN,), generator=rng
    )
    return ids.tolist()


def sorted_response(prompt: List[int]) -> List[int]:
    return sorted(prompt)


def sortedness(response: List[int]) -> float:
    """Reward proxy in [0, 1]: fraction of adjacent pairs in non-decreasing order."""
    if len(response) <= 1:
        return 1.0
    good = sum(1 for a, b in zip(response, response[1:]) if a <= b)
    return good / (len(response) - 1)


def is_compliant(prompt: List[int], response: List[int]) -> bool:
    return response == sorted_response(prompt)


def build_sequence(prompt: List[int], response: List[int]) -> List[int]:
    return [BOS] + prompt + [SEP] + response + [EOS]


def build_response_mask() -> torch.Tensor:
    """Boolean mask over full-sequence positions marking the response region."""
    mask = torch.zeros(SEQ_LEN, dtype=torch.bool)
    start = 1 + PROMPT_LEN + 1
    mask[start:] = True
    return mask


def encode_batch(prompts: List[List[int]], responses: List[List[int]]) -> torch.Tensor:
    seqs = [build_sequence(p, r) for p, r in zip(prompts, responses)]
    return torch.tensor(seqs, dtype=torch.long)


def prefix_tokens(prompt: List[int]) -> List[int]:
    return [BOS] + prompt + [SEP]


def make_demonstrations(
    n: int, rng: torch.Generator, p_correct: float = 0.55
) -> Tuple[List[List[int]], List[List[int]]]:
    """SFT data for the reference policy: (prompt, demonstrated response) pairs.

    Demonstrations are imperfect (only a fraction `p_correct` are ideal), so the
    SFT/reference policy is decent but not fully aligned — leaving clear room for
    DPO to improve it directly from preferences.
    """
    prompts, responses = [], []
    for _ in range(n):
        p = random_prompt(rng)
        prompts.append(p)
        if torch.rand((), generator=rng).item() < p_correct:
            responses.append(sorted_response(p))
        else:
            perm = torch.randperm(PROMPT_LEN, generator=rng).tolist()
            responses.append([p[i] for i in perm])
    return prompts, responses


def make_preference_pairs(
    n: int, rng: torch.Generator
) -> Tuple[List[List[int]], List[List[int]], List[List[int]]]:
    """DPO data: (prompt, chosen, rejected) triples labeled by the rule.

    Chosen is the fully sorted prompt; rejected is a random permutation that is
    strictly less sorted.
    """
    prompts, chosen, rejected = [], [], []
    for _ in range(n):
        p = random_prompt(rng)
        good = sorted_response(p)
        bad = good[:]
        for _ in range(8):
            perm = torch.randperm(PROMPT_LEN, generator=rng).tolist()
            cand = [p[i] for i in perm]
            if sortedness(cand) < sortedness(good):
                bad = cand
                break
        prompts.append(p)
        chosen.append(good)
        rejected.append(bad)
    return prompts, chosen, rejected
