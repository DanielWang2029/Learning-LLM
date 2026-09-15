"""The toy alignment task and its tokenizer.

InstructGPT aligns a language model to a hard-to-specify human preference.
To make that fully reproducible on CPU we replace the human with a *known
rule*: the "helpful" response to a prompt is the prompt's content tokens
sorted in ascending order.

    prompt:   3 1 2 0       (a small bag of tokens)
    good:     0 1 2 3       (sorted  -> reward 1.0)
    bad:      3 1 2 0       (unsorted -> low reward)

Every sequence the model sees has the layout

    [BOS] <prompt tokens> [SEP] <response tokens> [EOS]

so the model can be trained autoregressively and we can score just the
response region. `sortedness` is our stand-in for "human approval".
"""

from __future__ import annotations

from typing import List, Tuple

import torch

# Special tokens, then content tokens 0..NUM_CONTENT-1 offset by NUM_SPECIAL.
PAD, BOS, SEP, EOS = 0, 1, 2, 3
NUM_SPECIAL = 4
NUM_CONTENT = 6  # content symbols: ids NUM_SPECIAL .. NUM_SPECIAL+NUM_CONTENT-1
VOCAB_SIZE = NUM_SPECIAL + NUM_CONTENT
PROMPT_LEN = 4
RESP_LEN = PROMPT_LEN
# Full length: BOS + prompt + SEP + response + EOS
SEQ_LEN = 1 + PROMPT_LEN + 1 + RESP_LEN + 1


def content_ids() -> List[int]:
    return list(range(NUM_SPECIAL, NUM_SPECIAL + NUM_CONTENT))


def random_prompt(rng: torch.Generator) -> List[int]:
    """A prompt is a random multiset of content tokens (may repeat)."""
    ids = torch.randint(
        NUM_SPECIAL, NUM_SPECIAL + NUM_CONTENT, (PROMPT_LEN,), generator=rng
    )
    return ids.tolist()


def sorted_response(prompt: List[int]) -> List[int]:
    """The rule-following (rewarded) response: the prompt, sorted ascending."""
    return sorted(prompt)


def sortedness(response: List[int]) -> float:
    """Reward proxy in [0, 1]: fraction of adjacent pairs in non-decreasing order."""
    if len(response) <= 1:
        return 1.0
    good = sum(1 for a, b in zip(response, response[1:]) if a <= b)
    return good / (len(response) - 1)


def is_compliant(prompt: List[int], response: List[int]) -> bool:
    """Exact rule compliance: response is exactly the sorted prompt."""
    return response == sorted_response(prompt)


def build_sequence(prompt: List[int], response: List[int]) -> List[int]:
    return [BOS] + prompt + [SEP] + response + [EOS]


def build_response_mask() -> torch.Tensor:
    """Boolean mask over full-sequence positions marking the response region.

    True where the token is a response token or the trailing EOS, i.e. the
    positions whose prediction we actually train / score.
    """
    mask = torch.zeros(SEQ_LEN, dtype=torch.bool)
    start = 1 + PROMPT_LEN + 1  # index of first response token
    mask[start:] = True  # response tokens + EOS
    return mask


def encode_batch(
    prompts: List[List[int]], responses: List[List[int]]
) -> torch.Tensor:
    seqs = [build_sequence(p, r) for p, r in zip(prompts, responses)]
    return torch.tensor(seqs, dtype=torch.long)


def prefix_tokens(prompt: List[int]) -> List[int]:
    """The conditioning prefix the policy is asked to continue: [BOS,prompt,SEP]."""
    return [BOS] + prompt + [SEP]


def make_demonstrations(
    n: int, rng: torch.Generator, p_correct: float = 0.55
) -> Tuple[List[List[int]], List[List[int]]]:
    """Stage-1 SFT data: (prompt, demonstrated response) pairs.

    Human demonstrations are *imperfect and expensive*: only a fraction
    `p_correct` are the ideal sorted answer; the rest are a careless random
    ordering. This mirrors a central InstructGPT finding — reliable preference
    comparisons are cheaper to collect than flawless demonstrations, so RLHF
    against the reward model can exceed the quality of the SFT data itself.
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
    """Stage-2 RM data: (prompt, chosen, rejected) triples.

    The labeler (our rule) prefers the more-sorted of two candidate responses.
    Chosen is the fully sorted prompt; rejected is a random permutation that is
    strictly less sorted, mirroring human preference labels over model samples.
    """
    prompts, chosen, rejected = [], [], []
    for _ in range(n):
        p = random_prompt(rng)
        good = sorted_response(p)
        # Draw a shuffled candidate that is less sorted than the ideal.
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
