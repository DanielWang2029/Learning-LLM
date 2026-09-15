"""A tiny multiple-choice eval harness (capability-vs-scale).

Mirrors how the GPT-4 report tracks *capabilities* (not just loss) as a
function of scale: here each question is a prompt plus several candidate
completions, exactly one of which is a genuine continuation of the Markov
language. A model "answers" by picking the completion it assigns the highest
log-likelihood — the same length-normalized scoring used by real
multiple-choice benchmarks (e.g. HellaSwag / MMLU cloze). Better models
(lower loss) pick the right completion more often, so accuracy rises with scale.
"""

from __future__ import annotations

import numpy as np
import torch

from .data import MarkovSource


def build_mc_questions(
    source: MarkovSource,
    n_questions: int,
    prompt_len: int,
    completion_len: int,
    n_choices: int,
    seed: int,
):
    """Return prompts, choices, and the index of the correct choice.

    The correct completion continues the prompt under the true source; the
    distractors are independent draws from the source that do *not* follow this
    particular prompt, so only genuine local structure distinguishes them.
    """
    rng = np.random.default_rng(seed)
    total = prompt_len + completion_len
    prompts, choices, answers = [], [], []
    for _ in range(n_questions):
        seq = source.sample(1, total, seed=int(rng.integers(0, 2**31)))[0]
        prompt = seq[:prompt_len]
        correct = seq[prompt_len:]
        opts = [correct]
        for _ in range(n_choices - 1):
            distract = source.sample(
                1, total, seed=int(rng.integers(0, 2**31))
            )[0, prompt_len:]
            opts.append(distract)
        order = rng.permutation(n_choices)
        opts = [opts[i] for i in order]
        answer = int(np.where(order == 0)[0][0])
        prompts.append(prompt)
        choices.append(np.stack(opts))
        answers.append(answer)
    return (
        np.stack(prompts),
        np.stack(choices),
        np.array(answers, dtype=np.int64),
    )


@torch.no_grad()
def evaluate_mc(model, prompts, choices, answers, device) -> float:
    """Length-normalized log-likelihood scoring; returns accuracy in [0, 1]."""
    model.eval()
    n_q, n_choices, comp_len = choices.shape
    prompt_len = prompts.shape[1]
    correct = 0
    for q in range(n_q):
        scores = []
        for c in range(n_choices):
            seq = np.concatenate([prompts[q], choices[q, c]])
            idx = torch.tensor(seq[None, :], dtype=torch.long, device=device)
            logits, _ = model(idx[:, :-1])
            logp = torch.log_softmax(logits, dim=-1)[0]
            tgt = idx[0, 1:]
            token_lp = logp[torch.arange(len(tgt)), tgt]
            # Only score the completion region (the graded answer span).
            scores.append(token_lp[prompt_len - 1 :].mean().item())
        if int(np.argmax(scores)) == answers[q]:
            correct += 1
    return correct / n_q
