"""The capability-probe harness: score a model on each task, report a profile.

This is the reusable piece that mirrors the paper's methodology — a battery
runner that turns raw model behavior into an interpretable pass/fail
"capability profile" per task and per scale.
"""

from __future__ import annotations

import numpy as np
import torch

# A task "passes" (capability considered present) above this exact-match rate.
PASS_THRESHOLD = 0.80


@torch.no_grad()
def evaluate_task(model, seqs: np.ndarray, masks: np.ndarray, device="cpu") -> float:
    """Teacher-forced exact-match accuracy over the answer region."""
    model.eval()
    x = torch.tensor(seqs, dtype=torch.long, device=device)
    logits = model(x[:, :-1])                 # predict token t+1 from t
    pred = logits.argmax(dim=-1)              # (N, T-1)
    tgt = x[:, 1:]
    m = torch.tensor(masks[:, 1:], dtype=torch.bool, device=device)
    correct_tok = (pred == tgt) | ~m          # ignore non-answer positions
    per_example = correct_tok.all(dim=1)      # all answer tokens correct
    return per_example.float().mean().item()


def capability_profile(model, eval_sets, device="cpu") -> dict:
    """eval_sets: {task: (seqs, masks)} -> {task: {'acc':..., 'pass':bool}}."""
    out = {}
    for task, (seqs, masks) in eval_sets.items():
        acc = evaluate_task(model, seqs, masks, device)
        out[task] = {"acc": acc, "pass": acc >= PASS_THRESHOLD}
    return out
