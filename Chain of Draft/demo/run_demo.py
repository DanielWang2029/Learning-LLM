"""Chain of Draft demo: match CoT accuracy with far fewer tokens.

One tiny decoder-only LM is trained on a multi-step arithmetic task rendered in
three prompting styles (Standard / Chain of Thought / Chain of Draft). At test
time we prompt the *same* model in each style and report, on held-out problems:

* **accuracy** — Standard (direct) is weak; CoT and CoD are both strong, and
* **tokens** — CoD uses a small fraction of CoT's reasoning tokens.

Writes ``data/cod_run.json`` for the visualization.

Run with:  python demo/run_demo.py
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import (  # noqa: E402
    TinyGPT, Tokenizer, STYLES, STYLE_NAME, STANDARD, COT, COD,
    make_problem, prompt_of, full_example, parse_answer, RENDER,
)

SEED = 0
NUM_TERMS = 4
STEPS = 450
SUB_BATCH = 24         # examples per style, per step
LR = 3e-3
MAX_LEN = 80
N_LAYERS = 3
N_TRAIN_PROBLEMS = 3000
N_EVAL = 150
# The direct-answer target (a 4-step chained sum in one shot) is essentially
# unlearnable for a tiny model; left at full weight its large, noisy gradient
# swamps the shared model and prevents CoT/CoD from learning. We down-weight it
# so the model still learns the Standard *format* while staying (correctly) weak.
STYLE_LOSS_W = {STANDARD: 0.1, COT: 1.0, COD: 1.0}
# Cap generation length per style (a little over each style's completion length;
# generation also stops early on end-of-sequence).
MAX_NEW = {STANDARD: 4, COT: 64, COD: 24}


def make_split(n, seed):
    rng = random.Random(seed)
    seen, probs = set(), []
    while len(probs) < n:
        terms, ans = make_problem(NUM_TERMS, rng)
        key = tuple(terms)
        if key in seen:
            continue
        seen.add(key)
        probs.append((terms, ans))
    return probs, seen


def encode_example(tok, style, terms):
    """Return (ids, loss_mask) where loss is applied only to completion tokens."""
    prompt = prompt_of(style, terms)
    full = full_example(style, terms)
    ids = tok.encode(full)
    mask = [0] * len(tok.encode(prompt)) + [1] * (len(ids) - len(tok.encode(prompt)))
    return ids, mask


def pad_batch(seqs, masks, pad_id):
    m = max(len(s) for s in seqs)
    x = torch.full((len(seqs), m), pad_id, dtype=torch.long)
    lm = torch.zeros((len(seqs), m), dtype=torch.float)
    for i, (s, k) in enumerate(zip(seqs, masks)):
        x[i, : len(s)] = torch.tensor(s)
        lm[i, : len(k)] = torch.tensor(k, dtype=torch.float)
    return x, lm


def _style_loss(model, tok, x, lm):
    """Masked next-token cross-entropy over the completion tokens of a batch."""
    logits = model(x[:, :-1])
    tgt = x[:, 1:]
    lossmask = lm[:, 1:]
    ce = F.cross_entropy(
        logits.reshape(-1, tok.vocab_size), tgt.reshape(-1), reduction="none"
    ).reshape(tgt.shape)
    return (ce * lossmask).sum() / lossmask.sum().clamp_min(1)


def train(model, tok, problems, gen):
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    for step in range(1, STEPS + 1):
        model.train()
        opt.zero_grad()
        last = {}
        # Gradient accumulation: one sub-batch per style, each padded only to its
        # OWN longest sequence. This avoids padding the short Standard/CoD batches
        # up to the verbose CoT length, and balances the per-style gradient signal.
        for style in STYLES:
            seqs, masks = [], []
            for _ in range(SUB_BATCH):
                terms, _ = problems[gen.randrange(len(problems))]
                ids, mk = encode_example(tok, style, terms)
                seqs.append(ids)
                masks.append(mk)
            x, lm = pad_batch(seqs, masks, tok.eos_id)
            loss = _style_loss(model, tok, x, lm) * STYLE_LOSS_W[style]
            loss.backward()
            last[style] = loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 150 == 0 or step == 1:
            parts = " ".join(f"{s}={last[s]:.3f}" for s in STYLES)
            print(f"  step {step:4d}/{STEPS} | loss {parts}")


@torch.no_grad()
def evaluate(model, tok, problems, style):
    """Greedy-decode all problems of a style together; return (acc, mean tokens).

    Every prompt of a given style has the same length (fixed ``NUM_TERMS``), so
    the whole eval set is decoded in a single batched generation pass."""
    model.eval()
    prompts = [prompt_of(style, terms) for terms, _ in problems]
    ids = torch.tensor([tok.encode(p) for p in prompts])
    plen = ids.size(1)
    out = model.generate(ids, max_new=MAX_NEW[style], eos_id=tok.eos_id)

    correct, total_tokens = 0, 0
    example = None
    for i, (terms, ans) in enumerate(problems):
        gen_ids = out[i, plen:].tolist()
        text = tok.decode(gen_ids)
        completion = text.split(".")[0]          # up to (not incl.) EOS
        n_tok = len(completion) + 1              # +1 for the terminating '.'
        total_tokens += n_tok
        if parse_answer(text) == ans:
            correct += 1
        if example is None:
            example = {"prompt": prompts[i], "completion": completion + ".",
                       "pred": parse_answer(text), "answer": ans}
    return correct / len(problems), total_tokens / len(problems), example


def main() -> None:
    torch.manual_seed(SEED)
    tok = Tokenizer()

    train_probs, seen = make_split(N_TRAIN_PROBLEMS, SEED)
    # Held-out problems the model never saw during training.
    eval_probs = []
    rng = random.Random(SEED + 999)
    while len(eval_probs) < N_EVAL:
        terms, ans = make_problem(NUM_TERMS, rng)
        if tuple(terms) not in seen:
            eval_probs.append((terms, ans))

    print("Chain of Draft: one model, three prompting styles")
    print(f"task: {NUM_TERMS}-term chained add mod 10 | train={len(train_probs)} eval={len(eval_probs)}\n")

    # dropout=0 — the task is deterministic (a lookup of (a+b) mod 10 per step),
    # so regularisation only slows convergence here.
    model = TinyGPT(tok.vocab_size, num_layers=N_LAYERS, max_len=MAX_LEN, dropout=0.0)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"training TinyGPT ({n_params:,} params) on all three styles ...")
    start = time.time()
    train(model, tok, train_probs, random.Random(SEED + 1))
    print(f"trained in {time.time() - start:.1f}s\n")

    results = {}
    print("Held-out results (greedy decode):")
    print("  style               accuracy   mean tokens")
    for s in STYLES:
        acc, mtok, ex = evaluate(model, tok, eval_probs, s)
        results[s] = {"name": STYLE_NAME[s], "acc": acc, "tokens": mtok, "example": ex}
        print(f"  {STYLE_NAME[s]:<18} {acc*100:6.1f}%      {mtok:6.1f}")

    cot_tok = results[COT]["tokens"]
    cod_tok = results[COD]["tokens"]
    frac = cod_tok / cot_tok * 100
    print(f"\n  → CoD uses {frac:.0f}% of CoT's tokens "
          f"({cod_tok:.1f} vs {cot_tok:.1f}) at accuracy "
          f"{results[COD]['acc']*100:.0f}% vs {results[COT]['acc']*100:.0f}%.")

    out = {
        "meta": {"num_terms": NUM_TERMS, "params": n_params, "n_eval": len(eval_probs)},
        "results": {s: {"name": STYLE_NAME[s], "acc": results[s]["acc"],
                        "tokens": results[s]["tokens"], "example": results[s]["example"]}
                    for s in STYLES},
        "cod_token_fraction": frac,
    }
    out_path = ROOT / "data" / "cod_run.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}")

    assert results[COT]["acc"] > 0.85, "CoT should be accurate"
    assert results[COD]["acc"] > 0.85, "CoD should be accurate"
    assert results[COD]["acc"] > results[STANDARD]["acc"] + 0.2, \
        "reasoning (CoD) should beat direct answering"
    assert cod_tok < cot_tok * 0.5, "CoD should use far fewer tokens than CoT"
    print("\nOK: Chain of Draft matches Chain of Thought accuracy with far fewer tokens.")


if __name__ == "__main__":
    main()
