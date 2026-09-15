"""LIMO demo: *Less Is More for Reasoning*.

At an **equal training budget** we supervised-fine-tune the same tiny reasoner
two ways and compare held-out reasoning accuracy:

* **Curated (few, high-quality)** — a small set of full step-by-step solutions.
* **Bulk (many, low-quality)**     — a much larger set of terse "shortcut"
  solutions that give the right answer but show no reasoning.

Both pools have correct answers; only the *quality of the demonstrated
reasoning* differs. The result reproduces LIMO's thesis: the small curated set
of high-quality reasoning traces generalizes better to unseen problems.

A JSON summary is written to ``data/limo_run.json`` for the visualization.

Run with:  python demo/run_demo.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import TinyGPT, Tokenizer, build_dataset, make_problem  # noqa: E402
from src.task import prompt_of, high_quality_trace, low_quality_trace, EOS  # noqa: E402
import random  # noqa: E402

SEED = 0
NUM_TERMS = 4
MAX_LEN = 40
CURATED_N = 200       # few (hundreds), high-quality demonstrations
BULK_N = 2000         # many (10x more), low-quality demonstrations
TRAIN_STEPS = 600     # identical budget for both runs
BATCH = 64
TEST_N = 300


def pad_batch(strings, tok, device):
    ids = [tok.encode(s) for s in strings]
    L = max(len(x) for x in ids)
    L = min(L, MAX_LEN)
    x = torch.full((len(ids), L), tok.eos_id, dtype=torch.long)
    lengths = []
    for i, seq in enumerate(ids):
        seq = seq[:L]
        x[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        lengths.append(len(seq))
    return x.to(device), lengths


def train(pool, tok, device, gen, tag):
    torch.manual_seed(SEED)
    model = TinyGPT(tok.vocab_size, d_model=128, num_layers=3, num_heads=4, d_ff=256, max_len=MAX_LEN).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01)
    tokens_seen = 0
    idx_pool = list(range(len(pool)))
    for step in range(1, TRAIN_STEPS + 1):
        model.train()
        pick = [pool[gen.randint(0, len(pool) - 1)] for _ in range(BATCH)]
        x, lengths = pad_batch(pick, tok, device)
        logits = model(x[:, :-1])
        target = x[:, 1:]
        # Ignore positions at/after EOS in the loss so padding is not learned.
        b, t = target.shape
        pos = torch.arange(t, device=device).unsqueeze(0)
        valid = pos < (torch.tensor(lengths, device=device).unsqueeze(1) - 1)
        loss = F.cross_entropy(
            logits.reshape(-1, tok.vocab_size), target.reshape(-1), reduction="none"
        ).view(b, t)
        loss = (loss * valid).sum() / valid.sum().clamp_min(1)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        tokens_seen += int(sum(lengths))
        if step % 300 == 0 or step == 1:
            print(f"  [{tag}] step {step:4d}/{TRAIN_STEPS} | loss {loss.item():.4f}")
    return model, tokens_seen


def parse_answer(text: str):
    """Extract the integer answer between '#' and the EOS '.'."""
    if "#" not in text:
        return None
    tail = text.split("#", 1)[1]
    tail = tail.split(".", 1)[0]
    return int(tail) if tail.isdigit() else None


@torch.no_grad()
def evaluate(model, tok, test_problems, device):
    """Batched greedy decoding over all held-out prompts (they share a length)."""
    model.eval()
    prompts = [prompt_of(terms) for terms, _ in test_problems]
    answers = [ans for _, ans in test_problems]
    idx = torch.tensor([tok.encode(p) for p in prompts], dtype=torch.long, device=device)
    plen = idx.size(1)
    out = model.generate(idx, max_new=MAX_LEN - plen, eos_id=tok.eos_id)

    correct = 0
    examples = []
    for k in range(len(prompts)):
        gen_text = tok.decode(out[k].tolist())
        # Trim everything after the first EOS following the prompt.
        body = gen_text[plen:]
        if EOS in body:
            body = body[: body.index(EOS) + 1]
        gen_text = prompts[k] + body
        pred = parse_answer(body)
        ok = pred == answers[k]
        correct += int(ok)
        if k < 6:
            examples.append({"prompt": prompts[k], "answer": answers[k],
                             "generated": gen_text, "pred": pred, "ok": ok})
    return correct / len(test_problems), examples


def main() -> None:
    torch.manual_seed(SEED)
    device = torch.device("cpu")
    tok = Tokenizer()

    curated = build_dataset(CURATED_N, NUM_TERMS, "high", seed=SEED + 1)
    bulk = build_dataset(BULK_N, NUM_TERMS, "low", seed=SEED + 2)

    # Held-out test set: unseen problems not present in either training pool.
    train_qs = {s.split("=", 1)[0] for s in curated} | {s.split("=", 1)[0] for s in bulk}
    rng = random.Random(SEED + 99)
    test_problems, seen = [], set()
    while len(test_problems) < TEST_N:
        terms, ans = make_problem(NUM_TERMS, rng)
        q = "+".join(str(t) for t in terms)
        if q in train_qs or q in seen:
            continue
        seen.add(q)
        test_problems.append((terms, ans))

    print("LIMO: few high-quality vs many low-quality reasoning demonstrations")
    print(f"task: sum of {NUM_TERMS} single digits | held-out problems: {TEST_N}")
    print(f"curated set: {CURATED_N} high-quality step-by-step traces")
    print(f"bulk set   : {BULK_N} low-quality shortcut traces")
    print(f"budget     : {TRAIN_STEPS} steps, batch {BATCH} (identical for both)\n")

    print("example curated (high-quality) trace:")
    print(f"    {high_quality_trace([3, 5, 2, 4])}")
    print("example bulk (low-quality) trace:")
    print(f"    {low_quality_trace([3, 5, 2, 4])}\n")

    gen = random.Random(SEED + 7)
    start = time.time()
    print("Training on CURATED high-quality set ...")
    m_hi, tok_hi = train(curated, tok, device, gen, "curated")
    print("Training on BULK low-quality set ...")
    m_lo, tok_lo = train(bulk, tok, device, gen, "bulk")
    print(f"trained both in {time.time() - start:.1f}s\n")

    acc_hi, ex_hi = evaluate(m_hi, tok, test_problems, device)
    acc_lo, ex_lo = evaluate(m_lo, tok, test_problems, device)

    print("Held-out reasoning accuracy (unseen problems):")
    print(f"  curated  ({CURATED_N:4d} high-quality, {tok_hi:>7d} train tokens): {acc_hi * 100:5.1f}%")
    print(f"  bulk     ({BULK_N:4d} low-quality , {tok_lo:>7d} train tokens): {acc_lo * 100:5.1f}%")
    print(f"  → curated set is {CURATED_N/BULK_N:.2f}x the size but wins by "
          f"{(acc_hi - acc_lo) * 100:.1f} points\n")

    print("Sample generations (curated model shows its work):")
    for e in ex_hi[:3]:
        print(f"  {e['generated']}   [{'OK' if e['ok'] else 'WRONG'}]")
    print("Sample generations (bulk model jumps to an answer):")
    for e in ex_lo[:3]:
        print(f"  {e['generated']}   [{'OK' if e['ok'] else 'WRONG'}]")

    out = {
        "meta": {
            "num_terms": NUM_TERMS, "curated_n": CURATED_N, "bulk_n": BULK_N,
            "steps": TRAIN_STEPS, "batch": BATCH, "test_n": TEST_N,
        },
        "metrics": {
            "curated_acc": acc_hi, "bulk_acc": acc_lo,
            "curated_tokens": tok_hi, "bulk_tokens": tok_lo,
        },
        "examples": {
            "curated_trace": high_quality_trace([3, 5, 2, 4]),
            "bulk_trace": low_quality_trace([3, 5, 2, 4]),
            "curated_gens": ex_hi, "bulk_gens": ex_lo,
        },
    }
    out_path = ROOT / "data" / "limo_run.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {out_path}")

    assert acc_hi > acc_lo + 0.15, (
        f"expected curated high-quality set to win clearly (hi={acc_hi:.2f} lo={acc_lo:.2f})"
    )
    print("OK: fewer high-quality reasoning traces beat many low-quality ones.")


if __name__ == "__main__":
    main()
