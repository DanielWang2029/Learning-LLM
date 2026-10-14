"""Reproduce the core claims of HES data selection (arXiv:2605.22389) on CPU.

Pipeline (mirrors the paper):
  1. Train a small REFERENCE LM on a mix of reasoning traces.
  2. Score every sample with HES (sum of the top-p% highest-entropy tokens) and
     with the baseline metrics the paper compares against (AvgE, AvgHE, ES).
     -> Claim 1: HES ranks GOOD vs low-quality samples far better than the
        averaging/total-entropy baselines (reproduces Figure 1).
  3. SFT data selection: rank a dataset by HES, train a fresh student on the
     top-20% vs the bottom-20% vs the full set, evaluate held-out accuracy.
     -> Claim 2: top-20% HES matches the full dataset, bottom-20% degrades it
        (reproduces the SFT Efficiency / De-noising tests).

Note on scale: the paper uses p = 0.5% because real long-CoT responses are
thousands of tokens; our synthetic traces are short, so we use a larger
percentile (top ~30%) to select the same handful of critical tokens. The
mechanism (SUM over the highest-entropy tokens) is identical.

Run:  python demo/run_demo.py
"""

from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import TinyLM, hes, avg_entropy, avg_high_entropy, entropy_sum, data  # noqa: E402
from src.data import VOCAB_SIZE, PAD, EOS, STOI, MAX_LEN, GOOD, HQ, METRIC_TIERS  # noqa: E402

SEED = 0
P_TOP = 0.50          # top-p% token fraction for this toy (short-trace) scale
DIGITS = [STOI[str(d)] for d in range(10)]


def pad_batch(seqs, device):
    m = max(len(s) for s in seqs)
    x = torch.full((len(seqs), m), PAD, dtype=torch.long, device=device)
    for i, s in enumerate(seqs):
        x[i, : len(s)] = torch.tensor(s, device=device)
    return x


def train_lm(model, samples, steps, batch, lr, device, resp_only=True):
    """Causal-LM training. If resp_only, only response tokens contribute loss
    (standard SFT masking of the prompt)."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    rng = random.Random(1234)
    for _ in range(steps):
        batch_s = [rng.choice(samples) for _ in range(batch)]
        seqs = [s["tokens"][:MAX_LEN] for s in batch_s]
        x = pad_batch(seqs, device)
        logits = model(x[:, :-1])
        target = x[:, 1:].clone()
        target[x[:, 1:] == PAD] = -100
        if resp_only:
            for i, s in enumerate(batch_s):
                rs = s["resp_start"]
                target[i, : max(0, rs - 1)] = -100  # ignore prompt-token prediction
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), target.reshape(-1), ignore_index=-100
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()


@torch.no_grad()
def response_entropies(model, sample, device):
    """Per-token predictive entropy for the RESPONSE tokens of one sample."""
    seq = sample["tokens"][:MAX_LEN]
    x = torch.tensor([seq], device=device)
    logits = model(x)[0]                      # (L, vocab); logits[t] predicts token t+1
    ent_all = -(F.softmax(logits, -1) * F.log_softmax(logits, -1)).sum(-1)  # (L,)
    rs = sample["resp_start"]
    # entropy predicting response token j (index rs..L-1) lives at position j-1
    return ent_all[rs - 1 : len(seq) - 1]


@torch.no_grad()
def hes_scores_batch(model, samples, device, p, bs=128):
    """HES for many samples (batched for speed)."""
    out = []
    for i in range(0, len(samples), bs):
        chunk = samples[i:i + bs]
        x = pad_batch([s["tokens"][:MAX_LEN] for s in chunk], device)
        logits = model(x)
        ent = -(F.softmax(logits, -1) * F.log_softmax(logits, -1)).sum(-1)  # (B, L)
        for j, s in enumerate(chunk):
            rs, L = s["resp_start"], len(s["tokens"][:MAX_LEN])
            out.append(hes(ent[j, rs - 1 : L - 1], p=p))
    return out


def auc(pos, neg):
    """Prob that a random GOOD score exceeds a random low-quality score."""
    pos, neg = np.asarray(pos), np.asarray(neg)
    wins = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(wins / (len(pos) * len(neg)))


@torch.no_grad()
def eval_em(model, probs, device):
    """Held-out exact-match: greedy-generate, read the answer (last 3 digits)."""
    correct = 0
    for a, b in probs:
        prompt = data.prompt_tokens(a, b)
        out = model.greedy_generate(
            torch.tensor(prompt, device=device), max_new=20, eos_id=EOS
        ).tolist()
        gen = out[len(prompt):]
        digs = [t for t in gen if t in DIGITS]
        pred = "".join(data.ITOS[t] for t in digs[-3:]) if len(digs) >= 3 else ""
        if pred == str(a + b).zfill(3):
            correct += 1
    return correct / len(probs)


def main():
    t0 = time.time()
    device = torch.device("cpu")
    rng = random.Random(SEED)
    torch.manual_seed(SEED)

    print("HES data selection for LLM reasoning  (arXiv:2605.22389, CPU repro)\n")

    # ---- 1. reference LM (knows the trace FORMAT, not the arithmetic) ----
    ref_data = [data.randomize_digits(s, rng)
                for s in data.gen_metric(400, rng) + data.gen_sft(400, rng)]
    ref = TinyLM(VOCAB_SIZE, d_model=64, n_layers=2, n_heads=4, max_len=MAX_LEN).to(device)
    train_lm(ref, ref_data, steps=400, batch=48, lr=3e-3, device=device, resp_only=False)
    print(f"[1] trained reference LM ({time.time()-t0:.0f}s)")

    # ---- 2. HES vs baseline metrics (Claim 1): reproduce Fig 1 / Table 4 ----
    pool = data.gen_metric(240, rng)
    metrics = {"HES": [], "AvgE": [], "AvgHE": [], "ES": []}
    tiers = []
    for s in pool:
        ent = response_entropies(ref, s, device)
        metrics["HES"].append(hes(ent, p=P_TOP))
        metrics["AvgE"].append(avg_entropy(ent))
        metrics["AvgHE"].append(avg_high_entropy(ent, p=P_TOP))
        metrics["ES"].append(entropy_sum(ent))
        tiers.append(s["tier"])
    tiers = np.array(tiers)
    good_mask = tiers == HQ
    aucs = {}
    print("\n[2] metric quality-ranking (AUC: P[score(HQ) > score(low-quality)])")
    for name, vals in metrics.items():
        vals = np.array(vals)
        aucs[name] = auc(vals[good_mask], vals[~good_mask])
        print(f"    {name:5s}  AUC = {aucs[name]:.3f}")
    tier_hes = {t: float(np.mean([metrics['HES'][i] for i in range(len(pool)) if pool[i]['tier'] == t]))
                for t in METRIC_TIERS}

    # ---- 3. SFT selection (Claim 2): HES-selected vs random vs lowest, equal budget ----
    sft = data.gen_sft(3000, rng)
    hscores = np.array(hes_scores_batch(ref, sft, device, p=P_TOP))
    order = np.argsort(-hscores)
    k = int(0.2 * len(sft))
    rand_idx = list(range(len(sft)))
    random.Random(SEED).shuffle(rand_idx)
    subsets = {
        "top-20% HES": [sft[i] for i in order[:k]],          # HES selection (paper)
        "random-20%": [sft[i] for i in rand_idx[:k]],         # the paper's baseline
        "bottom-20% HES": [sft[i] for i in order[-k:]],       # lowest-HES (should hurt)
        "full (100%)": sft,                                   # reference
    }
    test_probs = [(rng.randint(10, 99), rng.randint(10, 99)) for _ in range(150)]
    print("\n[3] SFT: train a fresh student on each subset (equal steps), held-out exact-match")
    sft_res = {}
    for name, subset in subsets.items():
        torch.manual_seed(SEED)
        stu = TinyLM(VOCAB_SIZE, d_model=64, n_layers=2, n_heads=4, max_len=MAX_LEN).to(device)
        train_lm(stu, subset, steps=900, batch=48, lr=3e-3, device=device, resp_only=True)
        em = eval_em(stu, test_probs, device)
        good_frac = np.mean([s["tier"] == GOOD for s in subset])
        sft_res[name] = {"em": em, "good_fraction": float(good_frac), "n": len(subset)}
        print(f"    {name:16s} n={len(subset):4d}  GOOD={good_frac*100:4.0f}%  held-out EM={em*100:5.1f}%")

    # ---- save for the visualization ----
    payload = {
        "p_top": P_TOP,
        "metric_auc": aucs,
        "tier_mean_hes": tier_hes,
        "metric_scatter": {
            name: [{"v": float(v), "good": bool(g)} for v, g in zip(metrics[name], good_mask)]
            for name in metrics
        },
        "sft": sft_res,
        "example": None,
    }
    ex = next(s for s in pool if s["tier"] == HQ)
    ex_ent = response_entropies(ref, ex, device).tolist()
    payload["example"] = {
        "prompt": data.decode(ex["prompt"]),
        "response_tokens": [data.ITOS[t] for t in ex["tokens"][ex["resp_start"]:][:len(ex_ent)]],
        "entropies": [round(e, 3) for e in ex_ent],
        "answer": ex["answer"],
    }
    data_dir = ROOT / "data"
    data_dir.mkdir(exist_ok=True)
    (data_dir / "demo_results.json").write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {data_dir / 'demo_results.json'}.  Total: {time.time()-t0:.0f}s")

    # ---- sanity checks ----
    top = sft_res["top-20% HES"]["em"]
    rnd = sft_res["random-20%"]["em"]
    bot = sft_res["bottom-20% HES"]["em"]
    full = sft_res["full (100%)"]["em"]
    ok = (
        aucs["HES"] > 0.9                      # HES separates quality (Claim 1)
        and aucs["HES"] > aucs["AvgE"] + 0.2   # ... far better than averaging
        and aucs["HES"] >= aucs["AvgHE"]
        and top > rnd + 0.15                   # HES selection >> random (Claim 2)
        and top >= full - 0.1                  # top-20% matches the full dataset
        and rnd > bot + 0.03                   # ... and lowest-HES data is worst
    )
    if not ok:
        raise SystemExit(
            f"Sanity failed: HES AUC {aucs['HES']:.2f} (AvgE {aucs['AvgE']:.2f}, AvgHE {aucs['AvgHE']:.2f}); "
            f"EM top {top:.2f} / full {full:.2f} / random {rnd:.2f} / bottom {bot:.2f}"
        )
    print("OK: HES ranks reasoning quality far better than averaging baselines; "
          "top-20% HES-selected SFT data matches the full set and beats random / lowest-HES.")


if __name__ == "__main__":
    main()
