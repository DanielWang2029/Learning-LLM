"""Constitutional AI in miniature (Bai et al., 2022).

Two stages, no humans in the loop:

  SL-CAI  — a toy model emits responses that may break the constitution; a
            CRITIQUE step names the violated principle and a REVISION step
            rewrites the response to comply, iterating until it does.
  RLAIF   — the revised responses are labeled preferred over the originals
            (AI feedback) and a preference model is trained on those pairs.

Success criteria (asserted at the end):
  * the constitutional violation rate DROPS sharply after critique+revision, and
  * the preference model trained on AI feedback ranks compliant responses above
    non-compliant ones with high accuracy.

Runs on CPU in a few seconds.  Run with:  python demo/run_demo.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
torch.set_num_threads(1)

from src.words import MAX_LEN, PAD, STOI, VOCAB_SIZE, encode
from src.constitution import (
    CONSTITUTION,
    critique_and_revise,
    is_compliant,
    violation_counts,
)
from src.generator import PROMPTS, sample_dataset
from src.preference_model import PreferenceModel, preference_loss

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SEED = 0
MODEL_MAXLEN = MAX_LEN + 2  # longest a raw response can be


def banner(t):
    print("\n" + "=" * 66 + f"\n{t}\n" + "=" * 66)


def enc(responses):
    return torch.tensor([encode(r, MODEL_MAXLEN) for r in responses], dtype=torch.long)


def main() -> None:
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    t0 = time.time()

    print("Constitutional AI in miniature — self-critique, revision, and RLAIF")
    print("  constitution:")
    for i, p in enumerate(CONSTITUTION, 1):
        print(f"    {i}. [{p.id}] {p.text}")

    # ---------------------------------------------------------- SL-CAI stage
    banner("STAGE 1 (SL-CAI) — critique & revise against the constitution")
    raw = sample_dataset(1000, rng)
    revised = [critique_and_revise(r)[0] for r in raw]

    before = violation_counts(raw)
    after = violation_counts(revised)
    print(f"  compliance BEFORE critique/revise: {before['compliance_rate']*100:5.1f}%")
    print(f"  compliance AFTER  critique/revise: {after['compliance_rate']*100:5.1f}%")
    print("  per-rule violations (before -> after):")
    for p in CONSTITUTION:
        print(f"    [{p.id:14s}] {before['per_rule'][p.id]:4d} -> {after['per_rule'][p.id]:4d}")

    # a concrete critique/revision trace — prefer a rich, multi-step one that
    # includes fixing harmful content.
    example, best_score = None, -1
    for i, r in enumerate(raw):
        final, trace = critique_and_revise(r)
        rules = {s["rule"] for s in trace}
        score = len(trace) + (2 if "harmless" in rules else 0)
        if score > best_score and len(trace) >= 1:
            best_score = score
            example = {"prompt": PROMPTS[i % len(PROMPTS)], "original": r,
                       "trace": trace, "final": final, "compliant": is_compliant(final)}
        if best_score >= 4:
            break

    # -------------------------------------------------------------- RLAIF
    banner("STAGE 2 (RLAIF) — train a preference model on AI feedback")
    # Build preference pairs (revised ≻ original) from responses that needed fixing.
    pref, disp = [], []
    for r in raw:
        rev = critique_and_revise(r)[0]
        if rev != r and is_compliant(rev):
            pref.append(rev)   # AI-preferred (revised, compliant)
            disp.append(r)     # AI-dispreferred (original, non-compliant)
    print(f"  built {len(pref)} AI-feedback preference pairs (revised ≻ original)")

    P = enc(pref); D = enc(disp)

    # Held-out set to test whether the model predicts *compliance*.
    holdout = sample_dataset(600, rng)
    comp = [r for r in holdout if is_compliant(r)]
    noncomp = [r for r in holdout if not is_compliant(r)]
    k = min(len(comp), len(noncomp))
    Hc = enc(comp[:k]); Hn = enc(noncomp[:k])

    model = PreferenceModel(VOCAB_SIZE, STOI[PAD], MODEL_MAXLEN)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    EPOCHS = 80
    curve = []

    def compliance_acc():
        with torch.no_grad():
            sc = model(Hc); sn = model(Hn)
            return (sc > sn).float().mean().item()

    for epoch in range(1, EPOCHS + 1):
        sp = model(P); sd = model(D)
        loss = preference_loss(sp, sd)
        opt.zero_grad(); loss.backward(); opt.step()
        if epoch % 10 == 0 or epoch == 1:
            with torch.no_grad():
                train_acc = (model(P) > model(D)).float().mean().item()
            c_acc = compliance_acc()
            curve.append({"epoch": epoch, "pair_acc": train_acc, "compliance_acc": c_acc})
            print(f"  epoch {epoch:3d}/{EPOCHS} | bt loss {loss.item():.4f} | "
                  f"pref-pair acc {train_acc*100:5.1f}% | compliance-pred acc {c_acc*100:5.1f}%")

    final_pair_acc = curve[-1]["pair_acc"]
    final_comp_acc = compliance_acc()

    # ------------------------------------------------------------- Results
    banner("RESULTS")
    print(f"  violation rate: {(1-before['compliance_rate'])*100:.1f}% (before) -> "
          f"{(1-after['compliance_rate'])*100:.1f}% (after critique/revise)")
    print(f"  preference model — AI-pair ranking accuracy    : {final_pair_acc*100:5.1f}%")
    print(f"  preference model — compliance-prediction acc.  : {final_comp_acc*100:5.1f}%")
    if example:
        print("\n  example critique/revision:")
        print(f"    prompt   : \"{example['prompt']}\"")
        print(f"    original : {' '.join(example['original'])}")
        for step in example["trace"]:
            print(f"    critique [{step['rule']}]: {step['critique']}")
            print(f"      revise -> {' '.join(step['after'])}")
        print(f"    final    : {' '.join(example['final'])}  (compliant={example['compliant']})")

    # ------------------------------------------------------------- Persist
    DATA_DIR.mkdir(exist_ok=True)
    payload = {
        "constitution": [{"id": p.id, "text": p.text} for p in CONSTITUTION],
        "before": before, "after": after,
        "pair_accuracy": final_pair_acc,
        "compliance_accuracy": final_comp_acc,
        "learning_curve": curve,
        "example": {
            "prompt": example["prompt"],
            "original": example["original"],
            "trace": example["trace"],
            "final": example["final"],
            "compliant": example["compliant"],
        } if example else None,
        "num_pairs": len(pref),
    }
    out = DATA_DIR / "cai_results.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"\n  wrote {out}")
    print(f"  total time: {time.time()-t0:.1f}s")

    # ------------------------------------------------------------- Asserts
    assert after["compliance_rate"] > 0.99, "revision failed to fix violations"
    assert before["compliance_rate"] < 0.8, "generator was already too compliant"
    assert final_comp_acc > 0.9, f"preference model can't predict compliance: {final_comp_acc:.2f}"
    print("\nOK: critique+revise removed violations and RLAIF preference model works.")


if __name__ == "__main__":
    main()
