"""End-to-end Search-R1 demo: RL for interleaved reasoning + search (CPU, <60s).

Trains a search policy with reinforcement learning (REINFORCE + GRPO-style group
baseline) to answer multi-hop questions by interleaving reasoning with real
retrieval calls over a small corpus. Compares it against a no-search classifier,
and shows the learned search procedure GENERALIZES to held-out persons — while the
no-search baseline, which can only memorize, does not.

Run with:  python demo/run_demo.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

torch.set_num_threads(1)  # shared CPU box: avoid thread oversubscription

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import Corpus, SearchPolicy, NoSearchClassifier, group_advantages

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@torch.no_grad()
def eval_policy(policy, corpus, persons):
    correct = searches = 0
    for p in persons:
        r = policy.rollout(corpus, p, sample=False)
        correct += int(r["correct"]); searches += r["n_searches"]
    return correct / len(persons), searches / len(persons)


def train_policy(policy, corpus, train_persons, steps, group_size, lr, gen):
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    acc0, srch0 = eval_policy(policy, corpus, train_persons)
    curve = [{"step": 0, "train_acc": round(acc0, 4), "avg_searches": round(srch0, 2)}]
    for step in range(1, steps + 1):
        # one question per step (cycled), G rollouts -> group-relative advantage
        p = train_persons[step % len(train_persons)]
        logps, rewards = [], []
        for _ in range(group_size):
            r = policy.rollout(corpus, p, sample=True, generator=gen)
            logps.append(r["logp"]); rewards.append(r["reward"])
        logps = torch.stack(logps)
        rewards_t = torch.tensor(rewards)
        adv = group_advantages(rewards_t)
        loss = -(adv.detach() * logps).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 5 == 0 or step == 1:
            acc, srch = eval_policy(policy, corpus, train_persons)
            curve.append({"step": step, "train_acc": round(acc, 4),
                          "avg_searches": round(srch, 2)})
    return curve


def train_classifier(clf, corpus, train_persons, steps, lr, gen):
    """Supervised no-search baseline on TRAIN persons only."""
    opt = torch.optim.Adam(clf.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    local = {p: i for i, p in enumerate(corpus.persons)}
    x = torch.tensor([local[p] for p in train_persons])
    y = torch.tensor([corpus.countries.index(corpus.answer_country(p))
                      for p in train_persons])
    for _ in range(steps):
        logits = clf(x)
        loss = loss_fn(logits, y)
        opt.zero_grad(); loss.backward(); opt.step()


@torch.no_grad()
def eval_classifier(clf, corpus, persons):
    local = {p: i for i, p in enumerate(corpus.persons)}
    x = torch.tensor([local[p] for p in persons])
    pred = clf(x).argmax(-1)
    y = torch.tensor([corpus.countries.index(corpus.answer_country(p)) for p in persons])
    return (pred == y).float().mean().item()


def trace_to_json(corpus, trace):
    """Serialize a rollout trace into simple dicts for the visualization."""
    out = []
    for ev in trace:
        if ev[0] == "search":
            _, e, results = ev
            out.append({"kind": "search", "entity": corpus.name_of[e],
                        "info": [f"{corpus.name_of[e]} {rel} {corpus.name_of[t]}"
                                 for rel, t in results]})
        else:
            _, ans = ev
            out.append({"kind": "answer",
                        "entity": corpus.name_of[ans] if ans is not None else "None"})
    return out


def render_trace(corpus, person, trace):
    lines = [f"Question: What country does {corpus.name_of[person]} live in?"]
    for ev in trace:
        if ev[0] == "search":
            _, e, results = ev
            lines.append(f"  <search>{corpus.name_of[e]}</search>")
            info = "; ".join(f"{corpus.name_of[e]} {rel} {corpus.name_of[t]}"
                             for rel, t in results)
            lines.append(f"  <information>{info}</information>")
        else:
            _, ans = ev
            lines.append(f"  <answer>{corpus.name_of[ans] if ans is not None else 'None'}</answer>")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-persons", type=int, default=60)
    ap.add_argument("--n-train", type=int, default=40)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--group-size", type=int, default=6)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--clf-steps", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    corpus = Corpus(n_persons=args.n_persons, generator=torch.Generator().manual_seed(args.seed))
    train_persons = corpus.persons[:args.n_train]
    test_persons = corpus.persons[args.n_train:]

    print(f"Corpus: {len(corpus.persons)} persons, {len(corpus.cities)} cities, "
          f"{len(corpus.countries)} countries (+ year/color distractors)")
    print(f"Question: multi-hop 'what country does Person_x live in?' "
          f"(person -> city -> country)")
    print(f"Split: {len(train_persons)} train persons, {len(test_persons)} held-out\n")

    t0 = time.time()
    policy = SearchPolicy()
    pre_acc, _ = eval_policy(policy, corpus, test_persons)
    print(f"Search policy (before RL) test accuracy: {pre_acc*100:.1f}%")
    curve = train_policy(policy, corpus, train_persons, args.steps,
                         args.group_size, args.lr, gen)
    for c in curve[::max(1, len(curve)//6)]:
        print(f"  step {c['step']:4d} | train acc {c['train_acc']*100:5.1f}% | "
              f"avg searches {c['avg_searches']:.2f}")

    search_train_acc, srch = eval_policy(policy, corpus, train_persons)
    search_test_acc, srch_test = eval_policy(policy, corpus, test_persons)

    clf = NoSearchClassifier(len(corpus.persons), len(corpus.countries))
    train_classifier(clf, corpus, train_persons, args.clf_steps, 1e-2, gen)
    clf_train_acc = eval_classifier(clf, corpus, train_persons)
    clf_test_acc = eval_classifier(clf, corpus, test_persons)

    print(f"\nTrained in {time.time()-t0:.1f}s\n")
    print("                          train persons    held-out persons")
    print(f"  WITH search (RL)        {search_train_acc*100:6.1f}%          "
          f"{search_test_acc*100:6.1f}%   ({srch_test:.2f} searches/q)")
    print(f"  NO search (classifier)  {clf_train_acc*100:6.1f}%          "
          f"{clf_test_acc*100:6.1f}%")

    # Interleaved trace on a held-out question.
    ex_person = test_persons[0]
    r = policy.rollout(corpus, ex_person, sample=False)
    trace_str = render_trace(corpus, ex_person, r["trace"])
    print("\nInterleaved trace on a HELD-OUT question:")
    print(trace_str)
    print(f"  correct answer: {corpus.name_of[corpus.answer_country(ex_person)]} "
          f"-> {'OK' if r['correct'] else 'WRONG'}")

    out = {
        "config": {"n_persons": len(corpus.persons), "n_train": len(train_persons),
                   "n_test": len(test_persons), "n_countries": len(corpus.countries),
                   "group_size": args.group_size, "steps": args.steps,
                   "seconds": round(time.time() - t0, 1)},
        "curve": curve,
        "results": {
            "search": {"train_acc": round(search_train_acc, 4),
                       "test_acc": round(search_test_acc, 4),
                       "avg_searches": round(srch_test, 2)},
            "no_search": {"train_acc": round(clf_train_acc, 4),
                          "test_acc": round(clf_test_acc, 4)},
            "pre_rl_test_acc": round(pre_acc, 4),
        },
        "example": {
            "person": corpus.name_of[ex_person],
            "true_country": corpus.name_of[corpus.answer_country(ex_person)],
            "correct": r["correct"], "n_searches": r["n_searches"],
            "trace": trace_to_json(corpus, r["trace"]),
        },
    }
    DATA_DIR.mkdir(exist_ok=True)
    out_path = DATA_DIR / "search_r1_results.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {out_path}")

    if not (search_test_acc > 0.9 and search_test_acc - clf_test_acc > 0.3):
        raise SystemExit("Search policy did not clearly beat the no-search baseline "
                         "on held-out persons.")
    print("\nOK: the RL-trained search procedure generalizes to unseen persons; the "
          "no-search baseline can only memorize the training set.")


if __name__ == "__main__":
    main()
