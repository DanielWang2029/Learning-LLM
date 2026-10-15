"""Synthetic long-CoT reasoning data with KNOWN quality tiers.

Two purpose-built generators (both over the same tiny 2-digit-addition world):

* METRIC data (`gen_metric`) -- for reproducing the paper's Figure 1 / Table 4
  finding that the *summation* over the highest-entropy tokens (HES) separates
  quality where averaging (AvgE/AvgHE) and length-confounded totals (ES) do not.
  Quality here changes the NUMBER of genuine high-entropy "fork" tokens and the
  length:
    - HQ        : a long, detailed derivation (many high-entropy digit tokens);
    - LQ_SHORT  : the answer only (few forks, short);
    - LQ_FILLER : the answer plus long low-entropy filler (few forks, long).

* SFT data (`gen_sft`) -- for the data-selection experiment. All samples share
  ONE fixed format `[scratch] SEP [answer] EOS` so that mixing subsets never
  introduces a *format* confound; only the reasoning QUALITY of the scratch
  region differs:
    - GOOD : the correct digit-by-digit scratch (teaches the algorithm);
    - BAD  : the same slots filled with low-information filler (teaches nothing
             computable), with the correct answer still appended.

A reference LM trained on *digit-randomised* versions of these traces (see
`randomize_digits`) is uncertain at every digit slot, so HES counts the genuine
reasoning tokens -- the CPU stand-in for the paper's pretrained reference model.
"""

from __future__ import annotations

import random
from typing import Dict, List

SPECIAL = ["PAD", "BOS", "EOS", "SEP", "+", "=", ";", "F"]
VOCAB: List[str] = SPECIAL + [str(d) for d in range(10)]
STOI: Dict[str, int] = {t: i for i, t in enumerate(VOCAB)}
ITOS: Dict[int, str] = {i: t for t, i in STOI.items()}
VOCAB_SIZE = len(VOCAB)

PAD, BOS, EOS, SEP = STOI["PAD"], STOI["BOS"], STOI["EOS"], STOI["SEP"]
PLUS, EQ, SEMI, FILL = STOI["+"], STOI["="], STOI[";"], STOI["F"]
_DIGIT_IDS = {STOI[str(d)] for d in range(10)}

# metric tiers (length / fork-count varying) and sft tiers (shared format)
HQ, LQ_SHORT, LQ_FILLER = "HQ", "LQ_short", "LQ_filler"
METRIC_TIERS = [HQ, LQ_SHORT, LQ_FILLER]
GOOD, BAD = "good", "bad"
SFT_TIERS = [GOOD, BAD]
SCRATCH_LEN = 8
MAX_LEN = 48


def _digits(n: int, width: int) -> List[int]:
    return [STOI[c] for c in str(n).zfill(width)]


def prompt_tokens(a: int, b: int) -> List[int]:
    """BOS a1 a0 + b1 b0 =   (operands as 2 digits, most-significant first)."""
    return [BOS] + _digits(a, 2) + [PLUS] + _digits(b, 2) + [EQ]


def _good_scratch(a: int, b: int) -> List[int]:
    """Correct per-digit addition: [d0 ; c0 ; d1 ; c1 ;]  (8 tokens)."""
    carry, toks = 0, []
    for i in range(2):
        ai, bi = (a // 10 ** i) % 10, (b // 10 ** i) % 10
        s = ai + bi + carry
        digit, carry = s % 10, s // 10
        toks += [STOI[str(digit)], SEMI, STOI[str(carry)], SEMI]
    return toks


def _verbose_derivation(a: int, b: int) -> List[int]:
    """Long HQ reasoning: per position restate operands, result, carry.

    [da db ; r c ;] x2  -> many genuine high-entropy digit tokens.
    """
    carry, toks = 0, []
    for i in range(2):
        ai, bi = (a // 10 ** i) % 10, (b // 10 ** i) % 10
        s = ai + bi + carry
        r, carry = s % 10, s // 10
        toks += [STOI[str(ai)], STOI[str(bi)], SEMI, STOI[str(r)], STOI[str(carry)], SEMI]
    return toks


# ---------- METRIC dataset (length / fork-count varying) ----------

def _metric_response(a: int, b: int, tier: str) -> List[int]:
    ans = _digits(a + b, 3)
    if tier == HQ:
        return _verbose_derivation(a, b) + [SEP] + ans + [EOS]
    if tier == LQ_SHORT:
        return ans + [EOS]
    if tier == LQ_FILLER:
        return ans + [FILL] * 12 + [EOS]
    raise ValueError(tier)


# ---------- SFT dataset (shared format, quality-only difference) ----------

def _sft_response(a: int, b: int, tier: str) -> List[int]:
    ans = _digits(a + b, 3)
    scratch = _good_scratch(a, b) if tier == GOOD else [FILL] * SCRATCH_LEN
    return scratch + [SEP] + ans + [EOS]


def _make(a: int, b: int, tier: str, resp_fn) -> Dict:
    prompt = prompt_tokens(a, b)
    resp = resp_fn(a, b, tier)
    return {"a": a, "b": b, "answer": a + b, "tier": tier, "prompt": prompt,
            "response": resp, "tokens": prompt + resp, "resp_start": len(prompt)}


def gen_metric(n: int, rng: random.Random) -> List[Dict]:
    return [_make(rng.randint(10, 99), rng.randint(10, 99), rng.choice(METRIC_TIERS),
                  _metric_response) for _ in range(n)]


def gen_sft(n: int, rng: random.Random, p_good: float = 1 / 3) -> List[Dict]:
    out = []
    for _ in range(n):
        tier = GOOD if rng.random() < p_good else BAD
        out.append(_make(rng.randint(10, 99), rng.randint(10, 99), tier, _sft_response))
    return out


def randomize_digits(sample: Dict, rng: random.Random) -> Dict:
    """Copy with every DIGIT token replaced by a random digit (keeps structure).

    The reference LM trains on these, so it learns the trace FORMAT but not the
    arithmetic -> every digit slot stays genuinely high-entropy under it.
    """
    def rnd(t):
        return rng.randint(STOI["0"], STOI["9"]) if t in _DIGIT_IDS else t
    s = dict(sample)
    s["prompt"] = [rnd(t) for t in sample["prompt"]]
    s["response"] = [rnd(t) for t in sample["response"]]
    s["tokens"] = s["prompt"] + s["response"]
    return s


def decode(ids: List[int]) -> str:
    return " ".join(ITOS[i] for i in ids)
