"""Synthetic multilingual data for the language-ID prompt demo.

Two synthetic "languages" reuse the SAME acoustic phone templates but map them to
different token sets. Therefore the identical audio has two valid transcriptions,
and only the language-ID prompt disambiguates which — exactly the prompt
conditioning the Nemotron 3.5 ASR model card describes ("language-specific
transcription from a single ASR model through language ID conditioning").
"""

from __future__ import annotations

import numpy as np

from .audio import NUM_PHONES, features, synth_utterance

NUM_LANGS = 2
LANG_NAMES = ["lang-A", "lang-B"]

BLANK = 0
# token id = lang * NUM_PHONES + phone + 1   (blank reserved at 0)
VOCAB = NUM_LANGS * NUM_PHONES + 1


def token_id(lang, phone):
    return lang * NUM_PHONES + phone + 1


def token_str(tok):
    if tok == BLANK:
        return "_"
    tok -= 1
    lang, phone = tok // NUM_PHONES, tok % NUM_PHONES
    return f"{LANG_NAMES[lang][-1]}{phone}"     # e.g. A0, B3


def gen_utterance(rng, n_phones, lang=None, noise=2.0):
    phones = [int(x) for x in rng.integers(0, NUM_PHONES, size=n_phones)]
    if lang is None:
        lang = int(rng.integers(0, NUM_LANGS))
    wav, nf = synth_utterance(phones, rng)
    feats = features(wav, nf)
    feats = feats + noise * rng.standard_normal(feats.shape).astype(np.float32)
    target = [token_id(lang, p) for p in phones]
    return feats, lang, phones, target


def make_dataset(n, rng, n_phones, noise=2.0):
    return [gen_utterance(rng, n_phones, noise=noise) for _ in range(n)]
