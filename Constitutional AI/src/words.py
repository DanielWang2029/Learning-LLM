"""The toy vocabulary and tokenizer.

Constitutional AI operates on natural-language responses. To keep the demo
fully reproducible on CPU we use a tiny word vocabulary split into "safe" and
"toxic" words, plus a single apologetic revision token. Responses are lists of
these words; the tokenizer maps them to integer ids for the preference model.
"""

from __future__ import annotations

from typing import List

# Words the assistant may safely use.
SAFE = [
    "hi", "hello", "thanks", "please", "sure", "okay",
    "yes", "help", "glad", "happy", "sorry", "here",
]
# Words the constitution forbids (harmful / rude).
TOXIC = ["stupid", "idiot", "hate", "dumb", "ugly"]

# The word a revision uses to replace harmful content.
REVISION_TOKEN = "sorry"

# Constitutional constraints referenced by the rules.
MAX_LEN = 6  # a response must be at most this many words (be concise)

PAD = "<pad>"
VOCAB: List[str] = [PAD] + SAFE + TOXIC
STOI = {w: i for i, w in enumerate(VOCAB)}
VOCAB_SIZE = len(VOCAB)
TOXIC_SET = set(TOXIC)


def encode(words: List[str], max_len: int) -> List[int]:
    """Words -> padded id list of length `max_len`."""
    ids = [STOI[w] for w in words[:max_len]]
    ids += [STOI[PAD]] * (max_len - len(ids))
    return ids
