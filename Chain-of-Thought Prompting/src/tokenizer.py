"""Character-level tokenizer for the arithmetic language.

The whole demo lives in a tiny fixed alphabet, so a character tokenizer is all
we need. Index 0 is reserved as a padding token that never appears inside a
real string.
"""

from __future__ import annotations

# Every character the DIRECT and CHAIN-OF-THOUGHT strings can contain.
#   digits, '+', '=', carry marker 'c', step separator '|',
#   answer marker '#', end-of-sequence '$'.
VOCAB_CHARS = "0123456789+=c|#$"

PAD = 0  # reserved padding id (no character maps to it)


class CharVocab:
    def __init__(self, chars: str = VOCAB_CHARS) -> None:
        # itos[0] is the pad slot; real characters start at index 1.
        self.itos = ["\u0000"] + list(chars)
        self.stoi = {c: i for i, c in enumerate(self.itos)}

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids if i != PAD)

    @property
    def eos_id(self) -> int:
        return self.stoi["$"]
