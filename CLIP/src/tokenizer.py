"""A minimal word-level tokenizer for the synthetic captions.

CLIP uses a byte-pair-encoding tokenizer over web text. Our captions are drawn
from a tiny closed vocabulary ("a photo of a {color} {shape}"), so a whitespace
word tokenizer with a fixed vocab is faithful in spirit and keeps the demo tiny.
"""

from __future__ import annotations

from typing import List

PAD = "<pad>"

# Fixed vocabulary covering every token that can appear in a caption.
VOCAB = [
    PAD,
    "a", "photo", "of",
    "red", "green", "blue", "yellow",
    "circle", "square", "triangle",
]
STOI = {tok: i for i, tok in enumerate(VOCAB)}
VOCAB_SIZE = len(VOCAB)
PAD_ID = STOI[PAD]
MAX_LEN = 6  # "a photo of a red circle" -> 6 tokens


def encode(caption: str, max_len: int = MAX_LEN) -> List[int]:
    """Turn a caption string into a fixed-length list of token ids (PAD-padded)."""
    toks = caption.lower().split()
    ids = [STOI[t] for t in toks][:max_len]
    ids = ids + [PAD_ID] * (max_len - len(ids))
    return ids
