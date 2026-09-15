"""Synthetic visual-question-answering data (numpy only).

Each example is (image, question, answer):
  * image    : a small RGB picture of one colored shape,
  * question : the fixed prompt "what color is the shape ?",
  * answer   : the color word (red / green / blue / yellow).

The answer is impossible to know from the text alone — a text-only LM can only
guess the marginal color (~25%). The information lives entirely in the image, so
the model must route it through the gated cross-attention layers to answer.

We also build the tiny token vocabulary used by the language model here.
"""

from __future__ import annotations

import numpy as np

IMG_SIZE = 24

COLORS = {
    "red": (0.90, 0.15, 0.15),
    "green": (0.15, 0.80, 0.25),
    "blue": (0.20, 0.35, 0.95),
    "yellow": (0.95, 0.85, 0.10),
}
COLOR_NAMES = list(COLORS.keys())
SHAPES = ["circle", "square", "triangle"]

# ---- Language: fixed prompt + color answers -------------------------------
PAD, BOS = "<pad>", "<bos>"
QUESTION = ["what", "color", "is", "the", "shape", "?"]
VOCAB = [PAD, BOS] + QUESTION + COLOR_NAMES
STOI = {t: i for i, t in enumerate(VOCAB)}
VOCAB_SIZE = len(VOCAB)
PAD_ID, BOS_ID = STOI[PAD], STOI[BOS]

# Sequence = <bos> what color is the shape ? <color>
PROMPT_IDS = [BOS_ID] + [STOI[w] for w in QUESTION]
SEQ_LEN = len(PROMPT_IDS) + 1        # + the answer token
ANSWER_POS = len(PROMPT_IDS) - 1     # position whose prediction is the answer


def make_sequence(color: str) -> list[int]:
    """Full token sequence for a given answer color."""
    return PROMPT_IDS + [STOI[color]]


def _draw(color: str, shape: str, rng: np.random.Generator) -> np.ndarray:
    bg = rng.uniform(0.02, 0.10, size=3)
    img = np.ones((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32) * bg
    yy, xx = np.mgrid[0:IMG_SIZE, 0:IMG_SIZE]
    r = int(rng.integers(5, 8))
    cy = int(rng.integers(r + 1, IMG_SIZE - r - 1))
    cx = int(rng.integers(r + 1, IMG_SIZE - r - 1))
    if shape == "circle":
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
    elif shape == "square":
        mask = (np.abs(yy - cy) <= r) & (np.abs(xx - cx) <= r)
    else:  # triangle
        dy = yy - (cy - r)
        half = (dy / (2 * r)) * r
        mask = (dy >= 0) & (dy <= 2 * r) & (np.abs(xx - cx) <= half)
    img[mask] = np.array(COLORS[color], dtype=np.float32)
    img = np.clip(img + rng.normal(0, 0.02, img.shape).astype(np.float32), 0, 1)
    return img.transpose(2, 0, 1)  # CHW


def make_dataset(n: int, seed: int = 0):
    """Return (images (n,3,H,W) float32, sequences (n,SEQ_LEN) int64, color_idx (n,))."""
    rng = np.random.default_rng(seed)
    imgs, seqs, colors = [], [], []
    for _ in range(n):
        ci = int(rng.integers(len(COLOR_NAMES)))
        color = COLOR_NAMES[ci]
        shape = SHAPES[int(rng.integers(len(SHAPES)))]
        imgs.append(_draw(color, shape, rng))
        seqs.append(make_sequence(color))
        colors.append(ci)
    return (
        np.stack(imgs).astype(np.float32),
        np.array(seqs, dtype=np.int64),
        np.array(colors, dtype=np.int64),
    )


if __name__ == "__main__":
    imgs, seqs, colors = make_dataset(4, seed=1)
    print("images", imgs.shape, "seqs", seqs.shape, "SEQ_LEN", SEQ_LEN)
    print("vocab", VOCAB)
    for i in range(4):
        toks = [VOCAB[t] for t in seqs[i]]
        print(f"  {toks}  (answer color = {COLOR_NAMES[colors[i]]})")
