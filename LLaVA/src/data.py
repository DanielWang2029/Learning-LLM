"""Synthetic vision-language data: colored shapes + instruction/answer triples.

LLaVA is trained on (image, instruction, answer) triples. We can't ship a real
multimodal corpus, so we *generate* one with NumPy: small RGB images each
containing exactly one shape of one color, paired with a natural-ish
instruction ("what color is the shape?") and the ground-truth answer token.

Everything is a fixed, tiny vocabulary so the whole pipeline stays CPU-friendly
and fully inspectable.
"""

from __future__ import annotations

import numpy as np

SHAPES = ["circle", "square", "triangle"]
COLORS = {
    "red": (1.0, 0.15, 0.15),
    "green": (0.15, 0.85, 0.25),
    "blue": (0.2, 0.35, 1.0),
    "yellow": (1.0, 0.85, 0.1),
}
COLOR_NAMES = list(COLORS)

# --- Token vocabulary shared by the language model -------------------------
# instruction tokens + answer tokens (colors and shapes).
SPECIAL = ["<pad>", "<img>", "ask_color", "ask_shape"]
ANSWER_TOKENS = COLOR_NAMES + SHAPES
VOCAB = SPECIAL + ANSWER_TOKENS
STOI = {t: i for i, t in enumerate(VOCAB)}
ITOS = {i: t for t, i in STOI.items()}


def _draw_shape(img: np.ndarray, shape: str, rgb, rng: np.random.Generator) -> None:
    S = img.shape[0]
    r = rng.integers(S // 4, S // 3)                       # radius / half-size
    cy = rng.integers(r + 1, S - r - 1)
    cx = rng.integers(r + 1, S - r - 1)
    ys, xs = np.mgrid[0:S, 0:S]
    if shape == "circle":
        mask = (ys - cy) ** 2 + (xs - cx) ** 2 <= r * r
    elif shape == "square":
        mask = (np.abs(ys - cy) <= r) & (np.abs(xs - cx) <= r)
    else:  # triangle (pointing up)
        dy = ys - (cy - r)
        halfw = (dy / (2 * r) * r).astype(int)
        mask = (dy >= 0) & (dy <= 2 * r) & (np.abs(xs - cx) <= np.maximum(halfw, 0))
    for c in range(3):
        img[..., c][mask] = rgb[c]


def make_image(shape: str, color: str, size: int, rng: np.random.Generator):
    img = np.zeros((size, size, 3), dtype=np.float32)
    _draw_shape(img, shape, COLORS[color], rng)
    # small amount of noise so the encoder can't cheat on exact pixels
    img += rng.normal(0, 0.02, img.shape).astype(np.float32)
    return np.clip(img, 0.0, 1.0)


def make_dataset(n: int, size: int, seed: int):
    """Return images (n,3,S,S), instruction ids (n,), answer ids (n,)."""
    rng = np.random.default_rng(seed)
    images, instr, answer = [], [], []
    for _ in range(n):
        shape = SHAPES[rng.integers(len(SHAPES))]
        color = COLOR_NAMES[rng.integers(len(COLOR_NAMES))]
        img = make_image(shape, color, size, rng)
        images.append(img.transpose(2, 0, 1))  # HWC -> CHW
        if rng.random() < 0.5:
            instr.append(STOI["ask_color"]); answer.append(STOI[color])
        else:
            instr.append(STOI["ask_shape"]); answer.append(STOI[shape])
    return (
        np.stack(images).astype(np.float32),
        np.array(instr, dtype=np.int64),
        np.array(answer, dtype=np.int64),
    )
