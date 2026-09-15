"""Synthetic (image, caption) dataset generator (numpy only).

CLIP trains on (image, text) pairs scraped from the web. We cannot download
anything, so we synthesize a tiny controllable analogue: small RGB images that
each contain one colored geometric shape, paired with a natural-language caption
built from the color + shape labels ("a photo of a red circle").

The dataset is deliberately compositional — 4 colors x 3 shapes = 12 classes —
so that after contrastive training the model can be probed *zero-shot*: given a
held-out image, pick the caption whose text embedding is closest.

Everything here is pure numpy; PyTorch only enters in `clip/`.
"""

from __future__ import annotations

import numpy as np

IMG_SIZE = 24
CHANNELS = 3

COLORS = {
    "red": (0.90, 0.15, 0.15),
    "green": (0.15, 0.80, 0.25),
    "blue": (0.20, 0.35, 0.95),
    "yellow": (0.95, 0.85, 0.10),
}
SHAPES = ["circle", "square", "triangle"]

COLOR_NAMES = list(COLORS.keys())


def caption_for(color: str, shape: str) -> str:
    """Human-readable caption, mirroring CLIP's "a photo of a {label}" prompts."""
    return f"a photo of a {color} {shape}"


def all_classes() -> list[tuple[str, str]]:
    """The 12 (color, shape) label combinations, in a fixed order."""
    return [(c, s) for c in COLOR_NAMES for s in SHAPES]


def _blank(rng: np.random.Generator) -> np.ndarray:
    # Dark, slightly noisy background so the task is not perfectly noiseless.
    bg = rng.uniform(0.02, 0.10, size=3)
    img = np.ones((IMG_SIZE, IMG_SIZE, CHANNELS), dtype=np.float32) * bg
    return img


def _draw_shape(img: np.ndarray, shape: str, color_rgb, rng: np.random.Generator):
    """Rasterize `shape` filled with `color_rgb` at a random position/size."""
    h = w = IMG_SIZE
    # Random center and radius, kept fully inside the frame.
    r = int(rng.integers(5, 8))
    cy = int(rng.integers(r + 1, h - r - 1))
    cx = int(rng.integers(r + 1, w - r - 1))
    yy, xx = np.mgrid[0:h, 0:w]
    color = np.array(color_rgb, dtype=np.float32)

    if shape == "circle":
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
    elif shape == "square":
        mask = (np.abs(yy - cy) <= r) & (np.abs(xx - cx) <= r)
    elif shape == "triangle":
        # Upward triangle: widening base toward the bottom.
        dy = yy - (cy - r)          # 0 at the apex, grows downward
        half = (dy / (2 * r)) * r   # allowed half-width at this row
        mask = (dy >= 0) & (dy <= 2 * r) & (np.abs(xx - cx) <= half)
    else:
        raise ValueError(f"unknown shape {shape!r}")

    img[mask] = color
    return img


def make_image(color: str, shape: str, rng: np.random.Generator) -> np.ndarray:
    """Return an (IMG_SIZE, IMG_SIZE, 3) float32 image in [0, 1]."""
    img = _blank(rng)
    _draw_shape(img, shape, COLORS[color], rng)
    # Mild per-pixel noise for realism / to avoid trivial memorization.
    img = np.clip(img + rng.normal(0, 0.02, img.shape).astype(np.float32), 0, 1)
    return img


def make_dataset(n_per_class: int, seed: int = 0):
    """Generate a balanced dataset over all 12 (color, shape) classes.

    Returns
        images : (N, 3, H, W) float32   (channels-first, ready for torch)
        labels : (N,) int64             index into all_classes()
    """
    rng = np.random.default_rng(seed)
    classes = all_classes()
    images, labels = [], []
    for _ in range(n_per_class):
        for idx, (color, shape) in enumerate(classes):
            img = make_image(color, shape, rng)
            images.append(img.transpose(2, 0, 1))  # HWC -> CHW
            labels.append(idx)
    images = np.stack(images).astype(np.float32)
    labels = np.array(labels, dtype=np.int64)
    # Shuffle so batches mix classes (important for the contrastive loss).
    perm = rng.permutation(len(labels))
    return images[perm], labels[perm]


if __name__ == "__main__":
    imgs, labs = make_dataset(2, seed=1)
    print("images", imgs.shape, imgs.dtype, "range", imgs.min(), imgs.max())
    print("labels", labs.shape, "unique", np.unique(labs))
    for i in range(3):
        c, s = all_classes()[labs[i]]
        print(f"  sample {i}: {caption_for(c, s)}")
