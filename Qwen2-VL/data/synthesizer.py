"""NumPy-only synthetic native-resolution images for the Qwen2-VL demo."""
from __future__ import annotations

import numpy as np


def make_batch(resolution: int, batch_size: int, rng: np.random.Generator):
    """Return RGB images and the quadrant of a colored square (0..3)."""
    images = np.full((batch_size, 3, resolution, resolution), 0.04, np.float32)
    labels = rng.integers(0, 4, size=batch_size, dtype=np.int64)
    patch = resolution // 4
    # Constant object appearance: only spatial position can solve the task.
    color = np.array([0.15, 0.95, 0.65], np.float32)
    for i, q in enumerate(labels):
        row, col = (q // 2) * 2 + 1, (q % 2) * 2 + 1
        y, x = row * patch - patch // 2, col * patch - patch // 2
        images[i, :, y:y + patch, x:x + patch] = color[:, None, None]
        images[i] += rng.normal(0, 0.008, images[i].shape).astype(np.float32)
    return images.clip(0, 1), labels
