"""NumPy images with a tiny diagonal detail whose global color average is equal."""
from __future__ import annotations
import numpy as np

def make_batch(batch_size: int, rng: np.random.Generator):
    images = rng.normal(.06, .006, (batch_size, 3, 8, 8)).astype(np.float32)
    labels = rng.integers(0, 2, batch_size, dtype=np.int64)
    # Both classes paint exactly two equal red pixels: only the local diagonal differs.
    for i, label in enumerate(labels):
        # Fix the local region: the task asks for its fine arrangement, not
        # translation invariance. Both choices still have equal global means.
        y, x = 3, 3
        if label == 0:
            pixels = [(y, x), (y+1, x+1)]
        else:
            pixels = [(y, x+1), (y+1, x)]
        for py, px in pixels:
            images[i, :, py, px] = (.98, .22, .38)
    return images.clip(0, 1), labels
