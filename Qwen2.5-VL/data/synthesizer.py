"""NumPy synthetic 8×8 images and normalized boxes, with no image libraries."""
from __future__ import annotations
import numpy as np

COLORS = np.array([[.95,.25,.35],[.25,.8,.98],[.55,.95,.35],[.98,.68,.2]], np.float32)

def make_batch(batch_size: int, rng: np.random.Generator):
    images = rng.normal(.035, .008, (batch_size, 3, 8, 8)).astype(np.float32)
    labels = rng.integers(0, 4, batch_size, dtype=np.int64)
    cells = rng.integers(0, 16, batch_size, dtype=np.int64)
    boxes = np.empty((batch_size, 4), np.float32)
    for i, (label, cell) in enumerate(zip(labels, cells)):
        row, col = divmod(cell, 4)
        y0, x0 = row * 2, col * 2
        images[i, :, y0:y0+2, x0:x0+2] = COLORS[label, :, None, None]
        boxes[i] = [x0 / 8, y0 / 8, (x0 + 2) / 8, (y0 + 2) / 8]
    return images.clip(0, 1), labels, boxes

def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    left, top = np.maximum(a[:,0], b[:,0]), np.maximum(a[:,1], b[:,1])
    right, bottom = np.minimum(a[:,2], b[:,2]), np.minimum(a[:,3], b[:,3])
    inter = np.maximum(0, right-left) * np.maximum(0, bottom-top)
    area = (a[:,2]-a[:,0])*(a[:,3]-a[:,1]) + (b[:,2]-b[:,0])*(b[:,3]-b[:,1])-inter
    return inter / np.maximum(area, 1e-8)
