"""NumPy-only synthetic prompt/image pairs: colored squares and crosses."""
import numpy as np

PROMPTS = [("red", "square"), ("red", "cross"), ("cyan", "square"), ("cyan", "cross")]
COLORS = {"red": np.array([0.96, 0.12, 0.20]), "cyan": np.array([0.08, 0.82, 0.92])}


def image_for(condition: int, size: int = 16) -> np.ndarray:
    color_name, shape = PROMPTS[condition]
    canvas = np.full((3, size, size), 0.035, dtype=np.float32)
    color = COLORS[color_name]
    if shape == "square":
        mask = np.zeros((size, size), bool); mask[4:12, 4:12] = True
    else:
        mask = np.zeros((size, size), bool); mask[6:10, 3:13] = True; mask[3:13, 6:10] = True
    canvas[:, mask] = color[:, None]
    return canvas


def batch(count: int, seed: int):
    rng = np.random.default_rng(seed)
    conditions = rng.integers(0, len(PROMPTS), count, dtype=np.int64)
    images = np.stack([image_for(int(c)) for c in conditions])
    # Small independent noise prevents a purely memorized scalar output from hiding the objective.
    images = np.clip(images + rng.normal(0, 0.01, images.shape), 0, 1).astype(np.float32)
    return images, conditions


def classify(image: np.ndarray) -> int:
    """An explicit attribute checker used only to evaluate generated pixels."""
    red = image[0].mean() > image[2].mean()
    foreground = image.mean(0) > 0.28
    horizontal = foreground[6:10].sum() > foreground[4:12, 4:12].sum() * 0.62
    shape = "cross" if horizontal and foreground.sum() < 75 else "square"
    return PROMPTS.index(("red" if red else "cyan", shape))
