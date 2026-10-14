"""A small compositional retrieval corpus expressed as integer tokens."""
import numpy as np

TOKENS = {"pad": 0, "color": 1, "shape": 2, "red": 3, "blue": 4,
          "green": 5, "circle": 6, "square": 7, "triangle": 8}
# One candidate per attribute value makes an instruction-selected answer unique.
# Every non-target candidate is still a hard semantic negative (it matches one
# plausible attribute but not the instruction-selected one).
DOCS = [("red", "square"), ("blue", "circle"), ("green", "triangle")]


def ids(words):
    return np.asarray([TOKENS[w] for w in words], dtype=np.int64)


def target_for(instruction, color, shape):
    """The instruction chooses which attribute determines the relevant document."""
    for i, (c, s) in enumerate(DOCS):
        if instruction == "color" and c == color:
            return i
        if instruction == "shape" and s == shape:
            return i
    raise ValueError("corpus needs an alternate matching document")


def batch(size, seed):
    rng = np.random.default_rng(seed)
    choices = []
    colors, shapes = ["red", "blue", "green"], ["circle", "square", "triangle"]
    for _ in range(size):
        inst = ("color", "shape")[rng.integers(2)]
        c, s = colors[rng.integers(3)], shapes[rng.integers(3)]
        choices.append((inst, c, s, target_for(inst, c, s)))
    return choices


def query_ids(inst, color, shape):
    return ids([inst, color, shape])


def document_ids(inst, index):
    color, shape = DOCS[index]
    return ids([inst, color, shape])
