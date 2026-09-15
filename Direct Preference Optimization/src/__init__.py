"""Direct Preference Optimization (Rafailov et al., 2023), in miniature.

- `data`   the toy "sort the tokens" preference task.
- `model`  `TinyLM`, the shared architecture for π_ref (frozen) and π_θ.
- `dpo`    the DPO loss and implicit-reward computation (paper Eq. 7).
"""

from . import data, dpo
from .model import TinyLM

__all__ = ["data", "dpo", "TinyLM"]
