"""A faithful, minimal reproduction of Qwen-Audio-3.1-Realtime's foundation
training (arXiv:2609.25176, §3): Core-Cocktail SFT + M2-OPD on-policy
distillation with a Text Teacher and a frozen Audio Reference.

- `AudioStudent`            the deployable audio policy pi_theta(. | x_a).
- `text_teacher_targets`    the Text Teacher oracle q_T, scoring the student's
                            own trajectory from clean text x_t.
- `sft_step` / `opd_step`   Core-Cocktail SFT, Multimodality OPD, and M2-OPD.
- `data`                    the synthetic audio-conditioned task (text skill +
                            audio-only attribute).
"""

from .model import AudioStudent, text_teacher_targets
from .opd import sft_step, opd_step, freeze_reference
from . import data

__all__ = [
    "AudioStudent",
    "text_teacher_targets",
    "sft_step",
    "opd_step",
    "freeze_reference",
    "data",
]
