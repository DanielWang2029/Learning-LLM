"""Constitutional AI (Bai et al., 2022), in miniature.

- `words`             the toy vocabulary + tokenizer.
- `constitution`      explicit rules, critique, and revision (the SL-CAI stage).
- `generator`         a toy model that sometimes violates the rules.
- `preference_model`  the RLAIF preference model trained on AI feedback.
"""

from . import words, constitution, generator, preference_model

__all__ = ["words", "constitution", "generator", "preference_model"]
