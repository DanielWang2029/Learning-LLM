"""A transparent, training-free streaming recognizer.

The Confucius4-R2T2 model card describes a *decoding paradigm* (Longest Stable
Prefix, append-only) that wraps an underlying ASR model. To study that paradigm
in isolation we use a fully transparent recognizer instead of a large neural
net: a nearest-template (matched-filter) classifier over our synthetic phones.

What matters for reproducing the paradigm is that the recognizer behaves like a
real streaming model at the trailing edge: a phone that has only just started is
frequently mis-recognized (its identifying formant has not arrived yet) and then
corrected once more audio is seen. Those trailing corrections are exactly what
an append-only decoder must avoid exposing to the user.
"""

from __future__ import annotations

import numpy as np

from .audio import FRAMES_PER_PHONE, PHONE_NAMES, log_mel, phone_templates


class MatchedFilterRecognizer:
    """Decode the audio observed *so far* into a phone-token hypothesis."""

    def __init__(self) -> None:
        self.templates = phone_templates()  # (num_phones, n_mels)
        # Normalise templates once for cosine matching.
        self._tn = self.templates / (np.linalg.norm(self.templates, axis=1, keepdims=True) + 1e-8)

    def hypothesize(self, wav_so_far: np.ndarray) -> list[int]:
        """Return the phone ids decoded from the currently available audio.

        A phone becomes hypothesizable as soon as at least one of its frames has
        arrived; partially observed phones are classified from whatever frames
        exist, which is why the last token can be wrong until the phone finishes.
        """
        if len(wav_so_far) < FRAMES_PER_PHONE:
            return []
        feats = log_mel(wav_so_far)
        n_frames = feats.shape[0]
        n_phones = int(np.ceil(n_frames / FRAMES_PER_PHONE))
        hyp = []
        for p in range(n_phones):
            lo = p * FRAMES_PER_PHONE
            hi = min((p + 1) * FRAMES_PER_PHONE, n_frames)
            if hi - lo < 1:
                break
            v = feats[lo:hi].mean(axis=0)
            v = v / (np.linalg.norm(v) + 1e-8)
            hyp.append(int(np.argmax(self._tn @ v)))
        return hyp


def tokens_to_text(token_ids) -> str:
    return " ".join(PHONE_NAMES[t] for t in token_ids)
