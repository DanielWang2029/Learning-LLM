"""Longest Stable Prefix (LSP) append-only streaming decoder.

This is the core mechanism of Confucius4-R2T2 ("R2T2 — Real Real-Time
Transcription"). From the model card:

    "The model operates in append-only output mode: committing transcript text
     permanently without revising previous words ... By exposing only stable
     prefixes, the model provides high-quality context that conditions
     subsequent predictions while guaranteeing that previously emitted text
     remains unchanged."

Given a stream of hypotheses (one per arriving audio chunk), LSP commits only
the prefix that is *stable* — i.e. agrees across the most recent hypotheses —
minus a small trailing guard (`unfixed_token_num`, the model card's rollback
window). Committed tokens are append-only and are never revised.

For contrast we also provide a NaiveDecoder that always shows the whole current
hypothesis, and therefore revises earlier tokens whenever the model changes its
mind — the flicker that LSP is designed to eliminate.
"""

from __future__ import annotations

from dataclasses import dataclass, field


def longest_common_prefix(seqs) -> list:
    if not seqs:
        return []
    out = []
    for col in zip(*seqs):
        first = col[0]
        if all(c == first for c in col):
            out.append(first)
        else:
            break
    return out


@dataclass
class Emission:
    """One committed token and when it was committed."""

    token: int
    chunk_index: int            # which streaming step committed it
    audio_end_frame: int        # frame at which this token's audio finished


@dataclass
class LSPDecoder:
    """Append-only decoder that emits the longest stable prefix.

    stability_window : how many consecutive hypotheses must agree on a token
                       before it is considered stable (>=2 means "unchanged
                       since the previous chunk").
    unfixed_token_num: trailing tokens held back even if they look stable
                       (the model card's rollback window).
    """

    stability_window: int = 2
    unfixed_token_num: int = 1
    committed: list = field(default_factory=list)      # committed token ids
    emissions: list = field(default_factory=list)      # list[Emission]
    revisions: int = 0
    _recent: list = field(default_factory=list)        # recent hypotheses

    def step(self, hypothesis, chunk_index, frames_per_token) -> list:
        """Feed a new hypothesis; return the tokens newly committed this step."""
        self._recent.append(list(hypothesis))
        if len(self._recent) > self.stability_window:
            self._recent.pop(0)

        stable = longest_common_prefix(self._recent)
        # Hold back the rollback window: never commit the last few stable tokens.
        commit_len = max(0, len(stable) - self.unfixed_token_num)
        commit_len = min(commit_len, len(hypothesis))

        newly = []
        while len(self.committed) < commit_len:
            idx = len(self.committed)
            tok = stable[idx]
            self.committed.append(tok)
            end_frame = (idx + 1) * frames_per_token
            self.emissions.append(Emission(tok, chunk_index, end_frame))
            newly.append(tok)
        return newly

    def finalize(self, hypothesis, chunk_index, frames_per_token) -> list:
        """Flush the remaining trailing tokens once the audio has ended."""
        newly = []
        while len(self.committed) < len(hypothesis):
            idx = len(self.committed)
            tok = hypothesis[idx]
            self.committed.append(tok)
            end_frame = (idx + 1) * frames_per_token
            self.emissions.append(Emission(tok, chunk_index, end_frame))
            newly.append(tok)
        return newly


@dataclass
class NaiveDecoder:
    """Baseline: re-display the whole current hypothesis every chunk.

    Any position whose token changes versus what was previously shown counts as
    a revision — the visible flicker R2T2 avoids.
    """

    displayed: list = field(default_factory=list)
    revisions: int = 0
    emissions: list = field(default_factory=list)   # first-display time per position

    def step(self, hypothesis, chunk_index, frames_per_token) -> None:
        for i, tok in enumerate(hypothesis):
            if i < len(self.displayed):
                if self.displayed[i] != tok:
                    self.revisions += 1
                    self.displayed[i] = tok
            else:
                self.displayed.append(tok)
                # A naive decoder shows a token the instant it appears (low
                # latency), but that token may later be revised (see revisions).
                self.emissions.append(Emission(tok, chunk_index, (i + 1) * frames_per_token))

    def finalize(self, hypothesis, chunk_index, frames_per_token) -> None:
        self.step(hypothesis, chunk_index, frames_per_token)
