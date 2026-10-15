"""Audio Student + Text-Teacher oracle for the M2-OPD recipe (§3).

* `AudioStudent` -- the deployable policy pi_theta(. | x_a). It reads audio
  features and produces (a) a STYLE prediction from a global acoustic summary
  (the audio-only attribute) and (b) an autoregressive STATE prediction
  s_j = f(audio_j, previous state) -- the text-representable skill. A frozen
  copy of this same module is the *Audio Reference* q_ref (§3.2).

* `text_teacher_targets` -- the *Text Teacher* oracle q_T. Reading the clean
  content x_t, it returns, for the student's CURRENT state at each position, the
  correct next state T[state, c_j]. It scores the student's own trajectory
  (on-policy); it never hands over a pre-written answer (§3.2).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .data import OUT_VOCAB, V, BASE, FEAT_DIM, T as TRANS


class AudioStudent(nn.Module):
    def __init__(self, d: int = 48) -> None:
        super().__init__()
        self.audio_enc = nn.Linear(FEAT_DIM, d)
        self.state_emb = nn.Embedding(V + 1, d)          # states 0..V-1 and BASE
        self.style_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, OUT_VOCAB))
        self.sum_head = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, OUT_VOCAB))
        self.d = d

    def encode(self, feats: torch.Tensor) -> torch.Tensor:
        return self.audio_enc(feats)                     # (B, L, d)

    def style_logits(self, h: torch.Tensor) -> torch.Tensor:
        return self.style_head(h.mean(dim=1))            # (B, OUT_VOCAB)

    def sum_logits(self, h: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """h (B,L,d); states (B,L) = previous state per position -> (B,L,OUT_VOCAB)."""
        se = self.state_emb(states)
        return self.sum_head(torch.cat([h, se], dim=-1))

    @torch.no_grad()
    def rollout(self, h: torch.Tensor, sample: bool = False, temp: float = 1.0):
        """Free-run the state chain from BASE; return (states_prev, emitted_sums).

        states_prev[:, j] is the state fed at position j (BASE at j=0, else the
        emitted sum at j-1) -- i.e. the states the student actually visits.
        """
        B, L, _ = h.shape
        dev = h.device
        states_prev = torch.empty(B, L, dtype=torch.long, device=dev)
        emitted = torch.empty(B, L, dtype=torch.long, device=dev)
        state = torch.full((B,), BASE, dtype=torch.long, device=dev)
        for j in range(L):
            states_prev[:, j] = state
            logits = self.sum_head(torch.cat([h[:, j], self.state_emb(state)], dim=-1))
            if sample:
                nxt = torch.multinomial((logits / temp).softmax(-1), 1).squeeze(1)
            else:
                nxt = logits.argmax(-1)
            emitted[:, j] = nxt
            state = nxt.clamp(max=V - 1)                 # non-state tokens -> clamp
        return states_prev, emitted


def _state_eff(states: torch.Tensor) -> torch.Tensor:
    """Map BASE (and any non-state token) to state 0 for the transition lookup."""
    return torch.where(states < V, states, torch.zeros_like(states))


def text_teacher_targets(content: torch.Tensor, states_prev: torch.Tensor) -> torch.Tensor:
    """q_T: correct next state T[state, c_j] for the student's current states.

    content (B,L), states_prev (B,L) -> target state ids (B,L) in {0..V-1}.
    """
    Tt = torch.as_tensor(TRANS, device=content.device)
    s = _state_eff(states_prev)
    return Tt[s, content]
