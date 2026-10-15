"""The three foundation-training objectives (§3.1-3.3).

  1. Core-Cocktail SFT  (off-policy)  -- an initial audio policy, trained with
     teacher-forcing on GOLD states. Produces the model that also becomes the
     frozen Audio Reference.
  2. Multimodality OPD  (on-policy, Text Teacher only) -- distils the Text
     Teacher onto the student's OWN sampled trajectories. The teacher cannot
     supervise the audio-only STYLE, so its style target is uniform.
  3. M2-OPD            (on-policy, Text Teacher + frozen Audio Reference) -- same
     state supervision, but the audio-only STYLE is supervised by the frozen
     Audio Reference q_ref, preserving information beyond the transcript (§3.2).

All on-policy losses score the student on the states it actually visits.
"""

from __future__ import annotations

import copy

import torch
import torch.nn.functional as F

from .data import OUT_VOCAB, BASE, V
from .model import AudioStudent, text_teacher_targets


def _gold_states_prev(target: torch.Tensor) -> torch.Tensor:
    """states fed at each sum position under teacher forcing: BASE then gold s_{j-1}."""
    B, Lp1 = target.shape
    L = Lp1 - 1
    sp = torch.full((B, L), BASE, dtype=torch.long, device=target.device)
    sp[:, 1:] = target[:, 1:L]            # gold s_1..s_{L-1} as previous states
    return sp


def sft_step(student, batch, opt) -> float:
    """Core-Cocktail SFT: off-policy teacher forcing on gold states + styles."""
    feats = batch["feats"]; target = batch["target"]; style_tok = batch["style_token"]
    h = student.encode(feats)
    style_logits = student.style_logits(h)
    loss_style = F.cross_entropy(style_logits, style_tok)
    states_prev = _gold_states_prev(target)
    sum_logits = student.sum_logits(h, states_prev)       # (B,L,OUT_VOCAB)
    gold_sums = target[:, 1:]
    loss_sum = F.cross_entropy(sum_logits.reshape(-1, OUT_VOCAB), gold_sums.reshape(-1))
    loss = loss_style + loss_sum
    opt.zero_grad(); loss.backward(); opt.step()
    return float(loss.detach())


def opd_step(student, reference, batch, opt, use_audio_reference: bool, temp: float = 1.0) -> float:
    """On-policy distillation on the student's sampled trajectory.

    use_audio_reference=False -> Multimodality OPD (Text Teacher only; style
    target is uniform). True -> M2-OPD (style target = frozen Audio Reference).
    """
    feats = batch["feats"]; content = batch["content"]
    h = student.encode(feats)

    # on-policy: sample the student's own state trajectory (no grad through sampling)
    states_prev, _ = student.rollout(h, sample=True, temp=temp)

    # --- state supervision from the Text Teacher (scores the student's states) ---
    tgt_states = text_teacher_targets(content, states_prev)       # (B,L) ids
    sum_logits = student.sum_logits(h, states_prev)
    loss_sum = F.cross_entropy(sum_logits.reshape(-1, OUT_VOCAB), tgt_states.reshape(-1))

    # --- style supervision ---
    style_logits = student.style_logits(h)
    log_p = F.log_softmax(style_logits, dim=-1)
    if use_audio_reference:
        with torch.no_grad():                                     # q_ref from frozen reference
            q_ref = F.softmax(reference.style_logits(reference.encode(feats)), dim=-1)
        loss_style = F.kl_div(log_p, q_ref, reduction="batchmean")
    else:
        # Text Teacher has no audio-only signal -> uniform target (forgets style)
        loss_style = -(log_p.mean(dim=-1)).mean()

    loss = loss_sum + loss_style
    opt.zero_grad(); loss.backward(); opt.step()
    return float(loss.detach())


def freeze_reference(student: AudioStudent) -> AudioStudent:
    ref = copy.deepcopy(student)
    for p in ref.parameters():
        p.requires_grad_(False)
    ref.eval()
    return ref
