"""wav2vec 2.0, from scratch (paper Sections 2 and 3).

Pipeline:

    raw waveform
      │  FeatureEncoder (multi-layer 1-D CNN, Section 2.1)
      ▼
    latents z_t  ──────────────► GumbelVectorQuantizer (Section 2.2)  ──► q_t
      │                                                                    (targets)
      │  mask a subset of time steps (replace with a learned mask vector)
      ▼
    ContextTransformer (Section 2.1)
      ▼
    context c_t
      │  contrastive InfoNCE over masked steps (Section 3.2):
      ▼  identify the true q_t among K distractors.

Everything is tiny and CPU-friendly. No torchaudio / librosa: the model sees the
raw numpy waveform directly, exactly as wav2vec 2.0 does.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureEncoder(nn.Module):
    """Multi-layer temporal CNN over the raw waveform (Section 2.1).

    Each layer is Conv1d -> GroupNorm -> GELU. Strides downsample the 16 kHz
    waveform to a ~200 Hz latent sequence (here stride product = 80).
    """

    def __init__(self, dim: int = 128) -> None:
        super().__init__()
        # (out_channels, kernel, stride) per layer.
        specs = [(64, 10, 5), (128, 8, 4), (dim, 4, 2), (dim, 4, 2)]
        layers, in_ch = [], 1
        for out_ch, k, s in specs:
            layers += [
                nn.Conv1d(in_ch, out_ch, k, s),
                nn.GroupNorm(1, out_ch),
                nn.GELU(),
            ]
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        """wave (B, samples) -> latents (B, T, dim)."""
        z = self.conv(wave.unsqueeze(1))  # (B, dim, T)
        return z.transpose(1, 2)


class GumbelVectorQuantizer(nn.Module):
    """Product quantization with a Gumbel-softmax codebook (Section 2.2).

    The latent is split into ``num_groups`` groups; each group picks one of
    ``num_vars`` codewords via a straight-through Gumbel-softmax. The chosen
    codewords are concatenated to form the discrete target q_t.
    """

    def __init__(self, dim: int, num_groups: int = 2, num_vars: int = 32, temp: float = 2.0) -> None:
        super().__init__()
        assert dim % num_groups == 0
        self.num_groups = num_groups
        self.num_vars = num_vars
        self.temp = temp
        self.var_dim = dim // num_groups
        self.proj = nn.Linear(dim, num_groups * num_vars)
        # Codebook: (1, num_groups * num_vars, var_dim).
        self.codebook = nn.Parameter(torch.randn(1, num_groups * num_vars, self.var_dim) * 0.1)

    def forward(self, z: torch.Tensor):
        B, T, _ = z.shape
        logits = self.proj(z).view(B * T * self.num_groups, self.num_vars)

        if self.training:
            probs = F.gumbel_softmax(logits, tau=self.temp, hard=True)
        else:
            idx = logits.argmax(-1, keepdim=True)
            probs = torch.zeros_like(logits).scatter_(-1, idx, 1.0)

        # Codebook usage / diversity (soft probs, for the perplexity metric).
        soft = torch.softmax(logits.view(B * T, self.num_groups, self.num_vars).float(), dim=-1)
        avg = soft.mean(0)                                   # (G, V)
        perplexity = torch.exp(-(avg * torch.log(avg + 1e-9)).sum(-1)).sum()

        vars_ = self.codebook.view(self.num_groups * self.num_vars, self.var_dim)
        vars_ = vars_.view(self.num_groups, self.num_vars, self.var_dim)
        probs_g = probs.view(B * T, self.num_groups, self.num_vars)
        # weighted sum over codewords per group, then concat groups.
        q = torch.einsum("ngv,gvd->ngd", probs_g, vars_).reshape(B, T, -1)

        used = (avg > (1.0 / self.num_vars) * 0.5).float().sum(-1).mean().item()
        return q, perplexity, used


class ContextTransformer(nn.Module):
    """Small Transformer context network (Section 2.1)."""

    def __init__(self, dim: int, num_layers: int = 2, num_heads: int = 4, ff: int = 256) -> None:
        super().__init__()
        self.pos = nn.Conv1d(dim, dim, kernel_size=15, padding=7, groups=16)  # conv positional emb
        layer = nn.TransformerEncoderLayer(
            dim, num_heads, ff, dropout=0.0, activation="gelu", batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pos(x.transpose(1, 2)).transpose(1, 2)
        return self.encoder(x)


def compute_mask_indices(
    T: int, mask_prob: float, mask_len: int, rng: np.random.Generator, min_masks: int = 8
) -> np.ndarray:
    """Sample span-mask start points and expand to a boolean frame mask (Section 3.1)."""
    mask = np.zeros(T, dtype=bool)
    num_spans = max(min_masks // mask_len, int(mask_prob * T / mask_len + rng.random()))
    starts = rng.choice(max(1, T - mask_len), size=num_spans, replace=False)
    for s in starts:
        mask[s : s + mask_len] = True
    if mask.sum() < min_masks:  # guarantee enough masked frames for distractors
        extra = rng.choice(np.where(~mask)[0], size=min(min_masks, (~mask).sum()), replace=False)
        mask[extra] = True
    return mask


class Wav2Vec2(nn.Module):
    """Full model: encoder + quantizer + context net + contrastive head."""

    def __init__(self, dim: int = 128, num_groups: int = 2, num_vars: int = 32) -> None:
        super().__init__()
        self.feature_encoder = FeatureEncoder(dim)
        self.layer_norm = nn.LayerNorm(dim)
        self.quantizer = GumbelVectorQuantizer(dim, num_groups, num_vars)
        self.mask_emb = nn.Parameter(torch.randn(dim) * 0.1)
        self.context = ContextTransformer(dim)
        self.project_ctx = nn.Linear(dim, dim)
        self.project_q = nn.Linear(dim, dim)
        self.kappa = 0.1  # contrastive temperature

    def forward(self, wave: torch.Tensor, masks: List[np.ndarray], rng, num_negatives: int = 20):
        z = self.layer_norm(self.feature_encoder(wave))       # (B, T, D)
        q, perplexity, used = self.quantizer(z)               # targets from *unmasked* z
        q = self.project_q(q)

        # Build masked transformer input (replace masked frames with mask_emb),
        # done functionally to keep autograd happy (no in-place edits of z).
        B, T, D = z.shape
        mask_t = torch.stack([torch.from_numpy(m) for m in masks]).bool()  # (B, T)
        mf = mask_t.unsqueeze(-1).float()                     # (B, T, 1)
        x = z * (1.0 - mf) + self.mask_emb.view(1, 1, D) * mf
        c = self.project_ctx(self.context(x))                 # (B, T, D)

        total_loss, correct, count = 0.0, 0, 0
        for b in range(B):
            idx = torch.where(mask_t[b])[0]
            if len(idx) < 2:
                continue
            c_b = c[b, idx]                                    # (m, D)
            q_pos = q[b, idx]                                  # (m, D)
            m = len(idx)
            # sample distractors from other masked positions in the same utterance
            neg = torch.empty(m, num_negatives, dtype=torch.long)
            for i in range(m):
                choices = torch.cat([idx[:i], idx[i + 1 :]])
                sel = choices[torch.randint(len(choices), (num_negatives,))]
                neg[i] = sel
            q_neg = q[b, neg]                                  # (m, K, D)
            cands = torch.cat([q_pos.unsqueeze(1), q_neg], dim=1)  # (m, K+1, D)
            sim = F.cosine_similarity(c_b.unsqueeze(1), cands, dim=-1) / self.kappa
            target = torch.zeros(m, dtype=torch.long)         # positive is index 0
            total_loss = total_loss + F.cross_entropy(sim, target)
            correct += (sim.argmax(-1) == 0).sum().item()
            count += m

        loss = total_loss / max(1, B)
        # Diversity loss encourages full codebook usage (Section 3.2).
        num_codewords = self.quantizer.num_groups * self.quantizer.num_vars
        diversity = (num_codewords - perplexity) / num_codewords
        loss = loss + 0.1 * diversity
        acc = correct / max(1, count)
        return loss, acc, perplexity.item(), used, mask_t
