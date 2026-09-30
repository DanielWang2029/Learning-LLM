"""Split residual vector quantization (Moshi tech report, §3.3.2).

The paper's key twist on SoundStream/EnCodec-style RVQ:

    "Rather than a single RVQ with 8 levels, we distill semantic information into
    a plain VQ and apply an RVQ with 7 levels in parallel. We sum their outputs,
    such that while both can be used for reconstruction, we remove the constraint
    that acoustic information should be conserved in the residual of the semantic
    quantizer."

So the bottleneck is:

    z  ->  semantic VQ (1 codebook, distilled to a semantic target)   = q_sem
       ->  acoustic RVQ (K residual codebooks, carry acoustic detail)  = q_ac
    z_q = q_sem + q_ac                     (both feed the decoder)

Each codebook is a straight-through VQ with an EMA-updated codebook (the standard
SoundStream / VQ-VAE-2 recipe), which trains stably on CPU without tuning a
codebook-loss weight.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizerEMA(nn.Module):
    """Single VQ codebook with exponential-moving-average updates.

    Forward returns the quantized vectors (with a straight-through gradient to
    the encoder), the integer codes, and the commitment loss. Codebook entries
    are updated with EMA statistics during training rather than by gradient.
    """

    def __init__(self, dim: int, codebook_size: int, decay: float = 0.99) -> None:
        super().__init__()
        self.dim = dim
        self.codebook_size = codebook_size
        self.decay = decay
        self.eps = 1e-5
        self.register_buffer("codebook", torch.randn(codebook_size, dim) * 0.1)
        self.register_buffer("cluster_size", torch.zeros(codebook_size))
        self.register_buffer("embed_avg", self.codebook.clone())
        self._inited = False

    @torch.no_grad()
    def _maybe_init(self, flat: torch.Tensor) -> None:
        """k-means-lite: seed the codebook from the first batch of inputs."""
        if self._inited or not self.training:
            return
        n = flat.shape[0]
        idx = torch.randint(0, n, (self.codebook_size,), device=flat.device)
        self.codebook.copy_(flat[idx])
        self.embed_avg.copy_(self.codebook)
        self.cluster_size.fill_(1.0)
        self._inited = True

    def _distances(self, flat: torch.Tensor) -> torch.Tensor:
        # ||x - e||^2 = ||x||^2 - 2 x.e + ||e||^2
        return (
            flat.pow(2).sum(1, keepdim=True)
            - 2 * flat @ self.codebook.t()
            + self.codebook.pow(2).sum(1)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """x: (B, D, T) -> (quantized (B, D, T), codes (B, T), commit loss)."""
        b, d, t = x.shape
        flat = x.permute(0, 2, 1).reshape(-1, d)  # (B*T, D)
        self._maybe_init(flat)

        codes_flat = self._distances(flat).argmin(dim=1)  # (B*T,)
        quant_flat = F.embedding(codes_flat, self.codebook)  # (B*T, D)

        if self.training:
            self._ema_update(flat, codes_flat)

        commit = F.mse_loss(quant_flat, flat.detach())
        # Straight-through: gradients flow to the encoder unchanged.
        quant_flat = flat + (quant_flat - flat).detach()

        quant = quant_flat.view(b, t, d).permute(0, 2, 1)
        codes = codes_flat.view(b, t)
        return quant, codes, commit

    @torch.no_grad()
    def _ema_update(self, flat: torch.Tensor, codes: torch.Tensor) -> None:
        onehot = F.one_hot(codes, self.codebook_size).type(flat.dtype)  # (N, K)
        counts = onehot.sum(0)  # (K,)
        embed_sum = onehot.t() @ flat  # (K, D)
        self.cluster_size.mul_(self.decay).add_(counts, alpha=1 - self.decay)
        self.embed_avg.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
        n = self.cluster_size.sum()
        smoothed = (self.cluster_size + self.eps) / (n + self.codebook_size * self.eps) * n
        self.codebook.copy_(self.embed_avg / smoothed.unsqueeze(1))

    @torch.no_grad()
    def lookup(self, codes: torch.Tensor) -> torch.Tensor:
        """codes (B, T) -> quantized vectors (B, D, T)."""
        return F.embedding(codes, self.codebook).permute(0, 2, 1)


class ResidualVQ(nn.Module):
    """RVQ: iteratively quantize the residual (Zeghidour et al., 2022)."""

    def __init__(self, dim: int, num_levels: int, codebook_size: int) -> None:
        super().__init__()
        self.levels = nn.ModuleList(
            VectorQuantizerEMA(dim, codebook_size) for _ in range(num_levels)
        )

    def forward(
        self, x: torch.Tensor, n_active: int | None = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        """Quantize x with the first ``n_active`` levels (default: all).

        Returns (summed quantized, list of per-level codes, commitment loss).
        """
        if n_active is None:
            n_active = len(self.levels)
        residual = x
        quant_sum = torch.zeros_like(x)
        codes: List[torch.Tensor] = []
        commit = x.new_zeros(())
        for i, vq in enumerate(self.levels):
            if i >= n_active:
                break
            q, c, l = vq(residual)
            residual = residual - q
            quant_sum = quant_sum + q
            codes.append(c)
            commit = commit + l
        return quant_sum, codes, commit


class SplitRVQ(nn.Module):
    """Split RVQ bottleneck: 1 semantic VQ ∥ K acoustic RVQ levels (§3.3.2).

    ``project_in`` maps the encoder latent (dim D) down to ``rvq_dim`` before
    quantization and ``project_out`` maps the summed codes back to D — mirroring
    the paper's 512->256->512 projections. The semantic and acoustic quantizers
    both read the *same* projected latent (they run in parallel) and their codes
    are summed, so the acoustic path is never forced to spend bits preserving the
    semantic residual.
    """

    def __init__(
        self,
        latent_dim: int,
        rvq_dim: int,
        num_acoustic: int,
        codebook_size: int,
    ) -> None:
        super().__init__()
        self.num_acoustic = num_acoustic
        self.codebook_size = codebook_size
        self.project_in = nn.Conv1d(latent_dim, rvq_dim, 1)
        self.project_out = nn.Conv1d(rvq_dim, latent_dim, 1)
        self.semantic = VectorQuantizerEMA(rvq_dim, codebook_size)
        self.acoustic = ResidualVQ(rvq_dim, num_acoustic, codebook_size)

    def forward(self, latent: torch.Tensor, n_acoustic: int | None = None):
        """latent (B, D, T) -> dict with quantized latent, codes, losses.

        ``n_acoustic`` selects how many acoustic levels to use (bitrate scaling).
        """
        z = self.project_in(latent)
        q_sem, sem_codes, sem_commit = self.semantic(z)
        q_ac, ac_codes, ac_commit = self.acoustic(z, n_active=n_acoustic)
        z_q = q_sem + q_ac
        z_out = self.project_out(z_q)
        return {
            "z_q": z_out,             # quantized latent for the decoder (B, D, T)
            "semantic_embed": q_sem,  # semantic code vectors (for distillation)
            "semantic_codes": sem_codes,       # (B, T)
            "acoustic_codes": ac_codes,        # list of (B, T)
            "commit": sem_commit + ac_commit,
        }
