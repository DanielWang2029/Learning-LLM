"""SEANet-style causal convolutional autoencoder (Moshi tech report, §3.3.1).

Mimi's encoder "projects a single-channel waveform to a latent representation by
cascading residual convolutional blocks that interleave dilated and strided
convolutions along with ELU non-linearities. All convolutions are causal, such
that this autoencoder can run in a streaming fashion." With striding factors
(4, 5, 6, 8) plus a final stride-2 conv, the 24 kHz waveform is projected to a
12.5 Hz latent sequence.

This is a *tiny*, CPU-friendly reproduction of exactly that structure:

* ``CausalConv1d``      — left-padded 1-D convolution + explicit streaming state,
                          so chunk-by-chunk inference equals the offline pass.
* ``CausalConvTranspose`` is avoided on purpose: the decoder upsamples by integer
  repetition followed by a causal conv, which keeps the *whole* pipeline
  streamable with one uniform state mechanism (still a causal conv decoder).
* ``MimiEncoder`` / ``MimiDecoder`` — the SEANet down/up-sampling stacks.

The point of the demo is the *mechanism* (causal streaming + a low frame rate),
not the trained 24 kHz weights, so channels and depth are deliberately small.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    """1-D convolution that only ever looks left (causal), with streaming state.

    Offline mode left-pads the input by ``(kernel - 1) * dilation`` and convolves.
    Streaming mode keeps a buffer of the samples not yet fully consumed; feeding
    contiguous chunks and carrying that buffer reconstructs the exact same padded
    stream, so ``streaming(chunks)`` == ``forward(full)`` to numerical precision
    for any chunking (paper §3.3.1: "run in a streaming fashion").
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            in_ch, out_ch, kernel_size, stride=stride, dilation=dilation
        )
        self.left_pad = (kernel_size - 1) * dilation
        self.stride = stride
        self._state: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Offline pass over a full sequence ``(B, C, T)``."""
        return self.conv(F.pad(x, (self.left_pad, 0)))

    def reset_stream(self, batch: int, device: torch.device) -> None:
        """Start a new stream: the buffer is the causal left-padding (zeros)."""
        self._state = torch.zeros(
            batch, self.conv.in_channels, self.left_pad, device=device
        )

    def step(self, x: torch.Tensor) -> torch.Tensor:
        """Streaming pass over one chunk, carrying the conv receptive field."""
        assert self._state is not None, "call reset_stream() before step()"
        cat = torch.cat([self._state, x], dim=-1)
        n_out = (cat.shape[-1] - self.left_pad - 1) // self.stride + 1
        n_out = max(n_out, 0)
        consumed = n_out * self.stride
        y = self.conv(cat) if n_out > 0 else cat[..., :0]
        # Keep everything the next chunk still needs to combine with.
        self._state = cat[..., consumed:]
        return y


class ResidualUnit(nn.Module):
    """Dilated causal residual block (SEANet building block, §3.3.1).

    A dilated conv widens the receptive field cheaply; the 1x1 conv mixes
    channels; ELU is the paper's non-linearity. Wrapped in a residual add.
    """

    def __init__(self, ch: int, dilation: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.conv1 = CausalConv1d(ch, ch, kernel_size, dilation=dilation)
        self.conv2 = CausalConv1d(ch, ch, 1)
        self.act = nn.ELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv2(self.act(self.conv1(self.act(x))))
        return x + y

    def reset_stream(self, batch: int, device: torch.device) -> None:
        self.conv1.reset_stream(batch, device)
        self.conv2.reset_stream(batch, device)

    def step(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv2.step(self.act(self.conv1.step(self.act(x))))
        return x + y


class EncoderBlock(nn.Module):
    """Residual units followed by a strided down-sampling conv (§3.3.1)."""

    def __init__(self, in_ch: int, out_ch: int, stride: int) -> None:
        super().__init__()
        self.res = ResidualUnit(in_ch, dilation=1)
        self.res2 = ResidualUnit(in_ch, dilation=3)
        self.act = nn.ELU()
        # Down-sample: kernel = 2*stride keeps a clean causal footprint.
        self.down = CausalConv1d(in_ch, out_ch, kernel_size=2 * stride, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(self.act(self.res2(self.res(x))))

    def reset_stream(self, batch: int, device: torch.device) -> None:
        self.res.reset_stream(batch, device)
        self.res2.reset_stream(batch, device)
        self.down.reset_stream(batch, device)

    def step(self, x: torch.Tensor) -> torch.Tensor:
        return self.down.step(self.act(self.res2.step(self.res.step(x))))


class DecoderBlock(nn.Module):
    """Upsample by integer repetition, then a causal conv (mirror of encoder).

    Using repeat + causal conv instead of a transposed conv keeps the decoder
    fully streamable with the *same* state mechanism as the encoder, while still
    being a causal convolutional decoder that reconstructs the waveform.
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int) -> None:
        super().__init__()
        self.stride = stride
        self.conv = CausalConv1d(in_ch, out_ch, kernel_size=2 * stride + 1)
        self.act = nn.ELU()
        self.res = ResidualUnit(out_ch, dilation=1)
        self.res2 = ResidualUnit(out_ch, dilation=3)

    @staticmethod
    def _upsample(x: torch.Tensor, stride: int) -> torch.Tensor:
        return x.repeat_interleave(stride, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(self._upsample(x, self.stride))
        return self.res2(self.res(self.act(y)))

    def reset_stream(self, batch: int, device: torch.device) -> None:
        self.conv.reset_stream(batch, device)
        self.res.reset_stream(batch, device)
        self.res2.reset_stream(batch, device)

    def step(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv.step(self._upsample(x, self.stride))
        return self.res2.step(self.res.step(self.act(y)))


class MimiEncoder(nn.Module):
    """Waveform (B,1,L) -> latent (B, D, S) at a low frame rate (§3.3.1)."""

    def __init__(
        self,
        strides: List[int],
        channels: int = 32,
        latent_dim: int = 64,
    ) -> None:
        super().__init__()
        self.strides = strides
        self.total_stride = 1
        for s in strides:
            self.total_stride *= s
        self.in_conv = CausalConv1d(1, channels, kernel_size=7)
        self.blocks = nn.ModuleList(
            EncoderBlock(channels, channels, s) for s in strides
        )
        self.act = nn.ELU()
        self.out_conv = CausalConv1d(channels, latent_dim, kernel_size=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_conv(x)
        for blk in self.blocks:
            x = blk(x)
        return self.out_conv(self.act(x))

    def reset_stream(self, batch: int, device: torch.device) -> None:
        self.in_conv.reset_stream(batch, device)
        for blk in self.blocks:
            blk.reset_stream(batch, device)
        self.out_conv.reset_stream(batch, device)

    def step(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_conv.step(x)
        for blk in self.blocks:
            x = blk.step(x)
        return self.out_conv.step(self.act(x))


class MimiDecoder(nn.Module):
    """Latent (B, D, S) -> reconstructed waveform (B,1,L) (§3.3.1)."""

    def __init__(
        self,
        strides: List[int],
        channels: int = 32,
        latent_dim: int = 64,
    ) -> None:
        super().__init__()
        self.in_conv = CausalConv1d(latent_dim, channels, kernel_size=3)
        self.act = nn.ELU()
        self.blocks = nn.ModuleList(
            DecoderBlock(channels, channels, s) for s in reversed(strides)
        )
        self.out_conv = CausalConv1d(channels, 1, kernel_size=7)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.in_conv(x))
        for blk in self.blocks:
            x = blk(x)
        return torch.tanh(self.out_conv(x))

    def reset_stream(self, batch: int, device: torch.device) -> None:
        self.in_conv.reset_stream(batch, device)
        for blk in self.blocks:
            blk.reset_stream(batch, device)
        self.out_conv.reset_stream(batch, device)

    def step(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.in_conv.step(x))
        for blk in self.blocks:
            x = blk.step(x)
        return torch.tanh(self.out_conv.step(x))
