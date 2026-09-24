"""End-to-end Voxtral pipeline: encoder -> adapter -> language decoder.

    log-Mel  ->  Whisper encoder (50 Hz)  ->  adapter (4x, 12.5 Hz)  ->  LLM

The adapter's 4-frame concatenation is the whole point of the reproduction: it
quarters the number of audio tokens the decoder must attend over, which is what
lets Voxtral fit ~40 minutes of audio into a 32k context window (§2.2).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .adapter import Adapter
from .decoder import LMDecoder
from .encoder import WhisperEncoder

PAD, BOS, EOS = 0, 1, 2


class Voxtral(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_mels: int = 128,
        d_encoder: int = 64,
        d_llm: int = 64,
        enc_layers: int = 2,
        dec_layers: int = 2,
        num_heads: int = 4,
        d_ff: int = 128,
        adapter_stride: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoder = WhisperEncoder(
            n_mels=n_mels,
            d_model=d_encoder,
            num_layers=enc_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout,
        )
        self.adapter = Adapter(d_encoder, d_llm, stride=adapter_stride)
        self.decoder = LMDecoder(
            vocab_size,
            d_model=d_llm,
            num_layers=dec_layers,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout,
        )
        self.adapter_stride = adapter_stride

    def encode_audio(self, mel: torch.Tensor) -> torch.Tensor:
        """(B, n_mels, n_frames) -> (B, T_audio_12.5Hz, d_llm) prefix embeddings."""
        return self.adapter(self.encoder(mel))

    def forward(self, mel: torch.Tensor, text_in: torch.Tensor) -> torch.Tensor:
        """Return logits over the text positions only.

        text_in is [BOS, t1, ..., tN]; the returned logits predict the next
        token at each text position (targets are [t1, ..., tN, EOS]).
        """
        audio = self.encode_audio(mel)
        text = self.decoder.embed_tokens(text_in)
        seq = torch.cat([audio, text], dim=1)
        logits = self.decoder(seq)
        return logits[:, audio.size(1) :]  # keep only text-position predictions

    @torch.no_grad()
    def transcribe(self, mel: torch.Tensor, max_len: int = 32) -> list[int]:
        """Greedy-decode a single audio clip into a list of token ids."""
        self.eval()
        audio = self.encode_audio(mel)
        tokens = [BOS]
        for _ in range(max_len):
            text = self.decoder.embed_tokens(
                torch.tensor([tokens], device=mel.device)
            )
            seq = torch.cat([audio, text], dim=1)
            logits = self.decoder(seq)
            nxt = int(logits[0, -1].argmax())
            if nxt == EOS:
                break
            tokens.append(nxt)
        return tokens[1:]  # drop BOS
