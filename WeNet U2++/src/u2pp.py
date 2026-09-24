"""U2++: one shared encoder, a streaming CTC first pass, attention rescoring.

Architecture (U2++, §3.1):

    log-mel ─▶ shared Conformer encoder ─┬─▶ CTC head          (first pass, streaming)
                                         ├─▶ L2R attention dec  (second pass, rescore)
                                         └─▶ R2L attention dec  (second pass, rescore)

Trained jointly with a combined loss  L = λ·CTC + (1−λ)·(L2R + R2L)/2 under
dynamic chunk masking, so the *same* weights run offline or streaming. At
inference the CTC first pass proposes an n-best; the attention decoders rescore
it using full left/right label context.
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .conformer import ConformerEncoder
from .decoder import AttentionDecoder
from .search import BLANK, edit_distance, greedy_decode, prefix_beam_search

CTC_VOCAB = 5  # A, B, C, D, blank


class U2PP(nn.Module):
    def __init__(self, n_mels=40, d_model=64, num_heads=2, d_ff=128,
                 enc_blocks=3, dec_layers=2, kernel_size=9, dropout=0.1):
        super().__init__()
        self.encoder = ConformerEncoder(n_mels, d_model, num_heads, d_ff,
                                        enc_blocks, kernel_size, dropout)
        self.ctc_head = nn.Linear(d_model, CTC_VOCAB)
        self.l2r = AttentionDecoder(d_model, num_heads, d_ff, dec_layers,
                                    dropout, reverse=False)
        self.r2l = AttentionDecoder(d_model, num_heads, d_ff, dec_layers,
                                    dropout, reverse=True)

    def encode(self, feats, chunk_size=None):
        return self.encoder(feats, chunk_size)

    def ctc_log_probs(self, memory):
        return F.log_softmax(self.ctc_head(memory), dim=-1)

    # ---------------------------- training loss -----------------------------
    def loss(self, feats, tokens, token_lengths, chunk_size, ctc_weight=0.3):
        """Combined CTC + (L2R + R2L)/2 attention loss."""
        memory = self.encode(feats, chunk_size)
        log_probs = self.ctc_log_probs(memory)              # (B, T, V)
        B, T, _ = log_probs.shape
        ctc = nn.functional.ctc_loss(
            log_probs.transpose(0, 1), tokens,              # (T, B, V), (B, L)
            input_lengths=torch.full((B,), T, dtype=torch.long),
            target_lengths=token_lengths, blank=BLANK, zero_infinity=True,
        )
        att = 0.0
        for dec in (self.l2r, self.r2l):
            _, tgt, logp = dec(tokens, memory)
            ce = F.nll_loss(logp.reshape(-1, logp.size(-1)), tgt.reshape(-1),
                            reduction="mean")
            att = att + ce
        att = att / 2.0
        return ctc_weight * ctc + (1 - ctc_weight) * att, ctc.item(), att.item()

    # ------------------------------ decoding --------------------------------
    @torch.no_grad()
    def ctc_greedy(self, feats, chunk_size=None) -> List[List[int]]:
        memory = self.encode(feats, chunk_size)
        lp = self.ctc_log_probs(memory).cpu().numpy()
        return [greedy_decode(lp[b]) for b in range(lp.shape[0])]

    @torch.no_grad()
    def rescore(self, feats, chunk_size=None, beam_size=8, n_best=4,
                ctc_weight=0.3, att_weight=1.0):
        """CTC prefix-beam n-best, rescored by the two attention decoders.

        Returns (greedy_hyps, rescored_hyps) so the caller can compare the pure
        streaming first pass against the second-pass result on the SAME encoder
        output.
        """
        memory = self.encode(feats, chunk_size)             # (B, T, d)
        lp = self.ctc_log_probs(memory).cpu().numpy()
        B = lp.shape[0]
        greedy_hyps, rescored_hyps = [], []
        for b in range(B):
            greedy_hyps.append(greedy_decode(lp[b]))
            nbest = prefix_beam_search(lp[b], beam_size, n_best)
            if not nbest:
                rescored_hyps.append([])
                continue
            mem_b = memory[b : b + 1]
            best_seq, best_score = None, -1e30
            for tokens, ctc_score in nbest:
                if len(tokens) == 0:
                    att = 0.0
                else:
                    tok = torch.tensor([list(tokens)], dtype=torch.long,
                                       device=feats.device)
                    l2r = self.l2r.score(tok, mem_b).item()
                    r2l = self.r2l.score(tok, mem_b).item()
                    att = 0.5 * l2r + 0.5 * r2l
                score = ctc_weight * ctc_score + att_weight * att
                if score > best_score:
                    best_score, best_seq = score, list(tokens)
            rescored_hyps.append(best_seq)
        return greedy_hyps, rescored_hyps


def token_error_rate(hyps: List[List[int]], refs: List[List[int]]):
    """Aggregate token error rate and exact-sequence accuracy."""
    total_err = sum(edit_distance(h, r) for h, r in zip(hyps, refs))
    total_len = sum(len(r) for r in refs)
    exact = np.mean([h == r for h, r in zip(hyps, refs)])
    return total_err / max(total_len, 1), float(exact)
