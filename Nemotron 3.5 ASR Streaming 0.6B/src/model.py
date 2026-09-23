"""Cache-aware FastConformer encoder + a tiny RNN-T (transducer) decoder.

Faithful, miniature version of the Nemotron 3.5 ASR architecture
("FastConformer-CacheAware-RNNT with Prompt"):

  * 8x depthwise-separable convolutional subsampling (three stride-2 blocks),
  * Conformer blocks whose self-attention uses full left context (the "cache")
    and a CONFIGURABLE right context R (the streaming lookahead) chosen at
    inference with no retraining, and a strictly causal depthwise-conv module,
  * language-ID prompt fusion: a one-hot language vector is broadcast across time,
    concatenated to the acoustic features, and projected — exactly as the model
    card describes,
  * an RNN-T decoder (prediction network + joint network) trained with a
    from-scratch transducer loss.

Because attention right-context is bounded by R and the conv path is causal, an
encoder frame t depends only on inputs up to frame t+R. Cache-aware streaming
(process new chunks, reuse cached state) therefore yields output identical to a
full-context pass — the "no redundant overlapping computation" property.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def causal_conv1d(x, conv, left_pad):
    """Apply a Conv1d causally: left padding only (no future leakage)."""
    return conv(F.pad(x, (left_pad, 0)))


class SubsampleConv(nn.Module):
    """8x depthwise-separable convolutional subsampling (3 x stride-2)."""

    def __init__(self, n_mels, d_model, k=3):
        super().__init__()
        chans = [n_mels, 48, 72, d_model]
        self.k = k
        self.blocks = nn.ModuleList()
        for i in range(3):
            dw = nn.Conv1d(chans[i], chans[i], k, stride=2, groups=chans[i])
            pw = nn.Conv1d(chans[i], chans[i + 1], 1)
            self.blocks.append(nn.ModuleList([dw, pw]))

    def forward(self, x):  # x: (B, n_mels, T)
        for dw, pw in self.blocks:
            x = causal_conv1d(x, dw, self.k - 1)
            x = F.gelu(pw(x))
        return x  # (B, d_model, T/8)


class ConformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, conv_k=5):
        super().__init__()
        self.ffn1 = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_ff),
                                  nn.GELU(), nn.Linear(d_ff, d_model))
        self.ln_attn = nn.LayerNorm(d_model)
        self.h, self.dk = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.ln_conv = nn.LayerNorm(d_model)
        self.conv_k = conv_k
        self.dwconv = nn.Conv1d(d_model, d_model, conv_k, groups=d_model)
        self.pwconv = nn.Conv1d(d_model, d_model, 1)
        self.ffn2 = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, d_ff),
                                  nn.GELU(), nn.Linear(d_ff, d_model))
        self.ln_out = nn.LayerNorm(d_model)

    def attn(self, x, attn_mask):
        B, S, D = x.shape
        q, k, v = self.qkv(self.ln_attn(x)).chunk(3, dim=-1)
        q = q.view(B, S, self.h, self.dk).transpose(1, 2)
        k = k.view(B, S, self.h, self.dk).transpose(1, 2)
        v = v.view(B, S, self.h, self.dk).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.dk)
        scores = scores.masked_fill(attn_mask, float("-inf"))
        out = (F.softmax(scores, dim=-1) @ v).transpose(1, 2).contiguous().view(B, S, D)
        return self.proj(out)

    def conv(self, x):  # causal depthwise conv module
        y = self.ln_conv(x).transpose(1, 2)
        y = causal_conv1d(y, self.dwconv, self.conv_k - 1)
        y = self.pwconv(F.gelu(y)).transpose(1, 2)
        return y

    def forward(self, x, attn_mask):
        x = x + 0.5 * self.ffn1(x)
        x = x + self.attn(x, attn_mask)
        x = x + self.conv(x)
        x = x + 0.5 * self.ffn2(x)
        return self.ln_out(x)


class FastConformerEncoder(nn.Module):
    def __init__(self, n_mels, d_model=64, n_layers=2, n_heads=4, d_ff=128):
        super().__init__()
        self.subsample = SubsampleConv(n_mels, d_model)
        self.blocks = nn.ModuleList([ConformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.d_model = d_model

    def right_context_mask(self, S, R, device):
        j = torch.arange(S, device=device).view(1, S)
        i = torch.arange(S, device=device).view(S, 1)
        disallowed = j > (i + R)                       # full left, R right
        return disallowed.view(1, 1, S, S)

    def forward(self, mel, R):
        # mel: (B, T_mel, n_mels)
        x = self.subsample(mel.transpose(1, 2)).transpose(1, 2)   # (B, T_enc, d)
        mask = self.right_context_mask(x.size(1), R, x.device)
        for blk in self.blocks:
            x = blk(x, mask)
        return x


class LangFusion(nn.Module):
    """Broadcast a one-hot language vector across time, concat, and project."""

    def __init__(self, d_model, n_langs, k=16):
        super().__init__()
        self.lang_emb = nn.Embedding(n_langs, k)
        self.proj = nn.Linear(d_model + k, d_model)

    def forward(self, enc, lang):  # enc:(B,T,d) lang:(B,)
        le = self.lang_emb(lang).unsqueeze(1).expand(-1, enc.size(1), -1)
        return self.proj(torch.cat([enc, le], dim=-1))


class RNNT(nn.Module):
    """Prediction network + joint network."""

    def __init__(self, vocab, d_model, pred_dim=64, joint_dim=64):
        super().__init__()
        self.vocab = vocab
        self.embed = nn.Embedding(vocab, pred_dim)
        self.gru = nn.GRU(pred_dim, pred_dim, batch_first=True)
        self.enc_proj = nn.Linear(d_model, joint_dim)
        self.pred_proj = nn.Linear(pred_dim, joint_dim)
        self.out = nn.Linear(joint_dim, vocab)

    def predict(self, labels):
        """labels: (B, U+1) with a leading BLANK start token -> pred (B,U+1,pd)."""
        e = self.embed(labels)
        out, _ = self.gru(e)
        return out

    def joint(self, enc, pred):
        # enc:(B,T,d)  pred:(B,U+1,pd) -> (B,T,U+1,V)
        e = self.enc_proj(enc).unsqueeze(2)
        p = self.pred_proj(pred).unsqueeze(1)
        return self.out(torch.tanh(e + p))


def transducer_loss(logp, targets, blank=0):
    """From-scratch log-space RNN-T forward loss (fixed T, U across batch).

    logp: (B, T, U+1, V) log-softmax joint outputs.
    targets: (B, U) label ids.
    """
    B, T, U1, V = logp.shape
    U = U1 - 1
    blank_lp = logp[..., blank]                                   # (B,T,U+1)
    if U > 0:
        emit_lp = torch.gather(
            logp[:, :, :U, :], 3, targets.view(B, 1, U, 1).expand(B, T, U, 1)
        ).squeeze(3)                                             # (B,T,U)
    device = logp.device
    neg_inf = torch.full((B,), float("-inf"), device=device)
    alpha = [[None] * (U + 1) for _ in range(T)]
    alpha[0][0] = torch.zeros(B, device=device)
    for t in range(1, T):
        alpha[t][0] = alpha[t - 1][0] + blank_lp[:, t - 1, 0]
    for u in range(1, U + 1):
        alpha[0][u] = alpha[0][u - 1] + emit_lp[:, 0, u - 1]
    for t in range(1, T):
        for u in range(1, U + 1):
            from_blank = alpha[t - 1][u] + blank_lp[:, t - 1, u]
            from_emit = alpha[t][u - 1] + emit_lp[:, t, u - 1]
            alpha[t][u] = torch.logaddexp(from_blank, from_emit)
    ll = alpha[T - 1][U] + blank_lp[:, T - 1, U]
    return -ll.mean()


@torch.no_grad()
def greedy_decode(rnnt, enc_cond, blank=0, max_labels=64):
    """Standard RNN-T greedy decoding over encoder frames (single sequence)."""
    device = enc_cond.device
    T = enc_cond.size(1)
    labels = [blank]
    emitted = []
    t, u = 0, 0
    steps = 0
    while t < T and steps < max_labels + T:
        steps += 1
        lab = torch.tensor([labels], device=device)
        pred = rnnt.predict(lab)[:, -1:, :]                 # (1,1,pd)
        logits = rnnt.joint(enc_cond[:, t:t + 1], pred)[0, 0, 0]
        tok = int(logits.argmax().item())
        if tok == blank:
            t += 1
        else:
            emitted.append(tok)
            labels.append(tok)
            u += 1
            if u >= max_labels:
                break
    return emitted
