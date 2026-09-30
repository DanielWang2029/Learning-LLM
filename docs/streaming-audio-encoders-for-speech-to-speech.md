# Choosing an Audio Encoder for a Speech-to-Speech Simultaneous Interpreter

*A comparison of seven audio encoders (the six you asked about + the Mimi codec),
a survey of how the speech-to-speech (S2S) field picks encoders, an analysis of
the encoder families and their techniques/structures, and concrete
recommendations.*

Scope: building a **speech-to-speech LLM for simultaneous interpretation across
languages**. All specs below are grounded in the reproductions in this repo (see
each paper folder) plus the cited literature. Reproduced encoders referenced
here live in [`Conformer/`](../Conformer/), [`wav2vec 2.0/`](../wav2vec%202.0/),
[`Emformer/`](../Emformer/), [`Whisper/`](../Whisper/),
[`FastConformer/`](../FastConformer/), [`Zipformer/`](../Zipformer/),
[`Cache-Aware Streaming Conformer/`](../Cache-Aware%20Streaming%20Conformer/),
[`WeNet U2++/`](../WeNet%20U2++/), [`Voxtral/`](../Voxtral/),
[`Qwen2.5-Omni/`](../Qwen2.5-Omni/), [`Uni-ASR/`](../Uni-ASR/),
[`VibeVoice-ASR-Streaming/`](../VibeVoice-ASR-Streaming/),
[`Confucius4-R2T2/`](../Confucius4-R2T2/),
[`Nemotron 3.5 ASR Streaming 0.6B/`](../Nemotron%203.5%20ASR%20Streaming%200.6B/),
and [`Mimi/`](../Mimi/).

---

## 0. Framing: not all "audio encoders" are the same kind of thing

Before comparing, it matters *what role* each plays:

- **Reusable continuous encoders** you can bolt onto an S2S LLM via a projector:
  **Voxtral** (a Whisper-large-v3 encoder + adapter) and **Qwen2.5-Omni** (a
  block-wise streaming audio encoder). These output continuous embeddings.
- **Full streaming ASR systems** (speech→text) whose "encoder" is an internal
  component: **Confucius4-R2T2** (Qwen3-ASR), **Uni-ASR** (Conformer),
  **VibeVoice-ASR-Streaming** (speech encoder + LLM), **Nemotron 3.5 ASR**
  (FastConformer). Using these steers you to a **cascaded** interpreter
  (ASR→MT/LLM→TTS).
- **A neural audio codec (tokenizer):** **Mimi** — it turns the waveform into
  **discrete tokens** at 12.5 Hz *and back*, so it can serve both the **input**
  and the **output** side of an end-to-end, full-duplex S2S model. This is a
  different mechanism from the semantic encoders above, and it is the encoder
  used by the current SOTA simultaneous-interpretation systems (Moshi, Hibiki).

This distinction drives the recommendations in §4.

---

## 1. The seven encoders, compared

Dimensions that matter for simultaneous interpretation: **streaming latency**,
**multilingual breadth**, **semantic richness** (translation, not just
transcription), **frame/token rate** into the LLM (lower = cheaper, longer
context, faster time-to-first-token), **speaker attribution**, **size /
on-device**, **license**, and **modularity** as an S2S encoder.

| Encoder (as used) | Streaming & latency | Multilingual | Semantics for translation | Token rate / size | License | Role for an S2S LLM |
|---|---|---|---|---|---|---|
| **Voxtral** (Whisper-large-v3 enc + 4-frame adapter) | Bidirectional full-attention over 30 s chunks → **not natively low-latency**; needs a streaming wrapper | Strong (Whisper + Mistral) | **Excellent** — ASR + translation + understanding | 50 Hz → **12.5 Hz** after adapter; enc ~0.6–1 B (models 3 B/24 B) | **Apache-2.0** | Clean, reusable **continuous encoder** module |
| **Qwen2.5-Omni** (block-wise audio encoder) | **Block-wise 2 s** attention → streaming prefill, ~2 s granularity | Broad (Whisper-init) | **Excellent**; it *is* an S2S model (Thinker-Talker) | ~12.5 Hz-class; enc ~0.6 B (model 7 B) | **Apache-2.0** | **Best template** — encoder already inside an end-to-end S2S stack |
| **Confucius4-R2T2** (Qwen3-ASR-1.7B) | **True streaming**, 80 ms–2 s chunks, ~200–600 ms; **append-only / no revision** | zh/en-centric (+others) | Good (LLM-ASR), **text output** | 1.7 B ASR system | **Restrictive** (NetEase) | Cascaded ASR front-end; borrow its **LSP decoding** |
| **VibeVoice-ASR-Streaming** | Streaming, **2.9 s chunk + 0.5 s lookahead**, ~2 s latency | 10 languages | Good; unique **speaker-attributed "who said what"** | 1.5 B / 7 B; text output | **MIT** | Cascaded; best when you need **diarization** |
| **Uni-ASR** (Conformer + Qwen3-1.7B) | Unified streaming/offline, **320–1000 ms** chunks; fallback decoding | Moderate | Good (LLM decoder), **text output** | ~2 B system | Unclear | Cascaded; **Conformer** is a reusable streaming encoder |
| **Nemotron 3.5 ASR 0.6B** (cache-aware FastConformer-RNNT) | **Best pure streaming**: cache-aware, **80–1120 ms** configurable, exact | **Widest (~40 locales)** | Weaker — **RNN-T → text**, no LLM/semantic backbone | **0.6 B, on-device** | **OpenMDW-1.1 (commercial)** | Cascaded/edge ASR front-end |
| **Mimi** (streaming split-RVQ neural codec) | **Fully streaming, causal, 80 ms frame**; the low-latency workhorse of full-duplex S2S | Language-agnostic (acoustic) | **Indirect** — codebook-0 is WavLM-distilled *semantic*; rest is acoustic | **12.5 Hz**, ~1.1 kbps, 8 codebooks; ~small | **CC-BY / Apache-family (Kyutai)** | **Discrete tokenizer for input *and* output** (full-duplex, voice-preserving) |

**One-line advantage / disadvantage each:**

- **Voxtral** — **+** strongest battle-tested multilingual *semantic* encoder, Apache-2.0, clean 12.5 Hz adapter; **−** offline-style 30 s bidirectional attention, needs re-engineering for low latency.
- **Qwen2.5-Omni** — **+** already a *streaming* S2S encoder feeding an LLM, Apache-2.0, closest to your target architecture; **−** ~2 s block latency floor, encoder entangled with the omni stack.
- **Confucius4-R2T2** — **+** lowest-latency, flicker-free **append-only** streaming (ideal interpreter UI); **−** speech→text only (cascaded), zh/en focus, restrictive license, no paper.
- **VibeVoice-ASR-Streaming** — **+** streaming **speaker attribution** (great for multi-speaker interpreting), MIT; **−** 2.9 s latency is coarse, only 10 languages, text bottleneck.
- **Uni-ASR** — **+** one model spans streaming↔offline with a reusable **Conformer** encoder and a no-cost accuracy "fallback"; **−** cascaded, moderate 320 ms–1 s latency.
- **Nemotron 3.5 ASR** — **+** tiny/on-device, **widest languages**, exact cache-aware streaming at 80 ms, commercial license; **−** transducer→text (no semantic embeddings, no voice/prosody).
- **Mimi** — **+** genuinely streaming at 12.5 Hz, preserves voice/prosody (acoustic tokens), enables **speech output + full-duplex**, and its semantic codebook aids intelligibility; **−** it is a *reconstruction* tokenizer, so its semantics are weaker than a dedicated ASR encoder — in practice it is **paired with** a semantic path, not used alone for translation understanding.

---

## 2. How the speech-to-speech field actually chooses encoders

Recent work clusters into three encoder families ([Speech LM survey, ACL 2025](https://aclanthology.org/2025.acl-long.682.pdf)):

- **Semantic "understanding" encoder → LLM (continuous embeddings).** The default
  is a **frozen Whisper (large-v3) encoder** + adapter feeding a text LLM. Used by
  Mini-Omni / SLAM-Omni, LLaMA-Omni, Kimi-Audio, and the 2026 translation system
  **S2ST-Omni** (frozen Whisper-large-v3 + hybrid "local-then-global" adapter +
  Qwen3). This is exactly Voxtral's design. ([S2ST-Omni, 2506.11160](https://arxiv.org/html/2506.11160v6))
- **Self-supervised representation encoders (w2v-BERT 2.0, HuBERT, WavLM, USM).**
  Meta's **Seamless / SeamlessStreaming** — the reference for *simultaneous*
  speech translation — uses **w2v-BERT 2.0**; unit-based SpeechLMs use HuBERT;
  **WavLM** is widely used to *distill* a semantic token.
- **Neural audio codecs at a constant low frame rate (Mimi, EnCodec, SNAC, DAC).**
  Needed when you must also **stream speech out** and preserve voice. **Moshi**
  (~200 ms full-duplex) and, most relevant here, **Hibiki** — *high-fidelity
  simultaneous S2S translation approaching human interpreters* — encode both
  source and target with **Mimi at 12.5 Hz** (its first RVQ level distilled from
  WavLM for semantics) in a batchable, on-device, decoder-only model.
  ([Hibiki, 2502.03382](https://arxiv.org/html/2502.03382))

Two clear trends in 2025–2026:

- **12.5 Hz is converging as the token rate** for S2S (Voxtral's adapter, Mimi,
  GLM-4-Voice, Qwen-Omni) — it balances semantic fidelity, LLM context length,
  and latency, and gets close to the ~3–4 Hz rate of text tokens.
- **The frontier is moving off frozen Whisper toward custom, streaming-native
  supervised encoders.** The clearest signal: **Qwen3-Omni (Sept 2025) replaced
  the Whisper encoder with its own "AuT" encoder** (~0.6 B, trained on 20 M hours,
  **block-wise dynamic-window attention 1–8 s for real-time prefill caching,
  12.5 Hz**) precisely to get Whisper-grade semantics *plus* native streaming.
  ([Qwen3-Omni, 2509.17765](https://ar5iv.labs.arxiv.org/html/2509.17765))
  SALMONN-omni similarly **distills a streaming Mamba encoder from a
  Whisper-large-v3 teacher**. ([2505.17060](https://arxiv.org/html/2505.17060v1))

**Net:** understanding/translation front-ends overwhelmingly pick **Whisper-style
semantic encoders** (increasingly retrofitted for streaming); genuine
**simultaneous, voice-preserving S2S** picks a **low-frame-rate neural codec
(Mimi)** or a self-supervised encoder (w2v-BERT 2.0).

---

## 3. Types of audio encoders: techniques and structures

It helps to separate four (overlapping) axes: **training objective**,
**representation/output type**, **temporal/streaming structure**, and
**building blocks**. Every model in §1 is a point in this space.

### 3.1 By training objective (what the encoder is optimized for)

| Family | Objective | Examples (repo) | Output | Best at | Weakness |
|---|---|---|---|---|---|
| **Supervised semantic / ASR** | CTC / RNN-T / attention on labeled speech→text | Conformer, FastConformer, Zipformer, Emformer, Whisper, Qwen-AuT | continuous embeddings (or text via a head) | meaning, translation, ASR | acoustic detail / voice not preserved |
| **Self-supervised (SSL)** | contrastive / masked prediction on unlabeled audio | wav2vec 2.0 (contrastive + product-VQ); (w2v-BERT +MLM, HuBERT masked-cluster, WavLM +denoising) | continuous features; often k-means → discrete units | data-efficient, robust, transferable | needs a downstream head; raw units are not "text" |
| **Neural audio codec / acoustic tokenizer** | reconstruction + adversarial + RVQ commitment | **Mimi**, (EnCodec, SoundStream, DAC, SNAC) | **discrete RVQ tokens** (multi-codebook) | fidelity, **speech output**, full-duplex, low latency | quantization loss; weaker standalone semantics; many tokens/step |
| **Supervised semantic tokenizer** | ASR-encoder features + VQ | (GLM-4-Voice tokenizer, CosyVoice S3) | discrete **semantic** tokens | intelligible generation from text scaffold | discards fine acoustics |

**Key idea — semantic vs acoustic tokens.** *Semantic* tokens capture meaning
(good for ASR/translation, very low bitrate); *acoustic* tokens capture
fidelity/timbre (needed to reproduce a voice). Codec-based S2S needs **both**.
Mimi's contribution is a **split RVQ** that puts a **WavLM-distilled semantic**
code in codebook-0 and lets a **parallel acoustic RVQ** carry the rest, so
distillation doesn't degrade audio quality — one tokenizer gives you semantic +
acoustic at once. ([Moshi/Mimi, 2410.00037](https://arxiv.org/abs/2410.00037))

### 3.2 By representation / output type

- **Continuous embeddings → projector/adapter → LLM.** (Whisper/Voxtral,
  Conformer, w2v-BERT features.) *Pros:* no quantization loss, semantically rich,
  simple to attach. *Cons:* not a discrete vocabulary, so the model can't
  *generate* speech from them — you still need a separate synthesizer.
- **Discrete tokens → LLM vocabulary.** *Single-codebook* (HuBERT units) vs
  *multi-codebook RVQ* (EnCodec/Mimi). *Pros:* fits the LLM token paradigm and
  enables **speech output** and full-duplex. *Cons:* quantization loss; RVQ depth
  means several tokens per frame, forcing tricks like **delay patterns**,
  **multistream/"RQ-Transformer"** (Moshi/Hibiki), or **multi-token prediction**
  to model them in real time.

### 3.3 By temporal structure / streaming mechanism (maps onto this repo)

| Mechanism | How future context is bounded | Latency | Examples (repo) |
|---|---|---|---|
| **Full bidirectional attention over a fixed window** | whole 30 s window | high (offline-ish) | Whisper, Voxtral |
| **Block / chunk-wise attention** | independent blocks (block-diagonal mask) | medium (~1–2 s) | Qwen2.5-Omni (2 s), Emformer, AuT (1–8 s dynamic) |
| **Cache-aware causal attention + conv-state cache** | left context only; cached, no recompute | **low (80–320 ms), exact** | FastConformer/Nemotron, Cache-Aware Conformer |
| **Causal convolution stacks (codec)** | receptive field of causal convs | **low (80 ms frame)** | Mimi / EnCodec / SoundStream encoder-decoder |
| **Recurrent / SSM (Mamba)** | fixed-size recurrent state | low | SALMONN-omni streaming encoder |

A related trick is **dynamic-chunk training** (WeNet U2++): randomize the chunk
size during training so a *single* model serves any latency at inference.

### 3.4 By structural building blocks

- **Front-end:** raw-waveform **CNN** (wav2vec 2.0, codecs) vs **log-mel + conv
  stem** (Whisper, Conformer). Codecs additionally learn the *inverse* (a decoder
  back to waveform) — encoders usually don't.
- **Core block:** **Transformer** (pure attention), **Conformer** (conv +
  attention, local + global), **SSM/Mamba** (linear-time recurrence), or a
  **RVQ bottleneck** (the defining piece of a codec).
- **Downsampling to a low frame rate:** strided conv (4× Conformer, **8×**
  FastConformer), **frame concatenation** (Voxtral concatenates 4 frames,
  50→12.5 Hz), or conv strides tuned to hit 12.5 Hz (Mimi).
- **Positional scheme:** absolute sinusoidal (Whisper), relative (Conformer),
  rotary/RoPE, or **TMRoPE** (time-aligned multimodal RoPE, Qwen-Omni).
- **Norms/activations that matter for speech:** GLU + depthwise conv + BatchNorm
  (Conformer conv module), **BiasNorm** and SwooshR/L (Zipformer).

### 3.5 The frame-rate spectrum (and why it matters)

```
raw mel ~100 Hz  →  Whisper / wav2vec2 ~50 Hz  →  ~25 Hz  →  Mimi / Voxtral-adapter / AuT / GLM-4-Voice 12.5 Hz  →  text ~3–4 Hz
```

Lower frame rate = **fewer LLM steps per second** = longer context, cheaper
compute, lower latency — but too low risks dropping detail. **12.5 Hz is the
emerging S2S sweet spot**: rich enough for speech, close enough to text rate to
keep autoregression tractable.

---

## 4. Recommendations for your simultaneous interpreter

Your encoder choice hinges on one architecture decision.

### A. End-to-end S2S LLM (audio→audio, text as internal scaffold) — recommended for true simultaneous interpretation

1. **Semantic/understanding path — pick the Qwen2.5-Omni audio encoder** (or, if
   weights are available, the newer **Qwen3-Omni AuT** encoder). It is the only
   one of the six that is a streaming, multilingual, ~12.5 Hz *semantic* encoder
   already wired into an end-to-end S2S stack (Apache-2.0). Strong fallback:
   **Voxtral's Whisper-large-v3 encoder + adapter** (richest multilingual
   translation semantics, clean 12.5 Hz module) — but add streaming (chunked /
   cache-aware attention, or Whisper-teacher distillation à la SALMONN-omni),
   since its native attention is 30 s bidirectional.
2. **Output / voice path — use Mimi** (or an EnCodec/SNAC-style RVQ codec).
   To speak in the target language *while preserving the speaker's voice* and to
   run **full-duplex**, you need a streaming codec on the output (and, in the
   Moshi/Hibiki design, on the input too). **Hibiki** — the current SOTA for
   simultaneous S2S interpretation, approaching human quality — is exactly
   **Mimi (12.5 Hz) + a multistream decoder-only LM**, batchable and on-device.
   This is the strongest blueprint for a from-scratch interpreter with voice
   transfer.

   *A clean, proven recipe:* **Mimi tokens for I/O + a multistream LM, with the
   semantic codebook (WavLM/Whisper-distilled) carrying meaning** — i.e. combine
   the semantic-encoder idea (§4.A.1) *inside* the codec, as Mimi does.

### B. Cascaded interpreter (streaming ASR → LLM/MT → streaming TTS) — simpler, faster to ship

- **Front-end encoder: Nemotron 3.5 ASR 0.6B** — widest languages (~40 locales),
  exact cache-aware streaming down to 80 ms, tiny/on-device, commercial license.
  (Or **Uni-ASR**'s Conformer if you want a streaming/offline-unified LLM-ASR.)
- **Adopt regardless of encoder:** **Confucius4-R2T2's Longest-Stable-Prefix
  append-only decoding** (so the on-screen translation never rewrites/flickers —
  critical UX for interpreting) and **Uni-ASR's fallback decoding** (free
  streaming-accuracy gain).
- **Multi-speaker (meetings/panels):** borrow **VibeVoice-ASR-Streaming's**
  speaker-attributed streaming or add a diarization head — but budget its ~2 s
  latency.

### Bottom line

- **Best fit among the six:** the **Qwen2.5-Omni / Qwen3-Omni-AuT streaming
  encoder** for understanding, with **Voxtral's Whisper encoder** as the robust
  fallback.
- **For genuine simultaneous, voice-preserving S2S:** add **Mimi** (the codec you
  just asked to include) on the I/O side and follow the **Hibiki/Moshi
  multistream** recipe — this is the field's current best answer to
  *simultaneous interpretation*.
- **For a low-latency, many-language cascade:** **Nemotron 3.5 ASR**, plus
  **Confucius** append-only decoding and **VibeVoice** attribution as needed.

---

## References

- Whisper — Robust Speech Recognition via Large-Scale Weak Supervision, arXiv 2212.04356
- Conformer, arXiv 2005.08100 · wav2vec 2.0, arXiv 2006.11477 · Emformer, arXiv 2010.10759
- FastConformer, arXiv 2305.05084 · Zipformer, arXiv 2310.11230 · Cache-Aware Streaming Conformer, arXiv 2312.17279 · WeNet U2++, arXiv 2106.05642
- Voxtral, arXiv 2507.13264 · Qwen2.5-Omni, arXiv 2503.20215 · Qwen3-Omni, arXiv 2509.17765
- Uni-ASR, arXiv 2603.11123 · VibeVoice-ASR-Streaming, arXiv 2609.02812 · Confucius4-R2T2 (NetEase model card) · Nemotron 3.5 ASR (NVIDIA model card)
- **Mimi / Moshi**, arXiv 2410.00037 · **Hibiki** (simultaneous S2S), arXiv 2502.03382 · Seamless / SeamlessStreaming
- S2ST-Omni, arXiv 2506.11160 · SALMONN-omni, arXiv 2505.17060 · Recent Advances in Speech Language Models: A Survey, ACL 2025 (arXiv 2410.03751)
