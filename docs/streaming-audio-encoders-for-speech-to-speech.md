# Streaming Audio Encoders for Speech-to-Speech — Comparison & Recommendation

> Scope: choosing an **audio encoder** for a **streaming speech-to-speech (S2S)**
> system — specifically a **simultaneous interpreter** that listens in one
> language and speaks another with bounded, continuous latency. This document
> compares the streaming encoders reproduced in this repo, surveys how the S2S
> field actually picks encoders, and ends with a concrete recommendation.

Each paper named here has a self-contained, runnable reproduction in its own
folder; this file lives at the repo root level (under `docs/`) because it is a
*cross-paper* comparison rather than the internals of any single paper.

---

## TL;DR recommendation

For a **streaming S2S simultaneous interpreter**:

- **Input understanding encoder → use a natively-causal, block/sliding-window
  streaming encoder at ~12.5 Hz that feeds the LLM as continuous embeddings.**
  Top pick: a **Voxtral-Realtime-style causal sliding-window encoder** (causal
  attention + bounded window + RoPE/RMSNorm/SwiGLU) because it (a) streams
  *exactly* with bounded memory, (b) emits ~12.5 Hz tokens so the LLM keeps
  context budget for two languages, and (c) exposes a **latency dial**
  (AdaRMSNorm delay conditioning) — the single most useful knob for an
  interpreter, where you tune the accuracy/latency trade-off per deployment.
- **Strong, battle-tested alternative:** the **Qwen3-Omni AuT block-wise
  streaming encoder** (12.5 Hz) if you want to build on a proven end-to-end
  omni **Thinker-Talker** stack that already does speech translation.
- **Output speech generation → use a streaming neural codec**, i.e.
  **Mimi** (12.5 Hz, fully causal, split-RVQ). The input encoder understands;
  the codec is what makes the *spoken output* streamable. S2S needs both.
- **On-device / tiny-footprint variant:** a **cache-aware FastConformer**
  (Nemotron 3.5 ASR / Cache-Aware Streaming Conformer) encoder — excellent
  latency/compute, "one model, many latencies," at the cost of needing an
  explicit LLM bridge for translation.

Details, trade-offs, and why *not* the others are below.

---

## 1. What "streaming" has to mean for an interpreter

An offline encoder may let any frame attend to the entire clip (past **and**
future). That is fine for file transcription but fatal for simultaneous
interpretation, which needs:

1. **Causality / bounded look-ahead.** Output for time *t* must depend only on
   audio up to *t* (+ a small, fixed look-ahead). No "wait for the sentence to
   end."
2. **Bounded memory & compute per step.** Cost must not grow with how long the
   speaker talks — the session is effectively infinite.
3. **A latency knob.** Interpreters live or die on the accuracy↔latency curve;
   you want to *set* that trade-off, ideally without retraining.
4. **LLM-friendly, token-efficient output.** The representation feeds a
   translation LLM; a low frame rate (~12.5 Hz) preserves context budget for
   *two* languages plus generation.
5. **Multilingual robustness.** The encoder must represent many source
   languages well.

These five axes drive the whole comparison.

---

## 2. The encoder families (taxonomy)

- **Conformer backbone (conv + attention).** The de-facto ASR encoder; local
  convolution + global attention. Streamable by causal convs (left-padding) and
  masked/windowed attention. Everything below is a variation on this idea.
  → `Conformer/`, `FastConformer/`, `Zipformer/`
- **Streaming Transformers with memory/caches.** Block-wise processing that
  caches left context and/or a memory bank for low latency.
  → `Emformer/`, `Cache-Aware Streaming Conformer/`, `WeNet U2++/`,
  `Nemotron 3.5 ASR Streaming 0.6B/`
- **Self-supervised encoders.** Pretrained representation extractors (not
  natively streaming), often reused as frozen features.
  → `wav2vec 2.0/`
- **Offline encoder-decoder ASR reused by speech-LLMs.** Robust, multilingual,
  but **bidirectional / offline**.
  → `Whisper/`, `Voxtral/` (Whisper encoder + adapter)
- **Natively-streaming speech-LLM encoders.** Causal/block-wise encoders built
  *for* online LLM decoding.
  → `Voxtral Realtime/`, `Qwen2.5-Omni/`, `Qwen3-Omni/`, `Uni-ASR/`,
  `VibeVoice-ASR-Streaming/`, `Confucius4-R2T2/`
- **Neural audio codecs (discrete tokens).** Encode audio into discrete tokens;
  used when a single LLM models audio directly (full-duplex S2S), and for
  **output** speech generation.
  → `Mimi/`

---

## 3. Side-by-side comparison

Frame rate = encoder output tokens per second (higher = more acoustic detail but
more LLM tokens). "Native streaming" = designed to run causally/online out of
the box (vs. requiring surgery). "Latency dial" = one model serves multiple
latencies.

| Encoder (folder) | Year | Family | Native streaming | Frame rate | Latency dial | Representation | Best role in S2S |
|---|---|---|---|---|---|---|---|
| **Conformer** | 2020 | conv+attn backbone | with masking | ~25 Hz (4× sub) | no | continuous | backbone to build on |
| **wav2vec 2.0** | 2020 | SSL encoder | no (bidirectional) | 50 Hz | no | continuous (SSL) | frozen features, pretraining |
| **Emformer** | 2021 | memory Transformer | **yes** (blocks + memory) | ~25 Hz | partial (block size) | continuous | low-latency on-device ASR |
| **WeNet U2++** | 2021 | dynamic-chunk Conformer | **yes** (dynamic chunk) | ~25 Hz | **yes** (chunk size) | continuous (+CTC) | unified stream/non-stream ASR |
| **Whisper** | 2022 | offline enc-dec | no (30 s chunks) | 50 Hz | no | continuous | robust multilingual features (offline) |
| **FastConformer** | 2023 | efficient Conformer | with cache | ~12.5 Hz (8× sub) | partial | continuous | efficient streaming backbone |
| **Zipformer** | 2023 | U-Net multi-rate | with masking | multi-rate | partial | continuous | high-accuracy efficient ASR |
| **Cache-Aware Streaming Conformer** | 2023 | stateful Conformer | **yes** (state cache) | ~12.5–25 Hz | **yes** (many latencies) | continuous | exact chunked streaming ASR |
| **Voxtral** | 2025 | Whisper enc + adapter | no (offline enc) | 12.5 Hz (4× adapter) | no | continuous → LLM | offline speech-LLM |
| **Qwen2.5-Omni** | 2025 | block-wise omni enc | **yes** (2 s blocks) | ~12.5 Hz | partial (block) | continuous → LLM | omni Thinker-Talker S2S |
| **Qwen3-Omni** | 2025 | AuT block-wise enc | **yes** | 12.5 Hz | partial | continuous → LLM (MoE) | **proven S2S translation stack** |
| **Voxtral Realtime** | 2026 | causal sliding-window | **yes** (exact) | 12.5 Hz (80 ms) | **yes** (AdaRMSNorm τ) | continuous → LLM | **interpreter input encoder** |
| **Uni-ASR** | 2026 | LLM-based unified | **yes** (interleave) | LLM-tokenized | **yes** (fallback) | tokens → LLM | unified stream/non-stream ASR |
| **VibeVoice-ASR-Streaming** | 2026 | LLM speaker-attributed | **yes** (chunk+lookahead) | chunked | partial | tokens → LLM | multi-speaker "who said what" |
| **Confucius4-R2T2** | 2026 | stable-prefix policy | **yes** (append-only) | inherits base | policy-level | text commit policy | flicker-free display layer |
| **Nemotron 3.5 ASR 0.6B** | 2026 | cache-aware FC-RNNT | **yes** | ~12.5 Hz | **yes** (configurable chunk) | continuous (RNNT) | tiny on-device streaming ASR |
| **Mimi (codec)** | 2024 | neural audio codec | **yes** (fully causal) | 12.5 Hz | no | **discrete tokens** | input tokens *and* output speech gen |

---

## 4. Per-encoder notes (pros / cons for S2S)

**Conformer** — The backbone. *Pros:* captures local (conv) + global (attn)
structure; everything streaming descends from it. *Cons:* vanilla version is
offline; you must add causal convs + masked attention yourself. Use as a
building block, not off-the-shelf.

**wav2vec 2.0** — *Pros:* superb self-supervised representations, label-efficient.
*Cons:* bidirectional and non-streaming by design; best as frozen features or
for pretraining, not as the online encoder of an interpreter.

**Emformer** — *Pros:* purpose-built low-latency streaming via augmented-memory
blocks + cached left context; proven on-device. *Cons:* ASR-oriented;
representations are tuned for a CTC/RNNT head, not for feeding a translation LLM.

**WeNet U2++** — *Pros:* **dynamic chunk training** gives one encoder that runs
at any chunk size (streaming *and* offline) with CTC + attention rescoring —
the chunk size *is* a latency dial. *Cons:* two-pass rescoring assumes you can
re-read; pure left-to-right interpretation uses only the first pass.

**Whisper** — *Pros:* extremely robust, massively multilingual; its encoder is
reused by many speech-LLMs (incl. Voxtral). *Cons:* **bidirectional, 30 s
chunked = offline**; cannot stream without architectural surgery. Great teacher
/ feature source, wrong online encoder.

**FastConformer** — *Pros:* 8× subsampling → ~12.5 Hz, big speed/memory win,
cache-aware streaming variants exist. *Cons:* needs the cache-aware recipe +
an LLM bridge for translation; ASR-first.

**Zipformer** — *Pros:* multi-rate U-Net, state-of-the-art accuracy/efficiency
on ASR benchmarks. *Cons:* more complex; streaming variants trail the
cache-aware Conformer's "exact chunk" guarantees for interpreter use.

**Cache-Aware Streaming Conformer** — *Pros:* caches attention/conv state across
chunks for **exact** chunked streaming, and **one model serves many latencies**.
*Cons:* ASR encoder — you still bolt on the translation LLM and output synthesis.

**Voxtral** — *Pros:* clean speech-LLM recipe; the **4-frame adapter (50→12.5 Hz)**
is the token-efficiency trick worth copying. *Cons:* the encoder is **offline
Whisper** — not streamable. Voxtral *Realtime* is the streaming answer.

**Qwen2.5-Omni** — *Pros:* block-wise (2 s) streaming audio encoder feeding a
**Thinker-Talker** LLM that already supports speech in/out. *Cons:* 2 s blocks
imply ~seconds of chunk latency unless tuned down.

**Qwen3-Omni** — *Pros:* from-scratch **AuT block-wise streaming encoder at
12.5 Hz** + MoE Thinker-Talker + multi-codebook MTP talker; a **complete,
proven streaming S2S translation stack**, strongly multilingual. *Cons:* big and
complex; adopting it means adopting the whole omni system.

**Voxtral Realtime** — *Pros:* **natively, exactly streaming** causal
sliding-window encoder; 12.5 Hz; and **AdaRMSNorm delay conditioning gives a true
latency dial from one model** — ideal for an interpreter. Modern backbone
(RMSNorm/RoPE/SwiGLU). *Cons:* newest; you integrate the translation decoder +
output codec yourself (Delayed Streams Modeling shows the decoding pattern).

**Uni-ASR** — *Pros:* one LLM-based model for streaming *and* non-streaming via
interleaved sequences, with a zero-latency fallback decode. *Cons:* ASR, not
translation; adapt the interleaving to S2S yourself.

**VibeVoice-ASR-Streaming** — *Pros:* streaming **speaker-attributed** ASR
("who said what") — valuable for multi-party interpretation. *Cons:* niche;
speaker attribution, not translation.

**Confucius4-R2T2** — *Pros:* **longest-stable-prefix** commit policy =
append-only, flicker-free output — exactly what you want for a live interpreter
*display/commit* layer. *Cons:* it is a decoding/commit policy over a base model,
not an encoder per se; pair it with one of the encoders above.

**Nemotron 3.5 ASR Streaming 0.6B** — *Pros:* **tiny (0.6B)**, cache-aware
FastConformer-RNNT, **configurable chunk sizes** (latency dial), language-ID
conditioning. Excellent edge/on-device option. *Cons:* RNNT ASR; translation LLM
is separate.

**Mimi (codec)** — *Pros:* **fully causal, 12.5 Hz, split-RVQ** neural codec;
powers full-duplex S2S (Moshi-style) and, crucially, is what makes **output
speech generation streamable**. Carries prosody/speaker identity. *Cons:*
discrete *acoustic* tokens are less semantically dense than a semantic encoder's
embeddings, so for *understanding* the source it complements rather than replaces
a semantic encoder.

---

## 5. How the S2S field actually chooses encoders

Looking across the recent systems reproduced here (and the broader literature),
a few clear patterns emerge:

1. **Convergence on ~12.5 Hz.** Mimi, the Voxtral adapter, Voxtral Realtime, and
   the Qwen-Omni AuT encoder all land near **12.5 Hz**. It is the empirical sweet
   spot: enough acoustic resolution for ASR/translation, few enough tokens that
   an LLM can hold long context across *two* languages and still generate.

2. **Two paradigms for feeding the LLM:**
   - **(A) Continuous embeddings as a soft prefix.** The encoder outputs vectors
     the LLM attends to (Whisper→Voxtral, Qwen-Omni, Voxtral Realtime). Highest
     semantic fidelity for understanding; dominant for ASR-/translation-centric
     S2S.
   - **(B) Discrete audio tokens modeled directly.** A codec (Mimi) turns audio
     into tokens the *same* LLM predicts, enabling **full-duplex** speech-in /
     speech-out (Moshi). Simpler single-stack duplex, slightly lower semantic
     density per token.

3. **Streaming = causal + windowed/block-wise attention + state caching.** The
   winning recipe is causal convolutions (left-padding) and bounded-context
   attention with per-layer K/V (and conv-state) caches, giving **exact**
   streaming equal to the offline forward with bounded memory.

4. **"One model, many latencies" is now expected.** Dynamic-chunk training
   (WeNet U2++), cache-aware chunks (Cache-Aware Conformer, Nemotron 3.5), and
   delay conditioning (Voxtral Realtime's AdaRMSNorm) all let a single checkpoint
   be tuned at inference — essential because interpreters are deployed under
   different latency budgets.

5. **End-to-end S2S translation stacks** (Qwen-Omni Thinker-Talker; Moshi;
   SeamlessM4T-style systems in the wider field) increasingly pair a **block-wise
   streaming semantic encoder** on the input with a **streaming codec talker** on
   the output — i.e. they use *both* an encoder (understanding) and a codec
   (generation), not one or the other.

6. **Output generation is a first-class constraint.** A streaming *input* encoder
   is necessary but not sufficient; the system is only as "simultaneous" as its
   *output* path, which is why a causal codec (Mimi) for the talker is part of
   essentially every modern full-duplex S2S design.

---

## 6. Recommendation for a streaming S2S simultaneous interpreter

### Primary recommendation

**Input (understanding) encoder:** a **Voxtral-Realtime-style causal
sliding-window encoder at 12.5 Hz**, producing continuous embeddings as a soft
prefix to a multilingual translation LLM.

Why it wins on the five axes from §1:

- **Causality / bounded look-ahead:** causal + sliding-window attention →
  exactly streamable, no future peeking.
- **Bounded memory:** receptive field is `L·(W−1)+1` frames, constant regardless
  of session length.
- **Latency knob:** **AdaRMSNorm delay conditioning** — set the interpreter's
  accuracy↔latency trade-off at inference from *one* trained model. No other
  reproduced encoder makes this as clean.
- **Token efficiency:** 12.5 Hz keeps context budget for source + target
  language and generation.
- **Multilingual:** Mistral/Voxtral lineage is strongly multilingual.

**Output (generation) codec:** **Mimi** (12.5 Hz, causal, split-RVQ) as the
talker's speech tokenizer, so synthesized target-language speech streams out with
the same bounded latency.

**Commit/display policy:** layer **Confucius4-R2T2's longest-stable-prefix** idea
on top so emitted translation is append-only and flicker-free.

### If you prefer a proven, integrated stack

Adopt **Qwen3-Omni**: its **AuT block-wise streaming encoder (12.5 Hz)** +
**Thinker-Talker MoE** already performs streaming speech translation end to end
and is strongly multilingual. You trade bespoke control (the AdaRMSNorm latency
dial) for a complete, validated system.

### If you are constrained to edge / tiny footprint

Use **Nemotron 3.5 ASR Streaming 0.6B** (or a **Cache-Aware Streaming
Conformer**) as the encoder: configurable chunk sizes give the latency dial, the
model is tiny, and streaming is exact — then bridge to a compact translation LLM
and a Mimi talker.

### What to avoid as the *online* encoder

- **Whisper / Voxtral (offline Whisper encoder) / wav2vec 2.0** — bidirectional
  or 30 s-chunked; they cannot stream without major surgery. Keep Whisper as a
  **teacher / multilingual feature source**, and copy Voxtral's **4-frame
  adapter** trick for 12.5 Hz token efficiency — but do not put an offline
  encoder on the live path.

### Concrete starter recipe

1. **Encoder:** causal sliding-window Conformer/Transformer (RMSNorm/RoPE/SwiGLU),
   causal depthwise convs (left-padded), per-layer K/V + conv-state caches,
   adapter down to **12.5 Hz**. (See `Voxtral Realtime/src/encoder.py` for the
   exact-streaming pattern and `Cache-Aware Streaming Conformer/` for conv-state
   caching.)
2. **Latency conditioning:** AdaRMSNorm on the decoder FFN, train with the target
   delay τ sampled across your deployment range (`Voxtral Realtime/src/adarmsnorm.py`).
3. **Translation decoder:** stream-synchronous (Delayed Streams Modeling): one
   step per frame, fusing audio embedding + previous target token
   (`Voxtral Realtime/src/model.py`).
4. **Output speech:** a Mimi-style causal 12.5 Hz codec talker for streaming TTS
   of the target language.
5. **Commit policy:** longest-stable-prefix (Confucius4-R2T2) for flicker-free
   output.

---

## 7. Streaming ASR (text) output from the same encoders

**Yes.** The Thinker-Talker (speech→speech) is only *one* decoder you can attach
to these encoders. Every recommended streaming encoder emits
**frame-synchronous embeddings** as the audio arrives, and a streaming ASR
system is just "that same encoder + a streaming text decoder/head." You can even
run **ASR text and translated speech at the same time off one shared encoder**
(§7.3).

### 7.1 Pick a streaming text decoder/head

| Head | Cadence | Latency | Models output deps? | Repo example | When to use |
|---|---|---|---|---|---|
| **CTC** | frame-synchronous | lowest (per frame) | no (conditional independence) | `WeNet U2++/` (1st pass) | ultra-low-latency partial captions; cheap extra head |
| **RNN-T / Transducer** | frame-synchronous (0+ tokens/frame via blank) | low | yes (prediction net) | `Nemotron 3.5 ASR Streaming 0.6B/`, `Cache-Aware Streaming Conformer/` | best standalone streaming ASR accuracy/latency |
| **DSM / stream-synchronous LLM** | 1 token / frame (`[P]`/`[W]`/word) | tunable via τ | yes (autoregressive) | `Voxtral Realtime/` | you already have an LLM decoder; want a latency dial |
| **Chunk-synchronous attention (AED, monotonic)** | per chunk | chunk-sized | yes | `Qwen2.5-Omni/`, `Qwen3-Omni/` (Thinker) | reuse an attention decoder you already train |
| **Two-pass (streaming CTC + attention rescoring)** | partials per chunk, final at endpoint | low partials, higher finals | yes (2nd pass) | `WeNet U2++/` | best quality finals while keeping instant partials |
| **LLM interleaving** | per audio chunk | chunk-sized | yes | `Uni-ASR/`, `VibeVoice-ASR-Streaming/` | one LLM, source audio+text interleaved; speaker labels |

All of these consume the *same* streaming encoder frames — you choose the head by
your latency/accuracy/engineering budget.

### 7.2 How to run it (the streaming loop)

1. **Run the encoder incrementally.** Feed audio in chunks and carry the
   per-layer **K/V cache + conv-state cache** so each step is *exact* and uses
   bounded memory (`Voxtral Realtime/src/encoder.py` `stream()`;
   `Cache-Aware Streaming Conformer/` for conv-state caching). Chunk size = your
   latency budget.
2. **Attach a head** (above) and decode frame-by-frame or chunk-by-chunk.
3. **Partial vs. final hypotheses.** Emit *partials* every chunk for
   responsiveness; mark text *final* once it stabilizes or an endpoint fires.
4. **Flicker-free display.** Commit only the **longest stable prefix**
   (`Confucius4-R2T2/`) so on-screen text is append-only and never rewrites.
5. **Endpointing / segmentation.** Use CTC/RNN-T blank runs (or a small VAD) to
   detect pauses and finalize segments.
6. **Timestamps.** CTC/RNN-T give frame-level alignment for free (frame index ×
   frame period, e.g. 80 ms), so each word carries a start time.
7. **Latency dial.** Turn one knob: **chunk size** (cache-aware Conformer, WeNet
   U2++, Nemotron) or **delay τ** (Voxtral Realtime AdaRMSNorm). Smaller =
   faster/less accurate.

### 7.3 Running ASR + S2S together off ONE encoder

The encoder is the expensive, shared trunk; heads are cheap. Tap it once and run
two (or more) heads in parallel:

```
                         ┌──► CTC / RNN-T head ──► streaming SOURCE-language captions (low latency)
audio ─► streaming ──────┤
         encoder (12.5Hz) ├──► Thinker (translation) ─► Talker + Mimi codec ─► TARGET-language speech
                         └──► (optional) DSM head ───► streaming TARGET-language text (subtitles)
```

- The **Thinker already produces text** (its intermediate target-language
  tokens), so translated **subtitles are essentially free** — just read the
  Thinker's text stream.
- For **source-language captions**, add a lightweight **CTC or RNN-T head** on
  the shared encoder. Marginal extra compute, since the encoder is reused.
- This is exactly how omni stacks (`Qwen2.5-Omni/`, `Qwen3-Omni/`) expose
  transcription *and* translation *and* speech output from one model.

### 7.4 Minimal recipes (pseudocode)

**CTC — zero-latency greedy partials:**

```python
cache, prev = None, BLANK
for chunk in audio_stream:                  # e.g. 1–8 frames
    emb, cache = encoder.step(chunk, cache) # exact streaming, bounded memory
    for logit in ctc_head(emb):             # per frame
        y = logit.argmax()
        if y != BLANK and y != prev:        # collapse blanks + repeats
            emit_partial(y)
        prev = y
```

**RNN-T — standard streaming transducer:**

```python
cache, state, hist = None, pred_net.init(), []
for chunk in audio_stream:
    enc, cache = encoder.step(chunk, cache)
    for t in range(enc.frames):
        while True:                         # emit 0+ labels on this frame
            j = joiner(enc[t], pred_net(hist, state))
            y = j.argmax()
            if y == BLANK: break            # advance time
            emit(y); hist.append(y)         # emit label, stay on frame
```

**DSM (Voxtral Realtime) — one token per frame, with a latency dial:**

```python
prev, cache = PAD, None
for frame in audio_stream:
    e, cache = encoder.step(frame, cache)
    x = adapter(e) + embed(prev)            # fuse audio + previous token
    tok = decoder_step(x, g(tau)).argmax()  # tau = AdaRMSNorm latency knob
    if tok != PAD: emit(tok)                # [W] then word
    prev = tok
```

**Two-pass (WeNet U2++) — instant partials, accurate finals:**

```python
# pass 1 (always on): streaming CTC → partial text every chunk
# on endpoint: attention decoder rescores CTC n-best → final text
```

Wrap the emitted text in the **longest-stable-prefix** policy
(`Confucius4-R2T2/`) so what the user sees only ever grows.

### 7.5 Which head for which recommendation

- **Voxtral Realtime encoder** → **DSM head** (same per-frame cadence; τ gives the
  latency dial) for text, *or* a parallel **CTC/RNN-T** head for ultra-low-latency
  captions.
- **Qwen3-Omni / Qwen2.5-Omni** → the **Thinker** already emits streaming text;
  add a **CTC/RNN-T** head if you want sub-chunk source captions.
- **Nemotron 3.5 ASR / Cache-Aware Streaming Conformer** → **RNN-T** is the native
  head; chunk size is the latency dial. These are *already* streaming ASR systems.
- **WeNet U2++ / Uni-ASR** → **two-pass CTC+attention** / **LLM interleaving**;
  one model covers streaming partials and non-streaming-quality finals.

**Bottom line:** streaming ASR text and the S2S Thinker-Talker are not either/or.
Build the streaming encoder once, then attach whichever text head(s) you need —
CTC/RNN-T for low-latency captions, DSM/attention for LLM-grade text — and read
the Thinker's stream for translated subtitles, all from the one shared encoder.

## 8. Decision cheat-sheet

| Your priority | Pick |
|---|---|
| Best latency dial + modern streaming encoder | **Voxtral Realtime** (AdaRMSNorm τ) |
| Proven end-to-end S2S translation stack | **Qwen3-Omni** (AuT enc + Thinker-Talker) |
| Full-duplex single-LLM speech-in/speech-out | **Mimi** tokens (Moshi-style) |
| Tiny / on-device | **Nemotron 3.5 ASR 0.6B** or **Cache-Aware Conformer** |
| Unified streaming + offline from one model | **WeNet U2++** or **Uni-ASR** |
| Multi-speaker "who said what" | **VibeVoice-ASR-Streaming** |
| Flicker-free committed output | **Confucius4-R2T2** policy (on top of any encoder) |
| Robust multilingual features (offline/teacher) | **Whisper** / **wav2vec 2.0** |
| Token-efficient LLM input (copy the trick) | **Voxtral** 4-frame adapter → 12.5 Hz |

---

## 9. References (reproductions in this repo)

`Conformer/`, `wav2vec 2.0/`, `Emformer/`, `WeNet U2++/`, `Whisper/`,
`FastConformer/`, `Zipformer/`, `Cache-Aware Streaming Conformer/`, `Voxtral/`,
`Voxtral Realtime/`, `Qwen2.5-Omni/`, `Qwen3-Omni/`, `Uni-ASR/`,
`VibeVoice-ASR-Streaming/`, `Confucius4-R2T2/`, `Nemotron 3.5 ASR Streaming 0.6B/`,
`Mimi/`. Cross-paper relationships are registered in the root `papers.json`
(category *Audio & Speech Encoders*) and visualized in `citation_graph.html`.
