# Two Ways to Train an Audio Encoder: ASR "specialize-then-freeze" vs Omni "from-scratch"

> Scope: how the audio **encoder** is *trained* in two Qwen-family lineages
> reproduced in this repo — the **ASR-encoder recipe** (used by
> `Qwen-Audio-3.1-Realtime/`, which cites a *Qwen-Audio-3.0-ASR* technical
> report) and the **from-scratch AuT encoder** of `Qwen3-Omni/`
> (arXiv:2509.17765) — and exactly how the two training processes differ.

This file lives at the repo root (under `docs/`) because it compares *across*
papers rather than documenting the internals of one.

**Sourcing & honesty.** The ASR recipe below is the three-stage process stated in
Qwen-Audio-3.1-Realtime §3.1 (which defers architectural detail to the cited
*Qwen-Audio-3.0-ASR Technical Report*, Bian et al., 2026). The AuT details come
from the Qwen3-Omni technical report. Where a figure is reported (e.g. ~20M hours)
it is labelled as such; where the newest public lineage ("Qwen3-ASR-Flash") is
involved, the initialization is noted as inferred, not asserted.

---

## TL;DR

- **ASR encoder = "specialize-then-freeze."** Self-supervised pretraining →
  supervised **encoder–decoder ASR** training → **discard the decoder** → connect
  the **frozen** encoder to an LLM via an adapter → cooldown. Objective is narrow
  (transcription); the encoder is then held fixed while an LLM adapts on top.
- **Omni AuT encoder = "from-scratch, system-first."** Trained from scratch at
  massive scale (~20M hours) for **general audio understanding**, block-wise and
  streaming at **12.5 Hz**, then embedded in a **Thinker–Talker** omni model whose
  training actively prevents **modality interference** and extends to speech
  generation.
- Both end up feeding a Qwen LLM with semantic audio features, but one is a
  *narrow, frozen* ASR front end and the other is a *broad, jointly-trained* omni
  front end.

---

## Paradigm A — the ASR encoder: specialize, then freeze

Used by the Qwen-Audio ASR lineage that `Qwen-Audio-3.1-Realtime/` builds on. It
is **encoder-first** and driven by a transcription objective.

### Stage 1 — Self-supervised pretraining
The encoder learns general speech representations from unlabeled audio
(masked/predictive SSL). This gives a strong acoustic/phonetic prior before any
labels are used.

### Stage 2 — Supervised encoder–decoder (ASR) training
A **temporary decoder** is attached and the encoder+decoder are trained on
**paired audio–text** with an ASR objective. This shapes the encoder's outputs to
be transcription-relevant (semantic, content-focused). The decoder exists only to
provide this training signal.

### Stage 3 — Discard the decoder, align to the LLM, cooldown
- The temporary decoder is **discarded**.
- The now-**frozen** encoder is connected to a pretrained **Qwen LLM** through an
  **adapter**; only the **adapter + LLM** are trained for audio-text alignment.
- A final **cooldown** uses higher-quality, rebalanced multilingual and
  multi-task data to refine alignment and instruction following.

### Defining characteristics
- **Objective:** narrow — ASR / transcription (text-recoverable content).
- **Decoder:** throwaway (used only to train the encoder).
- **After alignment:** encoder is **frozen**; downstream systems adapt the LLM,
  not the encoder. (In `Qwen-Audio-3.1-Realtime/`, the agentic foundation training
  — Core-Cocktail SFT + M²-OPD — trains the LLM/adapter "Audio Student," *not* the
  encoder.)

---

## Paradigm B — the Omni (AuT) encoder: train from scratch, as part of the system

Used by `Qwen3-Omni/`. It is **system-first** and driven by general audio
understanding, not just transcription.

### Stage 1 — Large-scale pretraining from scratch
**AuT (Audio Transformer)** is trained **from scratch** on a very large corpus
(the report cites ~**20M hours**) with supervised/weakly-supervised objectives
spanning **ASR *and* broader audio understanding**. It outputs a low **~12.5 Hz**
representation and uses **block-wise (causal) attention** so it can run streaming.
(It replaces the Whisper front end used in earlier Qwen-Audio models.)

### Stage 2 — Alignment into an omni LLM (mixed modalities)
The encoder feeds a **Thinker–Talker MoE** via adapters. Multimodal pretraining
mixes **text, audio, image, and video**, deliberately keeping a strong **text**
share so that adding audio/vision does **not regress** text reasoning — the
"no modality interference" property is a headline training claim.

### Stage 3 — Post-training (and the generation path)
SFT + RL across modalities. The **Talker** learns streaming speech via
multi-codebook **MTP** (multi-token prediction) and a lightweight causal
**Code2Wav** vocoder, giving low first-packet latency.

### Defining characteristics
- **Objective:** broad — general audio understanding (ASR + reasoning, scenes,
  paralinguistics).
- **No throwaway decoder:** the encoder is built directly for the omni LLM.
- **Training scope:** system-centric; the encoder is one component of an
  end-to-end speech/text/vision-in, speech/text-out model, trained with heavy
  mixed-modality data.

---

## Side-by-side differences

| Dimension | ASR encoder (specialize-then-freeze) | Omni AuT encoder (from-scratch) |
|---|---|---|
| **Primary objective** | ASR / transcription (narrow) | General audio understanding (broad) |
| **Pretraining** | SSL **then** supervised encoder–decoder ASR | **From scratch**, ~20M hr supervised/weak audio |
| **Temporary decoder** | Yes — attach for ASR, then **discard** | No throwaway decoder |
| **Encoder after alignment** | **Frozen**; adapt the LLM only | Aligned/trained within the omni stack |
| **Data mixture** | Paired audio–text (+ rebalanced cooldown) | **Mixed modalities** with anti-interference text share |
| **Frame rate / streaming** | ASR-tuned latency | **12.5 Hz**, **block-wise causal** streaming |
| **Role downstream** | Feed a Qwen **text** LLM for transcription | Feed a **Thinker–Talker MoE**; pairs with MTP + Code2Wav for speech out |
| **Scope of training** | Encoder-centric, then freeze | System-centric, end-to-end |
| **Representation type** | Semantic (content) | Semantic, but broader (understanding) |

---

## Why the differences exist (design rationale)

- **Specialization vs generality.** An ASR encoder only needs to preserve *what
  was said*, so an encoder-decoder ASR objective (then freeze) is efficient and
  robust. An omni encoder must support reasoning, audio scenes, and paralinguistic
  cues, so it is trained from scratch on far broader data and objectives.
- **Freeze vs joint training.** Freezing the ASR encoder keeps a stable,
  well-understood front end and makes downstream LLM training cheap and
  non-destructive. Omni models train the encoder as part of the system because the
  representation must serve many tasks and co-adapt with the LLM.
- **Streaming as a first-class goal.** The omni encoder bakes in block-wise causal
  attention and a 12.5 Hz rate for real-time I/O; a classic ASR encoder optimizes
  latency within the ASR task but is not necessarily built for an omni duplex loop.
- **Modality interference.** Only the omni recipe has to defend text/vision
  quality while adding audio, hence its explicit mixed-modality data strategy; the
  ASR recipe has no such constraint.

---

## The modern convergence

The two paradigms are merging. The public **Qwen3-ASR-Flash** is described as
**built on the Qwen3-Omni foundation** and then ASR-specialized — i.e. in the
newest lineage the "ASR encoder" may **start from the AuT encoder** and be further
trained with the specialize-then-freeze recipe above, rather than being an
independent from-scratch encoder. (This initialization is inferred from public
descriptions, not asserted by the reproduced reports.) The practical reading:

- Train a **broad, from-scratch** encoder once (AuT-style), then
- **ASR-specialize + freeze** a copy of it for transcription-centric products.

So "ASR encoder" and "omni encoder" increasingly denote **two training *stages* of
one lineage** rather than two unrelated encoders.

---

## Implications for a speech-to-speech / voice-agent build

- Want **transcription-grade** semantic features and a stable front end? The
  **specialize-then-freeze ASR encoder** is the efficient choice; adapt the LLM on
  top (as Qwen-Audio-3.1-Realtime does with M²-OPD).
- Want one front end for **understanding + reasoning + streaming + generation**?
  The **from-scratch omni AuT encoder** (12.5 Hz, block-wise) embedded in a
  Thinker–Talker stack is the better fit.
- Either way the encoder is **semantic**; acoustic/voice information for *output*
  comes from the codec/vocoder path (see
  `docs/streaming-audio-encoders-for-speech-to-speech.md`).

---

## References (reproductions in this repo)

- `Qwen-Audio-3.1-Realtime/` — agentic voice model; §3.1 states the ASR-encoder
  three-stage recipe and the Core-Cocktail SFT + M²-OPD foundation training.
- `Qwen3-Omni/` — AuT block-wise streaming encoder (12.5 Hz) + Thinker–Talker MoE
  + MTP talker; the from-scratch omni paradigm.
- `Whisper/`, `Voxtral/` — the earlier Whisper-encoder front end that AuT replaced.
- `docs/streaming-audio-encoders-for-speech-to-speech.md` — broader streaming
  encoder comparison and the semantic-vs-acoustic distinction.
