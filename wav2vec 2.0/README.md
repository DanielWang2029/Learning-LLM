# wav2vec 2.0: Self-Supervised Learning of Speech Representations

A faithful, minimal, self-contained reproduction of **wav2vec 2.0** from Baevski
et al., *"wav2vec 2.0: A Framework for Self-Supervised Learning of Speech
Representations"* (2020), arXiv:2006.11477. Everything needed to read, run, and
understand the core idea lives in **this folder**: the paper PDF, a from-scratch
implementation, a runnable CPU demo on synthetic audio, and an interactive
visualization.

- **Authors:** Alexei Baevski, Henry Zhou, Abdelrahman Mohamed, Michael Auli (Facebook AI)
- **Year:** 2020 &nbsp; **arXiv:** [2006.11477](https://arxiv.org/abs/2006.11477)

## Plain-English summary

wav2vec 2.0 learns speech representations from **unlabelled** audio. A CNN
**feature encoder** turns the raw waveform into latent frames. A random subset of
those frames is **masked**, and a Transformer **context network** must fill in the
gaps. The training signal is **contrastive**: at each masked step the model must
pick the correct **quantized** latent (its own product-quantized code) out of a
set of distractors. The quantization uses a Gumbel-softmax **codebook**, so the
targets are discrete. After pretraining, a thin layer fine-tuned on a little
labelled data reaches strong ASR accuracy — the representations do the heavy
lifting.

## What the demo shows

We run the full self-supervised objective on synthetic structured "speech"
(sequences of held tones). The demo prints, over training:

- **masked contrastive accuracy** — the fraction of masked frames where the true
  quantized latent beats all distractors — rising from ~5% (chance, 1/21) to
  **~90%**, and
- **codebook perplexity** — how many of the 64 codewords are actively used.

This demonstrates both halves of the method working: the context network learns
to predict masked content, and the Gumbel codebook stays diverse.

## Folder layout

```
wav2vec 2.0/
├── wav2vec_20.pdf             # the paper itself
├── requirements.txt           # pinned deps (CPU PyTorch + numpy)
├── src/
│   └── model.py               #   §2  FeatureEncoder, GumbelVectorQuantizer,
│                              #       ContextTransformer, contrastive Wav2Vec2
├── data/
│   └── generate_audio.py      #   numpy synthesizer for structured tone speech
├── demo/
│   └── run_demo.py            #   pretrains, writes data/demo_sample.json
└── visualization/
    └── index.html             #   interactive SSL pipeline + curves + codebook
```

## Setup

Requires Python 3.10+. Reuse the shared virtual-env at the repo root, or:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

## How to run

```bash
python demo/run_demo.py
```

Pretrains the full model on CPU in well under a minute and writes
`data/demo_sample.json` (accuracy/perplexity curves, codebook histogram, and one
masked example) for the visualization.

## Expected output

```
wav2vec 2.0 self-supervised pretraining on synthetic speech (CPU)
chance accuracy = 1/21 = 4.8%   codebook = 64 codewords
step  260/260 | loss 0.78 | masked-acc  78.3% | codebook perplexity 54.7/64 | temp 1.76
Final masked contrastive accuracy: ~90%  (chance 4.8%)
Final codebook perplexity: ~53 / 64 codewords in use
OK: masked contrastive accuracy is well above chance and the codebook is used.
```

## Explore the visualization

Open `visualization/index.html` in any browser (offline `file://` works) for the
masked contrastive pipeline (CNN → mask → Transformer → quantized targets), the
contrastive-accuracy and codebook-perplexity curves, the learned codebook usage
histogram, and a masked example utterance. To load fresh data instead of the
baked-in sample:

```bash
python -m http.server 8000   # then open http://localhost:8000/visualization/
```

## Code ↔ paper map

| Paper section | Component | File |
|---|---|---|
| §2.1 | CNN feature encoder (raw waveform → latents) | `src/model.py` → `FeatureEncoder` |
| §2.2 | Product quantization (Gumbel-softmax codebook) | `src/model.py` → `GumbelVectorQuantizer` |
| §2.1 | Transformer context network (+ conv positions) | `src/model.py` → `ContextTransformer` |
| §3.1 | Span masking of latent time steps | `src/model.py` → `compute_mask_indices` |
| §3.2 | Contrastive (InfoNCE) + diversity objective | `src/model.py` → `Wav2Vec2.forward` |

### Note on faithfulness

This is a small pretraining-only reproduction: the CNN, codebook, mask lengths
and codeword counts are scaled far down, and there is no fine-tuning/CTC stage or
LibriSpeech data. The learning mechanism — masked, product-quantized, contrastive
representation learning — is implemented exactly as in the paper.
