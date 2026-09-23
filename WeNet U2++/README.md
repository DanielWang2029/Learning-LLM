# WeNet U2++

A faithful, minimal, CPU-only reproduction of the core ideas in **"U2++:
Unified Two-pass Bidirectional End-to-end Model for Speech Recognition"** —
Di Wu, Binbin Zhang, Chao Yang, Zhendong Peng, Wenjing Xia, Xiaoyu Chen,
Xin Lei (WeNet), 2021.
arXiv: [2106.05642](https://arxiv.org/abs/2106.05642).

## The idea in plain English

U2++ builds one model that is **both** streaming and non-streaming, and decodes
in **two passes**:

1. **Dynamic chunk masking.** During training each batch uses a *random* chunk
   size, and attention is restricted so a frame sees all left history but only
   up to the end of its own chunk. A large chunk is full-context (offline); a
   small chunk is low-latency streaming. The *same* weights therefore run at any
   latency.
2. **Two-pass decoding.** A **CTC** head runs frame-synchronously in the first
   pass (streaming). Its n-best hypotheses are then **rescored** by two
   autoregressive attention decoders — a left-to-right (L2R) and a
   right-to-left (R2L) — that model label dependencies CTC cannot (CTC is
   conditionally independent across output positions). The R2L decoder is the
   "++" over the original U2: it brings *right* label context into rescoring.

Joint training loss and rescoring score:

```
L        = λ·CTC + (1−λ)·(L2R + R2L)/2
S_final  = λ·S_CTC + (1−α)·S_L2R + α·S_R2L
```

## What the demo shows

The task is an audio→token problem built to make the second pass matter:
tokens **C** and **D** are clean, distinct anchors; **A** and **B** differ only
by a *weak, noisy* side-tone, and the label grammar is deterministic
(`C→A, D→B`). CTC confuses the noisy A/B tokens and produces the occasional
insertion; the attention decoders learn the grammar and fix those errors.

The demo:

1. trains one U2++ model (shared encoder + CTC + L2R + R2L) with dynamic chunk
   masking,
2. shows the **same weights** decode across chunk sizes (streaming and offline),
3. shows **attention rescoring** lowers the streaming CTC token error (and
   prints a concrete hypothesis it repaired).

Expected: rescoring cuts streaming token error substantially (e.g. ~4.6% → ~2.8%,
a ~38% relative reduction) and raises exact-sequence accuracy (e.g. ~84% → ~90%).

## Folder layout

```
WeNet U2++/
├── wenet_u2++.pdf                # the paper
├── requirements.txt            # torch==2.8.0, numpy>=1.26 (CPU)
├── src/
│   ├── conformer.py            # shared encoder + dynamic chunk masking
│   ├── decoder.py              # L2R and R2L attention decoders
│   ├── search.py               # CTC greedy + prefix beam search + edit distance
│   └── u2pp.py                 # U2++ model: CTC first pass + rescoring
├── data/
│   └── audio_synth.py          # numpy audio synth + manual log-mel
├── demo/
│   └── run_demo.py             # train, chunk sweep, CTC vs rescoring
└── visualization/
    └── index.html              # two-pass diagram + dynamic-chunk mask + results
```

## Setup

Reuse the shared CPU virtual environment (Python 3.10+):

```bash
source "/workspace/Attention Is All You Need/.venv/bin/activate"
```

Or create a fresh one:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

## How to run

```bash
python demo/run_demo.py
```

Runs in ~50 s on one CPU thread. Flags include `--stream-chunk`, `--epochs`,
`--ctc-weight` (`--help`).

## Expected output

```
[2/3] Same weights, full-context vs streaming (CTC greedy) ...
      mode               token err   seq acc
      chunk=4                 5.2%     83.6%
      chunk=8                 4.6%     84.4%
      chunk=16                3.6%     89.1%
      full/offline            8.4%     71.1%

[3/3] Attention rescoring at streaming chunk=8 ...
decode mode                            token err     seq acc
CTC greedy (streaming)                      4.6%       84.4%
+ attention rescoring (U2++)                2.8%       89.8%

Rescoring cuts streaming token error by 38% relative (4.6% -> 2.8%).
```

(The same model decodes at every chunk size; exact numbers vary a little run to
run because the model is tiny and randomly initialized.)

It also writes `data/demo_results.json` for the visualization.

## Explore the visualization

Open `visualization/index.html` in any browser (offline via `file://`) for the
two-pass architecture (click each head), the dynamic-chunk attention masks
(streaming vs full context), and the CTC-vs-rescoring results with a repaired
example. For live data:

```bash
python -m http.server 8000    # then open http://localhost:8000/visualization/
```

## Code ↔ paper map

| Paper section | Contribution | File / symbol |
|---|---|---|
| §3.1 | Shared encoder, dynamic chunk masking | `src/conformer.py` → `ConformerEncoder`, `chunk_mask` |
| §3.1 | CTC first pass (streaming) | `src/u2pp.py` → `U2PP.ctc_greedy`; `src/search.py` |
| §3.1 | L2R + R2L attention decoders | `src/decoder.py` → `AttentionDecoder` |
| §3.3 | Two-pass attention rescoring | `src/u2pp.py` → `U2PP.rescore` |
| §3.1 | Joint CTC + AED training | `src/u2pp.py` → `U2PP.loss` |
| §3.3 | Streaming CTC vs rescored metric | `demo/run_demo.py` |
