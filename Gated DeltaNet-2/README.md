# Gated DeltaNet-2: Decoupling Erase and Write in Linear Attention

A minimal, self-contained reproduction of the **core operator** from
Hatamizadeh, Choi & Kautz, *"Gated DeltaNet-2: Decoupling Erase and Write in
Linear Attention"* (2026), arXiv:**2605.22791**. Everything needed to read,
run, and understand the mechanism lives in this folder: the paper PDF, a
from-scratch PyTorch implementation, a runnable CPU demo, and an interactive
visualization.

> **Honesty note.** This is a 2026 paper. The demo faithfully reproduces the
> paper's most characteristic *documented mechanism* — the **Gated Delta
> Rule-2** state update with **decoupled channel-wise erase and write gates**
> (paper §3, Eq. 8–10) — at tiny CPU scale on a toy associative-recall task.
> It does **not** reproduce the 1.3B-parameter model, the fused chunkwise/WY
> Triton kernels (§3.3), the hybrid sliding-window blocks (§3.5), or the
> FineWeb-Edu training. The plain O(L) recurrence here computes the same
> function the kernel does, just without the speed optimizations.

## What the paper is about

Linear attention replaces softmax attention's growing KV-cache with a
**fixed-size recurrent state** `S ∈ R^(d_k×d_v)`, giving linear-time sequence
mixing and constant-memory decoding. The hard part is *editing* this compressed
memory without scrambling existing associations. Delta-rule models
(DeltaNet, Gated DeltaNet, KDA) subtract the current read before writing, but
they use a **single scalar gate `β`** to control two different things at once:

- how much old content to **erase** (a *key-side* decision), and
- how much new content to **write** (a *value-side* decision).

Gated DeltaNet-2's contribution is to **decouple** these with two independent
channel-wise gates — an erase gate `b_t ∈ [0,1]^d_k` on the key axis and a
write gate `w_t ∈ [0,1]^d_v` on the value axis:

```
S_t = ( I − k_t (b_t ⊙ k_t)ᵀ ) D_t S_{t-1}  +  k_t (w_t ⊙ v_t)ᵀ      (Eq. 10)
o_t = S_tᵀ q_t
```

Tying `b = w = β·1` recovers KDA; further tying the decay `D_t` recovers Gated
DeltaNet — so the new rule is strictly more expressive.

## What the demo shows

`demo/run_demo.py` runs two parts on CPU in well under a minute:

1. **Mechanism check (no training).** Build a state with one-hot keys, write
   `(k₁,v₁)` and `(k₂,v₂)`, then **overwrite** `k₁` with a new value and read
   `k₁` back. The delta rule's erase step returns the *latest* value exactly,
   while plain additive linear attention (`S += k vᵀ`, no erase) returns a
   scrambled `v₁ + v₁_new`.
2. **Learned recall with overwrite.** Train two tiny models on a key-value
   recall task where keys are sometimes rewritten: one with **decoupled**
   channel-wise erase/write gates (the paper), one with a **tied scalar** gate
   (the KDA-style baseline). Recall accuracy is reported and split by whether
   the queried key was overwritten.

Both models use the same fixed-size `32×32` state, so runtime is linear in
sequence length with constant memory.

## Folder layout

```
Gated DeltaNet-2/
├── gated_deltanet-2.pdf        # the paper
├── requirements.txt            # pinned deps (CPU PyTorch + numpy)
├── src/
│   ├── gated_deltanet2.py      # §3  Gated Delta Rule-2 + tiny model + additive baseline
│   └── task.py                 #     key-value associative-recall-with-overwrite task
├── data/
│   ├── generate_demo_data.py   # write a human-readable sample of the task
│   └── results.json            # metrics + trace written by the demo (generated)
├── demo/
│   └── run_demo.py             # end-to-end CPU demo (mechanism check + training)
└── visualization/
    └── index.html              # interactive, offline diagram + equations + real data
```

## Setup

Requires Python 3.10+. This folder reuses the shared virtual environment that
ships with the repository:

```bash
source "../Attention Is All You Need/.venv/bin/activate"
```

To create a fresh environment instead:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

## How to run

```bash
python demo/run_demo.py                 # main demo (~25s on CPU)
python data/generate_demo_data.py       # optional: inspect the task's tokens
```

## Expected output

```
PART A  Mechanism check: overwrite one key, then read it back
  Gated Delta Rule-2 read(k1)   : [0.0, 0.0, 0.0, 5.0]   <- matches v1_new (old value erased)
  additive linear attn read(k1) : [1.0, 0.0, 2.0, 5.0]   <- v1 + v1_new (scrambled!)
  error vs latest value  |  gated-delta: 0.000   additive: 2.236

PART B  Learned recall with overwrite: decoupled vs tied gates
Recall accuracy on held-out sequences (800 sequences):
  model                      overall   overwritten   not-overwritten
  Gated DeltaNet-2 (decoupled)     85.1%         95.0%             75.2%
  KDA-style (tied scalar)      70.1%         93.0%             47.2%
Fixed-size recurrent state: 32x32 = 1024 scalars (... O(L) time, O(1) memory).
```

Exact numbers vary slightly with the environment, but the decoupled model
reliably (a) recalls overwritten values in the mechanism check with zero error
and (b) achieves higher overall recall than the tied-gate baseline. The demo
exits non-zero if this does not hold.

## Explore the visualization

Open `visualization/index.html` in any browser (works offline via `file://`).
It renders the six-stage Gated Delta Rule-2 update as a clickable pipeline, the
key equations, the mechanism-check vectors, the decoupled-vs-tied accuracy
bars, and a token-by-token trace with per-step erase/write magnitudes — all
from the real `data/results.json`. For live data, serve the folder:

```bash
python -m http.server 8000    # then open http://localhost:8000/visualization/
```

## Code ↔ paper map

| Paper section | Concept | Where in code |
|---|---|---|
| §2.1, Eq. 1 | Additive linear-attention recurrence (no erase) | `additive_linear_attention` in `src/gated_deltanet2.py` |
| §2.2, Eq. 5–7 | DeltaNet / Gated DeltaNet / KDA (scalar `β`) | tied-gate branch (`decouple=False`) in `GatedDeltaRule2` |
| §3.1, Eq. 8 | Gated erase `e=b⊙k` and write `z=w⊙v` | `GatedDeltaRule2.gates`, `forward` |
| §3.1, Eq. 9–10 | Gated Delta Rule-2 state update | `GatedDeltaRule2.forward` (the recurrence loop) |
| §3.1, Eq. 12 | Channel-wise log-decay `α=exp(g)` | `GatedDeltaRule2.gates` (`w_decay`, `a_decay`) |
| §3.5 | Block design (q/k L2-norm, SiLU output gate) | `GatedDeltaRule2` projections + `out_norm`/`w_out_gate` |
| Table 3 (RULER) | Multi-key recall under a fixed-size state | `src/task.py` (recall with overwrite) |
