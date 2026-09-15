# Attention Is All You Need

A faithful, minimal, and fully self-contained reproduction of the Transformer
from Vaswani et al., *"Attention Is All You Need"* (2017). Everything needed to
read, run, and understand this paper lives in **this folder**: the paper PDF,
the reference implementation, runnable demos with data, the development
environment, and an interactive visualization.

```
Attention Is All You Need/
├── attention is all you need.pdf   # the paper itself
├── requirements.txt                # pinned dependencies (CPU PyTorch)
├── transformer/                    # reference implementation (the paper, in code)
│   ├── attention.py                #   §3.2  scaled dot-product & multi-head attention
│   ├── positional.py               #   §3.5  sinusoidal positional encoding
│   ├── feedforward.py              #   §3.3  position-wise feed-forward network
│   ├── layers.py                   #   §3.1  encoder/decoder layers & stacks
│   └── model.py                    #   §3    full encoder–decoder Transformer
├── demo/                           # runnable code to experience the structure
│   ├── generate_demo_data.py       #   creates toy copy/reverse datasets
│   ├── inspect_structure.py        #   narrates one forward pass, stage by stage
│   └── copy_task.py                #   trains end-to-end to ~100% accuracy
├── data/                           # demo datasets & traces (generated)
└── visualization/
    └── index.html                  # colorful, interactive architecture diagram
```

## 1. Set up the development environment

Requires Python 3.10+. From inside this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

This installs the CPU build of PyTorch, so everything runs on a laptop with no
GPU. (On Debian/Ubuntu you may first need `sudo apt-get install python3-venv`.)

## 2. Generate the demo data

```bash
python demo/generate_demo_data.py
```

Writes small, human-readable `copy_dataset.json` and `reverse_dataset.json`
files to `data/`. Each example is a short integer sequence with `BOS`/`EOS`
markers — concrete input you can watch flow through the model.

## 3. Experience the structure of the paper

```bash
python demo/inspect_structure.py
```

Runs a single forward pass through a small Transformer and prints the tensor
shape after **every** stage (embedding → positional encoding → each encoder and
decoder sub-layer → generator), annotated with the paper section it comes from.
It also prints a real multi-head attention matrix as an ASCII heatmap and writes
`data/forward_trace.json`.

## 4. Train it end-to-end

```bash
python demo/copy_task.py
```

Trains the Transformer on the classic sequence-**copy** sanity task. With the
default settings it reaches ~100% exact-sequence accuracy in about half a minute
on CPU, then prints a worked example. All hyperparameters are flags
(`python demo/copy_task.py --help`).

## 5. Explore the visualization

Open `visualization/index.html` in any browser for a color-coded, clickable map
of the full architecture: click any block to see its role, equations, and
implementation file; inspect real trained attention heatmaps (the cross-attention
shows the diagonal alignment the model learns); and see the positional-encoding
sinusoids plotted live.

For live (rather than baked-in) attention data, serve the folder so the page can
fetch the generated JSON:

```bash
python -m http.server 8000        # then open http://localhost:8000/visualization/
```

## Code ↔ paper map

| Paper section | Component | File |
|---|---|---|
| §3.1 | Encoder/decoder layers, Add & Norm | `transformer/layers.py` |
| §3.2.1 | Scaled dot-product attention | `transformer/attention.py` |
| §3.2.2 | Multi-head attention | `transformer/attention.py` |
| §3.2.3 | Masked & cross attention | `transformer/layers.py`, `transformer/model.py` |
| §3.3 | Position-wise feed-forward | `transformer/feedforward.py` |
| §3.4 | Embeddings, generator, softmax | `transformer/model.py` |
| §3.5 | Positional encoding | `transformer/positional.py` |
