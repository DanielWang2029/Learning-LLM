# Qwen2.5-Math

**Qwen Team · 2024 · arXiv:2409.12122**

Qwen2.5-Math supports both natural-language Chain-of-Thought (CoT) and
**Tool-Integrated Reasoning (TIR)**.  In TIR, the model emits a program/tool
call during its reasoning, an executor returns the result, and the model
continues from that observation.  The report describes dedicated TIR data that
uses a Python interpreter for exact computation.

This is a tiny, transparent CPU reproduction of that interaction pattern—not a
reproduction of the paper's models, reward-model pipeline, training data, or
mathematical benchmark scores.  It compares three interpretable arithmetic
agents on synthetic word problems: a direct shortcut, a manual CoT trace with
a controlled large-number carrying slip, and TIR with a safe exact calculator.

## What the demo shows

The demo measures exact final-answer accuracy and prints a complete TIR trace:
the expression inside `<tool>…</tool>`, its actual executed result, and the
answer using that result.  Small arithmetic lets the CoT control succeed;
larger products make its deterministic manual-error model fail.  TIR remains
exact because execution, rather than language-only calculation, supplies the
value.

## Folder layout

```text
Qwen2.5-Math/
├── qwen25-math.pdf            # supplied paper
├── requirements.txt           # CPU-only PyTorch and NumPy
├── src/reasoner.py            # safe calculator and the three transparent agents
├── data/generate_data.py      # deterministic arithmetic word problems
├── demo/run_demo.py           # evaluation and JSON trace writer
└── visualization/index.html   # offline interactive CoT-versus-TIR view
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

## Run

```bash
python demo/run_demo.py
```

Expected output is `0.0%` direct, `50.0%` CoT, and `100.0%` TIR exact-match
accuracy on the eight controlled problems, followed by a visible executed tool
call.  The actual run output is saved to `data/tir_results.json`.

Open `visualization/index.html` directly with `file://` for its baked-in
offline example.  If the folder is served with `python -m http.server`, its
Load button can read the fresh JSON result.

## Code ↔ paper map

| Paper section | Reproduced idea | File |
|---|---|---|
| §3.1 | CoT and TIR supervision formats | `src/reasoner.py` |
| §3.1.2 | External Python-style executor for precise computation | `src/reasoner.py` |
| §3.3 | Exact answer as an outcome signal | `demo/run_demo.py` |
| §5.2 | Final-answer exact-match comparison | `demo/run_demo.py` |
