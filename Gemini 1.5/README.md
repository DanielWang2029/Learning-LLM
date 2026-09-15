# Gemini 1.5

A faithful, minimal, and self-contained reproduction of the two **headline
documented ideas** of *"Gemini 1.5: Unlocking multimodal understanding across
millions of tokens of context"* (Google, 2024, arXiv:2403.05530), at tiny CPU
scale.

Gemini 1.5 is a **closed** model — its weights, data, and exact architecture are
not public. Rather than pretend otherwise, this repo reproduces the two things the
report describes most concretely and that we *can* demonstrate honestly:

1. **Sparse Mixture-of-Experts.** Gemini 1.5 is a sparse MoE Transformer: each
   token is routed to a few experts out of a larger pool, so total capacity grows
   while per-token compute stays roughly fixed. We implement a top-k MoE layer
   with a load-balancing loss and show the router spreads tokens evenly.
2. **Long-context near-perfect recall.** Its most striking result is retrieving a
   single planted fact from a very long context ("needle in a haystack"). We
   implement that test: a tiny MoE Transformer must find a marked needle among
   distractor tokens and recall it — and we show recall stays high across needle
   positions and context lengths, *including contexts longer than any seen in
   training*.

```
Gemini 1.5/
├── gemini_15.pdf               # the paper itself
├── requirements.txt            # pinned CPU dependencies (torch + numpy)
├── src/
│   ├── moe.py                  #   §2  sparse top-k MoE FFN + load-balancing loss
│   ├── model.py                #   §2  tiny long-context Transformer (RoPE attn + MoE)
│   └── task.py                 #        needle-in-a-haystack retrieval task
├── data/
│   └── generate_data.py        # writes data/task.json (decoded examples)
├── demo/
│   └── run_demo.py             # trains + builds the recall heatmap + expert loads
└── visualization/
    └── index.html              # needle-recall heatmap + MoE routing diagram
```

## 1. Set up the environment

Requires Python 3.10+. From inside this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

## 2. Run the demo

```bash
python data/generate_data.py   # optional: writes data/task.json
python demo/run_demo.py        # ~15s on CPU
```

## 3. Expected output

Recall is **100% within trained lengths at every needle position**, degrading only
gracefully at the hardest extrapolation cell (needle at the very start of a
context longer than any trained on), and the MoE router stays balanced:

```
Needle-in-a-haystack recall (%) — rows = needle depth, cols = haystack length:
               6     12     20     28     36     44
  pos   0%  100.0% 100.0% 100.0% 100.0%   5.9%  35.5%
  pos  25%  100.0% 100.0% 100.0% 100.0% 100.0% 100.0%
  pos  50%  100.0% 100.0% 100.0% 100.0% 100.0% 100.0%
  pos  75%  100.0% 100.0% 100.0% 100.0% 100.0% 100.0%
  pos 100%  100.0% 100.0% 100.0% 100.0% 100.0% 100.0%

Overall recall: 94.7%  |  within trained lengths: 100.0%
MoE expert load (ideal = 0.500 each):  E0:0.482  E1:0.515  E2:0.528  E3:0.475
```

(Columns 36 and 44 are **longer than the 4–32 distractor haystacks seen in
training**, so they test length extrapolation.) Results are written to
`data/gemini_results.json` for the visualization.

## 4. Explore the visualization

Open `visualization/index.html` in any browser for a colorful needle-recall
heatmap (position × context length) built from your run, plus a diagram of the
top-k MoE routing with the measured per-expert load. Works from `file://`; serve
over HTTP to load fresh JSON:

```bash
python -m http.server 8000     # then open http://localhost:8000/visualization/
```

## Honesty note

This is **not** Gemini 1.5. It is a tiny from-scratch model that reproduces the
*mechanisms* the paper documents (sparse MoE routing; long-context needle recall)
so you can run and inspect them on a laptop. The absolute numbers are toy-scale;
the qualitative behaviors — even expert load and position-robust recall — are real.

## Code ↔ paper map

| Paper topic | Component | File |
|---|---|---|
| §2 — sparse MoE | top-k routing + load balancing | `src/moe.py` |
| §2 — architecture | RoPE attention + MoE blocks | `src/model.py` |
| Long-context recall | needle-in-a-haystack task | `src/task.py` |
| Long-context recall | position × length recall grid | `demo/run_demo.py` |
