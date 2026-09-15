# DeepSeek-V3 Technical Report

A faithful, minimal, and fully self-contained reproduction of the three signature
components of the *"DeepSeek-V3 Technical Report"* (2024, arXiv:2412.19437, §2),
at tiny CPU scale. Everything needed to read, run, and understand them lives in
**this folder**.

DeepSeek-V3 is a large Mixture-of-Experts LLM. This repo implements the three
architectural ideas that make it distinctive:

- **Multi-Head Latent Attention (MLA)** — compress keys/values into a small
  low-rank latent that is what gets cached, then up-project on the fly, with a
  *decoupled* RoPE key. This shrinks the KV cache (the memory bottleneck of
  long-context inference) while preserving quality.
- **DeepSeekMoE** — a feed-forward layer of many *fine-grained* routed experts
  plus a few *always-on shared* experts, balanced without an auxiliary loss (via a
  per-expert routing bias).
- **Multi-Token Prediction (MTP)** — a second head predicts the token *two* steps
  ahead, densifying the training signal and enabling speculative decoding.

```
DeepSeek-V3/
├── deepseek-v3.pdf             # the paper itself
├── requirements.txt            # pinned CPU dependencies (torch + numpy)
├── src/
│   ├── mla.py                  #   §2.1  Multi-Head Latent Attention (+ MHA baseline)
│   ├── moe.py                  #   §2.2  DeepSeekMoE: fine-grained + shared experts
│   ├── mtp.py                  #   §2.2  Multi-Token Prediction module
│   └── model.py                #   §2    the three combined in a small decoder LM
├── data/
│   └── generate_data.py        # writes data/task.json (copy-task layout)
├── demo/
│   └── run_demo.py             # trains, compares MLA vs MHA, reports MTP + MoE
└── visualization/
    └── index.html              # MLA compression + KV-cache savings + MoE + MTP
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
python demo/run_demo.py        # ~35s on CPU
```

The demo trains the full mini DeepSeek-V3 (MLA + DeepSeekMoE + MTP) on a toy copy
task, then retrains with plain multi-head attention to compare quality and
KV-cache size.

## 3. Expected output

The model reaches **100% next-token and 100% two-ahead (MTP) accuracy**, MLA
matches MHA quality while caching **2.67× less**, and the routed experts stay
balanced:

```
Full model: next-token 100.0% | MTP two-ahead 100.0%

DeepSeekMoE routed-expert load (ideal 0.250 each):
  E0:0.316 E1:0.261 E2:0.285 E3:0.164 E4:0.353 E5:0.228 E6:0.155 E7:0.237

MLA vs standard MHA (same budget, fresh models):
  MLA  : next-token acc 100.0% | KV cache  48 floats/token/layer
  MHA  : next-token acc 100.0% | KV cache 128 floats/token/layer
  => MLA matches quality and caches 2.67x less

MTP example — predict the next two tokens at once:
  true (t+1,t+2): (3, 19)   pred (t+1,t+2): (3, 19)
```

At this tiny width the KV-cache saving is 2.67×; in the full model (many wide
heads vs a 512-dim latent) the ratio is far larger. Results are written to
`data/dsv3_results.json` for the visualization.

## 4. Explore the visualization

Open `visualization/index.html` in any browser for the MLA latent-compression
diagram, the KV-cache savings bar chart, the DeepSeekMoE routing view (shared +
fine-grained experts with measured load), and the MTP two-token-ahead schematic —
all using the real numbers from your run. Works from `file://`; serve over HTTP to
load fresh JSON:

```bash
python -m http.server 8000     # then open http://localhost:8000/visualization/
```

## Code ↔ paper map

| Paper section | Component | File |
|---|---|---|
| §2.1 | Multi-Head Latent Attention (compressed KV cache) | `src/mla.py` |
| §2.1 | Decoupled RoPE key | `src/mla.py` |
| §2.2 | DeepSeekMoE (fine-grained + shared experts) | `src/moe.py` |
| §2.2 | Auxiliary-loss-free load balancing (router bias) | `src/moe.py` |
| §2.2 | Multi-Token Prediction | `src/mtp.py` |
| §2 | Combined decoder-only model | `src/model.py` |
