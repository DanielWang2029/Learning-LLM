# Llama 2

A faithful, minimal, and fully self-contained reproduction of the key
architecture change in **"Llama 2: Open Foundation and Fine-Tuned Chat Models"**
(Touvron et al., 2023, [arXiv:2307.09288](https://arxiv.org/abs/2307.09288)).
Everything needed to read, run, and understand the paper's headline *modeling*
change lives in **this folder**: the paper PDF, a small from-scratch
implementation, a runnable CPU demo with data, and an interactive visualization.

## What Llama 2 is (plain English)

Llama 2 is the successor to LLaMA: a family of open decoder-only language models
(7B/13B/70B) plus fine-tuned **chat** models. It keeps LLaMA's recipe (RMSNorm
pre-norm, RoPE, SwiGLU) and adds one important architecture change for the
larger models (paper Section 2.2):

- **Grouped-Query Attention (GQA)**: instead of one key/value head per query
  head (standard multi-head attention), the query heads are split into groups
  and each group shares a single key/value head. GQA sits between full
  multi-head attention (MHA) and multi-query attention (MQA):

  | Regime | # K/V heads | KV cache | Quality |
  |---|---|---|---|
  | MHA | = n_heads | largest | best |
  | **GQA** | a few (e.g. 8) | **small** | ~MHA |
  | MQA | 1 | smallest | slightly worse |

Fewer K/V heads means a smaller **KV cache** during generation — the dominant
memory cost of serving long contexts — with quality close to full MHA.

Beyond architecture, Llama 2-Chat is produced by a multi-stage alignment
pipeline — supervised fine-tuning (SFT) followed by **RLHF** with rejection
sampling and PPO, using two reward models for helpfulness and safety. That
pipeline is out of scope for this minimal repo (which focuses on the
*architecture*), but it is summarized in the visualization.

## What the demo shows

The demo trains a tiny Llama 2 with **GQA** on a toy **copy task** and reaches
**100% exact-copy accuracy**. It then:

- retrains the model as **MHA / GQA / MQA** and shows all three learn the task
  (so GQA does not hurt quality here), and
- **quantifies the KV-cache memory** of each regime, both for the demo config
  and at Llama-2-70B scale:

```
KV-cache at Llama-2-70B scale (n_layers=80, n_heads=64, head_dim=128, seq=4096, fp16):
  MHA (n_kv=64) | 10.00 GB |  1.0x
  GQA (n_kv= 8) |  1.25 GB |  8.0x smaller than MHA
  MQA (n_kv= 1) | 160.00 MB| 64.0x smaller than MHA
```

That 8× reduction (with near-MHA quality) is exactly why Llama 2 adopted GQA.

```
Llama 2/
├── llama_2.pdf               # the paper itself
├── requirements.txt          # pinned dependencies (CPU PyTorch)
├── src/                      # the paper's architecture, in code
│   ├── rope.py               #   RoPE rotary position embeddings (from LLaMA)
│   ├── normalization.py      #   RMSNorm (from LLaMA)
│   ├── feedforward.py        #   SwiGLU (from LLaMA)
│   ├── attention.py          #   Grouped-Query Attention (+ kv_cache_bytes)
│   ├── block.py              #   pre-norm block with GQA
│   └── model.py              #   full decoder-only Llama 2 (MHA/GQA/MQA via n_kv_heads)
├── data/                     # sample dataset + generated demo artifacts
│   └── generate_demo_data.py #   writes a human-readable copy_dataset.json
├── demo/
│   └── run_demo.py           #   train GQA + KV-cache comparison + JSON export
└── visualization/
    └── index.html            # MHA vs GQA vs MQA diagram + KV-cache chart
```

## 1. Set up the environment

Requires Python 3.10+. From inside this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

CPU build of PyTorch — no GPU needed.

## 2. (Optional) Generate a readable sample dataset

```bash
python data/generate_demo_data.py
```

## 3. Run the demo

```bash
python demo/run_demo.py
```

Expected output: the GQA model converging to **100%** exact-copy accuracy, a
regime table showing MHA/GQA/MQA all learn, and the KV-cache comparison above.
Total runtime is ~30 s on CPU. It writes `data/llama2_demo.json` for the
visualization. All hyperparameters are flags (`python demo/run_demo.py --help`).

## 4. Explore the visualization

Open `visualization/index.html` in any browser (offline via `file://`) for a
side-by-side **MHA vs GQA vs MQA** diagram, the real training-loss curve, the
**KV-cache memory chart** at 70B scale, and a high-level view of the RLHF chat
pipeline — all baked in from your demo run. To load fresh data, serve the
folder:

```bash
python -m http.server 8000    # then open http://localhost:8000/visualization/
```

## Code ↔ paper map

| Paper (Section 2.2) | Component | File |
|---|---|---|
| "Grouped-Query Attention" | GQA (MHA/GQA/MQA via n_kv_heads) | `src/attention.py` |
| KV-cache accounting | `kv_cache_bytes()` | `src/attention.py` |
| Pre-normalization (from LLaMA) | RMSNorm | `src/normalization.py` |
| RoPE (from LLaMA) | Rotary positions on Q/K | `src/rope.py` |
| SwiGLU (from LLaMA) | Gated feed-forward | `src/feedforward.py` |
| Section 2.2 (model) | Full decoder-only LM | `src/model.py` |
| Section 3 (RLHF) | Chat alignment pipeline (summary only) | see visualization |
