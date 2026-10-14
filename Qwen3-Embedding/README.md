# Qwen3-Embedding
**Qwen Team · 2025 · [arXiv:2506.05176](https://arxiv.org/abs/2506.05176)**

Qwen3-Embedding turns an instruction plus text into a retrieval vector using the
last-token hidden state of a causal LLM, then offers a stronger cross-encoder
reranker. This folder is a tiny synthetic CPU reproduction of those mechanisms,
not a Qwen checkpoint or benchmark result.

The demo trains a one-layer causal Transformer with full-corpus InfoNCE and hard
negatives. “color red circle” and “shape red circle” intentionally retrieve different
documents. To make the two-stage trade-off observable, dense search uses a compact
six-dimension projection before a joint query-document scorer reranks its top two.

## Layout
```text
src/models.py             # causal last-token embedder and cross-encoder scorer
data/generate.py          # NumPy compositional retrieval corpus
demo/run_demo.py          # InfoNCE training, retrieval, reranking
visualization/index.html  # offline, interactive diagram and results
```
## Setup and run
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
python demo/run_demo.py
```
The command prints embedding retrieval@1, reranked retrieval@1, and two
instruction-switch examples; it writes `data/embedding_results.json`. Open
`visualization/index.html` via `file://` for baked demo data (or serve the folder
to load fresh JSON).

## Code ↔ paper map
| Paper section | Mechanism | Code |
|---|---|---|
| §2–3 | instruction-aware text representation | `src/models.py:LastTokenEmbedder` |
| §3 | last-token causal-LM embedding | `LastTokenEmbedder.forward` |
| §3 | contrastive / hard-negative optimization | `demo/run_demo.py` |
| §4 | cross-encoder reranking | `src/models.py:CrossEncoderReranker` |
