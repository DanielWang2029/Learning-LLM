# Qwen2.5-1M
**Qwen Team · 2025 · [arXiv:2501.15383](https://arxiv.org/abs/2501.15383)**

Qwen2.5-1M extends long-context inference. Its training-free Dual Chunk Attention
(DCA) remaps relative positions so attention does not encounter distances outside
its pretrained range. This is a tiny CPU illustration, not the 7B/14B model or a
claim about million-token quality.

The demo trains one small relative-position attention retriever only at 64 tokens,
then asks it to find a unique needle at up to 1,024 tokens. Vanilla attention gives
unseen distances a bad bias; DCA combines intra-chunk, successive-chunk, and
repeated inter-chunk positions within the trained range.

## Layout
```text
src/dca.py              # DCA mapping and one-head relative attention
data/generate.py        # NumPy needle/haystack generator
demo/run_demo.py        # short training and long-context evaluation
visualization/index.html # offline interactive companion
```

## Setup and run
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
python demo/run_demo.py
```
It prints recall at 64–1,024 tokens and writes `data/dca_results.json`. The expected
effect is vanilla recall falling past the 64-token training length while DCA remains
high. Open `visualization/index.html` directly; it contains baked sample data and,
when served, can load the fresh JSON.

## Code ↔ paper map
| Paper section | Mechanism | Code |
|---|---|---|
| §5.1 | DCA relative-position remapping | `src/dca.py:dca_distance` |
| §5.1 | intra/inter/successive chunk cases | `src/dca.py:dca_distance` |
| §5.1 | short-context needle retrieval effect | `demo/run_demo.py` |
