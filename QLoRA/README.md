# QLoRA: Efficient Finetuning of Quantized LLMs

A faithful, minimal, fully self-contained reproduction of Dettmers et al.,
*"QLoRA: Efficient Finetuning of Quantized LLMs"* (2023),
arXiv:[2305.14314](https://arxiv.org/abs/2305.14314).

QLoRA makes fine-tuning cheap in **memory**: it stores the frozen base weights
in **4 bits** and trains only small **full-precision LoRA adapters** on top. The
forward pass dequantizes the 4-bit weights just for the matmul:

```
h = dequant(W0_4bit)·x + b  +  (alpha / r) · (B · A) · x
```

Two ingredients from the paper make the 4-bit base work:

- **Blockwise quantization** — split each weight tensor into blocks of 64,
  normalize each block by its own `absmax`, so one outlier can't wreck the whole
  tensor's resolution.
- **NF4 (NormalFloat-4)** — the 16 quantization levels are the quantiles of a
  standard normal, which is information-theoretically optimal for
  normally-distributed weights.

To keep everything CPU-reproducible, the "large model" is a small MLP and the
tasks are two related synthetic problems (same setup as the `LoRA/` folder). The
NF4 quantizer is implemented from scratch in numpy/torch — no bitsandbytes, no
CUDA.

```
QLoRA/
├── qlora.pdf                # the paper itself
├── requirements.txt         # pinned CPU dependencies (torch, numpy)
├── src/                     # the method, from scratch
│   ├── quant.py             #   §3  4-bit NF4 blockwise quantize / dequantize
│   ├── qlora.py             #   §3  QLoRALinear: 4-bit frozen base + fp32 LoRA
│   ├── model.py             #   the tiny MLP backbone we quantize + adapt
│   └── data.py              #   two related toy tasks (shared low-rank structure)
├── data/                    # results JSON written by the demo (generated)
├── demo/run_demo.py         # pretrain -> quantize -> QLoRA fine-tune -> memory
└── visualization/index.html # NF4 codebook, quant error, memory breakdown, accuracy
```

## 1. Set up the environment

Requires Python 3.10+. Reuse the shared virtual environment:

```bash
source "/workspace/Attention Is All You Need/.venv/bin/activate"
```

Or build a fresh one:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

## 2. Run the demo

```bash
python demo/run_demo.py
```

- **Stage 1 — Pretrain.** Train a small fp32 MLP on a base task and freeze it.
- **Stage 2 — Quantize.** Convert the frozen weights to 4-bit NF4 and report the
  quantization error and the accuracy the 4-bit model retains.
- **Stage 3 — QLoRA.** Fine-tune fp32 LoRA adapters on top of the 4-bit frozen
  base on a new task; only the adapters receive gradients.
- **Stage 4 — Memory.** Compare bytes for fp32 / fp16 base vs 4-bit base + fp32
  adapters.

## 3. Expected output

Seeded and reproducible. A representative run (≈14 s on CPU):

```
overall weight quant error (rel L2): 9.31%
base-task accuracy: fp32 86.9%  ->  4-bit 86.9%  (drop 0.0 pts)

new-task accuracy AFTER QLoRA: 82.1%

fp32 base                     : 2.36 MB
4-bit NF4 base (+absmax)      : 339.6 KB  (~4.50 bits/value)
+ fp32 LoRA adapters          : 24.3 KB
= QLoRA total (base+adapters) : 363.9 KB
memory vs fp32 base           : 6.6x smaller

OK: 4-bit base kept its accuracy, LoRA-on-quantized-base learned the new task ...
```

What this proves, all asserted by the demo:

1. the base weights quantize to **4 bits with small error** (~9% rel L2) and the
   4-bit model **keeps essentially all its accuracy**,
2. **LoRA on the quantized frozen base still learns** the new task to high
   accuracy (~82%, up from chance), and
3. the QLoRA footprint (4-bit base + fp32 adapters) is **several times smaller**
   than the fp32 (and fp16) base.

## 4. Explore the visualization

Open `visualization/index.html` in any browser (offline via `file://`). It shows
the NF4 codebook and blockwise scheme, per-layer quantization error, the
fp32-vs-fp16-vs-4-bit **memory breakdown**, and the **accuracy retained** — all
baked in from your demo run. Serve the folder to load live data:

```bash
python -m http.server 8000   # then open http://localhost:8000/visualization/
```

## Code ↔ paper map

| Paper section | Component | File |
|---|---|---|
| §3 (4-bit NormalFloat) | NF4 codebook (normal quantiles) | `src/quant.py` → `NF4_CODEBOOK` |
| §3 (blockwise) | Per-block absmax quantize / dequantize | `src/quant.py` → `quantize_nf4` / `dequantize_nf4` |
| §3 | 4-bit frozen base + fp32 LoRA on top | `src/qlora.py` → `QLoRALinear` |
| §3 | Dequantize for the matmul | `src/qlora.py` → `QLoRALinear.forward` |
| §4 (memory) | 4-bit base + adapters vs fp32/fp16 | `demo/run_demo.py` (Stage 4) |
