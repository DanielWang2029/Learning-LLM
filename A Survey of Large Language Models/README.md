# A Survey of Large Language Models — Taxonomy + Runnable Lifecycle

**Paper:** *A Survey of Large Language Models* — Zhao et al., 2023 —
[arXiv:2303.18223](https://arxiv.org/abs/2303.18223)

> **This is a survey, not an algorithm.** There is no single method to
> reimplement. This folder is honest about that and instead delivers the two
> things a survey is actually *for*: **(a)** an interactive **taxonomy** of the
> techniques it catalogs, and **(b)** a tiny **runnable lifecycle** that
> minimally touches every core stage the survey is organized around —
> pre-training → adaptation (SFT) → alignment (DPO) — printing a metric after
> each stage so you can watch a single model progress.

## The core idea, in plain English

Modern LLMs aren't built in one step. The survey organizes the field around a
**lifecycle**: you *pre-train* on a large corpus, *adapt* the model with
supervised instruction tuning, *align* it to preferences (RLHF / DPO), then
*use* it via prompting and *evaluate* it. This demo runs the first three stages
for real — on one small GPT — so the pipeline is concrete rather than abstract.

## What the demo shows

`demo/run_demo.py` carries **one tiny GPT** (13-token vocabulary) through the
lifecycle and prints a metric per stage:

| Stage | What happens | Metric (sample run) |
|---|---|---|
| **1. Pre-training** | next-token prediction on a synthetic corpus | LM loss **2.67 → 1.66** |
| **2. Adaptation (SFT)** | supervised fine-tuning on "sort 4 digits" | SORT exact-match **0% → 100%** |
| **3. Alignment (DPO)** | learn a *new* behavior ("max") from preference pairs only | preference accuracy **83% → 100%**, reward margin **+32.9** |

Two details make the alignment stage faithful:

- The `max` behavior is **never taught by SFT** — DPO instills a preference
  purely from `(chosen, rejected)` comparisons, exactly the RLHF/DPO premise.
- A small **SFT replay** runs alongside DPO (cf. InstructGPT's PPO-ptx), so the
  SORT skill is **retained (100%)** instead of being catastrophically forgotten.

Runs on CPU in ~18 seconds.

## Folder layout

```
A Survey of Large Language Models/
├── a_survey_of_large_language_models.pdf   # the paper (unchanged)
├── requirements.txt          # torch==2.8.0, numpy>=1.26 (CPU)
├── src/
│   ├── model.py              # the tiny GPT carried through every stage
│   └── pipeline.py           # data + train/score for pretrain, SFT, DPO
├── data/                     # generated: lifecycle.json (for the viz)
├── demo/run_demo.py          # pretrain -> SFT -> DPO, metric after each stage
└── visualization/index.html  # interactive technique taxonomy + lifecycle charts
```

## Setup

Reuse the shared CPU environment (do **not** create a new venv):

```bash
source "/workspace/Attention Is All You Need/.venv/bin/activate"
```

Or, standalone:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

## How to run

```bash
python demo/run_demo.py
```

Prints the per-stage metrics above and writes `data/lifecycle.json`.

## Explore the visualization

Open `visualization/index.html` in any browser (offline via `file://`). It has an
**interactive taxonomy** of LLM techniques (expand/collapse the five branches the
survey covers) and **charts of the mini-lifecycle** (SORT skill across stages;
preference accuracy before/after DPO). Serve over http to load live data:

```bash
python -m http.server 8000   # then open http://localhost:8000/visualization/
```

## Code ↔ paper map

| Survey topic | What we reproduce | Where |
|---|---|---|
| Pre-training | next-token LM training on a corpus | `src/pipeline.py` → `pretrain_batch`, `train_lm` |
| Adaptation / instruction tuning | SFT on an instruction→answer task | `src/pipeline.py` → `sft_batch` |
| Alignment / RLHF / DPO | direct preference optimization from comparisons | `src/pipeline.py` → `dpo_loss` |
| Anti-forgetting (PPO-ptx) | SFT replay during alignment | `demo/run_demo.py` |
| Full taxonomy of techniques | interactive tree (pretrain/adapt/use/align/eval) | `visualization/index.html` |
