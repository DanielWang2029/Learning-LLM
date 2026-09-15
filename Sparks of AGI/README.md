# Sparks of AGI — Capability Probing (a runnable, honest reproduction)

**Paper:** *Sparks of Artificial General Intelligence: Early experiments with
GPT-4* — Bubeck et al., 2023 — [arXiv:2303.12712](https://arxiv.org/abs/2303.12712)

> **This is an analysis paper, not an algorithm.** It studies the behavior of
> GPT-4 (a closed model) through a large battery of hand-designed tasks. There
> is no model, loss, or training procedure to reimplement. What *is*
> reproducible — and what this folder reproduces honestly — is the paper's
> **methodology**: build a battery of small, self-contained probes, turn model
> behavior into a **pass/fail capability profile**, and track how that profile
> changes with scale.

## The core idea, in plain English

Rather than a single benchmark number, the paper asks *which qualitatively
different things* a model can and cannot do, across a wide range of tasks. We
copy that lens: define six transparent probes spanning an easy→hard difficulty
range, train a family of tiny GPTs of increasing size on the whole battery, and
read off a **capability matrix** — which probes each model passes. The matrix
"fills in" as scale grows, and one deliberately hard probe (parity/XOR) stays
unsolved, mirroring the persistent capability gaps the paper documents.

## The probe battery

| probe | capability it tests | chance |
|---|---|---|
| `pattern` | continue an alternating sequence (induction) | 1% |
| `copy` | echo the input (memory / identity) | 0.01% |
| `reverse` | reverse the input (positional routing) | 0.01% |
| `sort` | sort digits ascending (comparison) | 0.01% |
| `add` | add two digits **with carry** (arithmetic) | ~5% |
| `parity` | XOR of a bit string (sequential counting) | 50% |

Each probe is a short token sequence `[TASK] inputs = answer`; only the answer
region is graded (exact match).

## What the demo shows

`demo/run_demo.py` trains four GPTs (`nano → micro → mini → small`) on the
battery and prints the capability matrix. A representative run:

```
probe         nano    micro     mini    small
---------------------------------------------
copy     ·    19% ✓    96% ✓    98% ✓   100%
reverse  ·    18% ✓    98% ✓    97% ✓    96%
sort     ·    66% ✓    94% ✓    86% ✓    94%
add      ·    45% ·    53% ·    77% ✓    94%   <- arithmetic emerges with scale
parity   ·    52% ·    47% ·    47% ·    52%   <- persistent gap (never solved)
pattern  ✓   100% ✓   100% ✓   100% ✓   100%

Capabilities present: nano:1/6 -> micro:4/6 -> mini:4/6 -> small:5/6
```

The profile fills in with scale, `add`-with-carry is a clear **emergent**
capability, and `parity` stays at chance — a transparent, tiny-scale echo of the
paper's qualitative findings. Runs on CPU in ~30 seconds.

## Folder layout

```
Sparks of AGI/
├── sparks_of_agi.pdf         # the paper (unchanged)
├── requirements.txt          # torch==2.8.0, numpy>=1.26 (CPU)
├── src/
│   ├── model.py              # tiny GPT (size dialed via GPTConfig)
│   ├── tasks.py              # the six-probe battery + encoding + chance rates
│   └── probe.py              # the harness: score a model -> capability profile
├── data/                     # generated: capability_profile.json (for the viz)
├── demo/run_demo.py          # train a family of GPTs -> capability matrix
└── visualization/index.html  # capability matrix + accuracy-vs-scale + counts
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

Prints the capability matrix above and writes `data/capability_profile.json`.

## Explore the visualization

Open `visualization/index.html` in any browser (offline via `file://`). It shows
the **capability matrix** (probes × scales, tiles shaded by accuracy), an
**accuracy-vs-scale** line chart per probe, and a **capabilities-passed vs.
parameters** chart. Serve over http to load live data:

```bash
python -m http.server 8000   # then open http://localhost:8000/visualization/
```

## Code ↔ paper map

| Paper element | What we reproduce | File |
|---|---|---|
| Battery of diverse probes | six transparent tasks spanning difficulty | `src/tasks.py` |
| Turning behavior into a capability judgment | pass/fail via exact-match threshold | `src/probe.py` |
| Capability profile across models | matrix over a family of scaled GPTs | `demo/run_demo.py` |
| "Sparks appear with scale" / persistent gaps | emergent `add`, never-solved `parity` | demo output + viz |
| A specific GPT-4 algorithm | *does not exist* — stated honestly above | — |
