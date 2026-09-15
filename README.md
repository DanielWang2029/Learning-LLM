# Papers, Read and Recreated

This repository reads and recreates influential papers on **Large Language
Models**. The goal for each paper is a hands-on, runnable reproduction that you
can actually experience — not just notes.

## How this repository is organized

**Each paper gets its own self-contained folder.** Inside that folder lives
everything about that paper and nothing about any other:

- the paper itself (PDF),
- all code that reproduces it,
- demo data and runnable demos,
- its development environment (dependencies / virtualenv),
- docs, and
- a colorful, interactive **HTML visualization** of how the paper's algorithm
  works in detail.

**General, cross-paper information lives at the repository root**, parallel to
the individual paper folders — because it describes relationships *between*
papers rather than the internals of any single one:

- [`papers.json`](papers.json) — a registry of every reproduced paper and the
  **citation graph** (which paper cites which).
- [`citation_graph.html`](citation_graph.html) — a visualization of that graph,
  rendered from `papers.json`.

```
.
├── README.md              # this file (general info)
├── papers.json            # cross-paper registry + citation graph (general info)
├── citation_graph.html    # visualization of the citation graph (general info)
└── Attention Is All You Need/   # one self-contained paper folder
    ├── attention is all you need.pdf
    ├── transformer/       # reference implementation
    ├── demo/              # runnable demos + data generator
    ├── visualization/     # interactive algorithm visualization
    └── README.md          # how to set up & run this paper
```

## Papers reproduced

| Paper | Year | Folder | Status |
|---|---|---|---|
| Attention Is All You Need | 2017 | [`Attention Is All You Need/`](Attention%20Is%20All%20You%20Need/) | reproduced |

## Reading list / roadmap

[`papers.json`](papers.json) also catalogs a broader roadmap of **key LLM papers**
(currently 28, marked `planned` until reproduced) together with the citation
edges that connect them. The list is drawn from widely-used curations — notably
*Essential LLM Papers* (Foundation Models Deep Dive) — spanning foundational
architecture (BERT, GPT), scaling laws (Kaplan, Chinchilla), architectural
innovations (RoPE, FlashAttention, Switch Transformers), model milestones (GPT-2/3,
T5, PaLM, LLaMA), alignment (InstructGPT, RLHF, Constitutional AI, DPO), retrieval
(RAG), efficient fine-tuning (LoRA, QLoRA), reasoning (Chain-of-Thought,
Self-Consistency, Tree of Thoughts, ReAct, DeepSeek-R1), and multimodal models
(CLIP, Flamingo). Open [`citation_graph.html`](citation_graph.html) to explore
them on a timeline colored by theme.

## Adding a new paper

1. Create a new top-level folder named after the paper and drop the PDF inside.
2. Add everything needed to reproduce it *within that folder* (code, data,
   demos, environment, docs, and an `index.html` visualization).
3. Register it in [`papers.json`](papers.json) and add any citation edges to
   papers already in the repository.

To view the citation graph, open `citation_graph.html` (serve the root folder
with `python -m http.server` so it can read `papers.json`).
