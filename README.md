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
├── Attention Is All You Need/   # one self-contained paper folder
│   ├── attention is all you need.pdf
│   ├── transformer/       # reference implementation
│   ├── demo/              # runnable demos + data generator
│   ├── visualization/     # interactive algorithm visualization
│   └── README.md          # how to set up & run this paper
└── <27 more paper folders, each with the same self-contained layout>
    ├── <paper>.pdf        # the paper
    ├── src/               # from-scratch reference implementation
    ├── data/              # tiny sample dataset (or generator)
    ├── demo/run_demo.py   # runnable CPU demo of the algorithm
    ├── visualization/     # interactive HTML visualization
    ├── requirements.txt   # development environment
    └── README.md          # how to set up & run this paper
```

## Running any paper

Every folder is self-contained and runs on CPU. From inside a paper folder:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
python demo/run_demo.py          # prints evidence the algorithm works
# then open visualization/index.html in a browser
```

## Papers reproduced

All 28 papers below have a self-contained folder with a from-scratch
implementation, a tiny sample dataset, a runnable CPU demo, docs, a development
environment, and an interactive HTML visualization. The list is drawn from
widely-used curations — notably *Essential LLM Papers* (Foundation Models Deep
Dive). Explore [`citation_graph.html`](citation_graph.html) to see how they
relate on a timeline colored by theme.

| Paper | Year | Category | Folder |
|---|---|---|---|
| Attention Is All You Need | 2017 | Foundational Architecture | [`Attention Is All You Need/`](Attention%20Is%20All%20You%20Need/) |
| Deep Reinforcement Learning from Human Preferences | 2017 | Alignment & Instruction Following | [`Deep RL from Human Preferences/`](Deep%20RL%20from%20Human%20Preferences/) |
| BERT: Pre-training of Deep Bidirectional Transformers | 2018 | Foundational Architecture | [`BERT/`](BERT/) |
| Improving Language Understanding by Generative Pre-Training (GPT-1) | 2018 | Foundational Architecture | [`GPT-1/`](GPT-1/) |
| Language Models are Unsupervised Multitask Learners (GPT-2) | 2019 | Model Milestones | [`GPT-2/`](GPT-2/) |
| Exploring the Limits of Transfer Learning (T5) | 2020 | Model Milestones | [`T5/`](T5/) |
| Language Models are Few-Shot Learners (GPT-3) | 2020 | Model Milestones | [`GPT-3/`](GPT-3/) |
| Retrieval-Augmented Generation | 2020 | Context & Retrieval | [`Retrieval-Augmented Generation/`](Retrieval-Augmented%20Generation/) |
| Scaling Laws for Neural Language Models | 2020 | Scaling Laws | [`Scaling Laws for Neural Language Models/`](Scaling%20Laws%20for%20Neural%20Language%20Models/) |
| CLIP: Learning Transferable Visual Models | 2021 | Multimodal | [`CLIP/`](CLIP/) |
| LoRA: Low-Rank Adaptation | 2021 | Efficient Fine-tuning | [`LoRA/`](LoRA/) |
| RoFormer: Rotary Position Embedding (RoPE) | 2021 | Architectural Innovations | [`RoFormer - Rotary Position Embedding/`](RoFormer%20-%20Rotary%20Position%20Embedding/) |
| Switch Transformers | 2021 | Architectural Innovations | [`Switch Transformers/`](Switch%20Transformers/) |
| Chain-of-Thought Prompting | 2022 | Reasoning | [`Chain-of-Thought Prompting/`](Chain-of-Thought%20Prompting/) |
| Constitutional AI | 2022 | Alignment & Instruction Following | [`Constitutional AI/`](Constitutional%20AI/) |
| Flamingo | 2022 | Multimodal | [`Flamingo/`](Flamingo/) |
| FlashAttention | 2022 | Architectural Innovations | [`FlashAttention/`](FlashAttention/) |
| PaLM: Scaling Language Modeling with Pathways | 2022 | Model Milestones | [`PaLM/`](PaLM/) |
| ReAct: Synergizing Reasoning and Acting | 2022 | Reasoning | [`ReAct/`](ReAct/) |
| Self-Consistency | 2022 | Reasoning | [`Self-Consistency/`](Self-Consistency/) |
| Training Compute-Optimal LLMs (Chinchilla) | 2022 | Scaling Laws | [`Chinchilla/`](Chinchilla/) |
| InstructGPT: Instructions with Human Feedback | 2022 | Alignment & Instruction Following | [`InstructGPT/`](InstructGPT/) |
| Direct Preference Optimization (DPO) | 2023 | Alignment & Instruction Following | [`Direct Preference Optimization/`](Direct%20Preference%20Optimization/) |
| LLaMA: Open and Efficient Foundation LMs | 2023 | Model Milestones | [`LLaMA/`](LLaMA/) |
| Llama 2 | 2023 | Model Milestones | [`Llama 2/`](Llama%202/) |
| QLoRA: Efficient Finetuning of Quantized LLMs | 2023 | Efficient Fine-tuning | [`QLoRA/`](QLoRA/) |
| Tree of Thoughts | 2023 | Reasoning | [`Tree of Thoughts/`](Tree%20of%20Thoughts/) |
| DeepSeek-R1 | 2025 | Reasoning | [`DeepSeek-R1/`](DeepSeek-R1/) |

## Adding a new paper

1. Create a new top-level folder named after the paper and drop the PDF inside.
2. Add everything needed to reproduce it *within that folder* (code, data,
   demos, environment, docs, and an `index.html` visualization).
3. Register it in [`papers.json`](papers.json) and add any citation edges to
   papers already in the repository.

To view the citation graph, open `citation_graph.html` (serve the root folder
with `python -m http.server` so it can read `papers.json`).
