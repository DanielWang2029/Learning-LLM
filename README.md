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
└── <62 more paper folders, each with the same self-contained layout>
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

All 63 papers here have a self-contained folder with a from-scratch
implementation, a tiny sample dataset, a runnable CPU demo, docs, a development
environment, and an interactive HTML visualization. Explore
[`citation_graph.html`](citation_graph.html) to see how they relate on a
timeline colored by theme.

The first batch below (through the DeepSeek-R1 row) is drawn from widely-used
foundational curations — notably *Essential LLM Papers* (Foundation Models Deep
Dive). The [**Top-cited by year (2023–2026)**](#top-cited-by-year-20232026)
section that follows adds the most-cited LLM papers of each recent year.

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

## Top-cited by year (2023–2026)

The **top-10 most-cited LLM papers of each year** from 2023 to 2026. Papers
already covered above are marked *(above)* and not duplicated. Rankings use
Google Scholar / Semantic Scholar counts as aggregated by Zeta Alpha, the NLLG
arXiv reports, Paper Digest influence lists, and *bestpapers.ai*. **2025 counts
are still maturing and the 2026 list is provisional** — only a partial year has
elapsed and citation data is not yet reliable, so 2026 is ranked by early
prominence/attention (Paper Digest, *1kpapers*, Sebastian Raschka's 2026 list)
rather than mature citation counts.

### 2023

| Paper | Category | Folder |
|---|---|---|
| GPT-4 Technical Report | Model Milestones | [`GPT-4/`](GPT-4/) |
| Visual Instruction Tuning (LLaVA) | Multimodal | [`LLaVA/`](LLaVA/) |
| Sparks of AGI: Early experiments with GPT-4 | Evaluation & Analysis | [`Sparks of AGI/`](Sparks%20of%20AGI/) |
| A Survey of Large Language Models | Survey | [`A Survey of Large Language Models/`](A%20Survey%20of%20Large%20Language%20Models/) |
| Mistral 7B | Architectural Innovations | [`Mistral 7B/`](Mistral%207B/) |
| LLaMA | Model Milestones | *(above)* |
| Llama 2 | Model Milestones | *(above)* |
| QLoRA | Efficient Fine-tuning | *(above)* |
| Direct Preference Optimization | Alignment | *(above)* |
| Tree of Thoughts | Reasoning | *(above)* |

### 2024

| Paper | Category | Folder |
|---|---|---|
| The Llama 3 Herd of Models | Model Milestones | [`Llama 3/`](Llama%203/) |
| Qwen2.5 Technical Report | Model Milestones | [`Qwen2.5/`](Qwen2.5/) |
| DeepSeekMath (GRPO) | Reasoning | [`DeepSeekMath/`](DeepSeekMath/) |
| Gemini 1.5 | Model Milestones | [`Gemini 1.5/`](Gemini%201.5/) |
| DeepSeek-V3 | Architectural Innovations | [`DeepSeek-V3/`](DeepSeek-V3/) |
| Gemma 2 | Model Milestones | [`Gemma 2/`](Gemma%202/) |
| Phi-3 | Model Milestones | [`Phi-3/`](Phi-3/) |
| Mixtral of Experts | Architectural Innovations | [`Mixtral of Experts/`](Mixtral%20of%20Experts/) |
| DoRA | Efficient Fine-tuning | [`DoRA/`](DoRA/) |
| GraphRAG | Context & Retrieval | [`GraphRAG/`](GraphRAG/) |

### 2025

| Paper | Category | Folder |
|---|---|---|
| DeepSeek-R1 | Reasoning | *(above)* |
| Qwen3 Technical Report | Model Milestones | [`Qwen3/`](Qwen3/) |
| Gemini 2.5 | Model Milestones | [`Gemini 2.5/`](Gemini%202.5/) |
| s1: Simple Test-time Scaling | Reasoning | [`s1 - Simple Test-Time Scaling/`](s1%20-%20Simple%20Test-Time%20Scaling/) |
| Gemma 3 | Model Milestones | [`Gemma 3/`](Gemma%203/) |
| Search-R1 | Reasoning | [`Search-R1/`](Search-R1/) |
| Large Language Diffusion Models (LLaDA) | Architectural Innovations | [`LLaDA/`](LLaDA/) |
| LIMO: Less Is More for Reasoning | Reasoning | [`LIMO/`](LIMO/) |
| Kimi K1.5 | Reasoning | [`Kimi K1.5/`](Kimi%20K1.5/) |
| Native Sparse Attention | Architectural Innovations | [`Native Sparse Attention/`](Native%20Sparse%20Attention/) |
| Chain of Draft | Reasoning | [`Chain of Draft/`](Chain%20of%20Draft/) |

### 2026 (provisional)

| Paper | Category | Folder |
|---|---|---|
| GLM-5: from Vibe Coding to Agentic Engineering | Model Milestones | [`GLM-5/`](GLM-5/) |
| DeepSeek-V4 | Architectural Innovations | [`DeepSeek-V4/`](DeepSeek-V4/) |
| Kimi K3 | Architectural Innovations | [`Kimi K3/`](Kimi%20K3/) |
| Qwen-AgentWorld | Reasoning | [`Qwen-AgentWorld/`](Qwen-AgentWorld/) |
| Nemotron 3 Super | Architectural Innovations | [`Nemotron 3 Super/`](Nemotron%203%20Super/) |
| Gated DeltaNet-2 | Architectural Innovations | [`Gated DeltaNet-2/`](Gated%20DeltaNet-2/) |
| Step 3.5 Flash | Architectural Innovations | [`Step 3.5 Flash/`](Step%203.5%20Flash/) |
| Scaling Embeddings vs Experts | Scaling Laws | [`Scaling Embeddings vs Experts/`](Scaling%20Embeddings%20vs%20Experts/) |
| Deep Delta Learning | Architectural Innovations | [`Deep Delta Learning/`](Deep%20Delta%20Learning/) |
| ZAYA1-8B | Architectural Innovations | [`ZAYA1-8B/`](ZAYA1-8B/) |

## Streaming audio encoders

A focused set on **audio encoders, especially streaming** ASR encoders — the
canonical top papers plus six specifically requested recent models. Each has the
same self-contained layout; demos synthesize audio with NumPy and implement
log-Mel/STFT from scratch (no `torchaudio`/`librosa`), so they run offline on CPU.

| Paper | Year | Streaming idea | Folder |
|---|---|---|---|
| Conformer | 2020 | conv-augmented Transformer encoder (backbone) | [`Conformer/`](Conformer/) |
| wav2vec 2.0 | 2020 | self-supervised contrastive speech encoder | [`wav2vec 2.0/`](wav2vec%202.0/) |
| Emformer | 2021 | augmented-memory block streaming Transformer | [`Emformer/`](Emformer/) |
| WeNet U2++ | 2021 | dynamic-chunk unified streaming/non-streaming | [`WeNet U2++/`](WeNet%20U2%2B%2B/) |
| Whisper | 2022 | log-mel + conv + Transformer encoder (30s chunks) | [`Whisper/`](Whisper/) |
| FastConformer | 2023 | 8× depthwise-separable subsampling | [`FastConformer/`](FastConformer/) |
| Zipformer | 2023 | U-Net multi-rate encoder + BiasNorm | [`Zipformer/`](Zipformer/) |
| Cache-Aware Streaming Conformer | 2023 | KV/conv cache, exact chunked streaming | [`Cache-Aware Streaming Conformer/`](Cache-Aware%20Streaming%20Conformer/) |
| Qwen2.5-Omni *(requested)* | 2025 | block-wise (2s) streaming audio encoder | [`Qwen2.5-Omni/`](Qwen2.5-Omni/) |
| Voxtral *(requested)* | 2025 | Whisper encoder + 4-frame-concat adapter (50→12.5 Hz) | [`Voxtral/`](Voxtral/) |
| Uni-ASR *(requested)* | 2026 | unified streaming + fallback decoding | [`Uni-ASR/`](Uni-ASR/) |
| VibeVoice-ASR-Streaming *(requested)* | 2026 | streaming speaker-attributed "who said what" | [`VibeVoice-ASR-Streaming/`](VibeVoice-ASR-Streaming/) |
| Confucius4-R2T2 *(requested)* | 2026 | Longest Stable Prefix, append-only streaming | [`Confucius4-R2T2/`](Confucius4-R2T2/) |
| Nemotron 3.5 ASR Streaming 0.6B *(requested)* | 2026 | cache-aware FastConformer-RNNT, configurable chunks | [`Nemotron 3.5 ASR Streaming 0.6B/`](Nemotron%203.5%20ASR%20Streaming%200.6B/) |

The six requested models are marked *(requested)*. Four have arXiv papers
(Voxtral, Qwen2.5-Omni, Uni-ASR, VibeVoice-ASR-Streaming); **Confucius4-R2T2**
and **Nemotron 3.5 ASR** have no formal paper yet, so each folder includes the
official model card as its reference document.

## Adding a new paper

1. Create a new top-level folder named after the paper and drop the PDF inside.
2. Add everything needed to reproduce it *within that folder* (code, data,
   demos, environment, docs, and an `index.html` visualization).
3. Register it in [`papers.json`](papers.json) and add any citation edges to
   papers already in the repository.

To view the citation graph, open `citation_graph.html` (serve the root folder
with `python -m http.server` so it can read `papers.json`).
