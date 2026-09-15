# Retrieval-Augmented Generation (RAG)

A faithful, minimal, fully self-contained reproduction of Lewis et al.,
*"Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"* (2020),
arXiv:[2005.11401](https://arxiv.org/abs/2005.11401).

RAG couples a **retriever** with a **generator**: instead of relying only on
knowledge baked into a model's parameters, it first *retrieves* relevant
passages from a corpus and then *generates* the answer conditioned on them
(paper Eq. 1). This grounds outputs in real text and sharply reduces
hallucination on knowledge-intensive questions.

To make the effect unmistakable and CPU-reproducible, the corpus is a set of
**fictional** facts (e.g. *"The capital of Zubland is Marn."*). A closed-book
model has no way to know these from its parameters — so it hallucinates — while
RAG retrieves the supporting sentence and answers correctly.

- **Retriever `p_η`** — a from-scratch **TF-IDF** index (numpy) that ranks
  documents by cosine similarity to the query (a transparent stand-in for the
  paper's learned DPR bi-encoder; same top-k-over-a-corpus interface).
- **Generator `p_θ`** — a faithful **extractive reader** that pulls the answer
  span straight out of the retrieved passage, so the answer is provably grounded
  in retrieved text.

```
Retrieval-Augmented Generation/
├── retrieval-augmented_generation.pdf   # the paper itself
├── requirements.txt                     # pinned CPU dependencies (torch, numpy)
├── src/                                 # the method, from scratch
│   ├── corpus.py                        #   tiny knowledge corpus + questions
│   ├── retriever.py                     #   §2  TF-IDF retriever p_η (top-k)
│   └── generator.py                     #   §2  grounded reader p_θ + closed-book
├── data/                                # results JSON written by the demo (generated)
├── demo/run_demo.py                     # RAG vs closed-book, with metrics + evidence
└── visualization/index.html            # retriever→generator pipeline, top-k, grounding
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

For each question it retrieves the top-k passages (marking the gold one),
reads a grounded answer, and contrasts it with a closed-book guess — then
reports retrieval accuracy and answer accuracy with vs without retrieval.

## 3. Expected output

Seeded and reproducible (< 1 s on CPU):

```
Q: What is the capital of Zubland?
   ★ top1 [sim 0.727]  The capital of Zubland is Marn.
     top2 [sim 0.427]  The currency of Zubland is zub.
     top3 [sim 0.427]  The capital of Marovia is Deltra.
   -> RAG answer         : 'Marn'   [OK ]  (gold: 'Marn')
   -> closed-book answer : 'Berlin' [HALLUCINATED]

retrieval accuracy (top-1)         : 100.0%  (36/36)
answer accuracy WITH retrieval     : 100.0%  (36/36)   <- RAG
answer accuracy WITHOUT retrieval  :   0.0%  (0/36)    <- closed-book (hallucinates)

OK: retrieval grounds the generator — RAG answers correctly where the
closed-book baseline hallucinates.
```

What this proves, all asserted by the demo:

1. the retriever finds the **gold supporting passage at top-1** (100%),
2. the generator, **conditioned on the retrieved text, answers correctly**
   (100%), and
3. the same model **without retrieval hallucinates** (0% on the invented
   facts) — retrieval is what closes the gap.

## 4. Explore the visualization

Open `visualization/index.html` in any browser (offline via `file://`). It draws
the retriever→generator pipeline, shows a real **query → top-k retrieved docs →
grounded answer** example (baked in from your demo run) beside the hallucinated
closed-book answer, and plots the with-vs-without-retrieval accuracy. Serve the
folder to load live data:

```bash
python -m http.server 8000   # then open http://localhost:8000/visualization/
```

## Code ↔ paper map

| Paper section | Component | File |
|---|---|---|
| §2.1 / Eq. 1 | Retriever p_η (top-k over a corpus) | `src/retriever.py` → `TfidfRetriever` |
| §2 | Generator p_θ conditioned on retrieved docs | `src/generator.py` → `rag_answer` / `read_answer` |
| §1 | Closed-book baseline (parametric, hallucinates) | `src/generator.py` → `closed_book_answer` |
| §4 (setup) | Knowledge corpus + questions | `src/corpus.py` |
| §4 (eval) | Retrieval accuracy & answer accuracy | `demo/run_demo.py` |
