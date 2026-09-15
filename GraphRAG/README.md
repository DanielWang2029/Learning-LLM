# GraphRAG — From Local to Global

A faithful, minimal, dependency-light reproduction of the *GraphRAG* pipeline
from *"From Local to Global: A Graph RAG Approach to Query-Focused
Summarization"* (Microsoft, 2024 — arXiv
[2404.16130](https://arxiv.org/abs/2404.16130)). Everything needed to read, run,
and understand the idea lives in this folder: the paper PDF, a from-scratch
implementation, a runnable demo with data, and an interactive visualization.

Classic RAG retrieves the top-k chunks most similar to a query. That is great for
**local** questions ("Where is Helios located?"), but it fails at **global**
sensemaking questions ("What are the main themes across the whole corpus?") — a
fixed handful of chunks simply cannot represent an entire dataset. GraphRAG fixes
this by:

1. **extracting** an entity/relation knowledge graph from the corpus,
2. **detecting communities** in that graph,
3. **summarizing** each community, and
4. answering a global query by **map-reduce** over the community summaries.

```
GraphRAG/
├── graphrag.pdf             # the paper
├── requirements.txt         # torch (CPU) + numpy   (only numpy is used at runtime)
├── src/
│   ├── corpus.py            # a toy corpus of facts across 4 themes (+ bridge facts)
│   ├── extract.py           # §2.1  rule-based entity/relation extraction -> graph
│   ├── graph.py             # §2.2  Graph + connected-components + greedy modularity
│   └── rag.py               # §2-3  bag-of-words top-k RAG vs GraphRAG map-reduce
├── data/
│   └── generate_data.py     # dumps the corpus (sentences + triples) to corpus.json
├── demo/
│   └── run_demo.py          # runs the whole pipeline and compares the two approaches
└── visualization/
    └── index.html           # rendered graph, communities, and the head-to-head answer
```

## 1. Set up the environment

Requires Python 3.10+. From inside this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

(The pipeline itself only needs `numpy` + the standard library; `torch` is
imported solely to set the thread count, matching the other demos.)

## 2. Peek at the corpus (optional)

```bash
python data/generate_data.py
```

Writes `data/corpus.json`: each document is a short fact rendered as a sentence,
along with its underlying `(subject, relation, object, theme)` triple.

## 3. Run the demo

```bash
python demo/run_demo.py
```

Runs the full pipeline on CPU in well under a second and writes
`data/graphrag_results.json` for the visualization.

### Expected output (highlights)

```
graph : 30 entities, 28 relations
connected components : 2   (too coarse — bridge facts merge whole themes into big blobs)
modularity communities: 4  (recovers the 4 themes)

LOCAL  "Where is Helios located?"  -> "Helios is located in Nevada."   (vanilla RAG: correct)

GLOBAL "What are the main themes discussed across all of the documents?"
  vanilla top-4 RAG theme coverage :    25%   (only Solar Energy)
  GraphRAG theme coverage           :   100%   (all 4 themes, via map-reduce)
```

Vanilla top-k RAG returns four chunks that all happen to fall in one theme, so it
"covers" only 25% of the corpus. GraphRAG summarizes every community and reduces
those summaries into a global answer that names all four themes. (Results are
deterministic.)

## 4. Explore the visualization

Open `visualization/index.html` in any browser (works offline via `file://`).
It shows the extracted knowledge graph (nodes colored by community, bridge edges
highlighted), the per-community summaries, the local-vs-global question contrast,
and the head-to-head theme-coverage bars.

To load live data instead of the baked-in snapshot, serve the folder:

```bash
python -m http.server 8000    # then open http://localhost:8000/visualization/
```

## Why connected-components is not enough

The corpus has a few **bridge** facts (e.g., an investor funding companies in two
different sectors). These merge otherwise-separate themes, so
connected-components lumps the graph into big coarse blobs. Greedy **modularity**
maximization instead recovers the true thematic communities — which is exactly
what the summaries and the global answer are built on.

## Code ↔ paper map

| Paper stage | Component | File |
|---|---|---|
| Source documents → text chunks | `build_chunks` | `src/corpus.py` |
| Entity & relationship extraction | `extract_triples`, `build_graph` | `src/extract.py` |
| Graph construction | `Graph` | `src/graph.py` |
| Community detection (Leiden → here greedy modularity) | `greedy_modularity_communities` | `src/graph.py` |
| Community (element) summaries — MAP | `summarize_community` | `src/rag.py` |
| Global answer via map-reduce — REDUCE | `graphrag_global` | `src/rag.py` |
| Baseline: naive top-k vector RAG | `vanilla_rag`, `BagOfWords` | `src/rag.py` |
| Query-focused sensemaking evaluation | `theme_coverage` | `src/rag.py`, `demo/run_demo.py` |
