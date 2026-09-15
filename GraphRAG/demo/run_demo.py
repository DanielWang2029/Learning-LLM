"""GraphRAG in miniature — from local to global (arXiv 2404.16130).

Pipeline, end-to-end on CPU in well under a second:

  1. EXTRACT an entity/relation graph from a toy corpus of facts.
  2. DETECT communities (from-scratch greedy modularity).
  3. SUMMARIZE each community (the MAP step).
  4. ANSWER a query two ways:
       - a LOCAL question ("Where is Helios located?") — vanilla top-k RAG is fine.
       - a GLOBAL sensemaking question ("What are the main themes?") — vanilla
         top-k RAG misses most themes, while GraphRAG's map-reduce covers them all.

Run with:  python demo/run_demo.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402  (perf convention for these demos)
torch.set_num_threads(1)

from src.corpus import FACTS, THEMES, build_chunks  # noqa: E402
from src.extract import build_graph  # noqa: E402
from src.graph import greedy_modularity_communities, connected_components  # noqa: E402
from src.rag import (  # noqa: E402
    BagOfWords, vanilla_rag, graphrag_global, summarize_community, theme_coverage,
)

DATA_DIR = ROOT / "data"
K = 4                       # top-k for vanilla RAG
LOCAL_Q = "Where is Helios located?"
GLOBAL_Q = "What are the main themes discussed across all of the documents?"


def banner(t: str) -> None:
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


def main() -> None:
    chunks = build_chunks()
    sentences = [c[0] for c in chunks]

    banner("STEP 1 — Extract an entity/relation graph from the corpus")
    g, triples = build_graph(sentences)
    print(f"  corpus: {len(sentences)} facts/chunks across {len(THEMES)} themes")
    print(f"  graph : {g.num_nodes()} entities, {g.num_edges()} relations")
    print("  example triples:")
    for t in triples[:4]:
        print(f"    {t[0]} --{t[1]}--> {t[2]}")

    banner("STEP 2 — Detect communities (greedy modularity)")
    comps = connected_components(g)
    communities = greedy_modularity_communities(g)
    print(f"  connected components : {len(comps)}  (too coarse — bridge facts merge "
          f"whole themes into big blobs)")
    print(f"  modularity communities: {len(communities)}  (recovers the "
          f"{len(THEMES)} themes)")
    for i, c in enumerate(communities):
        print(f"    community {i}: {sorted(c)}")

    banner("STEP 3 — Summarize each community (the MAP step)")
    summaries = [summarize_community(c, g, FACTS) for c in communities]
    for i, s in enumerate(summaries):
        print(f"  [{i}] {s['summary']}")

    # ---- LOCAL question -----------------------------------------------------
    banner("STEP 4a — A LOCAL question (vanilla top-k RAG handles this well)")
    bow = BagOfWords(sentences)
    local_hits = vanilla_rag(LOCAL_Q, chunks, k=1, bow=bow)
    print(f"  Q: {LOCAL_Q}")
    print(f"  vanilla RAG top-1 -> \"{local_hits[0][0]}\"  (sim={local_hits[0][2]:.2f})")
    local_ok = "Helios" in local_hits[0][0] and "located" in local_hits[0][0]
    print(f"  answered correctly: {local_ok}")

    # ---- GLOBAL question ----------------------------------------------------
    banner("STEP 4b — A GLOBAL question (vanilla top-k RAG vs GraphRAG)")
    print(f"  Q: {GLOBAL_Q}\n")

    vanilla_hits = vanilla_rag(GLOBAL_Q, chunks, k=K, bow=bow)
    vanilla_themes = []
    print(f"  VANILLA RAG (top-{K} chunks):")
    for text, theme, sim in vanilla_hits:
        print(f"    - \"{text}\"  [{theme}]  (sim={sim:.2f})")
        if theme not in vanilla_themes:
            vanilla_themes.append(theme)
    vanilla_cov = theme_coverage(vanilla_themes, THEMES)
    print(f"  themes covered: {sorted(set(vanilla_themes))}  "
          f"-> {vanilla_cov*100:.0f}% of the {len(THEMES)} themes\n")

    gr = graphrag_global(GLOBAL_Q, communities, g, FACTS)
    graph_cov = theme_coverage(gr["themes"], THEMES)
    print(f"  GRAPHRAG (map-reduce over community summaries):")
    print(f"    reduced answer: {gr['answer']}")
    print(f"  themes covered: {sorted(set(gr['themes']))}  "
          f"-> {graph_cov*100:.0f}% of the {len(THEMES)} themes")

    banner("RESULT")
    print(f"  vanilla top-{K} RAG theme coverage : {vanilla_cov*100:5.0f}%")
    print(f"  GraphRAG theme coverage           : {graph_cov*100:5.0f}%")
    print(f"  GraphRAG advantage                : +{(graph_cov-vanilla_cov)*100:.0f} "
          f"percentage points on the global question")

    # ---- layout + persist for the visualization ----------------------------
    node_comm = {}
    for ci, c in enumerate(communities):
        for node in c:
            node_comm[node] = ci
    rng = np.random.default_rng(0)
    n_comm = len(communities)
    centers = {ci: (math.cos(2 * math.pi * ci / n_comm),
                    math.sin(2 * math.pi * ci / n_comm)) for ci in range(n_comm)}
    nodes_json = []
    for node in g.node_list():
        ci = node_comm[node]
        cx, cy = centers[ci]
        jitter = rng.normal(0, 0.18, size=2)
        nodes_json.append({
            "id": node, "community": ci, "degree": g.degree(node),
            "theme": summaries[ci]["dominant_theme"],
            "x": round(cx * 2.2 + jitter[0], 3),
            "y": round(cy * 2.2 + jitter[1], 3),
        })
    edges_json = [{"source": u, "target": v, "rel": rel,
                   "bridge": node_comm[u] != node_comm[v]} for u, v, rel in g.edges]

    DATA_DIR.mkdir(exist_ok=True)
    payload = {
        "themes": THEMES,
        "graph": {"nodes": nodes_json, "edges": edges_json,
                  "n_nodes": g.num_nodes(), "n_edges": g.num_edges(),
                  "n_connected_components": len(comps),
                  "n_communities": len(communities)},
        "communities": [{"id": i, **summaries[i], "members": sorted(communities[i])}
                        for i in range(len(communities))],
        "queries": {
            "local": {"q": LOCAL_Q, "answer": local_hits[0][0], "ok": local_ok},
            "global": {
                "q": GLOBAL_Q,
                "vanilla_hits": [{"text": t, "theme": th, "sim": round(s, 3)}
                                 for t, th, s in vanilla_hits],
                "vanilla_themes": sorted(set(vanilla_themes)),
                "vanilla_coverage": vanilla_cov,
                "graphrag_answer": gr["answer"],
                "graphrag_themes": gr["themes"],
                "graphrag_coverage": graph_cov,
                "k": K,
            },
        },
    }
    out = DATA_DIR / "graphrag_results.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"\n  wrote {out}")

    # ---- asserts ------------------------------------------------------------
    assert g.num_nodes() > 15, "graph should have many entities"
    assert len(communities) == len(THEMES), (
        f"expected {len(THEMES)} communities, got {len(communities)}")
    assert local_ok, "vanilla RAG should answer the local question"
    assert graph_cov == 1.0, "GraphRAG should cover all themes"
    assert vanilla_cov < graph_cov, (
        "vanilla top-k RAG should miss themes on the global question")
    print("\nOK: GraphRAG answers the global sensemaking question (all themes), "
          "where vanilla top-k RAG only covers a fraction.")


if __name__ == "__main__":
    main()
