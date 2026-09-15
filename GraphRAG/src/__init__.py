"""GraphRAG: From Local to Global (Microsoft 2024, arXiv 2404.16130).

A tiny, dependency-light reproduction of the GraphRAG pipeline: extract an
entity/relation graph from a corpus, detect communities, summarize each
community, then answer a GLOBAL query by map-reduce over the community summaries
— something vanilla top-k chunk RAG cannot do.
"""

from .corpus import FACTS, THEMES, render_sentence, build_chunks
from .extract import extract_triples, build_graph
from .graph import Graph, greedy_modularity_communities, connected_components
from .rag import (
    BagOfWords,
    vanilla_rag,
    graphrag_global,
    summarize_community,
    theme_coverage,
)

__all__ = [
    "FACTS", "THEMES", "render_sentence", "build_chunks",
    "extract_triples", "build_graph",
    "Graph", "greedy_modularity_communities", "connected_components",
    "BagOfWords", "vanilla_rag", "graphrag_global", "summarize_community",
    "theme_coverage",
]
