"""Entity + relation extraction from text, and graph construction (GraphRAG §2.1).

Real GraphRAG uses an LLM to extract (entity, relation, entity) triples from
source text.  Here we use a small deterministic rule-based extractor over the
controlled sentence templates, which is enough to build the same kind of
knowledge graph without any network calls.
"""

from __future__ import annotations

from .corpus import RELATIONS
from .graph import Graph

# Longest relation phrases first so "is located in" matches before "in", etc.
_REL_PHRASES = sorted(RELATIONS.items(), key=lambda kv: -len(kv[1]))


def extract_triples(sentences: list[str]):
    """Parse each sentence into a (subject, relation, object) triple."""
    triples = []
    for s in sentences:
        text = s.rstrip(".").strip()
        for rel_key, phrase in _REL_PHRASES:
            marker = f" {phrase} "
            if marker in text:
                subj, obj = text.split(marker, 1)
                triples.append((subj.strip(), rel_key, obj.strip()))
                break
    return triples


def build_graph(sentences: list[str]) -> tuple[Graph, list]:
    """Extract triples from text and assemble an undirected entity graph."""
    triples = extract_triples(sentences)
    g = Graph()
    for subj, rel, obj in triples:
        g.add_edge(subj, obj, rel)
    return g, triples
