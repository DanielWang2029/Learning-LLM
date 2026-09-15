"""Two ways to answer a query, and the metric that separates them (GraphRAG §2-3).

* ``vanilla_rag`` — classic top-k chunk retrieval: embed the chunks and the
  query with a bag-of-words model, return the k most similar chunks.  Great for
  *local* questions ("Where is Helios?"), but a fixed handful of chunks can never
  represent a whole corpus, so it fails *global* sensemaking questions.

* ``graphrag_global`` — GraphRAG's map-reduce: summarize every community (MAP),
  then aggregate those summaries into one global answer (REDUCE).  It sees the
  whole corpus by construction.
"""

from __future__ import annotations

import re
from collections import Counter

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BagOfWords:
    """A tiny bag-of-words embedder (numpy) — stands in for a dense retriever."""

    def __init__(self, texts: list[str]) -> None:
        vocab = sorted({t for text in texts for t in _tokens(text)})
        self.vocab = {w: i for i, w in enumerate(vocab)}

    def embed(self, text: str) -> np.ndarray:
        v = np.zeros(len(self.vocab))
        for t in _tokens(text):
            if t in self.vocab:
                v[self.vocab[t]] += 1.0
        return v

    def similarity(self, a: str, b: str) -> float:
        va, vb = self.embed(a), self.embed(b)
        na, nb = np.linalg.norm(va), np.linalg.norm(vb)
        if na == 0 or nb == 0:
            return 0.0
        return float(va @ vb / (na * nb))


def vanilla_rag(query: str, chunks: list[tuple[str, str]], k: int, bow: BagOfWords):
    """Return the top-k (text, theme, score) chunks by cosine similarity."""
    qv = bow.embed(query)
    qn = np.linalg.norm(qv)
    scored = []
    for text, theme in chunks:
        cv = bow.embed(text)
        cn = np.linalg.norm(cv)
        sim = 0.0 if qn == 0 or cn == 0 else float(qv @ cv / (qn * cn))
        scored.append((text, theme, sim))
    scored.sort(key=lambda t: t[2], reverse=True)
    return scored[:k]


def summarize_community(comm: set[str], g, facts) -> dict:
    """MAP step: a per-community summary derived from its facts."""
    facts_in = [f for f in facts if f[0] in comm and f[2] in comm]
    theme_counts = Counter(f[3] for f in facts_in)
    dominant = theme_counts.most_common(1)[0][0] if facts_in else "(unknown)"
    key_entities = sorted(comm, key=lambda n: g.degree(n), reverse=True)[:3]
    relations = sorted({f[1] for f in facts_in})
    text = (f"This community centers on {dominant}: key entities are "
            f"{', '.join(key_entities)}; relationships include "
            f"{', '.join(relations) or 'various links'}.")
    return {
        "dominant_theme": dominant,
        "size": len(comm),
        "key_entities": key_entities,
        "relations": relations,
        "summary": text,
    }


def graphrag_global(query: str, communities: list[set[str]], g, facts) -> dict:
    """MAP each community to a summary, then REDUCE to a global answer."""
    community_summaries = [summarize_community(c, g, facts) for c in communities]
    themes = []
    for s in community_summaries:
        if s["dominant_theme"] not in themes:
            themes.append(s["dominant_theme"])
    answer = ("Across the whole corpus there are "
              f"{len(themes)} main themes: {', '.join(themes)}.")
    return {"answer": answer, "themes": themes,
            "community_summaries": community_summaries}


def theme_coverage(found: list[str], all_themes: list[str]) -> float:
    found_set = {t for t in found if t in all_themes}
    return len(found_set) / len(all_themes)
