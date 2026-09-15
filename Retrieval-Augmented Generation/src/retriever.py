"""A TF-IDF retriever implemented from scratch in numpy (RAG's retriever p_η).

RAG factorizes into a *retriever* that fetches supporting passages and a
*generator* that conditions on them (paper Eq. 1, Section 2). Here the retriever
embeds the query and every document as TF-IDF vectors and ranks documents by
cosine similarity — a transparent, dependency-free stand-in for the paper's
learned DPR bi-encoder (the interface, top-k over an indexed corpus, is the
same).

TF-IDF:
    tf(t, d)   = count of term t in document d
    idf(t)     = log((1 + N) / (1 + df(t))) + 1        (smoothed)
    weight     = tf * idf, then L2-normalized per document
    score(q,d) = cosine(vec(q), vec(d))
"""

from __future__ import annotations

import re

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class TfidfRetriever:
    """Index a corpus and return the top-k documents for a query."""

    def __init__(self, docs: list[str]) -> None:
        self.docs = docs
        self.n = len(docs)
        tokenized = [tokenize(d) for d in docs]

        # Vocabulary.
        vocab = sorted({t for toks in tokenized for t in toks})
        self.vocab = {t: i for i, t in enumerate(vocab)}
        self.vocab_size = len(vocab)

        # Document frequency -> smoothed idf.
        df = np.zeros(self.vocab_size)
        for toks in tokenized:
            for t in set(toks):
                df[self.vocab[t]] += 1
        self.idf = np.log((1 + self.n) / (1 + df)) + 1.0

        # Document matrix (N x V), tf-idf, L2-normalized rows.
        self.doc_vectors = np.zeros((self.n, self.vocab_size), dtype=np.float64)
        for i, toks in enumerate(tokenized):
            self.doc_vectors[i] = self._vectorize(toks)

    def _vectorize(self, tokens: list[str]) -> np.ndarray:
        vec = np.zeros(self.vocab_size, dtype=np.float64)
        for t in tokens:
            idx = self.vocab.get(t)
            if idx is not None:
                vec[idx] += 1.0            # term frequency
        vec *= self.idf                    # * inverse document frequency
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def query_vector(self, query: str) -> np.ndarray:
        return self._vectorize(tokenize(query))

    def retrieve(self, query: str, k: int = 3):
        """Return the top-k as a list of ``(doc_id, score, text)`` tuples."""
        q = self.query_vector(query)
        scores = self.doc_vectors @ q          # cosine (rows already unit-norm)
        order = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i]), self.docs[i]) for i in order]
