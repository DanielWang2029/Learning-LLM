"""RAG: Retrieval-Augmented Generation (Lewis et al., 2020).

A minimal, CPU-friendly, from-scratch reproduction: a TF-IDF retriever over a
tiny knowledge corpus + an extractive generator that conditions on the retrieved
passage, contrasted with a closed-book (parametric-only) baseline.
"""

from . import corpus
from .retriever import TfidfRetriever, tokenize
from .generator import closed_book_answer, rag_answer, read_answer

__all__ = [
    "corpus",
    "TfidfRetriever",
    "tokenize",
    "rag_answer",
    "read_answer",
    "closed_book_answer",
]
