"""The generator p_θ: answer a question conditioned on retrieved text.

RAG's generator produces the answer *given the retrieved passage* (paper Eq. 1).
We use a faithful **extractive reader**: it pulls the answer span directly out of
the retrieved sentence, so the answer is provably grounded in the retrieved
text. If retrieval returns the wrong passage, the reader extracts the wrong
answer — which is precisely why retrieval quality drives answer quality.

For contrast we also provide a **closed-book** baseline that answers from a fixed
parametric "memory" of real-world associations. Because the corpus is about
fictional entities, that memory is useless and the baseline hallucinates.
"""

from __future__ import annotations

from .corpus import CLOSED_BOOK_PRIORS


def read_answer(retrieved_doc: str) -> str:
    """Extract the answer from a retrieved sentence "The R of E is V.".

    The reader conditions purely on the retrieved text: it returns the span
    after " is ", with the trailing period removed.
    """
    marker = " is "
    idx = retrieved_doc.find(marker)
    if idx == -1:
        return "(could not read answer from passage)"
    span = retrieved_doc[idx + len(marker):].strip()
    return span[:-1].strip() if span.endswith(".") else span


def rag_answer(question: str, retriever, k: int = 3):
    """Retrieve top-k, then read the answer from the top passage.

    Returns ``(answer, retrieved)`` where ``retrieved`` is the list of
    ``(doc_id, score, text)`` tuples the retriever returned.
    """
    retrieved = retriever.retrieve(question, k=k)
    top_doc = retrieved[0][2]
    return read_answer(top_doc), retrieved


def closed_book_answer(question: str, relation: str) -> str:
    """Answer WITHOUT retrieval, from fixed parametric priors → hallucinates.

    Deterministic per question (a stable hash picks one plausible-but-wrong
    real-world value), mimicking a confident parametric guess.
    """
    pool = CLOSED_BOOK_PRIORS.get(relation)
    if not pool:
        return "(unknown)"
    h = sum(ord(c) for c in question)
    return pool[h % len(pool)]
