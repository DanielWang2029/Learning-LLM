"""Retrieval-Augmented Generation end-to-end on CPU (Lewis et al., 2020).

Pipeline (paper Eq. 1):  question --> retriever p_η --> top-k docs --> generator
p_θ --> grounded answer.

The demo:
  1. Indexes a tiny corpus of *fictional* facts with a TF-IDF retriever.
  2. Answers questions two ways and compares:
        - RAG          : retrieve top-k, read the answer from the top passage.
        - Closed-book  : answer from parametric priors only (no retrieval).
  3. Reports retrieval accuracy and answer accuracy WITH vs WITHOUT retrieval.
     Because the entities are invented, the closed-book model can only
     hallucinate, while RAG stays grounded and correct.

Runs on CPU in well under a second.  Run with:  python demo/run_demo.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import corpus
from src.generator import closed_book_answer, rag_answer
from src.retriever import TfidfRetriever

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TOP_K = 3


def norm(s: str) -> str:
    return s.strip().lower()


def banner(t: str) -> None:
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


def main() -> None:
    retriever = TfidfRetriever(corpus.DOCS)
    questions = corpus.build_questions()

    print("Retrieval-Augmented Generation in miniature")
    print(f"  corpus: {len(corpus.DOCS)} fictional facts | vocab {retriever.vocab_size}"
          f" | retriever: TF-IDF cosine | top-k={TOP_K}")
    print("  entities are invented, so a closed-book model cannot know these facts")

    # ---------------------------------------------------------- Per-question
    n = len(questions)
    retr_hit_top1 = retr_hit_topk = 0
    rag_correct = closed_correct = 0
    records = []
    for q in questions:
        answer, retrieved = rag_answer(q["question"], retriever, k=TOP_K)
        top_ids = [d for d, _, _ in retrieved]
        top1_ok = top_ids[0] == q["gold_doc_id"]
        topk_ok = q["gold_doc_id"] in top_ids
        retr_hit_top1 += top1_ok
        retr_hit_topk += topk_ok

        rag_ok = norm(answer) == norm(q["answer"])
        cb = closed_book_answer(q["question"], q["relation"])
        cb_ok = norm(cb) == norm(q["answer"])
        rag_correct += rag_ok
        closed_correct += cb_ok

        records.append({
            "question": q["question"], "gold": q["answer"],
            "gold_doc_id": q["gold_doc_id"], "relation": q["relation"],
            "retrieved": [{"doc_id": d, "score": round(s, 4), "text": t}
                          for d, s, t in retrieved],
            "top1_correct": top1_ok, "topk_correct": topk_ok,
            "rag_answer": answer, "rag_correct": rag_ok,
            "closed_book_answer": cb, "closed_book_correct": cb_ok,
        })

    # ---------------------------------------------------------- Show examples
    banner("EXAMPLES — retrieved evidence, grounded answer vs closed-book")
    for qid in corpus.DEMO_QUESTION_IDS:
        r = records[qid]
        print(f"\nQ: {r['question']}")
        for rank, d in enumerate(r["retrieved"]):
            mark = "★" if d["doc_id"] == r["gold_doc_id"] else " "
            print(f"   {mark} top{rank+1} [sim {d['score']:.3f}]  {d['text']}")
        rag_tag = "OK " if r["rag_correct"] else "WRONG"
        cb_tag = "OK " if r["closed_book_correct"] else "HALLUCINATED"
        print(f"   -> RAG answer         : {r['rag_answer']!r:20s} [{rag_tag}]  (gold: {r['gold']!r})")
        print(f"   -> closed-book answer : {r['closed_book_answer']!r:20s} [{cb_tag}]")

    # ---------------------------------------------------------- Metrics
    banner("METRICS")
    print(f"  questions                          : {n}")
    print(f"  retrieval accuracy (top-1)         : {retr_hit_top1/n*100:5.1f}%  "
          f"({retr_hit_top1}/{n})")
    print(f"  retrieval recall   (top-{TOP_K})         : {retr_hit_topk/n*100:5.1f}%  "
          f"({retr_hit_topk}/{n})")
    print(f"  answer accuracy WITH retrieval     : {rag_correct/n*100:5.1f}%  "
          f"({rag_correct}/{n})   <- RAG")
    print(f"  answer accuracy WITHOUT retrieval  : {closed_correct/n*100:5.1f}%  "
          f"({closed_correct}/{n})   <- closed-book (hallucinates)")

    # ---------------------------------------------------------- Persist
    DATA_DIR.mkdir(exist_ok=True)
    example = records[corpus.DEMO_QUESTION_IDS[0]]
    payload = {
        "config": {"num_docs": len(corpus.DOCS), "vocab_size": retriever.vocab_size,
                   "top_k": TOP_K, "num_questions": n},
        "metrics": {
            "retrieval_top1": retr_hit_top1 / n,
            "retrieval_topk": retr_hit_topk / n,
            "answer_with_retrieval": rag_correct / n,
            "answer_without_retrieval": closed_correct / n,
        },
        "example": example,
        "records": records,
        "corpus_sample": corpus.DOCS[:8],
    }
    out = DATA_DIR / "rag_results.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"\n  wrote {out}")

    # ---------------------------------------------------------- Asserts
    assert retr_hit_top1 / n >= 0.9, "retriever should find the gold doc at top-1"
    assert rag_correct / n >= 0.9, "RAG answers should be grounded and correct"
    assert closed_correct / n <= 0.1, "closed-book should fail on invented facts"
    assert rag_correct > closed_correct + n * 0.5, "RAG must clearly beat closed-book"
    print("\nOK: retrieval grounds the generator — RAG answers correctly where the "
          "closed-book baseline hallucinates.")


if __name__ == "__main__":
    main()
