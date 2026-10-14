"""Instruction-aware contrastive retrieval plus a miniature cross-encoder reranker."""
import json
import sys
from pathlib import Path

import numpy as np
import torch
torch.set_num_threads(1)
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "data")]
from models import CrossEncoderReranker, LastTokenEmbedder  # noqa: E402
from generate import DOCS, TOKENS, batch, document_ids, query_ids  # noqa: E402


def tensor(rows):
    return torch.tensor(np.stack(rows), dtype=torch.long)


def all_docs(inst):
    return tensor([document_ids(inst, i) for i in range(len(DOCS))])


def evaluate(embedder, reranker, examples):
    embed_hits, rerank_hits = 0, 0
    rows = []
    with torch.no_grad():
        for inst, color, shape, target in examples:
            q = tensor([query_ids(inst, color, shape)])
            docs = all_docs(inst)
            # A deliberately compact ANN-style projection gives the reranker
            # ambiguous top-two candidates to resolve.
            scores = (embedder(q)[:, :6] @ embedder(docs)[:, :6].T).squeeze(0)
            retrieved = int(scores.argmax())
            top2 = torch.topk(scores, 2).indices
            pairs = q.repeat(2, 1)
            reranked = int(top2[reranker(pairs, docs[top2]).argmax()])
            embed_hits += retrieved == target
            rerank_hits += reranked == target
            if len(rows) < 8:
                rows.append({"instruction": inst, "query": f"{color} {shape}",
                             "target": " ".join(DOCS[target]),
                             "embedding_top1": " ".join(DOCS[retrieved]),
                             "reranked_top1": " ".join(DOCS[reranked])})
    total = len(examples)
    return embed_hits / total, rerank_hits / total, rows


def main():
    torch.manual_seed(12)
    embedder = LastTokenEmbedder(len(TOKENS))
    reranker = CrossEncoderReranker(len(TOKENS))
    opt_e = torch.optim.Adam(embedder.parameters(), lr=0.004)
    opt_r = torch.optim.Adam(reranker.parameters(), lr=0.01)

    # Paper-style instruction prefixes are placed before both query and candidate.
    for step in range(80):
        examples = batch(24, step)
        queries = tensor([query_ids(a, b, c) for a, b, c, _ in examples])
        targets = torch.tensor([d for _, _, _, d in examples])
        candidate_rows = torch.cat([all_docs("color"), all_docs("shape")], dim=0)
        candidate_ids = targets + torch.tensor([0 if a == "color" else len(DOCS) for a, *_ in examples])
        logits = embedder(queries) @ embedder(candidate_rows).T / 0.08
        loss = F.cross_entropy(logits, candidate_ids)  # in-batch/full-corpus InfoNCE
        opt_e.zero_grad(); loss.backward(); opt_e.step()

        # Cross-encoder sees the query and doc together, including all hard negatives.
        q_rows, d_rows, labels = [], [], []
        for inst, color, shape, target in examples:
            for candidate in range(len(DOCS)):
                q_rows.append(query_ids(inst, color, shape))
                d_rows.append(document_ids(inst, candidate))
                labels.append(float(candidate == target))
        rloss = F.binary_cross_entropy_with_logits(reranker(tensor(q_rows), tensor(d_rows)),
                                                   torch.tensor(labels))
        opt_r.zero_grad(); rloss.backward(); opt_r.step()

    heldout = batch(120, 919)
    emb_acc, rerank_acc, examples = evaluate(embedder, reranker, heldout)
    # This direct contrast also demonstrates that the same words can retrieve differently.
    q_color = ("color", "red", "circle", 0)
    q_shape = ("shape", "red", "circle", 1)
    _, _, instruction_examples = evaluate(embedder, reranker, [q_color, q_shape])

    print("Qwen3-Embedding — causal last-token embeddings with instruction prefixes")
    print(f"embedding retrieval@1: {emb_acc:.1%}  |  after cross-encoder reranking: {rerank_acc:.1%}")
    for row in instruction_examples:
        print(f"  instruction={row['instruction']:<5} query={row['query']:<10} "
              f"→ {row['reranked_top1']} (target: {row['target']})")
    result = {"embedding_retrieval_at_1": round(emb_acc, 3),
              "reranked_retrieval_at_1": round(rerank_acc, 3),
              "examples": examples, "instruction_switch": instruction_examples}
    (ROOT / "data" / "embedding_results.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
