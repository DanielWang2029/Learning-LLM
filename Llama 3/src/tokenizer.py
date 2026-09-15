"""A from-scratch byte-level BPE tokenizer (Llama 3, §3.1 "Tokenizer").

Llama 3's headline tokenizer change is its size: a **128K-token** vocabulary
(built with tiktoken-style byte-level BPE), up from Llama 2's 32K SentencePiece
vocabulary. A bigger vocabulary lets common byte sequences merge into single
tokens, so the *same* text is encoded in **fewer** tokens — the paper reports up
to ~15% better compression, which directly speeds up training and inference and
effectively lengthens the usable context.

This module implements the core idea end to end at small scale:

* start from the 256 raw bytes (every string is losslessly representable),
* repeatedly merge the most frequent adjacent token pair (Sennrich et al. 2016,
  "Byte-Pair Encoding") until the target vocabulary size is reached,
* encode new text by replaying the learned merges in order.

Training BPE to several vocabulary sizes and measuring tokens-per-text is what
lets the demo *quantify* the "larger vocab -> fewer tokens" effect.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Tuple


class ByteBPETokenizer:
    """Minimal byte-level Byte-Pair-Encoding tokenizer.

    The vocabulary is the 256 single bytes plus ``vocab_size - 256`` learned
    merges. Every merge maps a pair of existing token ids to one new id.
    """

    def __init__(self) -> None:
        # merges: (id_a, id_b) -> new_id, in the order they were learned.
        self.merges: Dict[Tuple[int, int], int] = {}
        self.vocab_size = 256

    @staticmethod
    def _pair_counts(seqs: List[List[int]]) -> Counter:
        counts: Counter = Counter()
        for seq in seqs:
            for pair in zip(seq, seq[1:]):
                counts[pair] += 1
        return counts

    @staticmethod
    def _merge(seq: List[int], pair: Tuple[int, int], new_id: int) -> List[int]:
        out: List[int] = []
        i = 0
        while i < len(seq):
            if i < len(seq) - 1 and (seq[i], seq[i + 1]) == pair:
                out.append(new_id)
                i += 2
            else:
                out.append(seq[i])
                i += 1
        return out

    def train(self, text: str, vocab_size: int) -> "ByteBPETokenizer":
        """Learn merges until the vocabulary reaches ``vocab_size`` (>= 256)."""
        if vocab_size < 256:
            raise ValueError("vocab_size must be >= 256 (the raw byte alphabet)")
        # Split on whitespace-preserving chunks so merges never cross word gaps
        # unrealistically; each chunk keeps its leading space (GPT-style).
        chunks = _pretokenize(text)
        seqs = [list(chunk.encode("utf-8")) for chunk in chunks]

        self.merges = {}
        next_id = 256
        while next_id < vocab_size:
            counts = self._pair_counts(seqs)
            if not counts:
                break
            best = max(counts, key=lambda p: (counts[p], -p[0], -p[1]))
            if counts[best] < 2:
                break  # nothing worth merging anymore
            self.merges[best] = next_id
            seqs = [self._merge(s, best, next_id) for s in seqs]
            next_id += 1

        self.vocab_size = next_id
        return self

    def encode(self, text: str) -> List[int]:
        """Encode text by replaying learned merges in the order they were added."""
        chunks = _pretokenize(text)
        ids: List[int] = []
        for chunk in chunks:
            seq = list(chunk.encode("utf-8"))
            for pair, new_id in self.merges.items():
                seq = self._merge(seq, pair, new_id)
            ids.extend(seq)
        return ids

    def compression_ratio(self, text: str) -> float:
        """Bytes per token: higher means the same text uses fewer tokens."""
        n_bytes = len(text.encode("utf-8"))
        n_tokens = len(self.encode(text))
        return n_bytes / max(1, n_tokens)


def _pretokenize(text: str) -> List[str]:
    """Split text into word-ish chunks, each keeping its leading whitespace.

    This mirrors the regex pre-tokenization that byte-level BPE tokenizers use so
    that merges stay inside words instead of spanning across spaces.
    """
    chunks: List[str] = []
    current = ""
    for ch in text:
        if ch == " " and current:
            chunks.append(current)
            current = ch
        else:
            current += ch
    if current:
        chunks.append(current)
    return chunks
