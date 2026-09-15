"""Write a tiny natural-language corpus used to train the BPE tokenizers.

The corpus is original, self-contained English prose (no downloads). It is large
enough and repetitive enough — in the way real language is — for byte-pair merges
to capture common subwords, so the demo can show that a larger vocabulary encodes
the same text in fewer tokens.

Run with:  python data/generate_demo_data.py
"""

from __future__ import annotations

from pathlib import Path

PARAGRAPHS = [
    "A language model reads text one token at a time and predicts the token that "
    "comes next. The better it predicts, the more it seems to understand the "
    "structure of language, from spelling and grammar to facts about the world.",
    "Tokenization is the very first step. Before a model can process a sentence, "
    "the sentence must be broken into tokens. A tokenizer with a small vocabulary "
    "breaks common words into many little pieces, while a tokenizer with a large "
    "vocabulary can represent those same common words as single tokens.",
    "When the vocabulary grows, frequent sequences of characters merge into one "
    "token. The word tokenization, the word understanding, and the word language "
    "then cost fewer tokens than before. Fewer tokens for the same text means "
    "faster training, faster generation, and a longer effective context window.",
    "Byte pair encoding builds this vocabulary from data. It starts from raw bytes "
    "and repeatedly merges the most frequent adjacent pair into a new token. After "
    "many merges the vocabulary contains whole words and common word fragments, and "
    "the encoded length of ordinary text shrinks accordingly.",
    "The transformer that consumes these tokens uses attention to mix information "
    "across positions. Rotary position embeddings tell the model where each token "
    "sits, grouped query attention shares keys and values across heads to save "
    "memory, and a gated feed forward network transforms each position in place.",
    "Training a large language model is mostly a matter of scale: more data, more "
    "parameters, and more compute, arranged so that the model keeps improving its "
    "next token predictions. Careful tokenization makes every one of those tokens "
    "carry more information, which is why the tokenizer matters so much.",
]


def build_corpus(repeats: int = 6) -> str:
    # Repeating the paragraphs mimics the natural recurrence of words in a real
    # corpus, giving byte-pair encoding enough signal to learn useful merges.
    block = "\n".join(PARAGRAPHS)
    return "\n".join(block for _ in range(repeats))


def main() -> None:
    data_dir = Path(__file__).resolve().parent
    corpus = build_corpus()
    (data_dir / "corpus.txt").write_text(corpus, encoding="utf-8")
    print(f"Wrote {data_dir / 'corpus.txt'} "
          f"({len(corpus)} chars, {len(corpus.encode('utf-8'))} bytes)")


if __name__ == "__main__":
    main()
