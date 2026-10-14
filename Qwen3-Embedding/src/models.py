"""Tiny causal-LM-style embedding and cross-encoder reranker."""
import torch
from torch import nn
import torch.nn.functional as F


class LastTokenEmbedder(nn.Module):
    """A causal Transformer whose final non-padding state is the representation."""
    def __init__(self, vocab_size: int, width: int = 32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, width)
        layer = nn.TransformerEncoderLayer(width, nhead=4, dim_feedforward=64,
                                           dropout=0.0, batch_first=True,
                                           activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        length = ids.shape[1]
        causal = torch.triu(torch.ones(length, length, device=ids.device), diagonal=1).bool()
        states = self.encoder(self.embedding(ids), mask=causal)
        # The causal final token sees its whole prefix; pooling it is a stable
        # tiny analogue of extracting the final hidden state from an LLM.
        return F.normalize(states[:, -1] + states.mean(1), dim=-1)


class CrossEncoderReranker(nn.Module):
    """Scores a query-document pair jointly; unlike a bi-encoder, tokens can interact."""
    def __init__(self, vocab_size: int, width: int = 24):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, width)
        self.scorer = nn.Sequential(nn.Linear(width * 3, width), nn.GELU(), nn.Linear(width, 1))

    def forward(self, query: torch.Tensor, document: torch.Tensor) -> torch.Tensor:
        q = self.embedding(query).mean(1)
        d = self.embedding(document).mean(1)
        return self.scorer(torch.cat([q, d, q * d], dim=-1)).squeeze(-1)
