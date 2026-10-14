"""A deliberately small relative-position retriever for illustrating DCA."""
import torch
from torch import nn


def dca_distance(query_pos: int, key_pos: int, chunk_size: int = 32,
                 local_window: int = 8) -> int:
    """Map a causal relative distance into the short-context position range.

    This mirrors the three DCA cases: preserve intra-chunk and close successive
    chunk distances; use a repeated, chunk-relative position for inter-chunk pairs.
    """
    distance = query_pos - key_pos
    q_chunk, k_chunk = query_pos // chunk_size, key_pos // chunk_size
    if q_chunk == k_chunk or (q_chunk == k_chunk + 1 and distance <= local_window):
        return distance
    return chunk_size + ((query_pos % chunk_size) - (key_pos % chunk_size)) % chunk_size


class TinyRelativeRetriever(nn.Module):
    """One-head content + learned-relative-bias attention, trained only to length 64."""
    def __init__(self, train_length: int = 64):
        super().__init__()
        self.train_length = train_length
        self.content_scale = nn.Parameter(torch.tensor(0.5))
        # The query follows a 64-token context, so its oldest key is distance 64.
        self.relative_bias = nn.Parameter(torch.zeros(train_length + 1))

    def logits(self, needle_positions: torch.Tensor, query_pos: int, use_dca: bool) -> torch.Tensor:
        """Score every candidate key. A needle's weak content match must survive position bias."""
        keys = torch.arange(query_pos, device=needle_positions.device)
        raw_dist = query_pos - keys
        if use_dca:
            remapped = torch.tensor(
                [dca_distance(query_pos, int(k)) for k in keys.tolist()],
                device=keys.device,
            )
            bias = self.relative_bias[remapped]
        else:
            # Relative positions unseen during short-context training are deliberately OOD.
            bias = torch.where(
                raw_dist <= self.train_length,
                self.relative_bias[raw_dist.clamp(max=self.train_length)],
                torch.full_like(raw_dist, -16.0, dtype=torch.float),
            )
        return needle_positions.float() * self.content_scale + bias

    def forward(self, needle_positions: torch.Tensor, query_pos: int, use_dca: bool) -> torch.Tensor:
        return torch.softmax(self.logits(needle_positions, query_pos, use_dca), dim=-1)
