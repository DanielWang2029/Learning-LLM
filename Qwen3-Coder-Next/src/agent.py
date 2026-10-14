"""Tiny MoE policy and deterministic executable environment."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class HybridAttention(nn.Module):
    """Tiny hybrid context block: local convolution plus global self-attention.

    Qwen3-Next uses a much more sophisticated hybrid-attention design.  This
    two-token educational version makes both routes explicit before the MoE.
    """

    def __init__(self, width: int) -> None:
        super().__init__()
        self.local = nn.Conv1d(width, width, kernel_size=3, padding=1)
        self.global_attention = nn.MultiheadAttention(width, num_heads=4, batch_first=True)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        local = self.local(sequence.transpose(1, 2)).transpose(1, 2)
        global_context, _ = self.global_attention(sequence, sequence, sequence, need_weights=False)
        return F.gelu(local + global_context)


class TopKMoE(nn.Module):
    """Sparse top-k expert FFN: all experts exist, only k contribute per token."""

    def __init__(self, width: int, experts: int = 4, top_k: int = 2) -> None:
        super().__init__()
        self.top_k = top_k
        self.gate = nn.Linear(width, experts, bias=False)
        self.experts = nn.ModuleList(
            [nn.Sequential(nn.Linear(width, width * 2), nn.GELU(), nn.Linear(width * 2, width)) for _ in range(experts)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights, selected = self.gate(x).topk(self.top_k, dim=-1)
        weights = weights.softmax(dim=-1)
        all_outputs = torch.stack([expert(x) for expert in self.experts], dim=-2)
        picked = all_outputs.gather(
            -2, selected.unsqueeze(-1).expand(*selected.shape, x.size(-1))
        )
        return (picked * weights.unsqueeze(-1)).sum(dim=-2)

    def total_and_active_params(self) -> tuple[int, int]:
        gate = sum(p.numel() for p in self.gate.parameters())
        each_expert = sum(p.numel() for p in self.experts[0].parameters())
        return gate + len(self.experts) * each_expert, gate + self.top_k * each_expert


class MoEPolicy(nn.Module):
    """A tiny task-conditioned policy that emits one program ID."""

    def __init__(self, tasks: int, programs: int, width: int = 32) -> None:
        super().__init__()
        self.embedding = nn.Embedding(tasks, width)
        self.control_token = nn.Parameter(torch.zeros(1, 1, width))
        self.hybrid = HybridAttention(width)
        self.global_route = nn.Linear(width, width)
        self.moe = TopKMoE(width, experts=4, top_k=2)
        self.head = nn.Linear(width, programs)

    def forward(self, task_ids: torch.Tensor) -> torch.Tensor:
        task = self.embedding(task_ids).unsqueeze(1)
        control = self.control_token.expand(task.size(0), -1, -1)
        # The local and global routes exchange task/control context here.
        x = self.hybrid(torch.cat([control, task], dim=1))[:, 1]
        x = F.gelu(self.global_route(x)) + self.moe(x)
        return self.head(x)

    def moe_counts(self) -> tuple[int, int]:
        """Return all MoE parameters versus parameters activated per decision."""
        return self.moe.total_and_active_params()


def execute(program_id: int, x: int) -> int:
    """Deterministic executor for six emitted one-line programs."""
    operations = (
        lambda n: n + 1, lambda n: n + 3, lambda n: n * 2,
        lambda n: n * 3, lambda n: n * n, lambda n: -n,
    )
    return operations[program_id](x)


def reward(program_id: int, target_id: int, tests: list[int]) -> float:
    """Fraction of hidden tests passed; no gradient flows through this verifier."""
    return sum(execute(program_id, x) == execute(target_id, x) for x in tests) / len(tests)
