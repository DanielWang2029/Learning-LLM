"""Minimal from-scratch reproduction of Step 3.5 Flash's efficiency idea.

StepFun Team, *"Step 3.5 Flash: Open Frontier-Level Intelligence with 11B
Active Parameters"* (2026), arXiv:2602.10604.

The headline is a **sparse Mixture-of-Experts (MoE)**: a large 196B-parameter
foundation from which only ~11B parameters are *active* per token, via top-k
routing over many experts. This package implements that sparse-routing idea at
a tiny CPU scale.

Public API:
- ``SparseMoE``       many experts, top-k router, weighted combine.
- ``MoEClassifier``   a tiny model that uses SparseMoE as its capacity layer.
- ``DenseClassifier`` a dense baseline (all parameters active every token).
- ``count_params`` / ``active_params`` for the total-vs-active comparison.
"""

from .moe import (
    SparseMoE,
    MoEClassifier,
    DenseClassifier,
    count_params,
    active_params,
)

__all__ = [
    "SparseMoE",
    "MoEClassifier",
    "DenseClassifier",
    "count_params",
    "active_params",
]
