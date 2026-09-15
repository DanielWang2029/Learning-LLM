"""A minimal undirected graph plus two community-detection methods (GraphRAG §2.2).

* ``connected_components`` — the simplest possible community notion.
* ``greedy_modularity_communities`` — a from-scratch Clauset-Newman-Moore style
  agglomerative modularity maximizer (numpy).  It recovers the thematic clusters
  even though a few bridge edges make the whole graph one connected component.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np


class Graph:
    def __init__(self) -> None:
        self.adj: dict[str, set[str]] = defaultdict(set)
        self.edges: list[tuple[str, str, str]] = []

    def add_edge(self, u: str, v: str, rel: str = "") -> None:
        self.adj[u].add(v)
        self.adj[v].add(u)
        self.edges.append((u, v, rel))

    def node_list(self) -> list[str]:
        return sorted(self.adj.keys())

    def degree(self, u: str) -> int:
        return len(self.adj[u])

    def num_nodes(self) -> int:
        return len(self.adj)

    def num_edges(self) -> int:
        return len(self.edges)


def connected_components(g: Graph) -> list[set[str]]:
    seen: set[str] = set()
    comps: list[set[str]] = []
    for start in g.node_list():
        if start in seen:
            continue
        stack, comp = [start], set()
        while stack:
            u = stack.pop()
            if u in seen:
                continue
            seen.add(u)
            comp.add(u)
            stack.extend(g.adj[u] - seen)
        comps.append(comp)
    return comps


def greedy_modularity_communities(g: Graph) -> list[set[str]]:
    """Agglomeratively merge communities to maximize modularity Q."""
    nodes = g.node_list()
    idx = {n: i for i, n in enumerate(nodes)}
    n = len(nodes)
    A = np.zeros((n, n))
    for u, v, _ in g.edges:
        A[idx[u], idx[v]] += 1.0
        A[idx[v], idx[u]] += 1.0
    m = A.sum() / 2.0
    k = A.sum(axis=1)

    comms: list[set[int]] = [{i} for i in range(n)]

    def gain(a: int, b: int) -> float:
        e_ab = sum(A[i, j] for i in comms[a] for j in comms[b]) / (2.0 * m)
        a_a = sum(k[i] for i in comms[a]) / (2.0 * m)
        a_b = sum(k[i] for i in comms[b]) / (2.0 * m)
        return 2.0 * (e_ab - a_a * a_b)

    improved = True
    while improved and len(comms) > 1:
        improved = False
        best, best_gain = None, 1e-9
        for a in range(len(comms)):
            for b in range(a + 1, len(comms)):
                connected = any(A[i, j] > 0 for i in comms[a] for j in comms[b])
                if not connected:
                    continue
                gg = gain(a, b)
                if gg > best_gain:
                    best_gain, best = gg, (a, b)
        if best is not None:
            a, b = best
            comms[a] |= comms[b]
            del comms[b]
            improved = True

    return [set(nodes[i] for i in c) for c in comms]
