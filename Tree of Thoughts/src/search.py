"""Tree-of-Thoughts BFS search, with a greedy chain-of-thought baseline.

The paper's ToT = a *thought generator* (candidate next steps), a *state
evaluator* (sure/maybe/impossible), and a deliberate *search* (BFS here) with
pruning and a beam. The greedy chain-of-thought baseline is the same search
with beam width 1 and no alternatives: it commits to the single best-looking
next step at every level and cannot backtrack — the failure mode the paper
highlights.

Every node generated is recorded so the search tree can be visualized.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .game24 import evaluate, moves, TARGET


@dataclass
class Node:
    id: int
    parent: int | None
    nums: tuple[int, ...]
    op: str | None       # operation applied to reach this node from its parent
    depth: int
    label: str
    value: float
    status: str = "explored"  # "explored" | "pruned" | "solution"


@dataclass
class SearchResult:
    solved: bool
    beam: int
    nodes: list[Node] = field(default_factory=list)
    solution_path: list[str] = field(default_factory=list)
    n_expanded: int = 0


def search(start: tuple[int, ...], beam: int) -> SearchResult:
    """BFS over states, keeping the top ``beam`` promising nodes per level.

    ``beam == 1`` reproduces a greedy left-to-right chain of thought.
    """
    label, value = evaluate(start)
    root = Node(id=0, parent=None, nums=start, op=None, depth=0, label=label, value=value)
    nodes = [root]
    next_id = 1
    frontier = [root]
    result = SearchResult(solved=False, beam=beam, nodes=nodes)

    for depth in range(len(start) - 1):  # 4 numbers -> 3 combination steps
        generated: list[Node] = []
        for node in frontier:
            result.n_expanded += 1
            for child_nums, desc in moves(node.nums):
                lab, val = evaluate(child_nums)
                child = Node(
                    id=next_id, parent=node.id, nums=child_nums, op=desc,
                    depth=depth + 1, label=lab, value=val,
                )
                next_id += 1
                nodes.append(child)
                generated.append(child)
        # Prune states the evaluator judges impossible (sound for len<=2).
        alive = []
        for c in generated:
            if c.label == "impossible":
                c.status = "pruned"
            else:
                alive.append(c)
        # Keep only the top-`beam` most promising as the next frontier.
        alive.sort(key=lambda c: c.value, reverse=True)
        keep = alive[:beam]
        for c in alive[beam:]:
            c.status = "pruned"  # generated but not expanded
        frontier = keep
        if not frontier:
            break

    # A solution is a surviving leaf equal to 24.
    for node in nodes:
        if node.depth == len(start) - 1 and node.nums == (TARGET,) and node.status != "pruned":
            result.solved = True
            # Mark the path back to the root as the solution.
            path_ops: list[str] = []
            cur: Node | None = node
            while cur is not None:
                cur.status = "solution"
                if cur.op is not None:
                    path_ops.append(cur.op)
                cur = nodes[cur.parent] if cur.parent is not None else None
            result.solution_path = list(reversed(path_ops))
            break
    return result
