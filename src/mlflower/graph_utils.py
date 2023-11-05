from __future__ import annotations

from .entry_point import EntryPoint


def _topological_sort_rec(
    node: str,
    nodes: dict[str, EntryPoint],
    unseen: dict[str, bool],
    sorted_nodes: list[str],
) -> list[str]:
    unseen[node] = False

    for n in filter(unseen.get, nodes[node].depends_on):
        sorted_nodes = _topological_sort_rec(n, nodes, unseen, sorted_nodes)

    return [*sorted_nodes, node]


def topological_sort(
    nodes: dict[str, EntryPoint], root: str | None = None
) -> list[str]:
    if root and nodes[root].depends_on:
        raise ValueError(
            f"Root cannot have dependencies: {root} -> {nodes[root].depends_on}"
        )

    unseen = {key: key != root for key in nodes}

    sorted_nodes = []
    for node in filter(unseen.get, nodes):
        sorted_nodes = _topological_sort_rec(node, nodes, unseen, sorted_nodes)

    return sorted_nodes
