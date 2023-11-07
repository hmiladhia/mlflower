from __future__ import annotations

from typing import Iterable

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


# -- mermaid visualisation--
def _get_line(
    source: str, target: str, edge: str | None = None, edge_type: str | None = None
) -> str:
    if edge_type == "artifact":
        arrow = "--o"
    elif edge_type == "parameter":
        arrow = "-->"
    else:
        arrow = "-.->"

    edge = f"{arrow}|{edge}|" if edge else arrow
    source = source.replace(" ", "-")
    target = target.replace(" ", "-")
    return f"{source} {edge} {target}"


def _get_node_names(graph: Iterable[str], root: str | None = None) -> list[str]:
    text = []
    for node_name in graph:
        if node_name == root:
            continue

        node_id = node_name.replace(" ", "-")
        text.append(f"{node_id}([{node_name}])")
    return text


def get_mermaid_graph(graph: dict[str, EntryPoint], root: str | None = None) -> str:
    text = ["flowchart TD"]

    text.extend(_get_node_names(graph, root))

    keys = ("id", "type")
    for node_name, node in graph.items():
        param_dependencies = {root}
        for key, parameter in node.workflow_parameters.items():
            source, edge_type = map(parameter.get, keys)

            if source == root:
                continue

            text.append(_get_line(source, node_name, key, edge_type))
            param_dependencies.add(source)

        for dependency in node.depends_on:
            if dependency in param_dependencies:
                continue

            text.append(_get_line(dependency, node_name))

    return "\n".join(text)


def to_link(
    text: str, format_: str = "svg", alt_text: str = "Graph Representation"
) -> str:
    import base64
    import zlib

    url = base64.urlsafe_b64encode(zlib.compress(text.encode("utf-8"), 9)).decode(
        "utf-8"
    )
    return f"![{alt_text}](https://kroki.io/mermaid/{format_}/{url})"
