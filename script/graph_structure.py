from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Iterable
import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D
from torch_geometric.datasets import EllipticBitcoinDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "EllipticBitcoinDataset"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "figures" / "elliptic_components"

LABEL_NAMES = {
    0: "licit",
    1: "illicit",
    2: "unknown",
}

LABEL_COLORS = {
    0: "#2a9d8f",
    1: "#e63946",
    2: "#8d99ae",
}


def load_elliptic_data(root: Path | str = DEFAULT_DATA_ROOT):
    """Download if needed and load the PyG Elliptic Bitcoin graph."""
    dataset = EllipticBitcoinDataset(root=str(root))
    return dataset[0]


def build_undirected_graph(data, only_labeled=False) -> nx.Graph:
    """Build an undirected graph for weak connected-component analysis."""
    graph = nx.Graph()
    if not only_labeled:
        graph.add_nodes_from(range(data.num_nodes))
        edge_pairs = zip(data.edge_index[0].tolist(), data.edge_index[1].tolist())
        graph.add_edges_from(edge_pairs)
    else:
        labels = data.y.numpy()
        labeled_nodes = np.flatnonzero(labels != 2)
        graph.add_nodes_from(labeled_nodes)
        labeled_edges = [
            (u, v)
            for u, v in zip(data.edge_index[0].tolist(), data.edge_index[1].tolist())
            if labels[u] != 2 and labels[v] != 2
        ]
        graph.add_edges_from(labeled_edges)
    return graph


def build_directed_graph(data, only_labeled=False) -> nx.DiGraph:
    """Build a directed graph."""
    graph = nx.DiGraph()
    if not only_labeled:
        graph.add_nodes_from(range(data.num_nodes))
        edge_pairs = zip(data.edge_index[0].tolist(), data.edge_index[1].tolist())
        graph.add_edges_from(edge_pairs)
    else:
        labels = data.y.numpy()
        labeled_nodes = np.flatnonzero(labels != 2)
        graph.add_nodes_from(labeled_nodes)
        labeled_edges = [
            (u, v)
            for u, v in zip(data.edge_index[0].tolist(), data.edge_index[1].tolist())
            if labels[u] != 2 and labels[v] != 2
        ]
        graph.add_edges_from(labeled_edges)
    return graph


def label_counts(labels: Iterable[int]) -> dict[str, int]:
    counts = Counter(labels)
    return {LABEL_NAMES[label]: counts.get(label, 0) for label in LABEL_NAMES}


def component_summary(graph: nx.Graph, data, limit: int = 10) -> list[dict[str, int]]:
    rows = []
    labels = data.y.tolist()

    for rank, nodes in enumerate(
        sorted(nx.connected_components(graph), key=len, reverse=True)[:limit],
        start=1,
    ):
        counts = label_counts(labels[node] for node in nodes)
        rows.append(
            {
                "rank": rank,
                "nodes": len(nodes),
                "edges": graph.subgraph(nodes).number_of_edges(),
                **counts,
            }
        )

    return rows


def _seed_for_component(nodes: set[int], labels: list[int]) -> int:
    """Prefer a labelled illicit/licit seed so the sample is informative."""
    for target_label in (1, 0, 2):
        labelled_nodes = [node for node in nodes if labels[node] == target_label]
        if labelled_nodes:
            return min(labelled_nodes)
    return min(nodes)


def connected_sample(
    graph: nx.Graph,
    nodes: set[int],
    labels: list[int],
    max_nodes: int,
) -> list[int]:
    """Return a deterministic connected BFS sample from one component."""
    if len(nodes) <= max_nodes:
        return sorted(nodes)

    seed = _seed_for_component(nodes, labels)
    sampled = []
    seen = {seed}
    queue = [seed]

    while queue and len(sampled) < max_nodes:
        current = queue.pop(0)
        sampled.append(current)
        for neighbor in sorted(graph.neighbors(current)):
            if neighbor in nodes and neighbor not in seen:
                seen.add(neighbor)
                queue.append(neighbor)
            if len(sampled) + len(queue) >= max_nodes:
                break

    return sampled


def plot_component(
    graph: nx.Graph,
    data,
    nodes: list[int],
    output_path: Path,
    title: str,
    layout_seed: int = 305,
) -> Path:
    subgraph = graph.subgraph(nodes).copy()
    labels = data.y.tolist()
    node_colors = [LABEL_COLORS[labels[node]] for node in subgraph.nodes()]

    width = 10
    height = 7
    plt.figure(figsize=(width, height))
    pos = nx.spring_layout(subgraph, seed=layout_seed, k=0.18, iterations=70)
    nx.draw_networkx_edges(subgraph, pos, alpha=0.18, width=0.55, edge_color="#4a4e69")
    nx.draw_networkx_nodes(
        subgraph,
        pos,
        node_color=node_colors,
        node_size=28,
        linewidths=0.2,
        edgecolors="#f8f9fa",
    )

    counts = label_counts(labels[node] for node in subgraph.nodes())
    subtitle = ", ".join(f"{name}: {count}" for name, count in counts.items())
    plt.title(f"{title}\n{subtitle}", fontsize=12)
    plt.axis("off")

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=name,
            markerfacecolor=LABEL_COLORS[label],
            markersize=8,
        )
        for label, name in LABEL_NAMES.items()
    ]
    plt.legend(handles=legend_handles, loc="lower left", frameon=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    return output_path


def plot_connected_components(
    data_root: Path | str = DEFAULT_DATA_ROOT,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    num_components: int = 4,
    max_nodes_per_component: int = 300,
) -> list[Path]:
    data = load_elliptic_data(data_root)
    graph = build_undirected_graph(data)
    labels = data.y.tolist()
    output_dir = Path(output_dir)

    generated_paths = []
    components = sorted(nx.connected_components(graph), key=len, reverse=True)

    for rank, component_nodes in enumerate(components[:num_components], start=1):
        sampled_nodes = connected_sample(
            graph,
            set(component_nodes),
            labels,
            max_nodes=max_nodes_per_component,
        )
        if len(sampled_nodes) == len(component_nodes):
            title = f"Elliptic Bitcoin component {rank} ({len(component_nodes)} nodes)"
        else:
            title = (
                f"Elliptic Bitcoin component {rank} "
                f"({len(sampled_nodes)} of {len(component_nodes)} nodes sampled)"
            )
        output_path = output_dir / f"component_{rank:02d}.png"
        generated_paths.append(plot_component(graph, data, sampled_nodes, output_path, title))

    return generated_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the Elliptic Bitcoin dataset and plot labelled graph components."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-components", type=int, default=4)
    parser.add_argument("--max-nodes-per-component", type=int, default=300)
    args = parser.parse_args()

    data = load_elliptic_data(args.data_root)
    graph = build_undirected_graph(data)

    print(data)
    print(f"Connected components: {nx.number_connected_components(graph)}")
    print("Largest component summaries:")
    for row in component_summary(graph, data, limit=args.num_components):
        print(row)

    generated_paths = plot_connected_components(
        data_root=args.data_root,
        output_dir=args.output_dir,
        num_components=args.num_components,
        max_nodes_per_component=args.max_nodes_per_component,
    )

    print("Generated figures:")
    for path in generated_paths:
        print(path)


if __name__ == "__main__":
    main()
