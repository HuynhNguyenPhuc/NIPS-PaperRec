"""Graph."""

from __future__ import annotations

import networkx as nx

from paperrec.config import LOUVAIN_SEED


def build_term_graph(postings: dict[str, list[int]], n: int) -> nx.Graph:
    """
    Build the weighted graph where two papers are linked by the terms they share.

    Args:
        postings: Kept term to the catalogue positions that contain it.
        n: Number of catalogue papers; every position becomes a node.

    Returns:
        graph: Undirected graph; the edge weight counts the shared kept terms.
    """
    g = nx.Graph()

    # Nodes are added first so that papers with no shared term are still present.
    g.add_nodes_from(range(n))

    for members in postings.values():
        # Each pair of papers that contains this term adds one unit of weight to their edge.
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                u, v = members[a], members[b]
                if g.has_edge(u, v):
                    g[u][v]["weight"] += 1
                else:
                    g.add_edge(u, v, weight=1)

    return g


def louvain_communities(graph: nx.Graph) -> list[set[int]]:
    """
    Partition the term graph with the Louvain method.

    Args:
        graph: Weighted graph from build_term_graph.

    Returns:
        communities: Disjoint sets of node positions, one per community.
    """
    # The fixed seed makes the partition, and so the community-restricted ranking, reproducible.
    return nx.community.louvain_communities(graph, weight="weight", seed=LOUVAIN_SEED)
