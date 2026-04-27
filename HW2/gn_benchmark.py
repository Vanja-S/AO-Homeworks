"""Girvan-Newman planted-partition benchmark graph generator.

Each node has expected degree ``k_avg``. A fraction ``1 - mu`` of its expected
degree is allocated to within-group neighbours and a fraction ``mu`` to
between-group neighbours. The within- and between-group edge probabilities are
chosen so that the expected total degree of every node equals ``k_avg``
regardless of ``mu``.
"""

from __future__ import annotations

import numpy as np
import networkx as nx


def gn_benchmark(
    n_groups: int = 3,
    group_size: int = 24,
    k_avg: float = 20.0,
    mu: float = 0.0,
    seed: int | None = None,
) -> tuple[nx.Graph, dict[int, int]]:
    """Generate one realisation of the GN planted-partition benchmark.

    Parameters
    ----------
    n_groups, group_size
        The graph has ``n_groups * group_size`` nodes split into equal groups.
    k_avg
        Expected node degree. Must satisfy
        ``k_avg <= group_size - 1 + group_size * (n_groups - 1)``.
    mu
        Mixing parameter in ``[0, 1]``. ``mu = 0`` -> all edges within groups,
        ``mu = 1`` -> all edges between groups.
    seed
        Seed for reproducibility.

    Returns
    -------
    G
        The generated graph on ``n = n_groups * group_size`` nodes.
    membership
        Mapping ``node -> group index``.
    """
    if not 0.0 <= mu <= 1.0:
        raise ValueError("mu must lie in [0, 1]")

    rng = np.random.default_rng(seed)
    n = n_groups * group_size

    # Within-group / between-group neighbour counts per node.
    n_in = group_size - 1
    n_out = group_size * (n_groups - 1)

    # Edge probabilities chosen so that E[degree] = k_avg for every node.
    p_in = (1.0 - mu) * k_avg / n_in
    p_out = mu * k_avg / n_out
    if not (0.0 <= p_in <= 1.0 and 0.0 <= p_out <= 1.0):
        raise ValueError(f"Edge probabilities out of [0,1]: p_in={p_in}, p_out={p_out}")

    membership = {v: v // group_size for v in range(n)}

    G = nx.Graph()
    G.add_nodes_from(range(n))
    for u in range(n):
        for v in range(u + 1, n):
            same = membership[u] == membership[v]
            p = p_in if same else p_out
            if rng.random() < p:
                G.add_edge(u, v)
    return G, membership


if __name__ == "__main__":
    G, mem = gn_benchmark(mu=0.2, seed=0)
    print(f"n={G.number_of_nodes()}, m={G.number_of_edges()}, "
          f"<k>={2 * G.number_of_edges() / G.number_of_nodes():.2f}")
