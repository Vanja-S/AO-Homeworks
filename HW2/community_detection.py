"""Run the three community-detection experiments for HW2 Section 4.

Algorithms compared:
    * Leiden (modularity, via :mod:`leidenalg`)
    * FLPA (fast label propagation, via NetworkX)
    * Infomap (map equation, via :mod:`igraph`)

Experiments:
    (i)   Girvan-Newman planted partition  -> NMI vs mu
    (ii)  LFR benchmark (n = 2500)         -> NMI vs mu
    (iii) Erdos-Renyi (no community)       -> NVI vs <k>
"""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from networkx.algorithms.community.label_propagation import (
    fast_label_propagation_communities,
)
from networkx.generators.community import LFR_benchmark_graph
from sklearn.metrics import normalized_mutual_info_score

import igraph as ig
import leidenalg

from gn_benchmark import gn_benchmark


# Algorithms.  Each takes a NetworkX graph and returns an integer label per
# node, indexed by the node id (which is always 0..n-1 in our experiments).


def _nx_to_igraph(G: nx.Graph) -> ig.Graph:
    n = G.number_of_nodes()
    edges = [(u, v) for u, v in G.edges() if u != v]
    g = ig.Graph(n=n, edges=edges, directed=False)
    return g


def _membership_to_labels(membership, n: int) -> np.ndarray:
    """Convert an iterable of clusters (each a collection of node ids) or a
    flat per-vertex membership list to a numpy array of shape (n,)."""
    labels = np.full(n, -1, dtype=int)
    if hasattr(membership, "__iter__") and not isinstance(membership, (list, tuple)):
        membership = list(membership)
    # flat case: list/array with one entry per vertex of integer type
    if (
        isinstance(membership, (list, tuple, np.ndarray))
        and len(membership) == n
        and np.isscalar(membership[0])
    ):
        return np.asarray(membership, dtype=int)
    # cluster-of-nodes case
    for c_idx, cluster in enumerate(membership):
        for v in cluster:
            labels[int(v)] = c_idx
    if (labels == -1).any():
        # singleton fallback
        for i in np.where(labels == -1)[0]:
            labels[i] = labels.max() + 1
    return labels


def run_leiden(G: nx.Graph, seed: int | None = None) -> np.ndarray:
    g = _nx_to_igraph(G)
    part = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition, seed=seed)
    return np.asarray(part.membership, dtype=int)


def run_flpa(G: nx.Graph, seed: int | None = None) -> np.ndarray:
    communities = fast_label_propagation_communities(G, seed=seed)
    return _membership_to_labels(list(communities), G.number_of_nodes())


def run_infomap(G: nx.Graph, seed: int | None = None) -> np.ndarray:
    import random as _random

    g = _nx_to_igraph(G)
    if seed is not None:
        ig.set_random_number_generator(_random.Random(seed))
    part = g.community_infomap()
    return np.asarray(part.membership, dtype=int)


ALGORITHMS = {
    "Leiden": run_leiden,
    "FLPA": run_flpa,
    "Infomap": run_infomap,
}


# Metrics


def nmi(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """Normalised mutual information (arithmetic mean normalisation)."""
    return float(
        normalized_mutual_info_score(
            labels_true, labels_pred, average_method="arithmetic"
        )
    )


def nvi(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    """Normalised variation of information, NVI = VI / log(n) in [0, 1]."""
    n = len(labels_a)
    if n != len(labels_b):
        raise ValueError("labels must have the same length")

    joint = Counter(zip(labels_a.tolist(), labels_b.tolist()))
    px = Counter(labels_a.tolist())
    py = Counter(labels_b.tolist())

    def H(counts):
        return -sum((c / n) * math.log(c / n) for c in counts.values() if c > 0)

    Hx, Hy = H(px), H(py)
    Ixy = 0.0
    for (x, y), c in joint.items():
        pxy = c / n
        Ixy += pxy * math.log(pxy / ((px[x] / n) * (py[y] / n)))
    vi = Hx + Hy - 2.0 * Ixy
    return float(vi / math.log(n))


# Helpers


def cc_labels(G: nx.Graph) -> np.ndarray:
    """Return a label per node giving its connected-component index."""
    labels = np.zeros(G.number_of_nodes(), dtype=int)
    for c_idx, comp in enumerate(nx.connected_components(G)):
        for v in comp:
            labels[v] = c_idx
    return labels


def make_lfr(n: int, mu: float, seed: int) -> nx.Graph:
    """Generate an LFR benchmark graph; retries a few seeds on failure."""
    last_err = None
    for attempt in range(20):
        try:
            G = LFR_benchmark_graph(
                n=n,
                tau1=2.5,
                tau2=1.5,
                mu=max(mu, 1e-4),  # mu = 0 fails inside networkx LFR
                average_degree=20,
                max_degree=50,
                min_community=20,
                max_community=100,
                seed=seed + attempt * 1000,
            )
            return G
        except (
            nx.ExceededMaxIterations,
            RuntimeError,
            Exception,
        ) as err:  # noqa: BLE001
            last_err = err
    raise RuntimeError(f"LFR generation failed for mu={mu}: {last_err}")


def lfr_planted_labels(G: nx.Graph) -> np.ndarray:
    """Extract the planted partition stored on LFR graph nodes."""
    n = G.number_of_nodes()
    labels = np.full(n, -1, dtype=int)
    comm_id: dict[frozenset, int] = {}
    for v in G.nodes():
        c = frozenset(G.nodes[v]["community"])
        if c not in comm_id:
            comm_id[c] = len(comm_id)
        labels[v] = comm_id[c]
    return labels


# Experiments

N_REPS = 25
OUT_DIR = Path(__file__).parent / "networks"
OUT_DIR.mkdir(exist_ok=True)


def experiment_gn() -> dict[str, list[float]]:
    mus = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    results: dict[str, list[float]] = {a: [] for a in ALGORITHMS}
    for mu in mus:
        per_algo = {a: [] for a in ALGORITHMS}
        for r in range(N_REPS):
            seed = 1000 * int(round(mu * 10)) + r
            G, membership = gn_benchmark(mu=mu, seed=seed)
            true_labels = np.asarray(
                [membership[v] for v in range(G.number_of_nodes())]
            )
            for name, algo in ALGORITHMS.items():
                pred = algo(G, seed=seed)
                per_algo[name].append(nmi(true_labels, pred))
        for name in ALGORITHMS:
            results[name].append(float(np.mean(per_algo[name])))
            print(f"[GN]  mu={mu:.1f}  {name:7s}  NMI={results[name][-1]:.3f}")
    return mus, results


def experiment_lfr() -> dict[str, list[float]]:
    mus = [0.0, 0.2, 0.4, 0.6, 0.8]
    results: dict[str, list[float]] = {a: [] for a in ALGORITHMS}
    for mu in mus:
        per_algo = {a: [] for a in ALGORITHMS}
        for r in range(N_REPS):
            seed = 2000 * int(round(mu * 10)) + r
            G = make_lfr(n=2500, mu=mu, seed=seed)
            true_labels = lfr_planted_labels(G)
            G_int = nx.convert_node_labels_to_integers(G, ordering="sorted")
            for name, algo in ALGORITHMS.items():
                pred = algo(G_int, seed=seed)
                per_algo[name].append(nmi(true_labels, pred))
        for name in ALGORITHMS:
            results[name].append(float(np.mean(per_algo[name])))
            print(f"[LFR] mu={mu:.1f}  {name:7s}  NMI={results[name][-1]:.3f}")
    return mus, results


def experiment_er() -> dict[str, list[float]]:
    avg_ks = [8, 16, 24, 32, 40]
    n = 1000
    results: dict[str, list[float]] = {a: [] for a in ALGORITHMS}
    for k_avg in avg_ks:
        p = k_avg / (n - 1)
        per_algo = {a: [] for a in ALGORITHMS}
        for r in range(N_REPS):
            seed = 100 * k_avg + r
            G = nx.fast_gnp_random_graph(n=n, p=p, seed=seed)
            cc = cc_labels(G)
            for name, algo in ALGORITHMS.items():
                pred = algo(G, seed=seed)
                per_algo[name].append(nvi(cc, pred))
        for name in ALGORITHMS:
            results[name].append(float(np.mean(per_algo[name])))
            print(f"[ER]  <k>={k_avg:>2d}  {name:7s}  NVI={results[name][-1]:.3f}")
    return avg_ks, results


# Plotting

PLOT_DIR = Path(__file__).parent / "plots"
PLOT_DIR.mkdir(exist_ok=True)


def plot_results(xs, results, xlabel: str, ylabel: str, title: str, fname: str) -> None:
    plt.figure(figsize=(6, 4))
    markers = {"Leiden": "o", "FLPA": "s", "Infomap": "^"}
    for name, ys in results.items():
        plt.plot(xs, ys, marker=markers.get(name, "o"), label=name, linewidth=1.6)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_DIR / fname, dpi=150)
    plt.close()


def main() -> None:
    print("=== (i) Girvan-Newman benchmark ===")
    gn_mus, gn_res = experiment_gn()
    plot_results(
        gn_mus,
        gn_res,
        xlabel=r"$\mu$",
        ylabel="NMI",
        title="GN benchmark (n=72, $\\langle k \\rangle$=20)",
        fname="gn_nmi.pdf",
    )

    print("\n=== (ii) LFR benchmark ===")
    lfr_mus, lfr_res = experiment_lfr()
    plot_results(
        lfr_mus,
        lfr_res,
        xlabel=r"$\mu$",
        ylabel="NMI",
        title="LFR benchmark (n=2500)",
        fname="lfr_nmi.pdf",
    )

    print("\n=== (iii) Erdos-Renyi robustness ===")
    er_ks, er_res = experiment_er()
    plot_results(
        er_ks,
        er_res,
        xlabel=r"$\langle k \rangle$",
        ylabel="NVI",
        title=r"Erd\H{o}s-R\'enyi robustness (n=1000)",
        fname="er_nvi.pdf",
    )

    print("\nWrote plots to", PLOT_DIR)
    # Also dump results to a small text file for reference.
    with open(PLOT_DIR / "results.txt", "w") as fh:
        fh.write("# GN benchmark NMI\n")
        fh.write("mu\t" + "\t".join(gn_res) + "\n")
        for i, mu in enumerate(gn_mus):
            fh.write(
                f"{mu}\t" + "\t".join(f"{gn_res[a][i]:.4f}" for a in gn_res) + "\n"
            )
        fh.write("\n# LFR benchmark NMI\n")
        fh.write("mu\t" + "\t".join(lfr_res) + "\n")
        for i, mu in enumerate(lfr_mus):
            fh.write(
                f"{mu}\t" + "\t".join(f"{lfr_res[a][i]:.4f}" for a in lfr_res) + "\n"
            )
        fh.write("\n# ER NVI\n")
        fh.write("k_avg\t" + "\t".join(er_res) + "\n")
        for i, k in enumerate(er_ks):
            fh.write(f"{k}\t" + "\t".join(f"{er_res[a][i]:.4f}" for a in er_res) + "\n")


if __name__ == "__main__":
    main()
