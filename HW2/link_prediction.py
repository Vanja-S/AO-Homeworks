from __future__ import annotations

import math
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable

import numpy as np
import networkx as nx

import igraph as ig
import leidenalg

NET_DIR = Path(__file__).parent / "networks"
N_RUNS = 10
TEST_FRAC = 0.1
N_AUC_SAMPLES = 10_000


# Loading


def load_pajek_undirected(path: Path) -> tuple[int, list[tuple[int, int]]]:
    """Parse a Pajek file and return (n, undirected edge list, 0-indexed)."""
    with open(path) as f:
        header = f.readline().split()
        n = int(header[1])
        for _ in range(n):
            f.readline()
        f.readline()  # *edges <m> or *arcs <m>
        edges: set[tuple[int, int]] = set()
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            u = int(parts[0]) - 1
            v = int(parts[1]) - 1
            if u == v:
                continue
            edges.add((min(u, v), max(u, v)))
    return n, sorted(edges)


def make_er(n: int, k_avg: float, seed: int) -> tuple[int, list[tuple[int, int]]]:
    p = k_avg / (n - 1)
    G = nx.fast_gnp_random_graph(n, p, seed=seed)
    return n, [(min(u, v), max(u, v)) for u, v in G.edges()]


# Helpers


def split_edges(edges, frac, rng):
    e = list(edges)
    rng.shuffle(e)
    k = int(round(len(e) * frac))
    return e[k:], e[:k]


def neighbours_and_degree(n, train_edges):
    nbrs: list[set[int]] = [set() for _ in range(n)]
    for u, v in train_edges:
        nbrs[u].add(v)
        nbrs[v].add(u)
    deg = np.fromiter((len(s) for s in nbrs), dtype=np.int32, count=n)
    return nbrs, deg


def run_leiden(n, train_edges, seed):
    g = ig.Graph(n=n, edges=train_edges, directed=False)
    g = g.simplify()
    part = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition, seed=seed)
    return np.asarray(part.membership, dtype=np.int32)


def community_stats(n, train_edges, membership):
    """Return (M, sizes) where M[(a,b)] is # train edges between community a and b
    (a <= b), and sizes[c] = number of nodes in community c."""
    sizes: dict[int, int] = defaultdict(int)
    for v in range(n):
        sizes[int(membership[v])] += 1
    M: dict[tuple[int, int], int] = defaultdict(int)
    for u, v in train_edges:
        a, b = int(membership[u]), int(membership[v])
        M[(min(a, b), max(a, b))] += 1
    return M, dict(sizes)


# Scores


def score_pa(deg):
    def f(u, v):
        return float(deg[u]) * float(deg[v])

    return f


def score_aa(nbrs, deg):
    log_term = np.zeros(len(deg), dtype=np.float64)
    for x, k in enumerate(deg):
        if k > 1:
            log_term[x] = 1.0 / math.log(float(k))

    def f(u, v):
        if len(nbrs[u]) > len(nbrs[v]):
            u, v = v, u
        s = 0.0
        nv = nbrs[v]
        for x in nbrs[u]:
            if x in nv:
                s += log_term[x]
        return s

    return f


def score_community(membership, M, sizes):
    def f(u, v):
        a = int(membership[u])
        b = int(membership[v])
        if a == b:
            n_a = sizes[a]
            denom = n_a * (n_a - 1) / 2.0
            return M.get((a, a), 0) / denom if denom > 0 else 0.0
        denom = sizes[a] * sizes[b]
        return M.get((min(a, b), max(a, b)), 0) / denom if denom > 0 else 0.0

    return f


# AUC by sampling


def sample_auc(score_fn, test_edges, full_edge_set, n, n_samples, rng):
    """Mann-Whitney AUC: draw n_samples (e_pos, e_neg) pairs."""
    test_arr = list(test_edges)
    hits = 0
    ties = 0
    for _ in range(n_samples):
        u_pos, v_pos = test_arr[rng.randrange(len(test_arr))]
        while True:
            u = rng.randrange(n)
            v = rng.randrange(n)
            if u == v:
                continue
            key = (u, v) if u < v else (v, u)
            if key in full_edge_set:
                continue
            break
        s_pos = score_fn(u_pos, v_pos)
        s_neg = score_fn(u, v)
        if s_pos > s_neg:
            hits += 1
        elif s_pos == s_neg:
            ties += 1
    return (hits + 0.5 * ties) / n_samples


# Per-network experiment


def run_network(name: str, loader: Callable, n_runs: int = N_RUNS) -> dict:
    print(f"\n=== {name} ===")
    t0 = time.time()
    n, edges = loader(seed=42)
    print(f"  n = {n:,}, m = {len(edges):,}  ({time.time()-t0:.1f}s)")
    full_edge_set = set(edges)
    results = {"PA": [], "AA": [], "Comm": []}

    for r in range(n_runs):
        run_t0 = time.time()
        rng = random.Random(7919 * (r + 1))
        # If ER, regenerate the graph for this run.
        if name == "ER":
            _, edges_run = loader(seed=10_000 + r)
            full_edge_set_run = set(edges_run)
        else:
            edges_run = edges
            full_edge_set_run = full_edge_set

        train, test = split_edges(edges_run, TEST_FRAC, rng)
        nbrs, deg = neighbours_and_degree(n, train)
        membership = run_leiden(n, train, seed=r + 1)
        M, sizes = community_stats(n, train, membership)

        f_pa = score_pa(deg)
        f_aa = score_aa(nbrs, deg)
        f_co = score_community(membership, M, sizes)

        rng_pa = random.Random(rng.randrange(1 << 30))
        rng_aa = random.Random(rng.randrange(1 << 30))
        rng_co = random.Random(rng.randrange(1 << 30))

        auc_pa = sample_auc(f_pa, test, full_edge_set_run, n, N_AUC_SAMPLES, rng_pa)
        auc_aa = sample_auc(f_aa, test, full_edge_set_run, n, N_AUC_SAMPLES, rng_aa)
        auc_co = sample_auc(f_co, test, full_edge_set_run, n, N_AUC_SAMPLES, rng_co)

        results["PA"].append(auc_pa)
        results["AA"].append(auc_aa)
        results["Comm"].append(auc_co)
        n_comm = len(set(membership.tolist()))
        print(
            f"  run {r+1:2d}: PA={auc_pa:.3f}  AA={auc_aa:.3f}  Comm={auc_co:.3f}  "
            f"(n_comm={n_comm}, {time.time()-run_t0:.1f}s)"
        )

    return results


# Main


def loader_er(seed):
    return make_er(25_000, 10, seed)


def loader_pajek(filename):
    def f(seed=None):
        return load_pajek_undirected(NET_DIR / filename)

    return f


def main():
    networks = [
        ("ER", loader_er),
        ("Circles", loader_pajek("circles.net")),
        ("Gnutella", loader_pajek("gnutella.net")),
        ("nec", loader_pajek("nec.net")),
    ]

    all_results: dict[str, dict[str, list[float]]] = {}
    for name, loader in networks:
        all_results[name] = run_network(name, loader)

    # Summary
    print("\n" + "=" * 72)
    print(f"{'Network':<10} {'PA mean':>12}  {'AA mean':>12}  {'Comm mean':>12}")
    print("-" * 72)
    for name, res in all_results.items():
        line = f"{name:<10}"
        for m in ("PA", "AA", "Comm"):
            mu = float(np.mean(res[m]))
            sd = float(np.std(res[m]))
            line += f"  {mu:.3f}+/-{sd:.3f}"
        print(line)
    print("=" * 72)

    out_path = Path(__file__).parent / "plots" / "link_prediction_results.txt"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as fh:
        fh.write("network\tmethod\tmean\tstd\truns\n")
        for name, res in all_results.items():
            for m, vals in res.items():
                fh.write(
                    f"{name}\t{m}\t{np.mean(vals):.4f}\t{np.std(vals):.4f}\t"
                    f"{','.join(f'{v:.4f}' for v in vals)}\n"
                )
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
