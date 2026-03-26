import random
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter


def parse_adj(path):
    adj = defaultdict(set)
    directed_edges = set()
    n_header = 0
    m_header = 0
    line_count = 0

    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#"):
                if "nodes" in line and "edges" in line:
                    parts = line.split()
                    for i, p in enumerate(parts):
                        if p == "nodes":
                            n_header = int(parts[i - 1].replace(",", ""))
                        if p == "edges":
                            m_header = int(parts[i - 1].replace(",", ""))
                continue
            if not line:
                continue
            parts = line.split()
            u, v = int(parts[0]), int(parts[1])
            adj[u].add(v)
            adj[v].add(u)
            directed_edges.add((u, v))
            line_count += 1

    vertices = set(adj.keys())
    n = len(vertices)
    degrees = [len(adj[v]) for v in vertices]
    m_undirected = sum(degrees) // 2

    # directed if line count > undirected edge count
    is_directed = line_count > m_undirected * 1.01
    reciprocity = (line_count - m_undirected) / line_count if line_count > 0 else 0

    return {
        "n": n,
        "m_undirected": m_undirected,
        "m_header": m_header,
        "line_count": line_count,
        "degrees": degrees,
        "is_directed": is_directed,
        "reciprocity": reciprocity,
        "adj": adj,
    }


def compute_stats(info):
    degrees = info["degrees"]
    n = info["n"]
    m = info["m_undirected"]

    avg_deg = np.mean(degrees)
    max_deg = np.max(degrees)
    med_deg = np.median(degrees)
    density = 2 * m / (n * (n - 1))

    deg_counts = Counter(degrees)
    ks = sorted(k for k in deg_counts if k > 0)
    pk = {k: deg_counts[k] / n for k in ks}

    # clustering coefficient (random sample)
    adj = info["adj"]
    all_nodes = list(adj.keys())
    random.seed(42)
    sample_nodes = random.sample(all_nodes, min(5000, len(all_nodes)))
    cc_values = []
    for v in sample_nodes:
        neighbors = list(adj[v])
        k = len(neighbors)
        if k < 2:
            continue
        if k > 500:
            # sample pairs for high-degree nodes
            pairs = 0
            triangles = 0
            sampled_pairs = min(5000, k * (k - 1) // 2)
            for _ in range(sampled_pairs):
                a, b = random.sample(neighbors, 2)
                pairs += 1
                if b in adj[a]:
                    triangles += 1
            cc_values.append(triangles / pairs if pairs > 0 else 0)
        else:
            triangles = 0
            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    if neighbors[j] in adj[neighbors[i]]:
                        triangles += 1
            cc_values.append(2 * triangles / (k * (k - 1)))
    avg_cc = np.mean(cc_values) if cc_values else 0

    return {
        "avg_deg": avg_deg,
        "max_deg": max_deg,
        "med_deg": med_deg,
        "density": density,
        "avg_cc": avg_cc,
        "ks": ks,
        "pk": pk,
    }


if __name__ == "__main__":
    networks = {}
    for i in range(1, 6):
        path = f"HW1/networks/network_{i}.adj"
        print(f"Parsing network_{i}...")
        info = parse_adj(path)
        stats = compute_stats(info)
        networks[i] = {**info, **stats}

    print(f"\n{'Net':>4} {'n':>10} {'lines':>12} {'m_undir':>12} {'<k>':>8} "
          f"{'max_k':>8} {'med_k':>8} {'density':>12} {'avg_cc':>8} {'directed':>10} {'recip':>8}")
    print("-" * 115)
    for i in range(1, 6):
        net = networks[i]
        print(f"{i:>4} {net['n']:>10,} {net['line_count']:>12,} {net['m_undirected']:>12,} "
              f"{net['avg_deg']:>8.2f} {net['max_deg']:>8} {net['med_deg']:>8.1f} "
              f"{net['density']:>12.2e} {net['avg_cc']:>8.4f} {str(net['is_directed']):>10} "
              f"{net['reciprocity']:>8.3f}")

    # degree distribution plots
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    for i in range(1, 6):
        ax = axes[i - 1]
        net = networks[i]
        ax.scatter(net["ks"], [net["pk"][k] for k in net["ks"]], s=5, c="black")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"network_{i}")
        ax.set_xlabel("$k$")
        ax.set_ylabel("$p_k$")
    plt.tight_layout()
    plt.savefig("HW1/build/five_networks_degree_dist.pdf")
    plt.savefig("HW1/build/five_networks_degree_dist.png", dpi=150)
    print("\nPlot saved to HW1/build/five_networks_degree_dist.pdf")
