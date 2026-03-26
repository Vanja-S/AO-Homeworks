import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter


def parse_net(path):
    in_degree = defaultdict(int)
    out_degree = defaultdict(int)
    vertices = set()

    with open(path) as f:
        section = None
        for line in f:
            line = line.strip()
            if line.startswith("*vertices"):
                section = "vertices"
                continue
            elif line.startswith("*arcs"):
                section = "arcs"
                continue

            if section == "vertices":
                vid = int(line.split()[0])
                vertices.add(vid)
                in_degree[vid] = in_degree.get(vid, 0)
                out_degree[vid] = out_degree.get(vid, 0)
            elif section == "arcs":
                parts = line.split()
                u, v = int(parts[0]), int(parts[1])
                out_degree[u] += 1
                in_degree[v] += 1

    return vertices, in_degree, out_degree


def degree_distribution(degree_dict):
    counts = Counter(degree_dict.values())
    n = len(degree_dict)
    ks = sorted(k for k in counts if k > 0)
    pk = {k: counts[k] / n for k in ks}
    return ks, pk


def mle_gamma(degrees, k_min):
    filtered = [k for k in degrees if k >= k_min]
    n = len(filtered)
    if n == 0:
        return None, 0
    gamma = 1 + n * (sum(np.log(k / (k_min - 0.5)) for k in filtered)) ** -1
    return gamma, n


def plot_distributions(total_ks, total_pk, in_ks, in_pk, out_ks, out_pk):
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(
        total_ks,
        [total_pk[k] for k in total_ks],
        c="black",
        s=20,
        label="Total degree $p_k$",
        zorder=3,
    )
    ax.scatter(
        in_ks,
        [in_pk[k] for k in in_ks],
        c="blue",
        s=20,
        label="In-degree $p_{k^{in}}$",
        zorder=3,
    )
    ax.scatter(
        out_ks,
        [out_pk[k] for k in out_ks],
        c="red",
        s=20,
        label="Out-degree $p_{k^{out}}$",
        zorder=3,
    )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Degree $k$")
    ax.set_ylabel("$p_k$")
    ax.set_title("Degree distributions of Lucene class dependency network")
    ax.legend()
    ax.grid(True, which="both", ls="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig("HW1/build/lucene_degree_distributions.pdf")
    plt.savefig("HW1/build/lucene_degree_distributions.png", dpi=150)
    print("Plot saved to HW1/build/lucene_degree_distributions.pdf")


if __name__ == "__main__":
    vertices, in_deg, out_deg = parse_net("HW1/networks/lucene.net")
    total_deg = {v: in_deg[v] + out_deg[v] for v in vertices}

    total_ks, total_pk = degree_distribution(total_deg)
    in_ks, in_pk = degree_distribution(in_deg)
    out_ks, out_pk = degree_distribution(out_deg)

    plot_distributions(total_ks, total_pk, in_ks, in_pk, out_ks, out_pk)

    print("\nMLE power-law exponents")
    k_min = 5
    for name, deg_dict in [("Total", total_deg), ("In", in_deg), ("Out", out_deg)]:
        degrees = [k for k in deg_dict.values() if k > 0]
        gamma, n_used = mle_gamma(degrees, k_min)
        if gamma:
            print(f"{name}-degree: gamma = {gamma:.3f} (k_min = {k_min}, n = {n_used})")
