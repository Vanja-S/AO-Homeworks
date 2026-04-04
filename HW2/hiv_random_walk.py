import networkx as nx
from networkx import Graph
import random
import numpy as np
from collections import Counter

random.seed(42)


def parse_pajek(path):
    G = nx.Graph()
    labels = {}
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip()]
    i = 0
    while i < len(lines) and not lines[i].lower().startswith("*vertices"):
        i += 1
    n = int(lines[i].split()[1])
    i += 1
    for _ in range(n):
        parts = lines[i].split()
        vid = int(parts[0])
        name = parts[1].strip('"')
        G.add_node(vid, label=name)
        labels[name] = vid
        i += 1
    while i < len(lines) and not lines[i].lower().startswith("*edges"):
        i += 1
    i += 1
    while i < len(lines) and not lines[i].startswith("*"):
        parts = lines[i].split()
        G.add_edge(int(parts[0]), int(parts[1]))
        i += 1
    return G, labels


def random_walk(G: Graph, threshold: float):
    start_node = random.choice(list(G.nodes))
    rw = [start_node]
    visited = {start_node}

    while len(visited) < int(threshold * G.number_of_nodes()):
        next_node = random.choice(list(G.neighbors(rw[-1])))
        rw.append(next_node)
        visited.add(next_node)

    return rw


def analyze_network(G, name):
    print(f"\n=== {name} ===")
    print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")

    avg_deg = 2 * G.number_of_edges() / G.number_of_nodes()
    print(f"Average degree: {avg_deg:.2f}")

    C = nx.average_clustering(G)
    print(f"Clustering coefficient: {C:.4f}")

    # Compare to random graph
    n = G.number_of_nodes()
    m = G.number_of_edges()
    p = 2 * m / (n * (n - 1))
    C_rand = p
    print(f"Expected C for random graph: {C_rand:.4f}")
    print(f"C / C_rand: {C / C_rand:.2f}")

    # Average shortest path length on largest connected component
    gcc = max(nx.connected_components(G), key=len)
    G_gcc = G.subgraph(gcc).copy()
    L = nx.average_shortest_path_length(G_gcc)
    L_rand = np.log(len(gcc)) / np.log(avg_deg) if avg_deg > 1 else float('inf')
    print(f"Avg shortest path (GCC, n={len(gcc)}): {L:.4f}")
    print(f"Expected L for random graph: {L_rand:.4f}")
    print(f"L / L_rand: {L / L_rand:.2f}")

    small_world = C > 10 * C_rand and L / L_rand < 2
    print(f"Small-world: C >> C_rand and L ~ L_rand => {small_world}")

    # Degree distribution for scale-free check
    degrees = [d for _, d in G.degree()]
    deg_count = Counter(degrees)
    ks = sorted(deg_count.keys())
    # Log-log linear regression on CCDF
    ccdf = []
    total = len(degrees)
    for k in ks:
        ccdf.append(sum(1 for d in degrees if d >= k) / total)
    log_k = np.log(np.array(ks, dtype=float))
    log_ccdf = np.log(np.array(ccdf))
    coeffs = np.polyfit(log_k, log_ccdf, 1)
    gamma_est = -coeffs[0]
    residuals = log_ccdf - np.polyval(coeffs, log_k)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((log_ccdf - np.mean(log_ccdf))**2)
    r_squared = 1 - ss_res / ss_tot
    print(f"Power-law fit: gamma ~ {gamma_est:.2f}, R^2 = {r_squared:.4f}")
    print(f"Max degree: {max(degrees)}, Min degree: {min(degrees)}")

    return {
        "n": n, "m": m, "avg_deg": avg_deg,
        "C": C, "C_rand": C_rand,
        "L": L, "L_rand": L_rand,
        "gamma": gamma_est, "r2": r_squared,
    }


G, labels = parse_pajek("networks/social.net")
rw = random_walk(G, 0.15)
visited = set(rw)
G_sampled = G.subgraph(visited).copy()

orig = analyze_network(G, "Original social network")
samp = analyze_network(G_sampled, "Sampled network (15% RW)")
