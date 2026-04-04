import networkx as nx
import sys


def parse_pajek(path):
    G = nx.Graph()
    labels = {}
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip()]
    i = 0
    # skip to *vertices
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
    # skip to *edges
    while i < len(lines) and not lines[i].lower().startswith("*edges"):
        i += 1
    i += 1
    while i < len(lines) and not lines[i].startswith("*"):
        parts = lines[i].split()
        G.add_edge(int(parts[0]), int(parts[1]))
        i += 1
    return G, labels


G, labels = parse_pajek("networks/dolphins.net")
target = "SN100"
tid = labels[target]

print(f"Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges\n")

# Degree centrality
dc = nx.degree_centrality(G)
dc_sorted = sorted(dc.items(), key=lambda x: -x[1])
rank_dc = next(i for i, (v, _) in enumerate(dc_sorted, 1) if v == tid)

# Betweenness centrality
bc = nx.betweenness_centrality(G)
bc_sorted = sorted(bc.items(), key=lambda x: -x[1])
rank_bc = next(i for i, (v, _) in enumerate(bc_sorted, 1) if v == tid)

# Closeness centrality
cc = nx.closeness_centrality(G)
cc_sorted = sorted(cc.items(), key=lambda x: -x[1])
rank_cc = next(i for i, (v, _) in enumerate(cc_sorted, 1) if v == tid)

# Eigenvector centrality
ec = nx.eigenvector_centrality(G, max_iter=1000)
ec_sorted = sorted(ec.items(), key=lambda x: -x[1])
rank_ec = next(i for i, (v, _) in enumerate(ec_sorted, 1) if v == tid)

print(f"=== {target} (node {tid}) ===")
print(
    f"  Degree:       {dc[tid]:.4f}  (deg={G.degree(tid)})  rank {rank_dc}/{G.number_of_nodes()}"
)
print(f"  Betweenness:  {bc[tid]:.4f}  rank {rank_bc}/{G.number_of_nodes()}")
print(f"  Closeness:    {cc[tid]:.4f}  rank {rank_cc}/{G.number_of_nodes()}")
print(f"  Eigenvector:  {ec[tid]:.4f}  rank {rank_ec}/{G.number_of_nodes()}")

print(f"\nNeighbors of {target}: {[G.nodes[v]['label'] for v in G.neighbors(tid)]}")

print("\n=== Top 10 by each centrality ===")
for name, ranked in [
    ("Degree", dc_sorted),
    ("Betweenness", bc_sorted),
    ("Closeness", cc_sorted),
    ("Eigenvector", ec_sorted),
]:
    print(f"\n{name}:")
    for v, val in ranked[:10]:
        marker = " <--" if v == tid else ""
        print(f"  {G.nodes[v]['label']:15s}  {val:.4f}{marker}")
