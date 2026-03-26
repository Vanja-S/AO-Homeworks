from collections import defaultdict


def parse_net(path):
    graph = defaultdict(set)
    reverse_graph = defaultdict(set)
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
                parts = line.split()
                vertices.add(int(parts[0]))
            elif section == "arcs":
                parts = line.split()
                u, v = int(parts[0]), int(parts[1])
                graph[u].add(v)
                reverse_graph[v].add(u)
                vertices.add(u)
                vertices.add(v)

    return vertices, graph, reverse_graph


def dfs(start, adj, allowed):
    visited = set()
    stack = [start]
    while stack:
        v = stack.pop()
        if v in visited or v not in allowed:
            continue
        visited.add(v)
        for neighbor in adj[v]:
            if neighbor not in visited and neighbor in allowed:
                stack.append(neighbor)
    return visited


def find_sccs(vertices, graph, reverse_graph):
    unvisited = set(vertices)
    sccs = []
    while unvisited:
        v = next(iter(unvisited))
        forward = dfs(v, graph, unvisited)
        backward = dfs(v, reverse_graph, unvisited)
        scc = forward & backward
        sccs.append(scc)
        unvisited -= scc
    return sccs


if __name__ == "__main__":
    vertices, graph, reverse_graph = parse_net("HW1/networks/enron.net")
    sccs = find_sccs(vertices, graph, reverse_graph)

    print(f"Number of SCCs: {len(sccs)}")
    largest = max(sccs, key=len)
    print(f"Largest SCC size: {len(largest)}")
