from __future__ import annotations

import re
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

NET_PATH = Path(__file__).parent / "networks" / "aps_2008_2013.net"
N_JOURNALS = 10
TRAIN_YEARS = (2008, 2009, 2010, 2011, 2012)
TEST_YEAR = 2013


# I/O


def load_pajek(path: Path = NET_PATH):
    """Parse the Pajek file into adjacency lists and per-node attributes."""
    with open(path) as f:
        n = int(f.readline().split()[1])
        journal = np.zeros(n + 1, dtype=np.int8)
        year = np.zeros(n + 1, dtype=np.int16)
        year_pat = re.compile(r"-(\d{4})\b")
        for _ in range(n):
            parts = f.readline().split()
            vid = int(parts[0])
            label = parts[1].strip('"')
            journal[vid] = int(parts[2])
            m = year_pat.search(label)
            year[vid] = int(m.group(1)) if m else -1
        m_arcs = int(f.readline().split()[1])
        out_adj: list[list[int]] = [[] for _ in range(n + 1)]
        in_adj: list[list[int]] = [[] for _ in range(n + 1)]
        for line in f:
            parts = line.split()
            if not parts:
                continue
            u, v = int(parts[0]), int(parts[1])
            out_adj[u].append(v)
            in_adj[v].append(u)
    und_adj: list[list[int]] = [
        list({u for u in out_adj[v] + in_adj[v]}) for v in range(n + 1)
    ]
    return {
        "n": n,
        "journal": journal,
        "year": year,
        "out_adj": out_adj,
        "in_adj": in_adj,
        "und_adj": und_adj,
        "n_arcs": m_arcs,
    }


# Feature extraction


def citation_profile_features(data) -> np.ndarray:
    """Return an (n+1, 30) matrix of [counts | log1p(counts) | proportions]
    over the 10 journals, computed from each paper's labelled out-neighbours."""
    n = data["n"]
    journal = data["journal"]
    year = data["year"]
    out_adj = data["out_adj"]

    counts = np.zeros((n + 1, N_JOURNALS), dtype=np.float32)
    for v in range(1, n + 1):
        for u in out_adj[v]:
            if 2008 <= year[u] <= 2012 and 1 <= journal[u] <= N_JOURNALS:
                counts[v, journal[u] - 1] += 1.0
    log_counts = np.log1p(counts)
    row_sums = counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    proportions = counts / row_sums
    return np.concatenate([counts, log_counts, proportions], axis=1)


# Baseline


def baseline_majority(data) -> np.ndarray:
    """Most frequent journal among labelled (2008-2012) neighbours.

    Tie-breaking: deterministic (smallest journal id wins) so the baseline is
    fully reproducible. Falls back to the global majority class for nodes
    that have no labelled neighbours."""
    journal = data["journal"]
    year = data["year"]
    und = data["und_adj"]
    n = data["n"]
    pred = np.zeros(n + 1, dtype=np.int8)
    global_majority = int(
        Counter(
            int(journal[v]) for v in range(1, n + 1) if 2008 <= year[v] <= 2012
        ).most_common(1)[0][0]
    )
    for v in range(1, n + 1):
        if year[v] != TEST_YEAR:
            continue
        cnt: Counter = Counter()
        for u in und[v]:
            if 2008 <= year[u] <= 2012:
                cnt[int(journal[u])] += 1
        if not cnt:
            pred[v] = global_majority
        else:
            top = max(cnt.values())
            pred[v] = min(j for j, c in cnt.items() if c == top)
    return pred


# Strategy


def predict_lr(data, X: np.ndarray, random_state: int) -> np.ndarray:
    """Fit a multinomial LR on 2008-2012 nodes and predict 2013 nodes."""
    journal = data["journal"]
    year = data["year"]
    n = data["n"]

    train_idx = np.where((year >= 2008) & (year <= 2012))[0]
    test_idx = np.where(year == TEST_YEAR)[0]

    Xtr = X[train_idx]
    ytr = journal[train_idx]
    has_features = Xtr[:, :N_JOURNALS].sum(axis=1) > 0

    scaler = StandardScaler(with_mean=False).fit(Xtr[has_features])
    clf = LogisticRegression(
        solver="saga",
        max_iter=400,
        C=1.0,
        random_state=random_state,
        n_jobs=-1,
    ).fit(scaler.transform(Xtr[has_features]), ytr[has_features])

    pred = np.zeros(n + 1, dtype=np.int8)
    pred[test_idx] = clf.predict(scaler.transform(X[test_idx]))
    return pred


# Evaluation


def accuracy(pred: np.ndarray, data) -> float:
    journal = data["journal"]
    year = data["year"]
    test_mask = year == TEST_YEAR
    return float(np.mean(pred[test_mask] == journal[test_mask]))


def run_experiment(n_runs: int = 10) -> None:
    print("Loading network ...")
    data = load_pajek()
    n_train = int(((data["year"] >= 2008) & (data["year"] <= 2012)).sum())
    n_test = int((data["year"] == TEST_YEAR).sum())
    print(f"  n_nodes = {data['n']:,}, arcs = {data['n_arcs']:,}")
    print(f"  train (2008-2012) = {n_train:,}, test (2013) = {n_test:,}")

    print("\nBuilding citation-profile features ...")
    X = citation_profile_features(data)
    print(f"  feature matrix: shape={X.shape}, dtype={X.dtype}")

    base_acc = accuracy(baseline_majority(data), data)
    print(f"\nBaseline (most-frequent labelled neighbour): {base_acc:.4f}")

    accs = []
    for r in range(n_runs):
        pred = predict_lr(data, X, random_state=1000 + r)
        a = accuracy(pred, data)
        accs.append(a)
        print(f"  run {r+1:2d}: accuracy = {a:.4f}")

    print()
    print(
        f"Strategy mean accuracy over {n_runs} runs = "
        f"{np.mean(accs):.4f}  (std {np.std(accs):.4f})"
    )
    print(
        f"Improvement over baseline = "
        f"{(np.mean(accs) - base_acc) * 100:+.2f} percentage points"
    )


if __name__ == "__main__":
    run_experiment(n_runs=10)
