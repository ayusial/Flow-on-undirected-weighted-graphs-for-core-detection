import argparse
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd


# ============================================================
# Graph loading
# ============================================================

def load_graph(path: str) -> nx.Graph:
    p = Path(path)

    if p.suffix.lower() == ".gexf":
        G = nx.read_gexf(p)

    else:
        # Generic whitespace-separated edge list.
        # Only the first two columns are required here because all
        # centrality baselines are evaluated on the unweighted topology.
        G = nx.Graph()

        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                if not line or line.startswith("#"):
                    continue

                parts = line.split()

                if len(parts) < 2:
                    continue

                u, v = parts[0], parts[1]

                if u != v:
                    G.add_edge(u, v)

    # Work with a simple undirected graph.
    G = nx.Graph(G)
    G.remove_edges_from(nx.selfloop_edges(G))

    components = list(nx.connected_components(G))
    largest = max((len(c) for c in components), default=0)

    print(
        f"Graph: n={G.number_of_nodes()}, "
        f"m={G.number_of_edges()}, "
        f"components={len(components)}, "
        f"largest_component={largest}"
    )

    return G


# ============================================================
# Centrality
# ============================================================

def centrality_scores(G: nx.Graph, method: str):
    method = method.lower()

    if method == "pagerank":
        return nx.pagerank(G, alpha=0.85, weight=None)

    if method == "degree":
        return nx.degree_centrality(G)

    if method == "betweenness":
        return nx.betweenness_centrality(G, weight=None)

    if method == "closeness":
        return nx.closeness_centrality(G)

    raise ValueError(f"Unknown centrality method: {method}")


def ranked_nodes(G: nx.Graph, method: str):
    """
    Follow the baseline logic of the reference implementation:
    sort nodes only by the specified centrality score.
    """
    score = centrality_scores(G, method)

    ranked = [
        node
        for node, _
        in sorted(
            score.items(),
            key=lambda x: x[1],
            reverse=True,
        )
    ]

    return ranked


# ============================================================
# top-k -> induced graph -> largest connected component
# ============================================================

def lcc_from_top_k(
    G: nx.Graph,
    ranked,
    k: int,
) -> nx.Graph:

    k = min(max(int(k), 1), len(ranked))

    selected = ranked[:k]
    H = G.subgraph(selected).copy()

    if H.number_of_nodes() == 0:
        return H

    largest_cc = max(
        nx.connected_components(H),
        key=len,
    )

    return H.subgraph(largest_cc).copy()


def find_best_k(
    G: nx.Graph,
    ranked,
    target_size: int,
):
    """
    Find a top-k cutoff such that the LCC of the induced top-k
    subgraph is as close as possible to target_size.

    Because the top-k vertex sets are nested, the LCC size is
    non-decreasing as k increases.  We therefore locate the first k
    whose LCC reaches/exceeds target_size by binary search and then
    compare it with the immediately preceding value.
    """

    n = len(ranked)
    target_size = int(target_size)

    if target_size <= 0:
        raise ValueError("target_size must be positive.")

    largest_component_size = max(
        (len(c) for c in nx.connected_components(G)),
        default=0,
    )

    if target_size > largest_component_size:
        raise ValueError(
            f"Target size {target_size} cannot be achieved by a connected "
            f"baseline core; largest graph component has only "
            f"{largest_component_size} vertices."
        )

    # k cannot be smaller than target_size if we want an LCC of target_size.
    lo = target_size
    hi = n

    # If even top-n does not reach target, something is inconsistent.
    H_hi = lcc_from_top_k(G, ranked, hi)

    if H_hi.number_of_nodes() < target_size:
        raise ValueError(
            f"Unable to reach target size {target_size}; "
            f"maximum LCC obtained is {H_hi.number_of_nodes()}."
        )

    # First k with LCC size >= target_size.
    while lo < hi:
        mid = (lo + hi) // 2

        H_mid = lcc_from_top_k(G, ranked, mid)
        size_mid = H_mid.number_of_nodes()

        if size_mid >= target_size:
            hi = mid
        else:
            lo = mid + 1

    k_hi = lo
    H_hi = lcc_from_top_k(G, ranked, k_hi)

    candidates = [(k_hi, H_hi)]

    if k_hi > target_size:
        k_lo = k_hi - 1
        H_lo = lcc_from_top_k(G, ranked, k_lo)
        candidates.append((k_lo, H_lo))

    # Choose the closest final LCC size.
    # If tied, prefer the smaller top-k cutoff.
    best_k, best_core = min(
        candidates,
        key=lambda x: (
            abs(x[1].number_of_nodes() - target_size),
            x[0],
        ),
    )

    return best_k, best_core


# ============================================================
# Metrics
# ============================================================

def compute_rd(
    G: nx.Graph,
    core: nx.Graph,
) -> float:

    if core.number_of_nodes() == 0:
        return float("nan")

    vals = []

    for v in core.nodes():
        d_full = G.degree(v)

        if d_full > 0:
            vals.append(core.degree(v) / d_full)
        else:
            vals.append(0.0)

    return float(np.mean(vals))


def compute_rs(
    G: nx.Graph,
    core_nodes,
    dist_G,
) -> float:

    residual_nodes = set(G.nodes()) - set(core_nodes)

    if len(residual_nodes) < 2:
        return float("nan")

    G_res = G.subgraph(residual_nodes).copy()

    dist_res = dict(
        nx.all_pairs_shortest_path_length(G_res)
    )

    nodes = list(residual_nodes)

    total = 0.0
    count = 0

    for i, u in enumerate(nodes):
        du_res = dist_res.get(u, {})
        du_full = dist_G.get(u, {})

        for v in nodes[i + 1:]:

            # Same convention as our previous experiments:
            # only pairs connected both originally and after removal
            # contribute to r_s.
            if (
                v in du_res
                and v in du_full
                and du_full[v] > 0
            ):
                total += du_res[v] / du_full[v]
                count += 1

    if count == 0:
        return float("nan")

    return total / count


# ============================================================
# Experiment
# ============================================================

def parse_targets(text: str):
    """
    Example:
        GEF:1190,LEF:1157,CF:1245
    """

    targets = []

    for item in text.split(","):
        item = item.strip()

        if not item:
            continue

        label, value = item.split(":", 1)

        targets.append(
            (label.strip(), int(value))
        )

    return targets


def run_baselines(
    G: nx.Graph,
    targets,
):

    methods = [
        "pagerank",
        "degree",
        "betweenness",
        "closeness",
    ]

    print("\nComputing original-graph shortest paths...")
    dist_G = dict(
        nx.all_pairs_shortest_path_length(G)
    )

    print("Shortest paths ready.\n")

    # Compute each centrality only once per dataset.
    rankings = {}

    for method in methods:
        print(f"[centrality] computing {method}")
        rankings[method] = ranked_nodes(G, method)

    rows = []

    for reference, target_size in targets:

        print(
            f"\n============================================\n"
            f"Reference: {reference}, target K={target_size}\n"
            f"============================================"
        )

        for method in methods:

            ranked = rankings[method]

            best_k, core = find_best_k(
                G,
                ranked,
                target_size,
            )

            actual_size = core.number_of_nodes()

            rd = compute_rd(G, core)
            rs = compute_rs(
                G,
                set(core.nodes()),
                dist_G,
            )

            row = {
                "reference": reference,
                "method": method,
                "target_size": target_size,
                "top_k_input": best_k,
                "core_nodes": actual_size,
                "size_gap": actual_size - target_size,
                "core_edges": core.number_of_edges(),
                "r_d": rd,
                "r_s": rs,
            }

            rows.append(row)

            print(
                f"[baseline] {reference} | {method:12s} | "
                f"target={target_size} | "
                f"top_k={best_k} | "
                f"actual={actual_size} | "
                f"gap={actual_size - target_size:+d} | "
                f"edges={core.number_of_edges()} | "
                f"r_d={rd:.6f} | "
                f"r_s={rs:.6f}"
            )

    return pd.DataFrame(rows)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--graph",
        required=True,
        help="Input graph (.gexf or whitespace-separated edge list).",
    )

    parser.add_argument(
        "--targets",
        required=True,
        help=(
            "Comma-separated reference:K pairs, e.g. "
            "'GEF:1190,LEF:1157,CF:1245'."
        ),
    )

    parser.add_argument(
        "--out",
        required=True,
        help="Output CSV file.",
    )

    args = parser.parse_args()

    G = load_graph(args.graph)
    targets = parse_targets(args.targets)

    df = run_baselines(
        G,
        targets,
    )

    print("\nFinal results:")
    print(df.to_string(index=False))

    df.to_csv(args.out, index=False)

    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()