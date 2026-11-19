from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import pickle
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from core_flows import evaluate_core


# ============================================================
# Load the author's Ricci-flow implementation
# ============================================================

def load_ricci_class(script_path: str):
    """
    Dynamically load RhoCurvature from the author's original script.
    This allows the filename to contain spaces.
    """
    path = Path(script_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Ricci-flow script not found: {path}"
        )

    spec = importlib.util.spec_from_file_location(
        "author_ricci_flow",
        path,
    )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "RhoCurvature"):
        raise AttributeError(
            "RhoCurvature was not found in the supplied Ricci script."
        )

    return module.RhoCurvature


# ============================================================
# Graph loading
# ============================================================

def load_graph(path: str) -> nx.Graph:
    p = Path(path)
    suffix = p.suffix.lower()

    if suffix == ".gexf":
        G = nx.read_gexf(p)

    elif suffix == ".graphml":
        G = nx.read_graphml(p)

    elif suffix in {".edgelist", ".txt"}:
        G = nx.read_edgelist(
            p,
            comments="#",
            delimiter=None,
            nodetype=str,
            data=False,
            create_using=nx.Graph,
        )

    elif suffix == ".edges":
        df = pd.read_csv(
            p,
            sep=r"\s+",
            header=None,
            comment="#",
        )

        if df.shape[1] < 2:
            raise ValueError(
                "Edge file must contain at least two columns."
            )

        df = df.iloc[:, :2]
        df.columns = ["source", "target"]

        G = nx.from_pandas_edgelist(
            df,
            "source",
            "target",
            create_using=nx.Graph(),
        )

    else:
        raise ValueError(
            f"Unsupported graph type: {suffix}"
        )

    G = nx.Graph(G)
    G = nx.convert_node_labels_to_integers(
        G,
        label_attribute="original_id"
    )
    G.remove_edges_from(nx.selfloop_edges(G))

    if G.number_of_nodes() == 0:
        raise ValueError("Input graph is empty.")

    # Fair comparison with our main experiments:
    # unit initial edge weights.
    for u, v in G.edges():

        # reset weight
        G[u][v]["weight"] = 1.0

        # remove old Ricci attributes
        for key in [
            "ricciCurvature",
            "original_RC",
        ]:
            if key in G[u][v]:
                del G[u][v][key]

    comps = list(nx.connected_components(G))

    print(
        f"Graph: "
        f"n={G.number_of_nodes()}, "
        f"m={G.number_of_edges()}, "
        f"components={len(comps)}, "
        f"largest_component={max(len(c) for c in comps)}",
        flush=True,
    )

    return G


# ============================================================
# Utilities
# ============================================================

def edge_key(u, v):
    return frozenset((u, v))


def graph_signature(G: nx.Graph) -> str:
    tokens = []

    for u, v in G.edges():
        a, b = sorted((str(u), str(v)))
        tokens.append(f"{a}\t{b}")

    tokens.sort()

    payload = (
        f"n={G.number_of_nodes()}|"
        f"m={G.number_of_edges()}|"
        + "\n".join(tokens)
    )

    return hashlib.sha1(
        payload.encode("utf-8")
    ).hexdigest()[:12]


# ============================================================
# Component-wise Ricci evolution
# ============================================================

def run_componentwise_ricci(
    G: nx.Graph,
    RhoCurvature,
    alpha: float,
    iterations: int,
    step: float,
    delta: float,
):
    """
    Run Ricci flow independently on every connected component
    containing at least one edge.

    The evolved edge weights are then merged into one global
    edge-weight dictionary.

    This avoids restricting a disconnected network such as
    Bio-CE-HT to its largest connected component.
    """

    evolved_weights = {}

    components = list(nx.connected_components(G))

    nontrivial_components = [
        set(c)
        for c in components
        if G.subgraph(c).number_of_edges() > 0
    ]

    print(
        f"Running Ricci flow on "
        f"{len(nontrivial_components)} nontrivial components.",
        flush=True,
    )

    for idx, nodes in enumerate(
        nontrivial_components,
        start=1,
    ):
        H = G.subgraph(nodes).copy()

        for u, v in H.edges():
            H[u][v]["weight"] = 1.0

        original_edges = {
            edge_key(u, v)
            for u, v in H.edges()
        }

        print(
            f"\n[component {idx}/"
            f"{len(nontrivial_components)}] "
            f"n={H.number_of_nodes()}, "
            f"m={H.number_of_edges()}",
            flush=True,
        )

        rho = RhoCurvature(
            copy.deepcopy(H),
            alpha=alpha,
            weight="weight",
        )

        # ricci_flow_iteration() itself computes the initial
        # curvature if necessary.
        rho.ricci_flow_iteration(
            iterations=iterations,
            step=step,
            delta=delta,
        )

        evolved_H = rho.G

        evolved_edge_map = {
            edge_key(u, v): float(
                evolved_H[u][v].get("weight", 1.0)
            )
            for u, v in evolved_H.edges()
        }

        missing = (
            original_edges
            - set(evolved_edge_map)
        )

        if missing:
            print(
                f"[warning] component {idx}: "
                f"{len(missing)} original edges disappeared "
                f"during Ricci contraction.",
                flush=True,
            )

            # Those edges were contracted only after their
            # weights became extremely small. Since the final
            # surgery removes the largest-weight edges, assign
            # them a very small ranking value.
            for e in missing:
                evolved_edge_map[e] = 0.0

        # We only need weights for edges that existed in
        # the original input graph.
        for u, v in H.edges():
            k = edge_key(u, v)
            evolved_weights[k] = evolved_edge_map[k]

    # Isolated vertices do not contribute edge weights.
    if len(evolved_weights) != G.number_of_edges():
        raise RuntimeError(
            f"Expected {G.number_of_edges()} evolved edge weights, "
            f"but obtained {len(evolved_weights)}."
        )

    return evolved_weights


# ============================================================
# Pure extraction
# ============================================================

def extract_core_pure(
    G: nx.Graph,
    evolved_weights,
    tau: float,
):
    """
    Same extraction convention as the main flow experiments:

    1. rank evolved edge weights in descending order;
    2. remove top tau fraction;
    3. retain all non-isolated vertices;
    4. reconstruct their induced subgraph in the original graph;
    5. return its largest connected component.

    No prescribed K and no degree-based node supplementation.
    """

    if not (0.0 <= tau < 1.0):
        raise ValueError("tau must lie in [0,1).")

    ranked_edges = sorted(
        G.edges(),
        key=lambda e: (
            evolved_weights[edge_key(*e)],
            str(e[0]),
            str(e[1]),
        ),
        reverse=True,
    )

    num_remove = int(
        len(ranked_edges) * tau
    )

    removed = ranked_edges[:num_remove]

    H = G.copy()
    H.remove_edges_from(removed)

    candidate_nodes = [
        v
        for v in H.nodes()
        if H.degree(v) > 0
    ]

    if not candidate_nodes:
        return nx.Graph(), removed, candidate_nodes

    reconstructed = G.subgraph(
        candidate_nodes
    ).copy()

    if reconstructed.number_of_nodes() == 0:
        return nx.Graph(), removed, candidate_nodes

    largest_cc = max(
        nx.connected_components(reconstructed),
        key=len,
    )

    core = reconstructed.subgraph(
        largest_cc
    ).copy()

    return core, removed, candidate_nodes


# ============================================================
# Matched-K search
# ============================================================

def scan_target(
    G: nx.Graph,
    weights,
    target_k: int,
    tau_min: float,
    tau_max: float,
    tau_step: float,
):
    """
    Select tau using ONLY the absolute core-size gap.

    r_d and r_s are evaluated only after the best tau
    has been selected.
    """

    taus = np.arange(
        tau_min,
        tau_max + tau_step / 2.0,
        tau_step,
    )

    rows = []

    for tau in taus:
        tau = round(float(tau), 6)

        core, removed, candidates = (
            extract_core_pure(
                G,
                weights,
                tau,
            )
        )

        k = core.number_of_nodes()

        rows.append(
            {
                "tau": tau,
                "target_k": target_k,
                "core_nodes": k,
                "size_gap": k - target_k,
                "abs_size_gap":
                    abs(k - target_k),
                "removed_edges":
                    len(removed),
                "non_isolated_nodes":
                    len(candidates),
            }
        )

    scan = pd.DataFrame(rows)

    best = (
        scan.sort_values(
            ["abs_size_gap", "tau"],
            ascending=[True, True],
        )
        .iloc[0]
        .copy()
    )

    best_tau = float(best["tau"])

    core, removed, candidates = (
        extract_core_pure(
            G,
            weights,
            best_tau,
        )
    )

    metrics = evaluate_core(
        G,
        core,
    )

    result = {
        "target_k": target_k,
        "core_nodes":
            metrics["core_nodes"],
        "size_gap":
            metrics["core_nodes"] - target_k,
        "abs_size_gap":
            abs(
                metrics["core_nodes"]
                - target_k
            ),
        "tau": best_tau,
        "removed_edges":
            len(removed),
        "non_isolated_nodes":
            len(candidates),
        "core_edges":
            metrics["core_edges"],
        "r_d":
            metrics["r_d"],
        "r_s":
            metrics["r_s"],
    }

    return result, scan


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser(
        description=(
            "Matched-size Ricci-flow baseline "
            "using pure core extraction."
        )
    )

    ap.add_argument(
        "--graph",
        required=True,
    )

    ap.add_argument(
        "--ricci-script",
        required=True,
        help=(
            "Path to the author's "
            "'core detectioin via Ricci flow.py'."
        ),
    )

    ap.add_argument(
        "--targets",
        required=True,
        help=(
            "Comma-separated target sizes, "
            "e.g. 1190,1162,1245"
        ),
    )

    ap.add_argument(
        "--labels",
        default="GEF,LEF,CF",
    )

    ap.add_argument(
        "--alpha",
        type=float,
        default=0.1,
    )

    ap.add_argument(
        "--iterations",
        type=int,
        default=12,
    )

    ap.add_argument(
        "--step",
        type=float,
        default=0.1,
    )

    ap.add_argument(
        "--delta",
        type=float,
        default=1e-6,
    )

    ap.add_argument(
        "--tau-min",
        type=float,
        default=0.0,
    )

    ap.add_argument(
        "--tau-max",
        type=float,
        default=0.95,
    )

    ap.add_argument(
        "--tau-step",
        type=float,
        default=0.001,
    )

    ap.add_argument(
        "--cache-dir",
        default="ricci_matched_cache",
    )

    ap.add_argument(
        "--out",
        default="ricci_matched_results.csv",
    )

    args = ap.parse_args()

    targets = [
        int(x.strip())
        for x in args.targets.split(",")
        if x.strip()
    ]

    labels = [
        x.strip()
        for x in args.labels.split(",")
        if x.strip()
    ]

    if len(labels) != len(targets):
        raise ValueError(
            "Number of labels must equal number of targets."
        )

    G = load_graph(args.graph)

    RhoCurvature = load_ricci_class(
        args.ricci_script
    )

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    sig = graph_signature(G)

    cache_path = cache_dir / (
        f"{Path(args.graph).stem}_"
        f"{sig}_"
        f"a{args.alpha}_"
        f"N{args.iterations}_"
        f"s{args.step}_"
        f"d{args.delta}.pkl"
    )

    if cache_path.exists():
        print(
            f"\n[cache] loading "
            f"{cache_path}",
            flush=True,
        )

        with cache_path.open("rb") as f:
            evolved_weights = pickle.load(f)

    else:
        evolved_weights = (
            run_componentwise_ricci(
                G=G,
                RhoCurvature=RhoCurvature,
                alpha=args.alpha,
                iterations=args.iterations,
                step=args.step,
                delta=args.delta,
            )
        )

        with cache_path.open("wb") as f:
            pickle.dump(
                evolved_weights,
                f,
            )

        print(
            f"\n[cache] saved "
            f"{cache_path}",
            flush=True,
        )

    result_rows = []

    out = Path(args.out)

    for label, target in zip(
        labels,
        targets,
    ):
        result, scan = scan_target(
            G=G,
            weights=evolved_weights,
            target_k=target,
            tau_min=args.tau_min,
            tau_max=args.tau_max,
            tau_step=args.tau_step,
        )

        result["reference"] = label
        result["method"] = "Ricci flow"
        result["alpha"] = args.alpha
        result["iterations"] = (
            args.iterations
        )
        result["step"] = args.step

        result_rows.append(result)

        scan_path = out.with_name(
            f"{out.stem}_"
            f"{label}_tau_scan.csv"
        )

        scan.to_csv(
            scan_path,
            index=False,
        )

        print(
            f"\nTarget {label}: "
            f"K={target}"
        )

        print(
            scan.sort_values(
                ["abs_size_gap", "tau"]
            )
            [
                [
                    "tau",
                    "target_k",
                    "core_nodes",
                    "size_gap",
                ]
            ]
            .head(10)
            .to_string(index=False)
        )

    results = pd.DataFrame(
        result_rows
    )

    columns = [
        "reference",
        "method",
        "target_k",
        "core_nodes",
        "size_gap",
        "tau",
        "core_edges",
        "r_d",
        "r_s",
        "alpha",
        "iterations",
        "step",
    ]

    results = results[columns]

    results.to_csv(
        out,
        index=False,
    )

    print("\nMatched Ricci results:")
    print(results.to_string(index=False))

    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()