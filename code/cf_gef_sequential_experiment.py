from __future__ import annotations

import argparse
import hashlib
import pickle
from pathlib import Path

import networkx as nx
import pandas as pd

from core_flows import (
    initialize_weights,
    run_flow,
    extract_core_data_driven,
    evaluate_core,
)


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

        if df.shape[1] >= 3:
            df = df.iloc[:, :3]
            df.columns = ["source", "target", "weight"]
            G = nx.from_pandas_edgelist(
                df,
                "source",
                "target",
                edge_attr="weight",
                create_using=nx.Graph(),
            )
        else:
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
            f"Unsupported graph file type: {suffix}"
        )

    G = nx.Graph(G)
    G.remove_edges_from(nx.selfloop_edges(G))

    if G.number_of_nodes() == 0:
        raise ValueError("The input graph is empty.")

    components = list(nx.connected_components(G))
    largest = max(len(c) for c in components)

    print(
        f"Raw graph used in experiment: "
        f"n={G.number_of_nodes()}, "
        f"m={G.number_of_edges()}, "
        f"components={len(components)}, "
        f"largest_component={largest}",
        flush=True,
    )

    # Same convention as the main experiments:
    # all benchmark graphs use unit initial weights.
    for u, v in G.edges():
        G[u][v]["weight"] = 1.0

    return G


# ============================================================
# Utilities
# ============================================================

def parse_pairs(text: str) -> list[tuple[int, int]]:
    """
    Example:
        "0:10,2:8,5:5,8:2,10:0"
    means:
        (CF iterations, GEF iterations)
    """
    ans = []

    for item in text.split(","):
        item = item.strip()

        if not item:
            continue

        a, b = item.split(":")
        n_cf = int(a)
        n_gef = int(b)

        if n_cf < 0 or n_gef < 0:
            raise ValueError(
                "Iteration numbers must be nonnegative."
            )

        ans.append((n_cf, n_gef))

    if not ans:
        raise ValueError("No valid CF:GEF pairs supplied.")

    return ans


def parse_float_list(text: str) -> list[float]:
    return [
        float(x.strip())
        for x in text.split(",")
        if x.strip()
    ]


def graph_signature(
    graph_path: str,
    G: nx.Graph,
) -> str:
    """
    Short signature to prevent accidental reuse of cache files
    from another version of the graph.
    """
    text = (
        f"{Path(graph_path).name}|"
        f"{G.number_of_nodes()}|"
        f"{G.number_of_edges()}|"
        f"{nx.number_connected_components(G)}"
    )

    return hashlib.sha1(
        text.encode("utf-8")
    ).hexdigest()[:10]


# ============================================================
# Sequential flow
# ============================================================

def sequential_cache_path(
    cache_dir: Path,
    graph_path: str,
    G: nx.Graph,
    n_cf: int,
    n_gef: int,
    s_cf: float,
    s_gef: float,
    alpha: float,
) -> Path:

    sig = graph_signature(graph_path, G)
    stem = Path(graph_path).stem

    name = (
        f"{stem}_{sig}"
        f"_CF{n_cf}_s{s_cf}"
        f"_GEF{n_gef}_s{s_gef}_a{alpha}.pkl"
    )

    # Avoid awkward dots in filenames.
    name = name.replace(".", "p")

    return cache_dir / name


def run_cf_then_gef(
    G: nx.Graph,
    graph_path: str,
    n_cf: int,
    n_gef: int,
    s_cf: float,
    s_gef: float,
    alpha: float,
    cache_dir: Path,
    use_cache: bool = True,
):
    """
    Sequential evolution:

        unit weights
          -> CF for n_cf iterations
          -> GEF for n_gef iterations

    The final CF weights are used directly as the
    initial weights of GEF.
    """

    cache_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cp = sequential_cache_path(
        cache_dir=cache_dir,
        graph_path=graph_path,
        G=G,
        n_cf=n_cf,
        n_gef=n_gef,
        s_cf=s_cf,
        s_gef=s_gef,
        alpha=alpha,
    )

    if use_cache and cp.exists():
        with cp.open("rb") as f:
            payload = pickle.load(f)

        print(
            f"[cache] CF={n_cf}, GEF={n_gef}",
            flush=True,
        )

        return (
            payload["weights"],
            payload["cf_runtime_sec"],
            payload["gef_runtime_sec"],
        )

    weights = initialize_weights(
        G,
        default=1.0,
    )

    # --------------------------------------------------------
    # Stage 1: CF
    # --------------------------------------------------------

    cf_runtime = 0.0

    if n_cf > 0:
        print(
            f"\n[stage 1] CF: "
            f"N={n_cf}, s={s_cf}",
            flush=True,
        )

        cf_result = run_flow(
            G=G,
            flow="CF",
            iterations=n_cf,
            step_size=s_cf,
            initial_weights=weights,
            keep_history=False,
        )

        weights = cf_result.weights
        cf_runtime = cf_result.runtime_seconds

    # --------------------------------------------------------
    # Stage 2: GEF
    # --------------------------------------------------------

    gef_runtime = 0.0

    if n_gef > 0:
        print(
            f"\n[stage 2] GEF: "
            f"N={n_gef}, s={s_gef}, alpha={alpha}",
            flush=True,
        )

        gef_result = run_flow(
            G=G,
            flow="GEF",
            iterations=n_gef,
            step_size=s_gef,
            alpha=alpha,

            # The important line:
            initial_weights=weights,

            keep_history=False,
        )

        weights = gef_result.weights
        gef_runtime = gef_result.runtime_seconds

    if use_cache:
        with cp.open("wb") as f:
            pickle.dump(
                {
                    "weights": weights,
                    "cf_runtime_sec": cf_runtime,
                    "gef_runtime_sec": gef_runtime,
                    "n_cf": n_cf,
                    "n_gef": n_gef,
                    "s_cf": s_cf,
                    "s_gef": s_gef,
                    "alpha": alpha,
                },
                f,
            )

    return (
        weights,
        cf_runtime,
        gef_runtime,
    )


# ============================================================
# Extraction
# ============================================================

def evaluate_at_tau(
    G: nx.Graph,
    weights,
    tau: float,
):
    cr = extract_core_data_driven(
        G=G,
        evolved_weights=weights,
        remove_ratio=tau,
    )

    metrics = evaluate_core(
        G,
        cr.core_graph,
    )

    return {
        "tau": tau,
        "removed_edges": len(cr.removed_edges),
        "non_isolated_nodes": len(cr.candidate_nodes),
        "core_nodes": metrics["core_nodes"],
        "core_ratio": (
            metrics["core_nodes"] / G.number_of_nodes()
            if G.number_of_nodes() > 0
            else float("nan")
        ),
        "core_edges": metrics["core_edges"],
        "r_d": metrics["r_d"],
        "r_s": metrics["r_s"],
    }


# ============================================================
# Main experiment
# ============================================================

def main():
    ap = argparse.ArgumentParser(
        description=(
            "Sequential CF -> GEF core-detection experiment."
        )
    )

    ap.add_argument(
        "--graph",
        required=True,
    )

    ap.add_argument(
        "--pairs",
        default="0:10,2:8,5:5,8:2,10:0",
        help=(
            "Comma-separated CF:GEF iteration pairs, "
            "e.g. 0:10,2:8,5:5,8:2,10:0."
        ),
    )

    ap.add_argument(
        "--taus",
        default="0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80",
    )

    ap.add_argument(
        "--s-cf",
        type=float,
        default=0.1,
    )

    ap.add_argument(
        "--s-gef",
        type=float,
        default=0.1,
    )

    ap.add_argument(
        "--alpha",
        type=float,
        default=0.1,
    )

    ap.add_argument(
        "--cache-dir",
        default="flow_cache_cf_gef_sequential",
    )

    ap.add_argument(
        "--no-cache",
        action="store_true",
    )

    ap.add_argument(
        "--out",
        default="cf_gef_sequential_results.csv",
    )

    args = ap.parse_args()

    G = load_graph(args.graph)

    pairs = parse_pairs(args.pairs)
    taus = parse_float_list(args.taus)

    for tau in taus:
        if not (0.0 <= tau < 1.0):
            raise ValueError(
                f"tau must lie in [0,1), got {tau}."
            )

    rows = []

    for n_cf, n_gef in pairs:

        print(
            "\n"
            "==================================================\n"
            f"Sequential experiment: "
            f"CF={n_cf}, GEF={n_gef}\n"
            "==================================================",
            flush=True,
        )

        weights, cf_runtime, gef_runtime = (
            run_cf_then_gef(
                G=G,
                graph_path=args.graph,
                n_cf=n_cf,
                n_gef=n_gef,
                s_cf=args.s_cf,
                s_gef=args.s_gef,
                alpha=args.alpha,
                cache_dir=Path(args.cache_dir),
                use_cache=not args.no_cache,
            )
        )

        for tau in taus:

            row = evaluate_at_tau(
                G=G,
                weights=weights,
                tau=tau,
            )

            row.update(
                {
                    "CF_iterations": n_cf,
                    "GEF_iterations": n_gef,
                    "total_iterations": n_cf + n_gef,
                    "s_CF": args.s_cf,
                    "s_GEF": args.s_gef,
                    "alpha_GEF": (
                        args.alpha
                        if n_gef > 0
                        else None
                    ),
                    "CF_runtime_sec": cf_runtime,
                    "GEF_runtime_sec": gef_runtime,
                    "total_runtime_sec": (
                        cf_runtime + gef_runtime
                    ),
                }
            )

            rows.append(row)

            print(
                f"[result] "
                f"CF={n_cf}, "
                f"GEF={n_gef}, "
                f"tau={tau:.3f}, "
                f"K={row['core_nodes']}, "
                f"rd={row['r_d']:.6f}, "
                f"rs={row['r_s']:.6f}",
                flush=True,
            )

    df = pd.DataFrame(rows)

    columns = [
        "CF_iterations",
        "GEF_iterations",
        "total_iterations",
        "s_CF",
        "s_GEF",
        "alpha_GEF",
        "tau",
        "removed_edges",
        "non_isolated_nodes",
        "core_nodes",
        "core_ratio",
        "core_edges",
        "r_d",
        "r_s",
        "CF_runtime_sec",
        "GEF_runtime_sec",
        "total_runtime_sec",
    ]

    df = df[columns]

    out = Path(args.out)
    df.to_csv(
        out,
        index=False,
    )

    print("\nFinal results:")
    print(df.to_string(index=False))

    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()