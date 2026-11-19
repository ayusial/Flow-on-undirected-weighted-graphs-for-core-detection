from __future__ import annotations

import argparse
import hashlib
import pickle
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd

from core_flows import (
    edge_key,
    evaluate_core,
    initialize_weights,
    run_flow,
)


# ---------------------------------------------------------------------
# Graph loading: use exactly the supplied topology (no LCC cropping)
# ---------------------------------------------------------------------
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
        df = pd.read_csv(p, sep=r"\s+", header=None, comment="#")
        if df.shape[1] >= 3:
            df = df.iloc[:, :3]
            df.columns = ["source", "target", "weight"]
            G = nx.from_pandas_edgelist(
                df, "source", "target", edge_attr="weight", create_using=nx.Graph()
            )
        else:
            df = df.iloc[:, :2]
            df.columns = ["source", "target"]
            G = nx.from_pandas_edgelist(
                df, "source", "target", create_using=nx.Graph()
            )
    else:
        raise ValueError(f"Unsupported graph file type: {suffix}")

    G = nx.Graph(G)
    G.remove_edges_from(nx.selfloop_edges(G))
    if G.number_of_nodes() == 0:
        raise ValueError("The input graph is empty.")

    components = list(nx.connected_components(G))
    largest = max(len(c) for c in components)
    print(
        f"Raw graph used in experiment: n={G.number_of_nodes()}, "
        f"m={G.number_of_edges()}, components={len(components)}, "
        f"largest_component={largest}"
    )

    # Main experiment convention: unit initial weights unless perturb mode
    # explicitly supplies a different initialization.
    for u, v in G.edges():
        G[u][v]["weight"] = 1.0
    return G


def parse_float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_pairs(text: str) -> list[Tuple[int, float]]:
    pairs: list[Tuple[int, float]] = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError("Each fixed-time pair must have form N:s, e.g. 20:0.05")
        n_text, s_text = item.split(":", 1)
        pairs.append((int(n_text), float(s_text)))
    return pairs


def graph_signature(G: nx.Graph) -> str:
    """Small deterministic topology signature used only for cache separation."""
    edge_tokens = []
    for u, v in G.edges():
        a, b = sorted((str(u), str(v)))
        edge_tokens.append(f"{a}\t{b}")
    edge_tokens.sort()
    payload = (
        f"n={G.number_of_nodes()}|m={G.number_of_edges()}|"
        f"c={nx.number_connected_components(G)}|" + "\n".join(edge_tokens)
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def _clean(x) -> str:
    return str(x).replace(".", "p").replace("-", "m")


def cache_path(
    cache_dir: Path,
    graph_path: str,
    G: nx.Graph,
    flow: str,
    N: int,
    s: float,
    alpha: float | None,
    init_tag: str,
) -> Path:
    stem = Path(graph_path).stem
    sig = graph_signature(G)
    if flow == "CF":
        name = f"{stem}_{sig}_CF_N{N}_s{_clean(s)}_{init_tag}.pkl"
    else:
        name = (
            f"{stem}_{sig}_{flow}_N{N}_s{_clean(s)}_"
            f"a{_clean(alpha)}_{init_tag}.pkl"
        )
    return cache_dir / name


def get_or_run_flow(
    G: nx.Graph,
    graph_path: str,
    flow: str,
    N: int,
    s: float,
    alpha: float | None,
    initial_weights: Mapping[frozenset, float],
    cache_dir: Path,
    init_tag: str,
    use_cache: bool = True,
):
    flow = flow.upper()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cp = cache_path(cache_dir, graph_path, G, flow, N, s, alpha, init_tag)

    if use_cache and cp.exists():
        with cp.open("rb") as f:
            payload = pickle.load(f)
        print(f"[cache] {flow}: N={N}, s={s}, alpha={alpha}, init={init_tag}")
        return payload["weights"], payload.get("runtime_sec", float("nan"))

    print(f"[flow] {flow}: N={N}, s={s}, alpha={alpha}, init={init_tag}")
    kwargs = dict(
        G=G,
        flow=flow,
        iterations=N,
        step_size=s,
        initial_weights=initial_weights,
        keep_history=False,
    )
    if flow != "CF":
        kwargs["alpha"] = alpha

    fr = run_flow(**kwargs)
    if use_cache:
        with cp.open("wb") as f:
            pickle.dump(
                {"weights": fr.weights, "runtime_sec": fr.runtime_seconds}, f
            )
    return fr.weights, fr.runtime_seconds


# ---------------------------------------------------------------------
# Pure flow-based extraction used in the main experiments
# ---------------------------------------------------------------------
def extract_pure(
    G: nx.Graph,
    weights: Mapping[frozenset, float],
    tau: float,
):
    if not (0.0 <= tau < 1.0):
        raise ValueError("tau must be in [0,1).")

    ranked = sorted(
        G.edges(),
        key=lambda uv: weights[edge_key(*uv)],
        reverse=True,
    )
    n_remove = int(len(ranked) * tau)

    pruned = G.copy()
    pruned.remove_edges_from(ranked[:n_remove])
    non_iso = {v for v in pruned if pruned.degree(v) > 0}

    if non_iso:
        induced = G.subgraph(non_iso).copy()
        largest = max(nx.connected_components(induced), key=len)
        core = induced.subgraph(largest).copy()
    else:
        core = nx.Graph()

    row = {
        "tau": tau,
        "removed_edges": n_remove,
        "non_isolated_nodes": len(non_iso),
        "core_nodes": core.number_of_nodes(),
        "core_ratio": core.number_of_nodes() / G.number_of_nodes(),
    }
    row.update(evaluate_core(G, core))
    return row, set(core.nodes())


def jaccard(A: set, B: set) -> float:
    U = A | B
    return len(A & B) / len(U) if U else 1.0


def rank_jaccard(
    G: nx.Graph,
    w1: Mapping[frozenset, float],
    w2: Mapping[frozenset, float],
    top_ratio: float,
) -> float:
    """Jaccard overlap of the top top_ratio fraction of evolved-weight edges."""
    k = max(1, int(G.number_of_edges() * top_ratio))
    r1 = sorted(G.edges(), key=lambda uv: w1[edge_key(*uv)], reverse=True)[:k]
    r2 = sorted(G.edges(), key=lambda uv: w2[edge_key(*uv)], reverse=True)[:k]
    A = {edge_key(*e) for e in r1}
    B = {edge_key(*e) for e in r2}
    return jaccard(A, B)


# ---------------------------------------------------------------------
# Sensitivity modes
# ---------------------------------------------------------------------
def run_n_sensitivity(args, G: nx.Graph) -> pd.DataFrame:
    rows = []
    flow = args.flow.upper()
    initial = initialize_weights(G, default=1.0)
    for N in parse_int_list(args.Ns):
        w, runtime = get_or_run_flow(
            G, args.graph, flow, N, args.s, args.alpha,
            initial, Path(args.cache_dir), "unit", not args.no_cache,
        )
        row, _ = extract_pure(G, w, args.tau)
        row.update({
            "mode": "N-sensitivity", "flow": flow, "N": N,
            "s": args.s, "alpha": None if flow == "CF" else args.alpha,
            "runtime_sec": runtime,
        })
        rows.append(row)
    return pd.DataFrame(rows)


def run_s_sensitivity(args, G: nx.Graph) -> pd.DataFrame:
    rows = []
    flow = args.flow.upper()
    initial = initialize_weights(G, default=1.0)
    for s in parse_float_list(args.ss):
        w, runtime = get_or_run_flow(
            G, args.graph, flow, args.N, s, args.alpha,
            initial, Path(args.cache_dir), "unit", not args.no_cache,
        )
        row, _ = extract_pure(G, w, args.tau)
        row.update({
            "mode": "s-sensitivity", "flow": flow, "N": args.N,
            "s": s, "alpha": None if flow == "CF" else args.alpha,
            "runtime_sec": runtime,
        })
        rows.append(row)
    return pd.DataFrame(rows)


def run_weight_perturbation(args, G: nx.Graph) -> pd.DataFrame:
    flow = args.flow.upper()
    rows = []

    # Uniform reference core.
    unit = initialize_weights(G, default=1.0)
    w_ref, runtime_ref = get_or_run_flow(
        G, args.graph, flow, args.N, args.s, args.alpha,
        unit, Path(args.cache_dir), "unit", not args.no_cache,
    )
    ref_row, ref_core = extract_pure(G, w_ref, args.tau)
    ref_row.update({
        "mode": "weight-perturbation", "flow": flow, "run": "uniform",
        "seed": None, "N": args.N, "s": args.s,
        "alpha": None if flow == "CF" else args.alpha,
        "runtime_sec": runtime_ref, "core_jaccard_vs_uniform": 1.0,
        "top_edge_jaccard_vs_uniform": 1.0,
    })
    rows.append(ref_row)

    edges = [edge_key(u, v) for u, v in G.edges()]
    for i in range(args.runs):
        seed = args.seed + i
        rng = np.random.default_rng(seed)
        vals = rng.uniform(args.weight_low, args.weight_high, size=len(edges))
        init = {e: float(v) for e, v in zip(edges, vals)}
        init_tag = f"uniformPerturb_seed{seed}_lo{_clean(args.weight_low)}_hi{_clean(args.weight_high)}"
        w, runtime = get_or_run_flow(
            G, args.graph, flow, args.N, args.s, args.alpha,
            init, Path(args.cache_dir), init_tag, not args.no_cache,
        )
        row, core = extract_pure(G, w, args.tau)
        row.update({
            "mode": "weight-perturbation", "flow": flow, "run": i + 1,
            "seed": seed, "N": args.N, "s": args.s,
            "alpha": None if flow == "CF" else args.alpha,
            "runtime_sec": runtime,
            "core_jaccard_vs_uniform": jaccard(core, ref_core),
            "top_edge_jaccard_vs_uniform": rank_jaccard(
                G, w, w_ref, args.edge_top_ratio
            ),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def run_fixed_time(args, G: nx.Graph) -> pd.DataFrame:
    rows = []
    flow = args.flow.upper()
    initial = initialize_weights(G, default=1.0)
    for N, s in parse_pairs(args.pairs):
        w, runtime = get_or_run_flow(
            G, args.graph, flow, N, s, args.alpha,
            initial, Path(args.cache_dir), "unit", not args.no_cache,
        )
        row, _ = extract_pure(G, w, args.tau)
        row.update({
            "mode": "fixed-time", "flow": flow, "N": N, "s": s,
            "T": N * s, "alpha": None if flow == "CF" else args.alpha,
            "runtime_sec": runtime,
        })
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Unified N/s/initial-weight sensitivity experiments for GEF, LEF and CF."
    )
    ap.add_argument("--graph", required=True)
    ap.add_argument("--flow", required=True, choices=["GEF", "LEF", "CF", "gef", "lef", "cf"])
    ap.add_argument(
        "--mode", required=True,
        choices=["N", "s", "weights", "fixed-time"],
        help="N: iteration sensitivity; s: step-size sensitivity; weights: initial-weight perturbation; fixed-time: compare (N,s) pairs with similar T=Ns.",
    )
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--tau", type=float, required=True)

    # N sensitivity
    ap.add_argument("--Ns", default="10,20,30,40,50")
    ap.add_argument("--s", type=float, default=0.1)

    # s sensitivity / perturbation baseline
    ap.add_argument("--N", type=int, default=10)
    ap.add_argument("--ss", default="0.025,0.05,0.1,0.2,0.3,0.5")

    # initial-weight perturbation
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--weight-low", type=float, default=0.5)
    ap.add_argument("--weight-high", type=float, default=1.5)
    ap.add_argument(
        "--edge-top-ratio", type=float, default=0.2,
        help="Top-edge fraction used only for edge-ranking Jaccard in weight perturbation mode.",
    )

    # fixed T = Ns checks
    ap.add_argument("--pairs", default="20:0.05,10:0.1,5:0.2")

    ap.add_argument("--cache-dir", default="flow_cache_sensitivity")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--out", default="sensitivity_results.csv")
    args = ap.parse_args()

    if not (0.0 < args.weight_low <= args.weight_high):
        raise ValueError("Require 0 < weight-low <= weight-high.")
    if not (0.0 < args.edge_top_ratio <= 1.0):
        raise ValueError("edge-top-ratio must be in (0,1].")

    G = load_graph(args.graph)
    flow = args.flow.upper()
    args.flow = flow

    if args.mode == "N":
        df = run_n_sensitivity(args, G)
    elif args.mode == "s":
        df = run_s_sensitivity(args, G)
    elif args.mode == "weights":
        df = run_weight_perturbation(args, G)
    else:
        df = run_fixed_time(args, G)

    out = Path(args.out)
    df.to_csv(out, index=False)
    print("\nResults:")
    print(df.to_string(index=False))

    # Helpful summaries for perturbation mode.
    if args.mode == "weights" and len(df) > 1:
        pert = df[df["run"] != "uniform"]
        summary_cols = [
            "core_nodes", "r_d", "r_s",
            "core_jaccard_vs_uniform", "top_edge_jaccard_vs_uniform",
        ]
        print("\nPerturbation mean/std:")
        print(pert[summary_cols].agg(["mean", "std"]).to_string())

    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
