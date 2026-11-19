
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Dict, Hashable, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import networkx as nx
import numpy as np


Node = Hashable
Edge = Tuple[Node, Node]


def edge_key(u: Node, v: Node) -> frozenset:
    """Undirected edge key robust to arbitrary node labels."""
    return frozenset((u, v))


def initialize_weights(
    G: nx.Graph,
    default: float = 1.0,
    weight_attr: str = "weight",
) -> Dict[frozenset, float]:
    """
    Read positive edge weights from G. If an edge has no usable weight,
    assign `default`.
    """
    w: Dict[frozenset, float] = {}
    for u, v, data in G.edges(data=True):
        val = data.get(weight_attr, default)
        try:
            val = float(val)
        except (TypeError, ValueError):
            val = default
        if val <= 0:
            val = default
        w[edge_key(u, v)] = val
    return w


def write_weights_to_graph(
    G: nx.Graph,
    weights: Mapping[frozenset, float],
    weight_attr: str = "weight",
) -> None:
    for u, v in G.edges():
        G[u][v][weight_attr] = float(weights[edge_key(u, v)])


def sym_kl(p: Mapping[Node, float], q: Mapping[Node, float], support: Iterable[Node]) -> float:
    """Symmetric KL (J-divergence), assuming strictly positive masses on support."""
    ans = 0.0
    for z in support:
        pz = float(p[z])
        qz = float(q[z])
        ans += pz * math.log(pz / qz) + qz * math.log(qz / pz)
    return ans


# ---------------------------------------------------------------------
# Global entropy flow (GEF)
# ---------------------------------------------------------------------

def outward_random_walk(
    G: nx.Graph,
    source: Node,
    alpha: float,
    weights: Mapping[frozenset, float],
) -> Dict[Node, float]:
    """
    Compute the alpha-lazy outward random walk R_source^alpha.

    This follows the definition in:
    "An Efficient Entropy Flow on Weighted Graphs: Theory and Applications"
    (arXiv:2604.08144v1).

    P(source,z) is computed recursively on the BFS-layer DAG. For a node
    z in layer j, the outward-path transition from u in layer j-1 is
    weighted by w_uz divided by the total weight from u to layer j.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0,1).")

    dist = nx.single_source_shortest_path_length(G, source)
    if len(dist) != G.number_of_nodes():
        raise ValueError("GEF outward walk requires a connected component.")

    max_d = max(dist.values(), default=0)
    layers: List[List[Node]] = [[] for _ in range(max_d + 1)]
    for z, d in dist.items():
        layers[d].append(z)

    # P(source,z): total probability over outward paths.
    P: Dict[Node, float] = {source: 1.0}
    for j in range(1, max_d + 1):
        for z in layers[j]:
            val = 0.0
            for u in G.neighbors(z):
                if dist[u] != j - 1:
                    continue
                denom = 0.0
                for a in G.neighbors(u):
                    if dist.get(a) == j:
                        denom += weights[edge_key(u, a)]
                if denom > 0:
                    val += P[u] * weights[edge_key(u, z)] / denom
            P[z] = val

    R: Dict[Node, float] = {source: alpha}
    for j in range(1, max_d + 1):
        next_layer = set(layers[j + 1]) if j < max_d else set()
        factor = (1.0 - alpha) ** j
        for z in layers[j]:
            has_outward_neighbor = bool(next_layer) and any(
                a in next_layer for a in G.neighbors(z)
            )
            stop_factor = alpha if has_outward_neighbor else 1.0
            R[z] = factor * stop_factor * P[z]

    # Tiny floating-point cleanup only; theory gives a probability measure.
    s = sum(R.values())
    if s <= 0:
        raise FloatingPointError("Invalid outward random-walk mass.")
    if abs(s - 1.0) > 1e-10:
        R = {z: val / s for z, val in R.items()}
    return R


def gef_driving_values(
    G: nx.Graph,
    alpha: float,
    weights: Mapping[frozenset, float],
) -> Dict[frozenset, float]:
    walks = {x: outward_random_walk(G, x, alpha, weights) for x in G.nodes()}
    support = list(G.nodes())
    D: Dict[frozenset, float] = {}
    for u, v in G.edges():
        D[edge_key(u, v)] = sym_kl(walks[u], walks[v], support)
    return D


# ---------------------------------------------------------------------
# Local entropy flow (LEF)
# ---------------------------------------------------------------------

def _paired_local_walk(
    G: nx.Graph,
    x: Node,
    y: Node,
    alpha: float,
    weights: Mapping[frozenset, float],
) -> Dict[Node, float]:
    """
    Local random walk mu_x^alpha associated with the oriented endpoint x
    of edge e=xy.
    """
    Nx = set(G.neighbors(x)) | {x}
    Ny = set(G.neighbors(y)) | {y}
    diff = Ny - Nx

    mu: Dict[Node, float] = {}

    denom_x = sum(weights[edge_key(x, u)] for u in G.neighbors(x))
    if denom_x <= 0:
        raise FloatingPointError("Non-positive weighted degree.")

    if diff:
        mu[x] = alpha
        for z in G.neighbors(x):
            mu[z] = (
                alpha
                * (1.0 - alpha)
                * weights[edge_key(x, z)]
                / denom_x
            )

        denom_y_diff = sum(weights[edge_key(y, u)] for u in diff)
        if denom_y_diff <= 0:
            raise FloatingPointError("Non-positive local normalization.")
        for z in diff:
            mu[z] = (
                (1.0 - alpha) ** 2
                * weights[edge_key(y, z)]
                / denom_y_diff
            )
    else:
        mu[x] = alpha
        for z in G.neighbors(x):
            mu[z] = (
                (1.0 - alpha)
                * weights[edge_key(x, z)]
                / denom_x
            )

    support = Nx | Ny
    # Formula guarantees positivity on the paired support.
    for z in support:
        if z not in mu or mu[z] <= 0:
            raise FloatingPointError(
                f"Local walk has zero mass at {z}; check graph assumptions."
            )
    return mu


def lef_driving_values(
    G: nx.Graph,
    alpha: float,
    weights: Mapping[frozenset, float],
) -> Dict[frozenset, float]:
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0,1).")

    theta: Dict[frozenset, float] = {}
    for x, y in G.edges():
        mux = _paired_local_walk(G, x, y, alpha, weights)
        muy = _paired_local_walk(G, y, x, alpha, weights)
        support = (
            set(G.neighbors(x)) | {x}
            | set(G.neighbors(y)) | {y}
        )
        theta[edge_key(x, y)] = sym_kl(mux, muy, support)
    return theta


# ---------------------------------------------------------------------
# Cohesion flow (CF)
# ---------------------------------------------------------------------

def cf_driving_values(
    G: nx.Graph,
    weights: Mapping[frozenset, float],
) -> Dict[frozenset, float]:
    C: Dict[frozenset, float] = {}
    for x, y in G.edges():
        Nx = set(G.neighbors(x)) | {x}
        Ny = set(G.neighbors(y)) | {y}
        common = set(G.neighbors(x)) & set(G.neighbors(y))

        A = sum(1.0 / weights[edge_key(x, u)] for u in (Nx - Ny))
        A += sum(1.0 / weights[edge_key(y, u)] for u in (Ny - Nx))

        B = sum(
            1.0 / weights[edge_key(x, z)]
            + 1.0 / weights[edge_key(y, z)]
            for z in common
        )

        tilde_c = (A - B) / (A + B) if A + B > 0 else 0.0
        C[edge_key(x, y)] = math.exp(tilde_c)
    return C


# ---------------------------------------------------------------------
# Unified discrete flow
# ---------------------------------------------------------------------

@dataclass
class FlowResult:
    flow: str
    weights: Dict[frozenset, float]
    history: List[Dict[frozenset, float]]
    runtime_seconds: float


def run_flow(
    G: nx.Graph,
    flow: str,
    iterations: int,
    step_size: float = 0.1,
    alpha: float = 0.5,
    initial_weights: Optional[Mapping[frozenset, float]] = None,
    keep_history: bool = False,
) -> FlowResult:
    """
    Run one of GEF / LEF / CF.

    Updates are synchronous:
        w_e^(j+1) = w_e^(j) + s F_e^(j).
    """
    if step_size <= 0:
        raise ValueError("step_size must be positive.")
    if iterations < 0:
        raise ValueError("iterations must be nonnegative.")

    H = nx.Graph(G)
    H.remove_edges_from(nx.selfloop_edges(H))
    if H.number_of_nodes() == 0:
        raise ValueError("Empty graph.")

    # Keep the full input graph.  If it is disconnected, each flow is
    # evaluated independently on each connected component and the edge-wise
    # driving values are merged before the synchronous global update.
    # This preserves every edge in the supplied dataset instead of silently
    # replacing the graph by its largest connected component.
    edge_keys = {edge_key(u, v) for u, v in H.edges()}
    if initial_weights is None:
        weights = initialize_weights(H, default=1.0)
    else:
        missing = [e for e in edge_keys if e not in initial_weights]
        if missing:
            raise KeyError(
                f"initial_weights is missing {len(missing)} graph edge(s)."
            )
        # Ignore stale/extra keys from a different graph version.
        weights = {e: float(initial_weights[e]) for e in edge_keys}

    history: List[Dict[frozenset, float]] = [dict(weights)] if keep_history else []

    flow = flow.upper()
    if flow not in {"GEF", "LEF", "CF"}:
        raise ValueError("flow must be one of {'GEF','LEF','CF'}.")

    # Materialize components once; topology does not change during a flow run.
    components = [
        H.subgraph(nodes).copy()
        for nodes in nx.connected_components(H)
        if H.subgraph(nodes).number_of_edges() > 0
    ]

    t0 = time.perf_counter()

    for j in range(iterations):
        iter_t0 = time.perf_counter()

        print(
            f"[{flow}] starting iteration {j + 1}/{iterations}",
            flush=True,
        )

        F: Dict[frozenset, float] = {}

        for C in components:
            if flow == "GEF":
                Fc = gef_driving_values(C, alpha, weights)
            elif flow == "LEF":
                Fc = lef_driving_values(C, alpha, weights)
            else:  # CF
                Fc = cf_driving_values(C, weights)

            F.update(Fc)

        if set(F) != edge_keys:
            missing = edge_keys - set(F)
            extra = set(F) - edge_keys
            raise RuntimeError(
                "Flow driving values do not match graph edges: "
                f"missing={len(missing)}, extra={len(extra)}."
            )

        weights = {
            e: weights[e] + step_size * F[e]
            for e in edge_keys
        }

        iter_runtime = time.perf_counter() - iter_t0

        print(
            f"[{flow}] finished iteration {j + 1}/{iterations} "
            f"in {iter_runtime:.2f} sec",
            flush=True,
        )

        if keep_history:
            history.append(dict(weights))

    return FlowResult(
        flow=flow,
        weights=weights,
        history=history,
        runtime_seconds=time.perf_counter() - t0,
    )


# ---------------------------------------------------------------------
# Core extraction and evaluation
# ---------------------------------------------------------------------

@dataclass
class CoreResult:
    core_graph: nx.Graph
    candidate_nodes: Set[Node]
    removed_edges: List[Tuple[Node, Node]]
    core_nodes: Set[Node]


def extract_core_data_driven(
    G: nx.Graph,
    evolved_weights: Mapping[frozenset, float],
    remove_ratio: float = 0.8,
) -> CoreResult:
    """
    Data-driven core extraction:
      1) remove top tau proportion of evolved-weight edges;
      2) retain all non-isolated vertices as the candidate set;
      3) induce that set in the ORIGINAL graph;
      4) return the largest connected component.

    No target core size K is used.
    """
    if not (0.0 <= remove_ratio < 1.0):
        raise ValueError("remove_ratio must be in [0,1).")

    H = nx.Graph(G)
    H.remove_edges_from(nx.selfloop_edges(H))

    ranked = sorted(
        H.edges(),
        key=lambda uv: evolved_weights[edge_key(*uv)],
        reverse=True,
    )
    n_remove = int(len(ranked) * remove_ratio)
    removed = ranked[:n_remove]

    pruned = H.copy()
    pruned.remove_edges_from(removed)

    candidate = {v for v in pruned.nodes() if pruned.degree(v) > 0}
    if not candidate:
        return CoreResult(
            core_graph=nx.Graph(),
            candidate_nodes=set(),
            removed_edges=removed,
            core_nodes=set(),
        )

    induced = H.subgraph(candidate).copy()
    largest = max(nx.connected_components(induced), key=len)
    core = induced.subgraph(largest).copy()
    return CoreResult(
        core_graph=core,
        candidate_nodes=candidate,
        removed_edges=removed,
        core_nodes=set(core.nodes()),
    )


def compute_rd(G: nx.Graph, core: nx.Graph) -> float:
    if core.number_of_nodes() == 0:
        return float("nan")
    vals = []
    for x in core.nodes():
        d0 = G.degree(x)
        vals.append(core.degree(x) / d0 if d0 > 0 else 0.0)
    return float(np.mean(vals))


def compute_rs(G: nx.Graph, core_nodes: Set[Node]) -> float:
    """
    r_s as defined in the Ricci-flow core-detection paper.
    Only pairs that remain connected in the residual graph are averaged.
    Returns NaN if no residual pair remains connected.
    """
    residual_nodes = set(G.nodes()) - set(core_nodes)
    if len(residual_nodes) < 2:
        return float("nan")

    G_res = G.subgraph(residual_nodes).copy()
    dist_G = dict(nx.all_pairs_shortest_path_length(G))
    dist_res = dict(nx.all_pairs_shortest_path_length(G_res))

    total = 0.0
    count = 0
    nodes = list(residual_nodes)
    for i, u in enumerate(nodes):
        du = dist_res.get(u, {})
        dg = dist_G.get(u, {})
        for v in nodes[i + 1:]:
            if v in du and v in dg and dg[v] > 0:
                total += du[v] / dg[v]
                count += 1

    return total / count if count else float("nan")


def evaluate_core(G: nx.Graph, core: nx.Graph) -> Dict[str, float]:
    nodes = set(core.nodes())
    return {
        "core_nodes": core.number_of_nodes(),
        "core_edges": core.number_of_edges(),
        "r_d": compute_rd(G, core),
        "r_s": compute_rs(G, nodes),
    }



def _centrality_scores(G: nx.Graph, method: str) -> Dict[Node, float]:
    method = method.lower()
    if method == "degree":
        return nx.degree_centrality(G)
    elif method == "pagerank":
        return nx.pagerank(G, alpha=0.85, weight=None)
    elif method == "betweenness":
        return nx.betweenness_centrality(G, weight=None)
    elif method == "closeness":
        return nx.closeness_centrality(G)
    else:
        raise ValueError("Unknown centrality baseline.")


def connected_centrality_core(
    G: nx.Graph,
    method: str,
    target_size: int,
) -> nx.Graph:
    """Matched-size connected centrality baseline for connected or disconnected G.

    The selected set is constrained to one connected component containing at
    least ``target_size`` vertices.  Within that component, start from the
    highest-ranked eligible vertex and repeatedly add the highest-ranked
    frontier vertex until exactly ``target_size`` vertices are selected.
    """
    target_size = int(target_size)
    if target_size <= 0:
        return nx.Graph()

    components = [set(c) for c in nx.connected_components(G)]
    feasible = [c for c in components if len(c) >= target_size]
    if not feasible:
        largest = max((len(c) for c in components), default=0)
        raise ValueError(
            f"No connected component contains {target_size} nodes; "
            f"largest component size is {largest}."
        )

    score = _centrality_scores(G, method)
    ranked = sorted(
        G.nodes(),
        key=lambda v: (score[v], G.degree(v), str(v)),
        reverse=True,
    )

    comp_of = {}
    for i, comp in enumerate(feasible):
        for v in comp:
            comp_of[v] = i

    seed = next(v for v in ranked if v in comp_of)
    allowed = feasible[comp_of[seed]]
    ranked_allowed = [v for v in ranked if v in allowed]
    rank_idx = {v: i for i, v in enumerate(ranked_allowed)}

    selected = {seed}
    frontier = set(G.neighbors(seed)) & allowed
    while len(selected) < target_size:
        frontier -= selected
        if not frontier:
            raise RuntimeError(
                "Frontier became empty inside a connected component before "
                "reaching target_size."
            )
        nxt = min(frontier, key=lambda v: rank_idx[v])
        selected.add(nxt)
        frontier.update(u for u in G.neighbors(nxt) if u in allowed)

    return G.subgraph(selected).copy()


def jaccard(A: Set[Node], B: Set[Node]) -> float:
    union = A | B
    return len(A & B) / len(union) if union else 1.0


# ---------------------------------------------------------------------
# Synthetic planted core-periphery generator
# ---------------------------------------------------------------------

def planted_core_periphery_graph(
    n_core: int = 60,
    n_periphery: int = 140,
    p_cc: float = 0.25,
    p_cp: float = 0.04,
    p_pp: float = 0.01,
    seed: Optional[int] = 0,
) -> Tuple[nx.Graph, Set[int]]:
    """
    Undirected planted core-periphery random graph.
    Returns (G, ground_truth_core).
    """
    rng = np.random.default_rng(seed)
    n = n_core + n_periphery
    core = set(range(n_core))
    G = nx.Graph()
    G.add_nodes_from(range(n))

    for i in range(n):
        for j in range(i + 1, n):
            if i in core and j in core:
                p = p_cc
            elif (i in core) ^ (j in core):
                p = p_cp
            else:
                p = p_pp
            if rng.random() < p:
                G.add_edge(i, j, weight=1.0)

    # Keep the generated graph exactly as sampled.  The real-network pipeline
    # likewise does not pre-extract the largest connected component.
    return G, core


def recovery_metrics(pred: Set[Node], truth: Set[Node]) -> Dict[str, float]:
    tp = len(pred & truth)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(truth) if truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "jaccard_truth": jaccard(pred, truth),
    }
