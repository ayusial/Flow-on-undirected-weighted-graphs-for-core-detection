from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from core_flows import (
    initialize_weights,
    run_flow,
    extract_core_data_driven,
    evaluate_core,
    planted_core_periphery_graph,
    recovery_metrics,
)


FLOWS = ("GEF", "LEF", "CF")


def parse_float_list(text: str):
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_int_list(text: str):
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def choose_tau_by_size(
    G,
    evolved_weights,
    target_size: int,
    taus,
):
    """
    Choose tau using ONLY the detected core size.

    Ground-truth node membership is deliberately not used here,
    so the recovery metrics do not leak into parameter selection.
    """
    candidates = []

    for tau in taus:
        result = extract_core_data_driven(
            G,
            evolved_weights,
            remove_ratio=tau,
        )

        k = len(result.core_nodes)

        candidates.append(
            (
                abs(k - target_size),
                tau,
                k,
                result,
            )
        )

    # Primary criterion: closest detected size to planted core size.
    # Tie-breaker: smaller tau, chosen deterministically.
    candidates.sort(
        key=lambda x: (
            x[0],
            x[1],
        )
    )

    _, best_tau, best_k, best_result = candidates[0]

    return best_tau, best_k, best_result


def run_one(
    flow: str,
    p_cp: float,
    seed: int,
    n_core: int,
    n_periphery: int,
    p_cc: float,
    p_pp: float,
    N: int,
    s: float,
    alpha: float,
    taus,
):
    G, true_core = planted_core_periphery_graph(
        n_core=n_core,
        n_periphery=n_periphery,
        p_cc=p_cc,
        p_cp=p_cp,
        p_pp=p_pp,
        seed=seed,
    )

    initial_weights = initialize_weights(
        G,
        default=1.0,
    )

    flow_result = run_flow(
        G=G,
        flow=flow,
        iterations=N,
        step_size=s,
        alpha=alpha,
        initial_weights=initial_weights,
        keep_history=False,
    )

    best_tau, detected_size, core_result = choose_tau_by_size(
        G=G,
        evolved_weights=flow_result.weights,
        target_size=len(true_core),
        taus=taus,
    )

    detected_core = set(core_result.core_nodes)

    recovery = recovery_metrics(
        detected_core,
        true_core,
    )

    structure = evaluate_core(
        G,
        core_result.core_graph,
    )

    return {
        "flow": flow,
        "p_cp": p_cp,
        "seed": seed,
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "n_components": 0 if G.number_of_nodes() == 0 else
                        __import__("networkx").number_connected_components(G),
        "true_core_size": len(true_core),
        "tau": best_tau,
        "detected_core_size": detected_size,
        "size_gap": detected_size - len(true_core),
        "precision": recovery["precision"],
        "recall": recovery["recall"],
        "f1": recovery["f1"],
        "jaccard": recovery["jaccard_truth"],
        "r_d": structure["r_d"],
        "r_s": structure["r_s"],
        "runtime_sec": flow_result.runtime_seconds,
        "N": N,
        "s": s,
        "alpha": None if flow == "CF" else alpha,
    }


def summarize(df: pd.DataFrame):
    metrics = [
        "detected_core_size",
        "size_gap",
        "precision",
        "recall",
        "f1",
        "jaccard",
        "r_d",
        "r_s",
        "runtime_sec",
        "tau",
    ]

    grouped = (
        df.groupby(["flow", "p_cp"])[metrics]
        .agg(["mean", "std"])
        .reset_index()
    )

    # Flatten MultiIndex column names.
    grouped.columns = [
        "_".join(str(x) for x in col if str(x) != "")
        for col in grouped.columns
    ]

    return grouped


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Synthetic planted core-periphery recovery experiment "
            "for GEF, LEF, and CF."
        )
    )

    ap.add_argument("--n-core", type=int, default=60)
    ap.add_argument("--n-periphery", type=int, default=140)

    ap.add_argument("--p-cc", type=float, default=0.25)
    ap.add_argument(
        "--p-cps",
        default="0.02,0.04,0.06,0.08,0.10",
    )
    ap.add_argument("--p-pp", type=float, default=0.01)

    ap.add_argument("--N", type=int, default=10)
    ap.add_argument("--s", type=float, default=0.1)
    ap.add_argument("--alpha", type=float, default=0.1)

    ap.add_argument(
        "--seeds",
        default="2026,2027,2028,2029,2030,2031,2032,2033,2034,2035",
    )

    ap.add_argument(
        "--tau-min",
        type=float,
        default=0.50,
    )

    ap.add_argument(
        "--tau-max",
        type=float,
        default=0.99,
    )

    ap.add_argument(
        "--tau-step",
        type=float,
        default=0.001,
    )

    ap.add_argument(
        "--out",
        default="synthetic_core_recovery_raw.csv",
    )

    ap.add_argument(
        "--summary-out",
        default="synthetic_core_recovery_summary.csv",
    )

    args = ap.parse_args()

    p_cps = parse_float_list(args.p_cps)
    seeds = parse_int_list(args.seeds)

    taus = np.arange(
        args.tau_min,
        args.tau_max + args.tau_step / 2,
        args.tau_step,
    )

    taus = [
        round(float(tau), 10)
        for tau in taus
        if 0.0 <= tau < 1.0
    ]

    rows = []

    total = len(FLOWS) * len(p_cps) * len(seeds)
    count = 0

    for p_cp in p_cps:
        for seed in seeds:
            for flow in FLOWS:
                count += 1

                print(
                    "\n"
                    "==================================================\n"
                    f"[{count}/{total}] "
                    f"flow={flow}, p_cp={p_cp}, seed={seed}\n"
                    "==================================================",
                    flush=True,
                )

                row = run_one(
                    flow=flow,
                    p_cp=p_cp,
                    seed=seed,
                    n_core=args.n_core,
                    n_periphery=args.n_periphery,
                    p_cc=args.p_cc,
                    p_pp=args.p_pp,
                    N=args.N,
                    s=args.s,
                    alpha=args.alpha,
                    taus=taus,
                )

                rows.append(row)

                print(
                    f"[result] flow={flow}, "
                    f"p_cp={p_cp}, "
                    f"seed={seed}, "
                    f"tau={row['tau']:.2f}, "
                    f"K={row['detected_core_size']}, "
                    f"precision={row['precision']:.4f}, "
                    f"recall={row['recall']:.4f}, "
                    f"F1={row['f1']:.4f}, "
                    f"J={row['jaccard']:.4f}",
                    flush=True,
                )

    df = pd.DataFrame(rows)

    out = Path(args.out)
    df.to_csv(out, index=False)

    summary = summarize(df)

    summary_out = Path(args.summary_out)
    summary.to_csv(summary_out, index=False)

    print("\nRaw results:")
    print(df.to_string(index=False))

    print("\nSummary:")
    print(summary.to_string(index=False))

    print(f"\nSaved raw results: {out}")
    print(f"Saved summary: {summary_out}")


if __name__ == "__main__":
    main()