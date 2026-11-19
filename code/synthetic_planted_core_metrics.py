"""Supplement the existing synthetic recovery CSV with planted-core r_d and r_s.

Run from the directory containing core_flows.py and the original recovery CSV.
No flow evolution is rerun. The graph is regenerated with the same seed and
model parameters; keep these identical to the original experiment.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx

from core_flows import planted_core_periphery_graph, evaluate_core


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', default='synthetic_core_recovery_raw.csv')
    ap.add_argument('--n-core', type=int, default=60)
    ap.add_argument('--n-periphery', type=int, default=140)
    ap.add_argument('--p-cc', type=float, default=0.25)
    ap.add_argument('--p-pp', type=float, default=0.01)
    ap.add_argument('--out', default='synthetic_core_recovery_with_truth.csv')
    ap.add_argument('--summary-out', default='synthetic_core_truth_comparison.csv')
    ap.add_argument('--truth-out', default='synthetic_planted_core_raw.csv')
    args = ap.parse_args()

    raw = pd.read_csv(args.raw)
    required = {'flow', 'p_cp', 'seed', 'r_d', 'r_s', 'f1', 'detected_core_size'}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f'Missing columns in recovery CSV: {sorted(missing)}')
    if raw.duplicated(['flow', 'p_cp', 'seed']).any():
        raise ValueError('Expected one result per (flow, p_cp, seed).')

    truth_rows = []
    for p_cp, seed in raw[['p_cp', 'seed']].drop_duplicates().itertuples(index=False, name=None):
        G, true_core = planted_core_periphery_graph(
            n_core=args.n_core,
            n_periphery=args.n_periphery,
            p_cc=args.p_cc,
            p_cp=float(p_cp),
            p_pp=args.p_pp,
            seed=int(seed),
        )
        core_graph = G.subgraph(true_core).copy()
        values = evaluate_core(G, core_graph)
        residual = G.subgraph(set(G.nodes()) - set(true_core))
        truth_rows.append({
            'p_cp': float(p_cp), 'seed': int(seed),
            'planted_core_size': len(true_core),
            'planted_core_edges': core_graph.number_of_edges(),
            'planted_r_d': values['r_d'],
            'planted_r_s': values['r_s'],
            'planted_residual_edges': residual.number_of_edges(),
            'planted_residual_components': nx.number_connected_components(residual),
            'n_nodes_regenerated': G.number_of_nodes(),
            'n_edges_regenerated': G.number_of_edges(),
        })
    truth = pd.DataFrame(truth_rows)
    truth.to_csv(args.truth_out, index=False)
    merged = raw.merge(truth, on=['p_cp', 'seed'], how='left', validate='many_to_one')
    if merged['planted_r_d'].isna().any():
        raise RuntimeError('Some planted metrics could not be merged.')
    if 'n_edges' in merged.columns:
        mismatch = merged['n_edges'] != merged['n_edges_regenerated']
        if mismatch.any():
            raise RuntimeError('Regenerated graph edge counts disagree with original CSV. Check model parameters and core_flows version.')
    merged['delta_r_d_vs_planted'] = merged['r_d'] - merged['planted_r_d']
    merged['delta_r_s_vs_planted'] = merged['r_s'] - merged['planted_r_s']
    merged.to_csv(args.out, index=False)

    # Means and standard deviations across independent graph realizations.
    metrics = ['f1', 'detected_core_size', 'r_d', 'r_s',
               'planted_r_d', 'planted_r_s',
               'delta_r_d_vs_planted', 'delta_r_s_vs_planted']
    summary = merged.groupby(['p_cp', 'flow'], as_index=False)[metrics].agg(['mean', 'std'])
    summary.columns = ['_'.join(str(x) for x in col if x != '') for col in summary.columns]
    summary.to_csv(args.summary_out, index=False)

    print('\nPlanted-core summary (one row per p_cp):')
    print(truth.groupby('p_cp')[['planted_r_d', 'planted_r_s']].agg(['mean', 'std']).to_string())
    print('\nFlow comparison summary:')
    print(summary.to_string(index=False))
    print(f'\nSaved: {args.truth_out}, {args.out}, {args.summary_out}')
    print('Note: NaN planted_r_s means the residual graph has no connected vertex pairs (xi=0).')

if __name__ == '__main__':
    main()
