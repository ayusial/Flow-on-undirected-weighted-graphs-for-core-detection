# Core Detection via Global Entropy, Local Entropy and Cohesion Flows on Weighted Graphs

Research code accompanying the paper **“Core Detection via Global Entropy, Local Entropy and Cohesion Flows on Weighted Graphs.”**

This repository contains implementations of global entropy flow (GEF), local entropy flow (LEF), cohesion flow (CF), and the core extraction procedure, together with scripts used for numerical experiments. The method evolves edge weights, removes high-weight edges, collects the non-isolated vertices, induces their subgraph **in the original graph**, and returns its largest connected component.

**Scope:** The repository does not include the benchmark network files or the original third-party Ricci-flow code. Download those separately (and observe their respective licenses). The example commands below show how to run the supplied scripts; exact reproduction of published numerical tables also requires the same input graph preprocessing and parameters.

## Files

| File | Purpose |
| --- | --- |
| `core_flows.py` | GEF, LEF, CF, synchronous iteration, pure/data-driven core extraction, `r_d`/`r_s`, synthetic graph generation. Two obsolete reference-ratio extraction functions have been removed from this release. |
| `sensitivity_experiments.py` | Iteration-number, step-size, and initial-weight sensitivity experiments; also offers fixed-total-time checks. |
| `synthetic_core_recovery.py` | Planted core–periphery simulations, size-matched `tau` selection, and recovery statistics. |
| `synthetic_planted_core_metrics.py` | Add structural metrics of the planted core to the synthetic experiment CSV. |
| `cf_gef_sequential_experiment.py` | Sequential CF then GEF, passing CF's final weights directly to GEF. |
| `matched_baselines_final.py` | Matched-size PageRank, degree, betweenness, and closeness baseline calculations. |
| `ricci_matched_baseline.py` | Matched-size Ricci-flow baseline, requiring an **external original Ricci implementation**. |

Some scripts include optional exploratory modes, but the scripts listed here cover the main computational tasks in the paper. In particular, the older `experiment.py` using reference-ratio extraction is **not** part of this release.

## Installation

Python 3.10 or newer is recommended. From this directory:

```bash
python -m venv .venv
# macOS / Linux:
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The optional Ricci baseline may require `cvxpy` and other dependencies of the original Ricci implementation. It is intentionally not included in the base requirements because that external implementation is not bundled.

## Input graphs

Place your own copies of **Cora**, **Citeseer**, and **Bio-CE-HT** in `data/`. These benchmark files are **not distributed here**. Verify that the graph topology, vertex/edge counts, self-loop handling, and preprocessing match the paper before comparing numerical values.

The flow experiment scripts support graph inputs such as `.gexf`, `.graphml`, and whitespace-separated edge lists, but **supported formats vary by script**. For the cross-script examples below, `.gexf` is the simplest common choice. Graphs are treated as simple undirected graphs; self-loops are removed. The benchmark flow experiments start from unit edge weights, unless a sensitivity experiment specifies otherwise. All connected components with edges are evolved separately, then combined for global edge ranking.

Example path below: `data/cora.gexf` (replace with your actual filename).

## Main experiment parameters (paper Table 2)

The main experiments use step size `s=0.1`. A dash means the flow does not require `alpha`.

| Dataset | Flow | Iterations `N` | `alpha` | Removal ratio `tau` |
| --- | --- | ---: | ---: | ---: |
| Cora | GEF | 10 | 0.1 | 0.78 |
| Cora | LEF | 20 | 0.1 | 0.76 |
| Cora | CF | 10 | — | 0.61 |
| Citeseer | GEF | 10 | 0.1 | 0.71 |
| Citeseer | LEF | 10 | 0.1 | 0.72 |
| Citeseer | CF | 10 | — | 0.65 |
| Bio-CE-HT | GEF | 20 | 0.2 | 0.45 |
| Bio-CE-HT | LEF | 10 | 0.5 | 0.50 |
| Bio-CE-HT | CF | 5 | — | 0.40 |

The main single-flow settings can be evaluated using the `N` mode of the sensitivity script with a **single** iteration value. For instance:

```bash
python sensitivity_experiments.py --graph data/cora.gexf --flow GEF --mode N --Ns 10 --s 0.1 --alpha 0.1 --tau 0.78 --out results/cora_gef.csv
python sensitivity_experiments.py --graph data/cora.gexf --flow LEF --mode N --Ns 20 --s 0.1 --alpha 0.1 --tau 0.76 --out results/cora_lef.csv
python sensitivity_experiments.py --graph data/cora.gexf --flow CF  --mode N --Ns 10 --s 0.1 --tau 0.61 --out results/cora_cf.csv
```

Repeat with the parameters in the table for the other datasets. The single-run CSV also includes core size, `r_d`, and `r_s`.

The number of edges removed by the data-driven extraction is `floor(tau * |E|)`. Ties in evolved weights preserve the input graph's edge iteration order (no independent label-based tie-break rule is applied).

## Sensitivity experiments

Iteration-count sensitivity (choose `--Ns` and the appropriate base parameters):

```bash
python sensitivity_experiments.py --graph data/cora.gexf --flow GEF --mode N --Ns 1,5,10,20,30,40,50 --s 0.1 --alpha 0.1 --tau 0.78 --out results/cora_gef_N.csv
```

Step-size sensitivity:

```bash
python sensitivity_experiments.py --graph data/citeseer.gexf --flow LEF --mode s --N 10 --ss 0.025,0.05,0.1,0.2,0.3,0.5 --alpha 0.1 --tau 0.72 --out results/citeseer_lef_s.csv
```

Initial-weight perturbations: sample each edge independently from `Unif(1-epsilon, 1+epsilon)` using `--weight-low` and `--weight-high` (with `0<epsilon<1`). For instance, `epsilon=0.1`:

```bash
python sensitivity_experiments.py --graph data/citeseer.gexf --flow LEF --mode weights --N 10 --s 0.1 --alpha 0.1 --tau 0.72 --runs 5 --seed 2026 --weight-low 0.9 --weight-high 1.1 --out results/citeseer_lef_weights_eps01.csv
```

Repeat for the other perturbation amplitudes and flows as needed. The `--runs` option controls the number of independently sampled weight vectors; ensure this agrees with the published experimental protocol. Cache files can be disabled with `--no-cache`.

## Synthetic core–periphery recovery

The paper uses 60 planted core vertices, 140 periphery vertices, `p_cc=0.25`, `p_pp=0.01`, and `p_cp` in `{0.02,0.04,0.06,0.08,0.10}`. The script defaults to ten independent seeds (`2026`–`2035`) and `N=10`, `s=0.1`, `alpha=0.1` for entropy flows. It scans `tau` to approximately match the planted core size.

```bash
python synthetic_core_recovery.py --out results/synthetic_core_recovery_raw.csv --summary-out results/synthetic_core_recovery_summary.csv
python synthetic_planted_core_metrics.py --raw results/synthetic_core_recovery_raw.csv --out results/synthetic_core_recovery_with_truth.csv --summary-out results/synthetic_core_truth_comparison.csv --truth-out results/synthetic_planted_core_raw.csv
```

The second command regenerates synthetic graphs from the recorded seeds/model parameters and adds planted-core structural metrics; its inputs must match those used for the first command.

## Sequential CF–GEF

CF evolves the unit initial weights for `N_CF` iterations. GEF then evolves the **CF output weights without resetting them** for `N_GEF` iterations. Example:

```bash
python cf_gef_sequential_experiment.py --graph data/cora.gexf --pairs 5:5 --taus 0.78 --s-cf 0.1 --s-gef 0.1 --alpha 0.1 --out results/cora_cf_gef.csv
```

`--pairs` accepts comma-separated `N_CF:N_GEF` pairs; `--taus` accepts comma-separated deletion ratios. **The example values illustrate the interface, not necessarily the final values of Appendix D**. Use the paper's Appendix D to reproduce the corresponding published CF–GEF results.

## Centrality-based matched baselines

`--targets` accepts the desired reference core vertex counts as `FLOW:K` entries. For instance, the Cora reference sizes in Table 3 are GEF 1190, LEF 1162, CF 1245:

```bash
python matched_baselines_final.py --graph data/cora.gexf --targets GEF:1190,LEF:1162,CF:1245 --out results/cora_matched_centralities.csv
```

This script selects the highest-ranked vertices by each centrality and retains the largest connected component of their induced subgraph. It is **not** the separate connectivity-constrained greedy baseline function also present in the core library.

## Ricci-flow matched baseline (external implementation required)

`ricci_matched_baseline.py` loads the original author's Ricci-flow script dynamically, expecting a `RhoCurvature` class. That external script is **not included** in this repository. Obtain it from its original author/source under the appropriate terms before running the baseline.

```bash
python ricci_matched_baseline.py --graph data/cora.gexf --ricci-script "/path/to/original_Ricci_flow.py" --targets 1190,1162,1245 --labels GEF,LEF,CF --out results/cora_ricci_matched.csv
```

Install the original script's required packages separately (potentially including `cvxpy`). Reproduction also depends on the Ricci flow settings and model implementation.

## Reproducibility notes

- **Core extraction:** The published flow method uses `extract_core_data_driven()` (Method A), not the older reference-ratio method that supplements isolated vertices. The two obsolete extraction functions have been removed from this copy of `core_flows.py`.
- **Connected components:** The core flow implementation processes all nontrivial connected components separately without discarding smaller components.
- **Ranking ties:** Tied edge weights preserve input edge order. Keep the source graph and its serialization consistent if exact ties may affect membership.
- **Randomness and caching:** Seed values and cache paths affect reproducibility. Existing `.pkl` cache files can be removed or bypassed with `--no-cache` in scripts that support it.
- **Baseline dependencies:** Ricci-flow results require the separately sourced original implementation; this seven-script package alone is **not sufficient** to regenerate that baseline.
- **Outputs:** Generated CSVs and caches are ignored by `.gitignore` by default. To publish selected final small CSVs, explicitly add them or adjust the ignore rules.

## Citation and license

If you use this code, please cite the associated paper and the original GEF/LEF/CF papers. Add the final bibliographic references or BibTeX information once available.

**License not yet selected.** No license is granted for third-party Ricci-flow implementations by this repository. The repository authors should select an appropriate license for their own code and verify any third-party reuse permissions before publishing a `LICENSE` file.
