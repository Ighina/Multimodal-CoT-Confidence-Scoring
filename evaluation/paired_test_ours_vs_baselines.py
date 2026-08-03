"""Paired significance tests of our aggregated method against every baseline.

Reads the per-run CSVs produced by multi_seed_evaluate.py (run_*/majority_weighted/*.csv),
pairs runs by seed index, and runs a paired t-test (plus Wilcoxon signed-rank as a
robustness check) of `weighted_majority_aggregated_optimal` against each baseline,
for every subset and metric.

Star convention matches multi_seed_evaluate.py:
    *   p < 0.01
    **  p < 0.001
    *** p < 0.0001

Paper: run-level paired t-tests behind the significance markers of
Tables 3, 10, 17 and 20.

Usage:
    python paired_test_ours_vs_baselines.py \
        --results-root gemini-majority-multirun-logistic-reg \
        --condition majority_weighted
"""

import argparse
import csv
import os

import numpy as np
from scipy.stats import ttest_rel, wilcoxon

METRICS = ["AUROC", "ECE", "Pearson", "AURAC", "TPR@10%FPR"]
LOWER_IS_BETTER = {"ECE"}
OUR_METHOD = "weighted_majority_aggregated_optimal"


def significance_stars(p: float) -> str:
    if p < 0.0001:
        return "***"
    elif p < 0.001:
        return "**"
    elif p < 0.01:
        return "*"
    return ""


def parse_run_csv(path: str) -> dict[str, dict[str, float | None]]:
    out: dict[str, dict[str, float | None]] = {}
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            method = row["Method"]
            out[method] = {}
            for m in METRICS:
                raw = (row.get(m) or "").strip()
                try:
                    out[method][m] = float(raw)
                except ValueError:
                    out[method][m] = None
    return out


def discover_runs(results_root: str) -> list[str]:
    runs = [d for d in os.listdir(results_root) if d.startswith("run_")]
    return sorted(runs, key=lambda d: int(d.split("_")[1]))


def discover_subsets(results_root: str, runs: list[str], condition: str) -> list[str]:
    for run in runs:
        run_dir = os.path.join(results_root, run, condition)
        if os.path.isdir(run_dir):
            subsets = [f[:-4] for f in os.listdir(run_dir) if f.endswith(".csv")]
            subsets.sort(key=lambda s: (s == "Overall", s))
            return subsets
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", default="gemini-majority-multirun-logistic-reg")
    parser.add_argument("--condition", default="majority_weighted")
    parser.add_argument("--our-method", default=OUR_METHOD)
    parser.add_argument("--output", default=None,
                        help="Output CSV path (default: <results-root>/statistics/<condition>/paired_tests_<our-method>.csv)")
    args = parser.parse_args()

    runs = discover_runs(args.results_root)
    subsets = discover_subsets(args.results_root, runs, args.condition)
    if not runs or not subsets:
        raise SystemExit(f"No runs or subsets found under {args.results_root}/{args.condition}")

    print(f"Found {len(runs)} runs, subsets: {subsets}")

    # data[subset][method][metric] -> list of per-run values (None-padded, aligned by run)
    data: dict[str, dict[str, dict[str, list[float | None]]]] = {}
    for run in runs:
        for subset in subsets:
            parsed = parse_run_csv(os.path.join(args.results_root, run, args.condition, f"{subset}.csv"))
            for method, metrics in parsed.items():
                for metric, value in metrics.items():
                    data.setdefault(subset, {}).setdefault(method, {}).setdefault(metric, []).append(value)

    rows = []
    for subset in subsets:
        methods = data.get(subset, {})
        ours = methods.get(args.our_method)
        if ours is None:
            print(f"[warn] {args.our_method} missing in subset {subset}; skipping")
            continue
        baselines = [m for m in methods if m != args.our_method]
        for baseline in sorted(baselines):
            for metric in METRICS:
                ours_vals = ours.get(metric, [])
                base_vals = methods[baseline].get(metric, [])
                pairs = [(o, b) for o, b in zip(ours_vals, base_vals) if o is not None and b is not None]
                if len(pairs) < 3:
                    continue
                a = np.array([p[0] for p in pairs])
                b = np.array([p[1] for p in pairs])
                diff = a - b
                # Mean per-run relative difference (%) w.r.t. the baseline
                with np.errstate(divide="ignore", invalid="ignore"):
                    rel = np.where(b != 0, diff / np.abs(b) * 100.0, np.nan)
                pct_diff = float(np.nanmean(rel))
                _, p_t = ttest_rel(a, b)
                try:
                    _, p_w = wilcoxon(a, b)
                except ValueError:  # all differences zero
                    p_w = 1.0
                better = diff.mean() < 0 if metric in LOWER_IS_BETTER else diff.mean() > 0
                rows.append({
                    "Subset": subset,
                    "Baseline": baseline,
                    "Metric": metric,
                    "N_runs": len(pairs),
                    "Ours_mean": round(a.mean(), 4),
                    "Baseline_mean": round(b.mean(), 4),
                    "Mean_diff": round(diff.mean(), 4),
                    "Pct_diff": round(pct_diff, 2),
                    "Ours_better": better,
                    "p_ttest": f"{p_t:.2e}",
                    "p_wilcoxon": f"{p_w:.2e}",
                    "stars_ttest": significance_stars(p_t),
                })

    output = args.output or os.path.join(
        args.results_root, "statistics", args.condition, f"paired_tests_{args.our_method}.csv"
    )
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} comparisons to {output}\n")

    # Console summary for the headline comparison: ours vs self-consistency
    print(f"{'Subset':<18}{'Metric':<12}{'Ours':>8}{'SelfCons':>10}{'Diff':>9}{'Pct':>8}{'p (t-test)':>12}  Stars")
    for row in rows:
        if row["Baseline"] == "majority_vote_selfconsistency":
            print(f"{row['Subset']:<18}{row['Metric']:<12}{row['Ours_mean']:>8}{row['Baseline_mean']:>10}"
                  f"{row['Mean_diff']:>9}{row['Pct_diff']:>7}%{row['p_ttest']:>12}  {row['stars_ttest']}")


if __name__ == "__main__":
    main()
