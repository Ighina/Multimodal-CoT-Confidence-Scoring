#!/usr/bin/env python3
"""
Sweep --max-candidates over a list of values, running multi_seed_evaluate.py for
each, and plot AUROC / AURAC / ECE as a function of N for majority-vote
self-consistency and the aggregated weighted-majority score.

Paper: Appendix L, Figure 6 (effect of the number of sampled candidates,
MiniCPM-generated answers, N in {2..10}).

Usage:
    python sweep_candidates_evaluate.py -o results/n_sweep --n-list 2 3 4 5 6 \\
        --num-runs 10 -- [any final_evaluate.py arguments]

    Wrapper arguments go BEFORE the ``--`` separator; everything after it is
    forwarded verbatim to final_evaluate.py (via multi_seed_evaluate.py), with
    --max-candidates overridden per sweep value.

Output structure:
    <output-dir>/
    ├── n_2/            (a full multi_seed_evaluate.py output root)
    ├── n_3/  ...
    ├── sweep_summary.csv
    ├── auroc_vs_n.png
    ├── aurac_vs_n.png
    └── ece_vs_n.png
"""

import argparse
import csv
import math
import os
import re
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PLOT_METRICS = ["AUROC", "AURAC", "ECE"]
METHODS = {
    "majority_vote_selfconsistency": {
        "label": "Majority Vote (Self-Consistency)",
        "color": "#2a78d6",
        "marker": "o",
        "linestyle": "-",
    },
    "weighted_majority_aggregated_optimal": {
        "label": r"$\mathcal{C}_{\mathrm{chain}}$ (weighted majority)",
        "color": "#008300",
        "marker": "s",
        "linestyle": "--",
    },
}
VALUE_RE = re.compile(r"([-\d.]+)\s*±\s*([\d.]+)")


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("-o", "--output-dir", required=True,
                   help="General directory holding one multi-seed output root per N plus the plots.")
    p.add_argument("--n-list", type=int, nargs="+", required=True,
                   help="Values of --max-candidates to sweep, e.g. --n-list 2 3 4 5 6.")
    p.add_argument("-n", "--num-runs", type=int, default=10,
                   help="Repetitions per N, forwarded to multi_seed_evaluate.py (default: 10).")
    p.add_argument("--base-seed", type=int, default=42,
                   help="Base seed forwarded to multi_seed_evaluate.py (default: 42).")
    p.add_argument("--setting", default="majority_weighted", choices=["majority_weighted", "no_majority"],
                   help="Which statistics folder to read (default: majority_weighted).")
    p.add_argument("--subset", default="Overall",
                   help="Which subset CSV to plot (default: Overall).")
    p.add_argument("--force", action="store_true",
                   help="Re-run evaluations even if statistics for an N already exist.")
    p.add_argument("--plots-only", action="store_true",
                   help="Skip all evaluation runs and just regenerate plots from existing statistics.")

    if "--" in sys.argv:
        idx = sys.argv.index("--")
        wrapper_argv, forward_argv = sys.argv[1:idx], sys.argv[idx + 1:]
    else:
        wrapper_argv, forward_argv = sys.argv[1:], []
    return p.parse_args(wrapper_argv), forward_argv


def strip_max_candidates(forward_argv: list[str]) -> list[str]:
    """Remove any user-supplied --max-candidates; the sweep manages it."""
    cleaned, skip = [], False
    for tok in forward_argv:
        if skip:
            skip = False
            continue
        if tok == "--max-candidates":
            skip = True
            continue
        if tok.startswith("--max-candidates="):
            continue
        cleaned.append(tok)
    return cleaned


def stats_csv(output_dir: str, n: int, setting: str, subset: str) -> str:
    return os.path.join(output_dir, f"n_{n}", "statistics", setting, f"{subset}.csv")


def run_one(n: int, args: argparse.Namespace, forward_argv: list[str]) -> None:
    root = os.path.join(args.output_dir, f"n_{n}")
    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "multi_seed_evaluate.py"),
        "-n", str(args.num_runs),
        "--base-seed", str(args.base_seed),
        "-o", root,
        "--",
        *forward_argv,
        "--max-candidates", str(n),
    ]
    print(f"\n=== max-candidates = {n} ===\n  {' '.join(cmd)}\n")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise SystemExit(f"multi_seed_evaluate.py failed for --max-candidates {n} "
                         f"(exit code {result.returncode})")


def read_metrics(path: str) -> dict[str, dict[str, tuple[float, float]]]:
    """Return {method: {metric: (mean, std)}} for the methods of interest."""
    out: dict[str, dict[str, tuple[float, float]]] = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            method = row["Method"]
            if method not in METHODS:
                continue
            out[method] = {}
            for metric in PLOT_METRICS:
                m = VALUE_RE.match(row.get(metric, "").strip())
                if m:
                    out[method][metric] = (float(m.group(1)), float(m.group(2)))
    return out


def plot_metric(metric: str, data: dict, n_list: list[int], args: argparse.Namespace) -> str:
    fig, ax = plt.subplots(figsize=(5.2, 3.4), dpi=200)
    for method, style in METHODS.items():
        points = [(n, *data[n][method][metric]) for n in n_list
                  if method in data.get(n, {}) and metric in data[n][method]]
        if not points:
            continue
        xs, means, stds = zip(*points)
        # Error bars show +/- 1 standard error of the mean over the num_runs
        # repetitions (std / sqrt(n)), not the raw run-to-run std.
        sems = [s / math.sqrt(args.num_runs) for s in stds]
        ax.errorbar(
            xs, means, yerr=sems,
            label=style["label"], color=style["color"],
            marker=style["marker"], markersize=6, linestyle=style["linestyle"],
            linewidth=2, capsize=3, elinewidth=1,
        )
    direction = "lower" if metric == "ECE" else "higher"
    ax.set_xlabel("Number of sampled candidates $N$")
    ax.set_ylabel(f"{metric} ({direction} is better)")
    ax.set_title(f"{args.subset} {metric} vs. $N$ ({args.setting})", fontsize=11)
    ax.set_xticks(n_list)
    ax.grid(axis="y", color="#e3e2d9", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    out = os.path.join(args.output_dir, f"{metric.lower()}_vs_n.png")
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> None:
    args, forward_argv = parse_args()
    forward_argv = strip_max_candidates(forward_argv)
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Run the evaluations (unless cached or plots-only)
    if not args.plots_only:
        for n in args.n_list:
            if not args.force and os.path.exists(stats_csv(args.output_dir, n, args.setting, args.subset)):
                print(f"[skip] statistics for N={n} already exist "
                      f"(use --force to re-run)")
                continue
            run_one(n, args, forward_argv)

    # 2. Collect statistics
    data: dict[int, dict] = {}
    for n in args.n_list:
        path = stats_csv(args.output_dir, n, args.setting, args.subset)
        if not os.path.exists(path):
            print(f"[warn] missing statistics for N={n} ({path}); excluded from plots")
            continue
        data[n] = read_metrics(path)

    if not data:
        raise SystemExit("No statistics found for any N; nothing to plot.")

    # 3. Summary CSV
    summary_path = os.path.join(args.output_dir, "sweep_summary.csv")
    with open(summary_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["N", "Method"] + [f"{m}_{s}" for m in PLOT_METRICS for s in ("mean", "std")])
        for n in sorted(data):
            for method in METHODS:
                if method not in data[n]:
                    continue
                row = [n, method]
                for metric in PLOT_METRICS:
                    mean, std = data[n][method].get(metric, (float("nan"), float("nan")))
                    row += [mean, std]
                writer.writerow(row)
    print(f"\nSummary written to {summary_path}")

    # 4. Plots
    for metric in PLOT_METRICS:
        out = plot_metric(metric, data, sorted(data), args)
        print(f"Plot written to {out}")


if __name__ == "__main__":
    main()
