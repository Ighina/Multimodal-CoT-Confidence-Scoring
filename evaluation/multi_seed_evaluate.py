#!/usr/bin/env python3
"""
Run final_evaluate.py multiple times with different random seeds and aggregate
the per-run CSV results into mean ± std statistics.

Paper: the "100 randomised runs" protocol behind every main table
(seeds 42..141; means reported in Tables 1, 2, 9 and 15-20).

Usage:
    python multi_seed_evaluate.py --num-runs 10 --output-root results/multi_run \\
        -- [any final_evaluate.py arguments]

    The wrapper's own arguments (--num-runs, --base-seed, --output-root) go BEFORE
    the ``--`` separator.  Everything after ``--`` is forwarded verbatim to
    final_evaluate.py, with the following automatic overrides applied per run:
      * --output-dir          → <output-root>/run_<i>
      * --output-eval         → <output-root>/run_<i>/eval.json
      * --output-eval-majority → <output-root>/run_<i>/eval_majority.json
      * --split-seed          → base_seed + i
      * --random-answer-seed  → base_seed + i

    If the user already passed any of these in the forwarded args they are
    silently replaced by the per-run values.

Examples:
    # 5 runs with gemini data, seeding from 100
    python multi_seed_evaluate.py -n 5 --base-seed 100 -o results/gemini_5runs \\
        -- --input-json gemini_cots_reformatted.json

    # 20 runs, majority-vote selection, gridded coarsely for speed
    python multi_seed_evaluate.py -n 20 -o results/coarse_20 \\
        -- --answer-selection-mode majority_vote --grid-step 0.2

Output structure:
    <output-root>/
    ├── run_0/
    │   ├── no_majority/       (Audio.csv, MC.csv, MO.csv, Visual.csv, Overall.csv,
    │   │                        metadata.json)
    │   └── majority_weighted/ (same)
    ├── run_1/  ...
    ├── ...
    └── statistics/
        ├── no_majority/
        │   ├── Audio.csv      (mean ± std across runs)
        │   ├── MC.csv
        │   ├── MO.csv
        │   ├── Visual.csv
        │   └── Overall.csv
        ├── majority_weighted/
        │   └── (same 5 files)
        └── metadata.json      (run parameters + per-run seed list)
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import ttest_rel

# ── Constants ──────────────────────────────────────────────────────────────────
SETTINGS = ["no_majority", "majority_weighted"]
METRICS = ["AUROC", "ECE", "Pearson", "AURAC", "TPR@10%FPR"]

# Arguments that the wrapper manages per-run — strip them from forwarded args
MANAGED_ARGS = {
    "--output-dir",
    "--output-eval",
    "--output-eval-majority",
    "--split-seed",
    "--random-answer-seed",
}


# ── Argument parsing ───────────────────────────────────────────────────────────
def parse_wrapper_args() -> tuple[argparse.Namespace, list[str]]:
    p = argparse.ArgumentParser(
        description="Multi-seed wrapper around final_evaluate.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "-n",
        "--num-runs",
        type=int,
        required=True,
        help="Number of independent repetitions with different seeds.",
    )
    p.add_argument(
        "--base-seed",
        type=int,
        default=42,
        help="Starting seed; run i uses base_seed + i for both --split-seed "
        "and --random-answer-seed (default: 42).",
    )
    p.add_argument(
        "-o",
        "--output-root",
        required=True,
        help="Root directory where per-run folders and the final statistics/ "
        "folder will be created.",
    )

    # Separate wrapper args from forwarded args using the standard "--" convention
    if "--" in sys.argv:
        idx = sys.argv.index("--")
        wrapper_argv = sys.argv[1:idx]
        forward_argv = sys.argv[idx + 1 :]
    else:
        wrapper_argv = sys.argv[1:]
        forward_argv = []

    args = p.parse_args(wrapper_argv)
    return args, forward_argv


def strip_managed_args(forward_argv: list[str]) -> list[str]:
    """Remove any occurrences of --managed-arg <value> from the forwarded list."""
    cleaned: list[str] = []
    skip_next = False
    for i, token in enumerate(forward_argv):
        if skip_next:
            skip_next = False
            continue
        if token in MANAGED_ARGS:
            # This token is a managed flag — skip it *and* its value
            skip_next = True
            continue
        if any(token.startswith(arg + "=") for arg in MANAGED_ARGS):
            # Handles --output-dir=foo style
            continue
        cleaned.append(token)
    return cleaned


# ── Run a single evaluation ────────────────────────────────────────────────────
def run_single(
    run_idx: int,
    seed: int,
    output_root: str,
    forward_argv: list[str],
) -> subprocess.CompletedProcess:
    """Execute final_evaluate.py for one seed; return the CompletedProcess."""
    run_dir = os.path.join(output_root, f"run_{run_idx}")
    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "final_evaluate.py"),
        *forward_argv,
        "--output-dir",
        run_dir,
        "--output-eval",
        os.path.join(run_dir, "eval.json"),
        "--output-eval-majority",
        os.path.join(run_dir, "eval_majority.json"),
        "--split-seed",
        str(seed),
        "--random-answer-seed",
        str(seed),
    ]
    print(f"\n{'=' * 70}")
    print(f"Run {run_idx + 1}  (seed={seed})")
    print(f"{'=' * 70}")
    print(f"  {' '.join(cmd)}\n")
    return subprocess.run(cmd)


# ── CSV parsing ─────────────────────────────────────────────────────────────────
def parse_results_csv(path: str) -> dict[str, dict[str, float | None]]:
    """Read a single CSV and return {method: {metric: value_or_None}}."""
    out: dict[str, dict[str, float | None]] = {}
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            method = row["Method"]
            out[method] = {}
            for m in METRICS:
                raw = row.get(m, "N/A").strip()
                if raw == "N/A" or raw == "":
                    out[method][m] = None
                else:
                    try:
                        out[method][m] = float(raw)
                    except ValueError:
                        out[method][m] = None
    return out


# ── Aggregation ─────────────────────────────────────────────────────────────────
def _significance_stars(p: float) -> str:
    """Return asterisk marker for a p-value."""
    if p < 0.0001:
        return " ***"
    elif p < 0.001:
        return " **"
    elif p < 0.01:
        return " *"
    return ""


def _best_vs_runner_up_pvalue(
    methods_data: dict, metric: str, higher_is_better: bool
) -> tuple[str | None, str, float]:
    """Paired t-test between the best and second-best method on *metric*.

    Returns (best_method_name, significance_stars, p_value).
    If fewer than 2 methods or fewer than 3 runs, returns (None, "", 1.0).
    """
    # Build {method: [values across runs]} filtering out Nones
    method_vals = {}
    for method, metrics in methods_data.items():
        vals = [v for v in metrics[metric] if v is not None]
        if len(vals) >= 2:
            method_vals[method] = np.array(vals, dtype=float)

    if len(method_vals) < 2:
        return None, "", 1.0

    # Rank by mean
    means = {m: float(arr.mean()) for m, arr in method_vals.items()}
    if higher_is_better:
        ranked = sorted(means.items(), key=lambda x: x[1], reverse=True)
    else:
        ranked = sorted(means.items(), key=lambda x: x[1])

    best_name, _best_mean = ranked[0]
    runner_up_name, _runner_mean = ranked[1]

    # Paired t-test on the aligned runs (rows where both have a value)
    best_arr = method_vals[best_name]
    runner_arr = method_vals[runner_up_name]
    # Trim to min length in case of mismatched run counts
    n = min(len(best_arr), len(runner_arr))
    if n < 3:
        return best_name, "", 1.0

    _, p = ttest_rel(best_arr[:n], runner_arr[:n])
    stars = _significance_stars(p)
    return best_name, stars, float(p)


def _discover_subsets(output_root: str, num_completed: int) -> list[str]:
    """Auto-detect the subsets present in the first completed run, with 'Overall' last."""
    for run_idx in range(num_completed):
        run_dir = os.path.join(output_root, f"run_{run_idx}", "no_majority")
        if not os.path.isdir(run_dir):
            continue
        csv_files = [
            f for f in os.listdir(run_dir) if f.endswith(".csv")
        ]
        if not csv_files:
            continue
        subsets = sorted(
            [os.path.splitext(f)[0] for f in csv_files if os.path.splitext(f)[0] != "Overall"]
        )
        if "Overall" in [os.path.splitext(f)[0] for f in csv_files]:
            subsets.append("Overall")
        print(f"Auto-detected subsets: {subsets}")
        return subsets
    # Fallback
    return ["Overall"]


def aggregate_runs(output_root: str, num_completed: int) -> None:
    """Collect all per-run CSVs and write mean ± std statistics."""
    stats_dir = os.path.join(output_root, "statistics")
    os.makedirs(stats_dir, exist_ok=True)

    subsets = _discover_subsets(output_root, num_completed)

    # ── Gather all values ──────────────────────────────────────────────────
    # collected[setting][subset][method][metric] = list of values across runs
    collected: dict = {
        setting: {
            subset: defaultdict(lambda: defaultdict(list))
            for subset in subsets
        }
        for setting in SETTINGS
    }

    for run_idx in range(num_completed):
        run_dir = os.path.join(output_root, f"run_{run_idx}")
        for setting in SETTINGS:
            for subset in subsets:
                csv_path = os.path.join(run_dir, setting, f"{subset}.csv")
                parsed = parse_results_csv(csv_path)
                for method, metrics in parsed.items():
                    for metric, value in metrics.items():
                        collected[setting][subset][method][metric].append(value)

    # ── Write combined CSVs ────────────────────────────────────────────────
    for setting in SETTINGS:
        setting_stats_dir = os.path.join(stats_dir, setting)
        os.makedirs(setting_stats_dir, exist_ok=True)

        for subset in subsets:
            methods_data = collected[setting][subset]
            if not methods_data:
                # No data for this subset — write header-only CSV
                path = os.path.join(setting_stats_dir, f"{subset}.csv")
                with open(path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(["Method"] + METRICS)
                continue

            # Compute best-vs-runner-up significance per metric
            sig_stars: dict[str, dict[str, str]] = defaultdict(dict)
            for metric in METRICS:
                higher = metric != "ECE"
                best_name, stars, p = _best_vs_runner_up_pvalue(
                    methods_data, metric, higher_is_better=higher
                )
                if best_name is not None and stars:
                    sig_stars[best_name][metric] = stars
                    print(
                        f"  [{setting}/{subset}] {metric}: best={best_name} "
                        f"p={p:.2e}{stars}"
                    )

            path = os.path.join(setting_stats_dir, f"{subset}.csv")
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["Method"] + METRICS)
                for method in sorted(methods_data.keys()):
                    row = [method]
                    for metric in METRICS:
                        vals = [
                            v
                            for v in methods_data[method][metric]
                            if v is not None
                        ]
                        if len(vals) >= 2:
                            mean = sum(vals) / len(vals)
                            # sample std
                            variance = sum((v - mean) ** 2 for v in vals) / (
                                len(vals) - 1
                            )
                            std = variance**0.5
                            cell = f"{mean:.4f} ± {std:.4f}"
                        elif len(vals) == 1:
                            cell = f"{vals[0]:.4f} ± N/A"
                        else:
                            cell = "N/A"
                        # Append significance stars if this method is best
                        cell += sig_stars.get(method, {}).get(metric, "")
                        row.append(cell)
                    writer.writerow(row)
            print(f"  Wrote {path}")

    # ── Write aggregate metadata ───────────────────────────────────────────
    meta = {
        "description": "Multi-seed aggregated statistics from final_evaluate.py",
        "num_runs_completed": num_completed,
        "subsets": subsets,
        "metrics": METRICS,
        "columns_format": "mean ± std (sample standard deviation across runs)",
    }
    meta_path = os.path.join(stats_dir, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"  Wrote {meta_path}")


# ── Main ────────────────────────────────────────────────────────────────────────
def main() -> None:
    args, forward_argv = parse_wrapper_args()

    if not forward_argv:
        print(
            "Warning: no arguments forwarded to final_evaluate.py. "
            "Use '--' to pass them, e.g.:\n"
            "  python multi_seed_evaluate.py -n 5 -o results -- --input-json data.json\n"
            "Running with final_evaluate.py defaults.",
            file=sys.stderr,
        )

    forward_argv = strip_managed_args(forward_argv)
    os.makedirs(args.output_root, exist_ok=True)

    # ── Run evaluations ────────────────────────────────────────────────────
    failed_runs: list[int] = []
    for i in range(args.num_runs):
        seed = args.base_seed + i
        result = run_single(i, seed, args.output_root, forward_argv)
        if result.returncode != 0:
            print(f"\n[ERROR] Run {i} (seed={seed}) failed with code {result.returncode}")
            failed_runs.append(i)

    num_completed = args.num_runs - len(failed_runs)

    if num_completed == 0:
        print("\nAll runs failed — cannot compute statistics.", file=sys.stderr)
        sys.exit(1)

    if failed_runs:
        print(
            f"\n{len(failed_runs)} run(s) failed (indices: {failed_runs}). "
            f"Statistics computed from the {num_completed} successful runs."
        )

    # ── Aggregate ──────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("Aggregating statistics across runs...")
    print(f"{'=' * 70}")
    aggregate_runs(args.output_root, args.num_runs)

    print(f"\nDone. Results in {args.output_root}/")
    print(f"  Per-run folders:  {args.output_root}/run_*/")
    print(f"  Statistics:       {args.output_root}/statistics/")


if __name__ == "__main__":
    main()
