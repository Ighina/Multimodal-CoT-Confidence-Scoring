#!/usr/bin/env python3
"""
Best-of-our-methods vs Self-Consistency, item-level.

For every dataset / subset / metric, selects the best of OUR methods
(the five embedding scores + the aggregated score, weighted-majority variants)
in two ways:
  * val-selected : best on the VALIDATION split (honest, pre-registerable);
  * test-selected: best on the TEST split (oracle upper bound, selection-biased).
and reports that method's paired-bootstrap delta vs self-consistency on the
test split (looked up from bootstrap-results/<ds>/paired_deltas.csv, which was
computed with shared resample indices, B=10k).

Run from the working root after bootstrap_significance.py has produced
bootstrap-results/.

Paper: supports the "best label-free single score" discussion of
Section 5.2 and the Limitations paragraph on discrimination margins.
"""

import csv
import json
import os

import numpy as np

import bootstrap_significance as bs

DATASETS = {
    "gemini": dict(
        input_json="gemini_cots_reformatted.json",
        cluster_cache="gemini-majority-vote-clusters_top6.json",
        run_dir="gemini-majority-multirun-logistic-reg/run_0",
        results="bootstrap-results/gemini",
    ),
    "lco-gemini": dict(
        input_json="lco_gemini_cots_reformatted.json",
        cluster_cache="gemini-majority-vote-clusters_top6.json",
        run_dir="lco-gemini-majority-multirun-logistic-reg/run_0",
        results="bootstrap-results/lco-gemini",
    ),
    "minicpm": dict(
        input_json="minicpm_cots_with_correctness_and_my_scores.json",
        cluster_cache="minicpm-majority-vote-clusters_top6.json",
        run_dir="minicpm-majority-multirun-logistic-reg/run_0",
        results="bootstrap-results/minicpm",
    ),
}
METRIC_KEYS = ["AUROC", "ECE", "AURAC"]  # ECE lower is better


def subset_metric(items, method, subset, uncertainty):
    d = items[method]
    cats = np.array(d["category"])
    mask = np.ones(len(cats), bool) if subset == "Overall" else cats == subset
    y_true = np.array(d["y_true"])[mask]
    y_prob = bs.minmax_transform(np.array(d["y_prob"])[mask], uncertainty)
    if len(np.unique(y_true)) < 2:
        return None
    return {
        "AUROC": bs.fast_auroc(y_true, y_prob),
        "ECE": bs.fast_ece(y_true, y_prob),
        "AURAC": bs.fast_aurac(y_true, y_prob),
    }


def main():
    all_rows = []
    for ds, cfg in DATASETS.items():
        meta = json.load(open(os.path.join(cfg["run_dir"], "majority_weighted",
                                           "metadata.json")))
        p = meta["parameters"]
        our_methods = [f"weighted_majority_{b}" for b in meta["features"]]
        our_methods.append("weighted_majority_aggregated_optimal")
        uncertainty = set(p["uncertainty_baselines"])

        records = json.load(open(cfg["input_json"]))
        cluster_info = json.load(open(cfg["cluster_cache"]))
        id_to_category = {}
        with open("unobench_processed.jsonl") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    id_to_category[item["question_id"]] = item.get(
                        "category", "Unknown")

        bw = meta["best_weights"]
        fb = bs.make_scorer(bw["fallback"], meta["features"])
        scorers = {c: bs.make_scorer(e, meta["features"]) or fb
                   for c, e in bw["per_category"].items()}
        test_records, val_records = bs.stratified_test_val_split(
            records, id_to_category, p["val_fraction"], p["split_seed"])
        _, val_items = bs.collect_per_item(
            val_records, cluster_info, meta, scorers, fb, id_to_category)
        _, test_items = bs.collect_per_item(
            test_records, cluster_info, meta, scorers, fb, id_to_category)

        # deltas lookup: (subset, method, metric) -> row
        deltas = {}
        with open(os.path.join(cfg["results"], "paired_deltas.csv")) as f:
            for r in csv.DictReader(f):
                if r["Reference"] == bs.SELF_CONSISTENCY_KEY:
                    deltas[(r["Subset"], r["Method"], r["Metric"])] = r

        subsets = sorted({c for c in test_items[our_methods[0]]["category"]})
        subsets.append("Overall")

        for subset in subsets:
            for metric in METRIC_KEYS:
                lower_better = metric == "ECE"
                picks = {}
                for mode, items in [("val", val_items), ("test", test_items)]:
                    best_m, best_v = None, None
                    for m in our_methods:
                        s = subset_metric(items, m, subset,
                                          m[len("weighted_majority_"):] in uncertainty)
                        if s is None or np.isnan(s[metric]):
                            continue
                        v = s[metric]
                        if best_v is None or ((v < best_v) if lower_better
                                              else (v > best_v)):
                            best_v, best_m = v, m
                    picks[mode] = best_m
                for mode in ["val", "test"]:
                    m = picks[mode]
                    r = deltas.get((subset, m, metric))
                    if r is None:
                        continue
                    all_rows.append([
                        ds, subset, metric, mode,
                        m.replace("weighted_majority_", ""),
                        float(r["Delta_mean"]), float(r["Delta_lo"]),
                        float(r["Delta_hi"]), float(r["p_boot"]),
                    ])
        print(f"{ds}: done")

    out = "bootstrap-results/best_vs_sc.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Dataset", "Subset", "Metric", "Selection", "Method",
                    "Delta_mean", "Delta_lo", "Delta_hi", "p_boot"])
        for r in all_rows:
            w.writerow([f"{x:.6f}" if isinstance(x, float) else x for x in r])
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
