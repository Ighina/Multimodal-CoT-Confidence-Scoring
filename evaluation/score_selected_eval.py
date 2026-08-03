#!/usr/bin/env python3
"""
Third answer-selection setting "score_selected", replayed over all stored runs.

Paper: the best-of-n selection experiment referenced at the end of
Appendix N (each method SELECTS the answer by its own per-candidate score).

Settings compared (per question, per method):
  * no_majority       -- majority-vote answer, method's own score as confidence
                         (stored in <run>/no_majority/*.csv, eval.json)
  * majority_weighted -- score-mass weighted majority answer, sum-share
                         confidence (stored in <run>/majority_weighted/*.csv,
                         eval_majority.json)
  * score_selected    -- NEW: the answer of the single CANDIDATE with the best
                         own score (argmax for confidence methods, argmin for
                         uncertainty methods, lowest index on ties); confidence
                         is that candidate's own score.  Question-level
                         baselines (umpire_normal) cannot select by score and
                         keep their no_majority row; Self-Consistency's own
                         score is the vote count, so its row is also identical
                         to no_majority (reference row).

Like bootstrap_significance.py, nothing is refit: each run's aggregator is
reconstructed from the lr_model/weights stored in that run's per-setting
metadata.json and the deterministic part of final_evaluate.py's Phase 2 is
replayed.  HARD GATE: for every run the replay must reproduce the stored
no_majority AND majority_weighted per-subset metrics (AUROC/ECE/AURAC) to
<= --tol (default 1e-6, checked against the full-precision eval.json /
eval_majority.json; the 4-decimal CSVs are additionally checked at rounding
precision).  Runs failing the gate are excluded, never approximated.

Outputs (per dataset, under bootstrap-results/score_selected/<dataset>/):
  all_runs.csv          one long CSV: run x setting x subset x method with
                        AUROC/ECE/AURAC (pipeline post-processing: per-split
                        min-max rescale, 1-p flip for uncertainty methods)
                        plus Accuracy (mean correctness of selected answers)
  means.csv             mean +/- std over passing runs
  bootstrap_deltas.csv  canonical run_0 item-level paired bootstrap
                        (B resamples, shared indices): score_selected vs
                        majority_weighted for aggregated_optimal + the five
                        embedding scores, and score_selected
                        aggregated_optimal vs Self-Consistency
  verification.csv      per-run gate outcome (worst |diff|, pass/fail)
  RESULTS.md            human-readable summary tables

Usage:
    python score_selected_eval.py                    # everything, all datasets
    python score_selected_eval.py --datasets gemini --num-runs 5 --num-boot 100
"""

import argparse
import csv
import json
import os
from collections import Counter, defaultdict

import numpy as np

import bootstrap_significance as bs

SC = bs.SELF_CONSISTENCY_KEY
METRIC_KEYS = ["AUROC", "ECE", "AURAC", "Accuracy"]
VERIFY_KEYS = ["AUROC", "ECE", "AURAC"]
SETTINGS = ["no_majority", "majority_weighted", "score_selected"]

DATASETS = {
    "gemini": "gemini-majority-multirun-logistic-reg",
    "minicpm": "minicpm-majority-multirun-logistic-reg",
    "lco-gemini": "lco-gemini-majority-multirun-logistic-reg",
}


# ----------------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------------
def load_dataset_inputs(name, jsonl_path):
    """input_json / cluster_cache come from the dataset's canonical
    bootstrap-results metadata (the same files bootstrap_significance.py used)."""
    meta_path = os.path.join("bootstrap-results", name, "metadata.json")
    ds_meta = json.load(open(meta_path))
    records = json.load(open(ds_meta["input_json"]))
    cluster_info = json.load(open(ds_meta["cluster_cache"]))
    id_to_category = {}
    with open(jsonl_path) as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                id_to_category[item["question_id"]] = item.get("category", "Unknown")
    return ds_meta, records, cluster_info, id_to_category


def make_scorers(meta):
    bw = meta["best_weights"]
    features = meta["features"]
    fallback = bs.make_scorer(bw["fallback"], features)
    scorers = {cat: bs.make_scorer(entry, features) or fallback
               for cat, entry in bw["per_category"].items()}
    return scorers, fallback


# ----------------------------------------------------------------------------------
# The new setting
# ----------------------------------------------------------------------------------
def collect_score_selected(records, cluster_info, meta, scorers, fallback_scorer,
                           id_to_category):
    """Per-item collection for the score_selected setting.  Mirrors
    bootstrap_significance.collect_per_item_nomaj's iteration/skip logic
    exactly; only the answer selection differs (best own candidate score
    instead of majority vote)."""
    p = meta["parameters"]
    features = meta["features"]
    baselines = p["baselines"]
    uncertainty = set(p["uncertainty_baselines"])
    qlevel = set(p["question_level_baselines"])
    threshold = p["correctness_threshold"]

    methods = [SC] + baselines + ["aggregated_optimal"]
    items = {m: {"qid": [], "category": [], "y_true": [], "y_prob": []}
             for m in methods}

    for rec in records:
        qid = str(rec.get("question_id"))
        ci = cluster_info.get(qid)
        if not ci or not ci.get("groups"):
            continue
        groups = ci["groups"]
        majority_ids = ci["majority_ids"]
        n = len(groups)
        confs = rec.get("generations_confidence", []) or []
        correctness = bs.get_correctness(rec, threshold)
        if correctness is None or len(confs) < n or len(correctness) < n:
            continue
        category = id_to_category.get(rec.get("question_id"), "Unknown")

        def emit(method, y_true, y_prob):
            items[method]["qid"].append(qid)
            items[method]["category"].append(category)
            items[method]["y_true"].append(int(y_true))
            items[method]["y_prob"].append(float(y_prob))

        # Self-Consistency: its own score IS the vote count, so score-based
        # selection == majority answer.  Identical to the no_majority row.
        counts_g = Counter(groups)
        sc_group = majority_ids[0]
        sc_index = groups.index(sc_group)
        emit(SC, correctness[sc_index], counts_g[sc_group] / n)

        for method in baselines + ["aggregated_optimal"]:
            if method == "aggregated_optimal":
                scorer = scorers.get(category, fallback_scorer)
                feat_matrix = np.array(
                    [[float(confs[i].get(f, 0.0)) for f in features]
                     for i in range(n)]
                )
                scores = np.asarray(scorer(feat_matrix), dtype=float)
                is_unc = False
            else:
                scores = np.array(
                    [float(confs[i].get(method, 0.0)) for i in range(n)],
                    dtype=float)
                is_unc = method in uncertainty

            if method in qlevel:
                # Question-level score (constant per question): cannot select
                # by score -> keep the majority-vote answer with the raw score
                # as confidence, i.e. identical to the no_majority row.
                winning_group = majority_ids[0]
                winning_index = groups.index(winning_group)
                emit(method, correctness[winning_index], float(scores[0]))
                continue

            # winner = candidate with the best OWN score; np.argmin/argmax
            # return the first (lowest-index) occurrence on ties.
            win = int(np.argmin(scores)) if is_unc else int(np.argmax(scores))
            emit(method, correctness[win], float(scores[win]))

    return methods, items


# ----------------------------------------------------------------------------------
# Metrics / verification
# ----------------------------------------------------------------------------------
def subset_mask(item_d, subset):
    cats = np.array(item_d["category"])
    return np.ones(len(cats), bool) if subset == "Overall" else cats == subset


def compute_setting_metrics(items, methods, subset_names, is_unc_fn):
    """{(subset, method): {N, AUROC, ECE, AURAC, Accuracy}} with the pipeline's
    per-split min-max rescale (+ post-rescale 1-p flip for uncertainty)."""
    out = {}
    for subset in subset_names:
        for m in methods:
            d = items[m]
            mask = subset_mask(d, subset)
            n = int(mask.sum())
            y_true = np.array(d["y_true"])[mask]
            y_prob = bs.minmax_transform(np.array(d["y_prob"])[mask],
                                         is_unc_fn(m))
            row = {"N": n}
            if n == 0:
                row.update({k: np.nan for k in METRIC_KEYS})
            else:
                row["AUROC"] = (bs.fast_auroc(y_true, y_prob)
                                if len(np.unique(y_true)) > 1 else np.nan)
                row["ECE"] = bs.fast_ece(y_true, y_prob)
                row["AURAC"] = bs.fast_aurac(y_true, y_prob)
                row["Accuracy"] = float(np.mean(y_true))
            out[(subset, m)] = row
    return out


def verify_against_stored(metrics, stored, subset_names, methods, tol):
    """Compare replayed metrics to the run's stored full-precision eval JSON."""
    worst, n_cmp, mismatches = 0.0, 0, []
    for subset in subset_names:
        for m in methods:
            st = stored.get(subset, {}).get(m)
            if not st:
                continue
            for k in VERIFY_KEYS:
                sv = st.get(k)
                rv = metrics[(subset, m)][k]
                if sv is None or rv is None or np.isnan(rv):
                    continue
                d = abs(rv - float(sv))
                worst = max(worst, d)
                n_cmp += 1
                if d > tol:
                    mismatches.append((subset, m, k, rv, float(sv), d))
    return worst, n_cmp, mismatches


def verify_against_csvs(metrics, run_dir, setting, subset_names, tol=5.0e-5 + 1e-9):
    """The stored per-subset CSVs are the eval JSON rounded to 4 decimals;
    check the replay matches them at rounding precision."""
    worst, n_cmp, mismatches = 0.0, 0, []
    for subset in subset_names:
        path = os.path.join(run_dir, setting, f"{subset}.csv")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for row in csv.DictReader(f):
                m = row["Method"]
                if (subset, m) not in metrics:
                    continue
                for k in VERIFY_KEYS:
                    sv = row.get(k)
                    rv = metrics[(subset, m)][k]
                    if sv in (None, "", "None") or rv is None or np.isnan(rv):
                        continue
                    d = abs(rv - float(sv))
                    worst = max(worst, d)
                    n_cmp += 1
                    if d > tol:
                        mismatches.append((subset, m, k, rv, float(sv), d))
    return worst, n_cmp, mismatches


def assert_reference_rows(ss_items, nm_items, meta):
    """Sanity gate: in score_selected, Self-Consistency and every
    question-level baseline (umpire_normal) must be identical to their
    no_majority rows."""
    qlevel = set(meta["parameters"]["question_level_baselines"])
    for m in [SC] + sorted(qlevel):
        a, b = ss_items[m], nm_items[m]
        if not (a["qid"] == b["qid"] and a["y_true"] == b["y_true"]
                and a["y_prob"] == b["y_prob"]):
            raise AssertionError(
                f"score_selected reference row for '{m}' differs from its "
                f"no_majority row")


# ----------------------------------------------------------------------------------
# Per-run replay
# ----------------------------------------------------------------------------------
def replay_run(run_dir, records, cluster_info, id_to_category, tol):
    """Replay one run in all three settings, verify the two stored settings,
    and return (metrics_by_setting, per_item_bundle, gate_info)."""
    nm_meta = json.load(open(os.path.join(run_dir, "no_majority",
                                          "metadata.json")))
    mw_meta = json.load(open(os.path.join(run_dir, "majority_weighted",
                                          "metadata.json")))
    p = nm_meta["parameters"]
    test_records, _val_records = bs.stratified_test_val_split(
        records, id_to_category, p["val_fraction"], p["split_seed"])

    nm_scorers, nm_fb = make_scorers(nm_meta)
    mw_scorers, mw_fb = make_scorers(mw_meta)

    nm_methods, nm_items = bs.collect_per_item_nomaj(
        test_records, cluster_info, nm_meta, nm_scorers, nm_fb, id_to_category)
    mw_methods, mw_items = bs.collect_per_item(
        test_records, cluster_info, mw_meta, mw_scorers, mw_fb, id_to_category)
    ss_methods, ss_items = collect_score_selected(
        test_records, cluster_info, nm_meta, nm_scorers, nm_fb, id_to_category)

    uncertainty = set(p["uncertainty_baselines"])

    def plain_is_unc(m):
        return m in uncertainty

    def mw_is_unc(m):
        base = m[len("weighted_majority_"):] \
            if m.startswith("weighted_majority_") else m
        return base in uncertainty

    subset_names = sorted(set(nm_items[SC]["category"])) + ["Overall"]

    metrics = {
        "no_majority": compute_setting_metrics(
            nm_items, nm_methods, subset_names, plain_is_unc),
        "majority_weighted": compute_setting_metrics(
            mw_items, mw_methods, subset_names, mw_is_unc),
        "score_selected": compute_setting_metrics(
            ss_items, ss_methods, subset_names, plain_is_unc),
    }

    # ---- gate 1: reproduce stored metrics of both stored settings ----
    stored_mw = json.load(open(os.path.join(run_dir, "eval_majority.json")))
    stored_nm = json.load(open(os.path.join(run_dir, "eval.json")))
    for subset, d in stored_mw.items():  # SC reference row lives here
        if subset in stored_nm and SC in d:
            stored_nm[subset][SC] = d[SC]

    worst_nm, n_nm, mm_nm = verify_against_stored(
        metrics["no_majority"], stored_nm, subset_names, nm_methods, tol)
    worst_mw, n_mw, mm_mw = verify_against_stored(
        metrics["majority_weighted"], stored_mw, subset_names, mw_methods, tol)
    worst_csv_nm, ncsv_nm, mmcsv_nm = verify_against_csvs(
        metrics["no_majority"], run_dir, "no_majority", subset_names)
    worst_csv_mw, ncsv_mw, mmcsv_mw = verify_against_csvs(
        metrics["majority_weighted"], run_dir, "majority_weighted", subset_names)

    mismatches = mm_nm + mm_mw + mmcsv_nm + mmcsv_mw
    n_compared = n_nm + n_mw
    if n_compared < 50:  # a run with (almost) nothing to compare cannot "pass"
        mismatches.append(("<gate>", "<too-few-comparisons>", "N",
                           n_compared, 50, n_compared))
    gate = {
        "worst_json": max(worst_nm, worst_mw),
        "worst_csv": max(worst_csv_nm, worst_csv_mw),
        "n_compared": n_compared,
        "n_compared_csv": ncsv_nm + ncsv_mw,
        "mismatches": mismatches,
        "passed": len(mismatches) == 0,
    }

    # ---- gate 2: reference rows identical to no_majority ----
    if gate["passed"]:
        assert_reference_rows(ss_items, nm_items, nm_meta)

    bundle = {
        "nm_items": nm_items, "mw_items": mw_items, "ss_items": ss_items,
        "nm_methods": nm_methods, "mw_methods": mw_methods,
        "ss_methods": ss_methods, "subset_names": subset_names,
        "plain_is_unc": plain_is_unc, "mw_is_unc": mw_is_unc,
        "features": nm_meta["features"],
        "qlevel": set(p["question_level_baselines"]),
    }
    return metrics, bundle, gate


# ----------------------------------------------------------------------------------
# Canonical run_0 item-level paired bootstrap
# ----------------------------------------------------------------------------------
def bootstrap_deltas(bundle, num_boot, boot_seed):
    """Paired bootstrap (shared resample indices across all arrays of a
    subset, as in bootstrap_significance.py) on canonical run_0:
      score_selected vs majority_weighted for aggregated_optimal + the five
      embedding scores (features), and score_selected aggregated_optimal vs
      Self-Consistency.  Metrics: AUROC, AURAC, Accuracy."""
    features = bundle["features"]
    cmp_methods = features + ["aggregated_optimal"]

    pairs = [(f"score_selected:{m}",
              f"majority_weighted:weighted_majority_{m}") for m in cmp_methods]
    pairs.append(("score_selected:aggregated_optimal",
                  f"no_majority:{SC}"))

    def get_items(tag):
        setting, m = tag.split(":", 1)
        items = {"no_majority": bundle["nm_items"],
                 "majority_weighted": bundle["mw_items"],
                 "score_selected": bundle["ss_items"]}[setting][m]
        is_unc = (bundle["mw_is_unc"] if setting == "majority_weighted"
                  else bundle["plain_is_unc"])(m)
        return items, is_unc

    tags = sorted({t for pr in pairs for t in pr})
    rows = []
    boot_metrics = ["AUROC", "AURAC", "Accuracy"]
    for subset in bundle["subset_names"]:
        arrs = {}
        n_items = None
        ref_qids = None
        for tag in tags:
            items, is_unc = get_items(tag)
            mask = subset_mask(items, subset)
            qids = [q for q, keep in zip(items["qid"], mask) if keep]
            if ref_qids is None:
                ref_qids = qids
            elif qids != ref_qids:
                raise AssertionError(
                    f"item misalignment in subset {subset} for {tag}")
            y_true = np.array(items["y_true"])[mask]
            ranked = bs.minmax_transform(np.array(items["y_prob"])[mask],
                                         is_unc)
            arrs[tag] = (y_true, ranked)
            n_items = len(y_true)

        def metric_vec(y_true, ranked):
            return np.array([bs.fast_auroc(y_true, ranked),
                             bs.fast_aurac(y_true, ranked),
                             float(np.mean(y_true))])

        point = {tag: metric_vec(*arrs[tag]) for tag in tags}

        rng = np.random.default_rng(boot_seed)
        B = num_boot
        boot = {tag: np.empty((B, len(boot_metrics))) for tag in tags}
        for b in range(B):
            idx = rng.integers(0, n_items, n_items)  # shared across all tags
            for tag in tags:
                y_true, ranked = arrs[tag]
                boot[tag][b] = metric_vec(y_true[idx], ranked[idx])

        for a, ref in pairs:
            for ki, k in enumerate(boot_metrics):
                da = boot[a][:, ki] - boot[ref][:, ki]
                da = da[~np.isnan(da)]
                if len(da) == 0:
                    continue
                lo, hi = np.percentile(da, 2.5), np.percentile(da, 97.5)
                p_two = min(1.0, 2 * min(np.mean(da <= 0), np.mean(da >= 0))
                            + 1.0 / len(da))
                rows.append([subset, a, ref, k, n_items,
                             point[a][ki], point[ref][ki],
                             point[a][ki] - point[ref][ki],
                             float(np.mean(da)), float(lo), float(hi), p_two])
        print(f"    [{subset}] bootstrap done (n={n_items}, B={B})")
    return rows


# ----------------------------------------------------------------------------------
# Outputs
# ----------------------------------------------------------------------------------
def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow([f"{x:.6f}" if isinstance(x, (float, np.floating))
                        else x for x in r])
    print(f"  wrote {path}")


def fmt_ms(mean, std):
    if np.isnan(mean):
        return "--"
    return f"{mean:.4f}±{std:.4f}"


def write_results_md(out_dir, ds_name, means, subset_names, methods_by_setting,
                     features, qlevel, gate_summary, boot_rows, num_boot):
    lines = [f"# score_selected evaluation — {ds_name}", ""]
    lines.append(f"Gate: {gate_summary['passed']}/{gate_summary['total']} runs "
                 f"reproduced their stored no_majority AND majority_weighted "
                 f"per-subset metrics (worst |diff| across passing runs: "
                 f"{gate_summary['worst']:.2e}; tolerance "
                 f"{gate_summary['tol']:.0e}).")
    if gate_summary["failed_runs"]:
        lines.append(f"Failed runs (excluded): {gate_summary['failed_runs']}")
    lines.append("")
    lines.append("`score_selected` = per question, take the answer of the "
                 "single candidate with the best own score (argmax for "
                 "confidence, argmin for uncertainty methods; lowest index on "
                 "ties); confidence = that candidate's own score, then the "
                 "pipeline's per-split min-max rescale (+1-p flip for "
                 "uncertainty methods).")
    lines.append("")
    lines.append(f"**FLAG:** question-level baselines ({sorted(qlevel)}) are "
                 "constant per question and CANNOT select by score — their "
                 "score_selected rows are identical to their no_majority rows "
                 "(majority-vote answer). Self-Consistency's own score is the "
                 "vote count, so its selection is the majority answer too "
                 "(reference row).")
    lines.append("")

    ss_methods = methods_by_setting["score_selected"]
    for metric in ["Accuracy", "AUROC", "AURAC", "ECE"]:
        lines.append(f"## {metric} (mean ± std over passing runs)")
        lines.append("")
        header = "| Method | " + " | ".join(
            f"{s} (sel/mw)" for s in subset_names) + " |"
        lines.append(header)
        lines.append("|" + "---|" * (len(subset_names) + 1))
        for m in ss_methods:
            mw_key = (SC if m == SC else f"weighted_majority_{m}")
            cells = []
            for subset in subset_names:
                a = means.get(("score_selected", subset, m))
                b = means.get(("majority_weighted", subset, mw_key))
                ca = fmt_ms(*a[metric]) if a else "--"
                cb = fmt_ms(*b[metric]) if b else "--"
                cells.append(f"{ca} / {cb}")
            label = m + (" [no-sel]" if m in qlevel or m == SC else "")
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
        lines.append("")

    lines.append(f"## Canonical run_0 item-level paired bootstrap "
                 f"(B={num_boot}, shared indices)")
    lines.append("")
    lines.append("| Subset | A | B | Metric | Delta(A-B) | 95% CI | p |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in boot_rows:
        subset, a, ref, k, _n, _pa, _pb, dpoint, dmean, lo, hi, p = r
        lines.append(f"| {subset} | {a} | {ref} | {k} | {dpoint:+.4f} | "
                     f"[{lo:+.4f}, {hi:+.4f}] | {p:.4f} |")
    lines.append("")
    path = os.path.join(out_dir, "RESULTS.md")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"  wrote {path}")


# ----------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS),
                    choices=list(DATASETS))
    ap.add_argument("--num-runs", type=int, default=100)
    ap.add_argument("--jsonl-path", default="unobench_processed.jsonl")
    ap.add_argument("--tol", type=float, default=1e-6,
                    help="Hard verification gate on |replayed - stored|.")
    ap.add_argument("--num-boot", type=int, default=10000)
    ap.add_argument("--boot-seed", type=int, default=0)
    ap.add_argument("--output-root",
                    default=os.path.join("bootstrap-results", "score_selected"))
    args = ap.parse_args()

    for ds_name in args.datasets:
        prefix = DATASETS[ds_name]
        out_dir = os.path.join(args.output_root, ds_name)
        os.makedirs(out_dir, exist_ok=True)
        print(f"\n=== {ds_name} ({prefix}) ===")
        ds_meta, records, cluster_info, id_to_category = load_dataset_inputs(
            ds_name, args.jsonl_path)
        print(f"input_json={ds_meta['input_json']} "
              f"cluster_cache={ds_meta['cluster_cache']} "
              f"records={len(records)}")

        long_rows, ver_rows = [], []
        acc = defaultdict(lambda: defaultdict(list))  # key -> metric -> [runs]
        failed_runs, worst_pass = [], 0.0
        run0_bundle, methods_by_setting = None, None
        subset_names, features, qlevel = None, None, None

        for r in range(args.num_runs):
            run_dir = os.path.join(prefix, f"run_{r}")
            metrics, bundle, gate = replay_run(
                run_dir, records, cluster_info, id_to_category, args.tol)
            ver_rows.append([r, gate["worst_json"], gate["worst_csv"],
                             gate["n_compared"], gate["n_compared_csv"],
                             "PASS" if gate["passed"] else "FAIL"])
            if not gate["passed"]:
                failed_runs.append(r)
                print(f"  run_{r}: GATE FAILED "
                      f"(worst_json={gate['worst_json']:.3e}, "
                      f"worst_csv={gate['worst_csv']:.3e}); first mismatches:")
                for mm in gate["mismatches"][:5]:
                    print(f"    {mm}")
                continue
            worst_pass = max(worst_pass, gate["worst_json"])

            if subset_names is None:
                subset_names = bundle["subset_names"]
                features = bundle["features"]
                qlevel = bundle["qlevel"]
                methods_by_setting = {
                    "no_majority": bundle["nm_methods"],
                    "majority_weighted": bundle["mw_methods"],
                    "score_selected": bundle["ss_methods"],
                }
            if r == 0:
                run0_bundle = bundle

            for setting in SETTINGS:
                meths = {"no_majority": bundle["nm_methods"],
                         "majority_weighted": bundle["mw_methods"],
                         "score_selected": bundle["ss_methods"]}[setting]
                for subset in bundle["subset_names"]:
                    for m in meths:
                        row = metrics[setting][(subset, m)]
                        long_rows.append(
                            [r, setting, subset, m, row["N"]]
                            + [row[k] for k in METRIC_KEYS])
                        for k in METRIC_KEYS:
                            acc[(setting, subset, m)][k].append(row[k])
            if (r + 1) % 10 == 0:
                print(f"  run_{r}: OK (worst |diff| so far {worst_pass:.2e})")

        n_passed = args.num_runs - len(failed_runs)
        print(f"Gate: {n_passed}/{args.num_runs} runs passed "
              f"(worst |diff| among passing = {worst_pass:.2e}); "
              f"failed: {failed_runs}")

        write_csv(os.path.join(out_dir, "all_runs.csv"),
                  ["Run", "Setting", "Subset", "Method", "N"] + METRIC_KEYS,
                  long_rows)
        write_csv(os.path.join(out_dir, "verification.csv"),
                  ["Run", "Worst_abs_diff_json", "Worst_abs_diff_csv",
                   "N_compared_json", "N_compared_csv", "Gate"], ver_rows)

        means = {}
        mean_rows = []
        for key in sorted(acc.keys()):
            setting, subset, m = key
            entry = {}
            row = [setting, subset, m, len(acc[key]["Accuracy"])]
            for k in METRIC_KEYS:
                v = np.array(acc[key][k], dtype=float)
                ok = v[~np.isnan(v)]
                mu = float(np.mean(ok)) if len(ok) else np.nan
                sd = float(np.std(ok)) if len(ok) else np.nan
                entry[k] = (mu, sd)
                row += [mu, sd, len(ok)]
            means[key] = entry
            mean_rows.append(row)
        header = ["Setting", "Subset", "Method", "N_runs"]
        for k in METRIC_KEYS:
            header += [f"{k}_mean", f"{k}_std", f"{k}_n"]
        write_csv(os.path.join(out_dir, "means.csv"), header, mean_rows)

        # ---- canonical run_0 bootstrap ----
        boot_rows = []
        if run0_bundle is not None:
            print(f"  canonical run_0 paired bootstrap (B={args.num_boot}) ...")
            boot_rows = bootstrap_deltas(run0_bundle, args.num_boot,
                                         args.boot_seed)
            write_csv(os.path.join(out_dir, "bootstrap_deltas.csv"),
                      ["Subset", "MethodA", "MethodB", "Metric", "N",
                       "PointA", "PointB", "Delta_point", "Delta_boot_mean",
                       "Delta_lo", "Delta_hi", "p_boot"], boot_rows)
        else:
            print("  run_0 failed the gate -> no canonical bootstrap.")

        gate_summary = {"passed": n_passed, "total": args.num_runs,
                        "failed_runs": failed_runs, "worst": worst_pass,
                        "tol": args.tol}
        write_results_md(out_dir, ds_name, means, subset_names,
                         methods_by_setting, features, qlevel, gate_summary,
                         boot_rows, args.num_boot)

        with open(os.path.join(out_dir, "metadata.json"), "w") as f:
            json.dump({
                "dataset": ds_name,
                "run_prefix": prefix,
                "input_json": ds_meta["input_json"],
                "cluster_cache": ds_meta["cluster_cache"],
                "jsonl_path": args.jsonl_path,
                "num_runs": args.num_runs,
                "runs_passed": n_passed,
                "failed_runs": failed_runs,
                "worst_abs_diff_passing": worst_pass,
                "verify_tol": args.tol,
                "num_boot": args.num_boot,
                "boot_seed": args.boot_seed,
                "notes": "score_selected: answer of the candidate with the "
                         "best own score (argmin for uncertainty methods); "
                         "confidence = that candidate's own score with the "
                         "pipeline's per-split minmax rescale (+flip). "
                         "Question-level baselines and Self-Consistency keep "
                         "their no_majority rows (cannot select by score). "
                         "no_majority uses no_majority/metadata.json weights; "
                         "majority_weighted uses majority_weighted/"
                         "metadata.json weights; score_selected uses the "
                         "no_majority weights.",
            }, f, indent=2)
    print("\nDone.")


if __name__ == "__main__":
    main()
