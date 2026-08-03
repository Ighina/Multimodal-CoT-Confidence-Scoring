#!/usr/bin/env python3
"""
Item-level statistics for the majority-weighted evaluation:

  (a) validation-calibrated ECE (Platt scaling fit on the validation split only,
      applied to the test split) for every method — review item "deployment-
      realistic calibration";
  (b) paired bootstrap over test questions (B resamples) giving per-method
      metric CIs and CIs + p-values on paired per-item deltas vs
      self-consistency — review item "item-level significance".

Paper: item-level comparisons of Section 5.1 and the variance / confidence
intervals / deployment-realistic calibration analysis of Appendix C
(Tables 4-6). Also a shared library of numerically identical ports of
final_evaluate.py's primitives, imported by most analysis scripts here.

Instead of refitting anything, the script reconstructs the EXACT aggregator of
a chosen canonical run (default run_0, seed 42) from the lr_model/weights
stored in <run>/majority_weighted/metadata.json, replays the deterministic
part of final_evaluate.py's Phase 2 to obtain per-question (y_true, y_prob)
for every majority-voting method, verifies the replication by comparing
against the run's stored eval_majority.json, and only then computes the new
statistics.

Usage (one dataset):
    python bootstrap_significance.py \
        --input-json gemini_cots_reformatted.json \
        --cluster-cache gemini-majority-vote-clusters_top6.json \
        --run-dir gemini-majority-multirun-logistic-reg/run_0 \
        --output-dir bootstrap-results/gemini \
        [--jsonl-path unobench_processed.jsonl] [--num-boot 10000] [--boot-seed 0]
"""

import argparse
import csv
import json
import os
import random
from collections import Counter, defaultdict

import numpy as np

SELF_CONSISTENCY_KEY = "majority_vote_selfconsistency"
METRICS = ["AUROC", "ECE", "AURAC"]


# ----------------------------------------------------------------------------------
# Ports of final_evaluate.py primitives (kept numerically identical)
# ----------------------------------------------------------------------------------
def get_correctness(rec, threshold):
    raw = rec.get("generations_uno_score")
    if not isinstance(raw, list):
        raw = rec.get(f"generations_uno_score_threshold_{threshold}")
    if not isinstance(raw, list):
        return None
    out = []
    for v in raw:
        try:
            out.append(int(float(v) >= threshold))
        except (TypeError, ValueError):
            out.append(0)
    return out


def aggregate_group_scores(groups_arr, scores):
    n_groups = int(groups_arr.max()) + 1
    group_sum = np.bincount(groups_arr, weights=scores, minlength=n_groups)
    group_count = np.bincount(groups_arr, minlength=n_groups)
    total_sum = group_sum.sum() + 1e-9
    norm_sum = group_sum / total_sum
    group_mean = group_sum / np.maximum(group_count, 1)
    total_mean = group_mean.sum() + 1e-9
    norm_mean = group_mean / total_mean
    return group_sum, norm_sum, norm_mean


def minmax_transform(y_prob, uncertainty=False):
    """The pipeline's prob_transform='minmax' + post-rescale flip."""
    y_prob = np.asarray(y_prob, dtype=float)
    if len(y_prob) > 0:
        p_min, p_max = float(np.min(y_prob)), float(np.max(y_prob))
        if p_max > p_min:
            y_prob = (y_prob - p_min) / (p_max - p_min)
        else:
            y_prob = np.zeros_like(y_prob)
        if uncertainty:
            y_prob = 1.0 - y_prob
    return y_prob


def stratified_test_val_split(records, id_to_category, val_fraction, seed):
    rng = random.Random(seed)
    by_stratum = defaultdict(list)
    for rec in records:
        key = rec.get("split") or id_to_category.get(rec.get("question_id"), "Unknown")
        by_stratum[key].append(rec)
    test_records, val_records = [], []
    for key, recs in by_stratum.items():
        recs = recs[:]
        rng.shuffle(recs)
        n_val = max(1, round(len(recs) * val_fraction)) if len(recs) >= 2 else 0
        val_records.extend(recs[:n_val])
        test_records.extend(recs[n_val:])
    return test_records, val_records


# ----------------------------------------------------------------------------------
# Aggregator reconstruction from stored metadata
# ----------------------------------------------------------------------------------
def make_scorer(entry, features):
    """Build per-sample score fn from a metadata best_weights entry
    ('lr_model' -> scaled logistic; 'weights' -> dot product)."""
    if entry is None:
        return None
    if "lr_model" in entry and entry["lr_model"] is not None:
        m = entry["lr_model"]
        coef = np.array([m["coefficients"][f] for f in features])
        mean = np.array([m["scaler_mean"][f] for f in features])
        scale = np.array([m["scaler_scale"][f] for f in features])
        intercept = m["intercept"]

        def score(feat_matrix):
            z = ((feat_matrix - mean) / scale) @ coef + intercept
            return 1.0 / (1.0 + np.exp(-z))

        return score
    if "weights" in entry:
        w = np.array([entry["weights"][f] for f in features])

        def score(feat_matrix):
            return feat_matrix @ w

        return score
    return None


# ----------------------------------------------------------------------------------
# Per-item replay of final_evaluate.py Phase 2 (majority-voting methods only)
# ----------------------------------------------------------------------------------
def collect_per_item(records, cluster_info, meta, wm_scorers, wm_fallback_scorer,
                     id_to_category):
    p = meta["parameters"]
    features = meta["features"]
    baselines = p["baselines"]
    mw_baselines = set(p["majority_weighted_baselines"])
    uncertainty = set(p["uncertainty_baselines"])
    qlevel = set(p["question_level_baselines"])
    threshold = p["correctness_threshold"]
    agg_key = "weighted_majority_aggregated_optimal"

    methods = ([SELF_CONSISTENCY_KEY]
               + [f"weighted_majority_{b}" for b in baselines] + [agg_key])
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
        correctness = get_correctness(rec, threshold)
        if correctness is None or len(confs) < n or len(correctness) < n:
            continue
        category = id_to_category.get(rec.get("question_id"), "Unknown")
        groups_arr = np.array(groups, dtype=int)

        def emit(method, y_true, y_prob):
            items[method]["qid"].append(qid)
            items[method]["category"].append(category)
            items[method]["y_true"].append(int(y_true))
            items[method]["y_prob"].append(float(y_prob))

        # ---- self-consistency ----
        counts_g = Counter(groups)
        sc_group = majority_ids[0]
        sc_index = groups.index(sc_group)
        emit(SELF_CONSISTENCY_KEY, correctness[sc_index], counts_g[sc_group] / n)

        # ---- weighted_majority_<baseline> ----
        for method in baselines:
            scores = [float(confs[i].get(method, 0.0)) for i in range(n)]
            is_unc = method in uncertainty
            scores_arr = np.array(scores, dtype=float)
            S_g, norm_sum, _norm_mean = aggregate_group_scores(groups_arr, scores_arr)
            conf_arr = norm_sum  # confidence_mode == "sum_share"
            norm_scores = {g: float(conf_arr[g]) for g in range(len(conf_arr))}
            wm_key = f"weighted_majority_{method}"

            if method in qlevel:
                # question-level score: majority answer + raw score
                winning_group = majority_ids[0]
                winning_index = groups.index(winning_group)
                emit(wm_key, correctness[winning_index], scores[0])
            elif method in mw_baselines:
                wm_group = int(np.argmin(S_g) if is_unc else np.argmax(S_g))
                wm_index = groups.index(wm_group)
                emit(wm_key, correctness[wm_index], norm_scores[wm_group])
            else:
                # replica of the plain baseline row (majority + tie-break)
                if len(majority_ids) == 1:
                    winning_group = majority_ids[0]
                elif is_unc:
                    winning_group = min(majority_ids, key=lambda g: norm_scores[g])
                else:
                    winning_group = max(majority_ids, key=lambda g: norm_scores[g])
                winning_index = groups.index(winning_group)
                emit(wm_key, correctness[winning_index], scores[winning_index])

        # ---- weighted_majority_aggregated_optimal ----
        scorer = wm_scorers.get(category, wm_fallback_scorer)
        feat_matrix = np.array(
            [[float(confs[i].get(f, 0.0)) for f in features] for i in range(n)]
        )
        wm_scores = np.asarray(scorer(feat_matrix), dtype=float)
        S_g, norm_sum, _norm_mean = aggregate_group_scores(groups_arr, wm_scores)
        conf_wm = norm_sum
        wm_group = int(np.argmax(S_g))
        wm_index = groups.index(wm_group)
        emit(agg_key, correctness[wm_index], float(conf_wm[wm_group]))

    return methods, items


def collect_per_item_nomaj(records, cluster_info, meta, scorers, fallback_scorer,
                           id_to_category):
    """Per-item replay of final_evaluate.py's TEST_METHODS loop (the
    'no_majority' setting: majority-vote answer selection with per-method
    tie-breaking, confidence from the method's own scores), plus the
    self-consistency reference row."""
    p = meta["parameters"]
    features = meta["features"]
    baselines = p["baselines"]
    mw = set(p["majority_weighted_baselines"]) | {"aggregated_optimal"}
    uncertainty = set(p["uncertainty_baselines"])
    qlevel = set(p["question_level_baselines"])
    threshold = p["correctness_threshold"]

    methods = [SELF_CONSISTENCY_KEY] + baselines + ["aggregated_optimal"]
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
        correctness = get_correctness(rec, threshold)
        if correctness is None or len(confs) < n or len(correctness) < n:
            continue
        category = id_to_category.get(rec.get("question_id"), "Unknown")
        groups_arr = np.array(groups, dtype=int)

        def emit(method, y_true, y_prob):
            items[method]["qid"].append(qid)
            items[method]["category"].append(category)
            items[method]["y_true"].append(int(y_true))
            items[method]["y_prob"].append(float(y_prob))

        counts_g = Counter(groups)
        sc_group = majority_ids[0]
        sc_index = groups.index(sc_group)
        emit(SELF_CONSISTENCY_KEY, correctness[sc_index], counts_g[sc_group] / n)

        for method in baselines + ["aggregated_optimal"]:
            if method == "aggregated_optimal":
                scorer = scorers.get(category, fallback_scorer)
                feat_matrix = np.array(
                    [[float(confs[i].get(f, 0.0)) for f in features]
                     for i in range(n)]
                )
                scores = [float(s) for s in scorer(feat_matrix)]
            else:
                scores = [float(confs[i].get(method, 0.0)) for i in range(n)]
            is_unc = method in uncertainty
            scores_arr = np.array(scores, dtype=float)
            _S_g, norm_sum, _norm_mean = aggregate_group_scores(
                groups_arr, scores_arr)
            norm_scores = {g: float(norm_sum[g]) for g in range(len(norm_sum))}

            if method in qlevel:
                winning_group = majority_ids[0]
                winning_index = groups.index(winning_group)
                emit(method, correctness[winning_index], scores[0])
                continue
            if len(majority_ids) == 1:
                winning_group = majority_ids[0]
            elif is_unc:
                winning_group = min(majority_ids, key=lambda g: norm_scores[g])
            else:
                winning_group = max(majority_ids, key=lambda g: norm_scores[g])
            winning_index = groups.index(winning_group)
            sel_y_prob = (norm_scores[winning_group] if method in mw
                          else scores[winning_index])
            emit(method, correctness[winning_index], sel_y_prob)

    return methods, items


# ----------------------------------------------------------------------------------
# Metrics (vectorized enough for the bootstrap loop)
# ----------------------------------------------------------------------------------
def fast_auroc(y_true, y_prob):
    """Rank-based AUROC with tie handling (equals sklearn's roc_auc_score)."""
    from scipy.stats import rankdata

    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.nan
    ranks = rankdata(y_prob)
    return (ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def fast_ece(y_true, y_prob, n_bins=10):
    """Same binning as final_evaluate.calculate_ece (identical np.linspace bin
    edges — including their float representation — with the last bin
    right-closed)."""
    edges = np.linspace(0, 1, n_bins + 1)
    bins = np.clip(np.searchsorted(edges, y_prob, side="right") - 1, 0, n_bins - 1)
    total = len(y_prob)
    ece = 0.0
    counts = np.bincount(bins, minlength=n_bins)
    sum_conf = np.bincount(bins, weights=y_prob, minlength=n_bins)
    sum_true = np.bincount(bins, weights=y_true, minlength=n_bins)
    nz = counts > 0
    ece = np.sum(np.abs(sum_conf[nz] / counts[nz] - sum_true[nz] / counts[nz])
                 * (counts[nz] / total))
    return float(ece)


def fast_aurac(y_true, y_prob):
    order = np.argsort(y_prob)[::-1]
    y_sorted = y_true[order]
    accs = np.cumsum(y_sorted) / np.arange(1, len(y_sorted) + 1)
    return float(np.mean(accs))


def compute_metrics(y_true, y_prob_ranked, y_prob_cal):
    """AUROC/AURAC on the pipeline-transformed scores; ECE both ways."""
    return {
        "AUROC": fast_auroc(y_true, y_prob_ranked),
        "ECE": fast_ece(y_true, y_prob_ranked),
        "ECE_cal": fast_ece(y_true, y_prob_cal) if y_prob_cal is not None else np.nan,
        "AURAC": fast_aurac(y_true, y_prob_ranked),
    }


# ----------------------------------------------------------------------------------
# Platt scaling on validation
# ----------------------------------------------------------------------------------
def fit_platt(scores, labels):
    """Return (predict_fn, slope). Unregularized-ish logistic fit on raw score."""
    from sklearn.linear_model import LogisticRegression

    scores = np.asarray(scores, dtype=float).reshape(-1, 1)
    labels = np.asarray(labels, dtype=int)
    if len(np.unique(labels)) < 2 or np.ptp(scores) == 0:
        base = labels.mean() if len(labels) else 0.5

        def predict(x):
            return np.full(len(x), base)

        return predict, 0.0
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=5000)
    lr.fit(scores, labels)

    def predict(x):
        return lr.predict_proba(np.asarray(x, dtype=float).reshape(-1, 1))[:, 1]

    return predict, float(lr.coef_[0][0])


# ----------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input-json", required=True)
    ap.add_argument("--cluster-cache", required=True)
    ap.add_argument("--run-dir", required=True,
                    help="Canonical run folder (e.g. .../run_0) whose "
                    "majority_weighted/metadata.json defines the aggregator.")
    ap.add_argument("--jsonl-path", default="unobench_processed.jsonl")
    ap.add_argument("--setting", choices=["majority_weighted", "no_majority"],
                    default="majority_weighted",
                    help="Which evaluation setting of final_evaluate.py to "
                    "replay. 'no_majority' evaluates the plain per-method "
                    "confidences (majority-vote answer selection, no weighted "
                    "voting) plus the self-consistency reference row.")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--num-boot", type=int, default=10000)
    ap.add_argument("--boot-seed", type=int, default=0)
    ap.add_argument("--verify-tol", type=float, default=5e-4,
                    help="Max |replicated - stored| metric difference allowed "
                    "in the run_0 verification.")
    args = ap.parse_args()

    meta = json.load(open(os.path.join(args.run_dir, args.setting,
                                       "metadata.json")))
    p = meta["parameters"]
    features = meta["features"]
    print(f"Aggregator: {meta['weight_search_method']}, features={features}")
    print(f"Split: seed={p['split_seed']}, val_fraction={p['val_fraction']}")

    records = json.load(open(args.input_json))
    cluster_info = json.load(open(args.cluster_cache))
    id_to_category = {}
    with open(args.jsonl_path) as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                id_to_category[item["question_id"]] = item.get("category", "Unknown")

    # aggregator scorers from stored metadata
    bw = meta["best_weights"]
    wm_fallback_scorer = make_scorer(bw["fallback"], features)
    wm_scorers = {cat: make_scorer(entry, features) or wm_fallback_scorer
                  for cat, entry in bw["per_category"].items()}

    test_records, val_records = stratified_test_val_split(
        records, id_to_category, p["val_fraction"], p["split_seed"]
    )
    print(f"Canonical split: {len(test_records)} test / {len(val_records)} val")

    collect = (collect_per_item if args.setting == "majority_weighted"
               else collect_per_item_nomaj)
    methods, test_items = collect(
        test_records, cluster_info, meta, wm_scorers, wm_fallback_scorer,
        id_to_category)
    _, val_items = collect(
        val_records, cluster_info, meta, wm_scorers, wm_fallback_scorer,
        id_to_category)

    uncertainty = set(p["uncertainty_baselines"])

    def wm_is_uncertainty(m):
        base = m[len("weighted_majority_"):] if m.startswith("weighted_majority_") else m
        return base in uncertainty

    # ---- organize per subset ----
    subset_names = sorted({c for c in test_items[methods[0]]["category"]})
    subset_names.append("Overall")

    def subset_mask(item_d, subset):
        cats = np.array(item_d["category"])
        return np.ones(len(cats), bool) if subset == "Overall" else cats == subset

    # ---- verification against the stored run metrics ----
    stored_majority = json.load(open(os.path.join(args.run_dir,
                                                  "eval_majority.json")))
    if args.setting == "majority_weighted":
        stored = stored_majority
    else:
        # TEST_METHODS live in eval.json; the SC reference row only exists in
        # eval_majority.json, so graft it in for verification.
        stored = json.load(open(os.path.join(args.run_dir, "eval.json")))
        for subset, d in stored_majority.items():
            if subset in stored and SELF_CONSISTENCY_KEY in d:
                stored[subset][SELF_CONSISTENCY_KEY] = d[SELF_CONSISTENCY_KEY]
    print("\n--- Verification vs stored run metrics ---")
    worst = 0.0
    for subset in subset_names:
        for m in methods:
            d = test_items[m]
            mask = subset_mask(d, subset)
            y_true = np.array(d["y_true"])[mask]
            y_prob = minmax_transform(np.array(d["y_prob"])[mask],
                                      wm_is_uncertainty(m))
            if len(np.unique(y_true)) < 2:
                continue
            rep = {"AUROC": fast_auroc(y_true, y_prob),
                   "ECE": fast_ece(y_true, y_prob),
                   "AURAC": fast_aurac(y_true, y_prob)}
            st = stored.get(subset, {}).get(m, {})
            for k in ["AUROC", "ECE", "AURAC"]:
                if st.get(k) is not None:
                    diff = abs(rep[k] - st[k])
                    worst = max(worst, diff)
                    if diff > args.verify_tol:
                        print(f"  MISMATCH {subset}/{m}/{k}: "
                              f"replicated={rep[k]:.6f} stored={st[k]:.6f}")
    print(f"Verification worst |diff| = {worst:.2e} "
          f"({'OK' if worst <= args.verify_tol else 'FAILED'})")
    if worst > args.verify_tol:
        raise SystemExit("Replication failed — refusing to compute statistics "
                         "on non-matching per-item scores.")

    # ---- Platt calibration (fit on validation, per subset w/ Overall fallback) ----
    os.makedirs(args.output_dir, exist_ok=True)
    calibrators = {}   # (subset, method) -> (predict_fn, slope, n_val)
    for subset in subset_names:
        for m in methods:
            dv = val_items[m]
            mask = subset_mask(dv, subset)
            s = np.array(dv["y_prob"])[mask]
            y = np.array(dv["y_true"])[mask]
            if len(np.unique(y)) < 2 and subset != "Overall":
                calibrators[(subset, m)] = None  # fall back to Overall later
                continue
            predict, slope = fit_platt(s, y)
            calibrators[(subset, m)] = (predict, slope, int(mask.sum()))
    for subset in subset_names:
        for m in methods:
            if calibrators[(subset, m)] is None:
                calibrators[(subset, m)] = calibrators[("Overall", m)]

    # ---- point estimates + bootstrap ----
    rng = np.random.default_rng(args.boot_seed)
    point_rows, ci_rows, delta_rows, slope_rows, pairwise_rows = [], [], [], [], []
    B = args.num_boot

    for subset in subset_names:
        # assemble aligned arrays (all methods share the same question order)
        ref = test_items[methods[0]]
        mask = subset_mask(ref, subset)
        n_items = int(mask.sum())
        y_true = np.array(ref["y_true"])[mask]  # per-method y_true may differ!
        arrs = {}
        for m in methods:
            d = test_items[m]
            m_mask = subset_mask(d, subset)
            yt = np.array(d["y_true"])[m_mask]
            raw = np.array(d["y_prob"])[m_mask]
            ranked = minmax_transform(raw, wm_is_uncertainty(m))
            predict, slope, n_val = calibrators[(subset, m)]
            cal = predict(raw)
            arrs[m] = (yt, ranked, cal)
            slope_rows.append([subset, m, slope, n_val])
            pm = compute_metrics(yt, ranked, cal)
            point_rows.append([subset, m, n_items, pm["AUROC"], pm["ECE"],
                               pm["ECE_cal"], pm["AURAC"]])

        # paired bootstrap: same index resample for every method
        boot = {m: {k: np.empty(B) for k in ["AUROC", "ECE", "ECE_cal", "AURAC"]}
                for m in methods}
        for b in range(B):
            idx = rng.integers(0, n_items, n_items)
            for m in methods:
                yt, ranked, cal = arrs[m]
                bm = compute_metrics(yt[idx], ranked[idx], cal[idx])
                for k, v in bm.items():
                    boot[m][k][b] = v

        for m in methods:
            row = [subset, m, n_items]
            for k in ["AUROC", "ECE", "ECE_cal", "AURAC"]:
                v = boot[m][k]
                v = v[~np.isnan(v)]
                row += [np.mean(v), np.percentile(v, 2.5), np.percentile(v, 97.5)]
            ci_rows.append(row)

        # paired deltas vs self-consistency (+ agg vs umpire & selfprobing)
        agg = "weighted_majority_aggregated_optimal"
        pairs = [(m, SELF_CONSISTENCY_KEY) for m in methods
                 if m != SELF_CONSISTENCY_KEY]
        for other in ["weighted_majority_umpire_normal",
                      "weighted_majority_confidence_score_selfprobing"]:
            if other in methods:
                pairs.append((agg, other))
        for a, bref in pairs:
            for k in ["AUROC", "ECE", "ECE_cal", "AURAC"]:
                da = boot[a][k] - boot[bref][k]
                da = da[~np.isnan(da)]
                if len(da) == 0:
                    continue
                lo, hi = np.percentile(da, 2.5), np.percentile(da, 97.5)
                p_le = np.mean(da <= 0)
                p_ge = np.mean(da >= 0)
                p_two = min(1.0, 2 * min(p_le, p_ge) + 1.0 / len(da))
                delta_rows.append([subset, a, bref, k, np.mean(da), lo, hi, p_two])

        # full pairwise p-value matrix (any best-vs-runner-up pair can be
        # looked up when starring tables)
        for i, a in enumerate(methods):
            for bref in methods[i + 1:]:
                for k in ["AUROC", "ECE", "ECE_cal", "AURAC"]:
                    da = boot[a][k] - boot[bref][k]
                    da = da[~np.isnan(da)]
                    if len(da) == 0:
                        continue
                    p_two = min(1.0, 2 * min(np.mean(da <= 0), np.mean(da >= 0))
                                + 1.0 / len(da))
                    pairwise_rows.append([subset, k, a, bref, np.mean(da), p_two])
        print(f"[{subset}] bootstrap done (n={n_items}, B={B})")

    # ---- write outputs ----
    def write_csv(name, header, rows):
        path = os.path.join(args.output_dir, name)
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            for r in rows:
                w.writerow([f"{x:.6f}" if isinstance(x, (float, np.floating))
                            else x for x in r])
        print(f"Wrote {path}")

    write_csv("point_estimates.csv",
              ["Subset", "Method", "N", "AUROC", "ECE_minmax", "ECE_calibrated",
               "AURAC"], point_rows)
    write_csv("bootstrap_cis.csv",
              ["Subset", "Method", "N",
               "AUROC_mean", "AUROC_lo", "AUROC_hi",
               "ECE_mean", "ECE_lo", "ECE_hi",
               "ECEcal_mean", "ECEcal_lo", "ECEcal_hi",
               "AURAC_mean", "AURAC_lo", "AURAC_hi"], ci_rows)
    write_csv("paired_deltas.csv",
              ["Subset", "Method", "Reference", "Metric", "Delta_mean",
               "Delta_lo", "Delta_hi", "p_boot"], delta_rows)
    write_csv("platt_slopes.csv",
              ["Subset", "Method", "Platt_slope", "N_val"], slope_rows)
    write_csv("pairwise_p.csv",
              ["Subset", "Metric", "MethodA", "MethodB", "Delta_mean",
               "p_boot"], pairwise_rows)

    meta_out = {
        "input_json": args.input_json,
        "cluster_cache": args.cluster_cache,
        "run_dir": args.run_dir,
        "num_boot": B,
        "boot_seed": args.boot_seed,
        "verification_worst_abs_diff": worst,
        "notes": "AUROC/AURAC use the pipeline's minmax-transformed scores "
                 "(rank-identical to the published tables). ECE_minmax is the "
                 "published split-rescaled ECE; ECE_calibrated applies Platt "
                 "scaling fit on the validation split only. Bootstrap is "
                 "paired over test questions.",
    }
    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(meta_out, f, indent=2)
    print("Done.")


if __name__ == "__main__":
    main()
