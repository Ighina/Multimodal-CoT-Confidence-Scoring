#!/usr/bin/env python3
"""
Transferability of the learned C_chain (aggregated_optimal) logistic-regression
aggregation weights in the paper's main ('no_majority') setting
(paper: Appendix G).

  (a) Cross-SPLIT transfer: fit the aggregator on the validation items of one
      split (source) of the canonical partition (seed 42, val_fraction 0.2)
      and evaluate on the test items of another split (target).
  (b) Cross-MODEL transfer: fit on Gemini validation items and evaluate on
      MiniCPM test items of the same split (features aligned by semantic
      role: S_smooth, S_goal, S_dens, G_avg, G_max), and vice versa.

Fitting replicates final_evaluate.py's published protocol exactly:
  StandardScaler + LogisticRegression(penalty='l1', solver='saga',
  max_iter=5000) on per-candidate features -> correctness, C grid
  {0.01,0.03,0.1,0.3,1.0,3.0,10.0}, model selection by question-level
  validation AUROC computed through the same sum_share group aggregation +
  majority-vote answer selection used everywhere, plus the safety net that
  falls back to the best single one-hot feature when the fitted LR does not
  strictly beat it on validation AUROC.

Verification gate: before any transfer number is produced, the stored run_0
aggregators are replayed on the canonical test split and the resulting
aggregated_optimal AUROC/AURAC must match bootstrap-results/*-nomaj/
point_estimates.csv within CSV rounding (6 decimals), for both datasets.

Outputs (bootstrap-results/transferability/):
  verification_gate.csv    replayed vs stored point estimates
  refit_vs_stored.csv      refit aggregator vs the stored run_0 aggregator
  cross_split_transfer.csv full source x target matrix per dataset
  cross_model_transfer.csv gemini<->minicpm per split (+ Overall, Pooled)
  best_single_context.csv  best single label-free score per split for context
  RESULTS.md               compact tables + takeaways
"""

import csv
import json
import os
import random
from collections import defaultdict

import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT, "bootstrap-results", "transferability")

SPLITS = ["UNOBench-Audio", "UNOBench-MC", "UNOBench-MO", "UNOBench-Visual"]
SHORT = {"UNOBench-Audio": "Audio", "UNOBench-MC": "MC",
         "UNOBench-MO": "MO", "UNOBench-Visual": "Visual"}
C_GRID = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
LR_PENALTY = "l1"

# Semantic roles, positionally aligned across the two datasets' feature lists.
SEMANTIC_ROLES = ["S_smooth", "S_goal", "S_dens", "G_avg", "G_max"]

CONFIG = {
    "gemini": {
        "input_json": os.path.join(ROOT, "gemini_cots_reformatted.json"),
        "cluster_cache": os.path.join(ROOT, "gemini-majority-vote-clusters_top6.json"),
        "run_dir": os.path.join(ROOT, "gemini-majority-multirun-logistic-reg", "run_0"),
        "point_csv": os.path.join(ROOT, "bootstrap-results", "gemini-nomaj",
                                  "point_estimates.csv"),
    },
    "minicpm": {
        "input_json": os.path.join(ROOT, "minicpm_cots_with_correctness_and_my_scores.json"),
        "cluster_cache": os.path.join(ROOT, "minicpm-majority-vote-clusters_top6.json"),
        "run_dir": os.path.join(ROOT, "minicpm-majority-multirun-logistic-reg", "run_0"),
        "point_csv": os.path.join(ROOT, "bootstrap-results", "minicpm-nomaj",
                                  "point_estimates.csv"),
    },
}
JSONL_PATH = os.path.join(ROOT, "unobench_processed.jsonl")


# ----------------------------------------------------------------------------------
# Ports of final_evaluate.py / bootstrap_significance.py primitives
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


def minmax_transform(y_prob):
    y_prob = np.asarray(y_prob, dtype=float)
    if len(y_prob) > 0:
        p_min, p_max = float(np.min(y_prob)), float(np.max(y_prob))
        if p_max > p_min:
            y_prob = (y_prob - p_min) / (p_max - p_min)
        else:
            y_prob = np.zeros_like(y_prob)
    return y_prob


def fast_auroc(y_true, y_prob):
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.nan
    ranks = rankdata(y_prob)
    return (ranks[y_true == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def fast_aurac(y_true, y_prob):
    order = np.argsort(y_prob)[::-1]
    y_sorted = y_true[order]
    accs = np.cumsum(y_sorted) / np.arange(1, len(y_sorted) + 1)
    return float(np.mean(accs))


def eval_metrics(y_true, y_prob):
    """Published AUROC/AURAC: minmax-transform the subset's scores first."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = minmax_transform(y_prob)
    return {"AUROC": float(fast_auroc(y_true, y_prob)),
            "AURAC": float(fast_aurac(y_true, y_prob))}


def make_scorer_from_entry(entry, features):
    """Scorer from a stored metadata best_weights entry (bootstrap port)."""
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
# Data loading / preparation
# ----------------------------------------------------------------------------------
def load_dataset(name):
    cfg = CONFIG[name]
    meta = json.load(open(os.path.join(cfg["run_dir"], "no_majority",
                                       "metadata.json")))
    p = meta["parameters"]
    records = json.load(open(cfg["input_json"]))
    cluster_info = json.load(open(cfg["cluster_cache"]))
    id_to_category = {}
    with open(JSONL_PATH) as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                id_to_category[item["question_id"]] = item.get("category", "Unknown")

    test_records, val_records = stratified_test_val_split(
        records, id_to_category, p["val_fraction"], p["split_seed"])

    features = meta["features"]
    threshold = p["correctness_threshold"]

    def prepare(recs):
        """Per-question prepared items (same admission rules as the pipeline)."""
        items = []
        for rec in recs:
            qid = str(rec.get("question_id"))
            ci = cluster_info.get(qid)
            if not ci or not ci.get("groups"):
                continue
            groups = ci["groups"]
            n = len(groups)
            confs = rec.get("generations_confidence", []) or []
            correctness = get_correctness(rec, threshold)
            if correctness is None or len(confs) < n or len(correctness) < n:
                continue
            feat = np.array(
                [[float(confs[i].get(f, 0.0)) for f in features] for i in range(n)],
                dtype=float)
            items.append({
                "qid": qid,
                "category": id_to_category.get(rec.get("question_id"), "Unknown"),
                "groups": groups,
                "groups_arr": np.array(groups, dtype=int),
                "majority_ids": ci["majority_ids"],
                "correctness": np.array(correctness[:n], dtype=int),
                "feat": feat,
            })
        return items

    return {
        "name": name,
        "meta": meta,
        "features": features,
        "test_items": prepare(test_records),
        "val_items": prepare(val_records),
        "point_csv": cfg["point_csv"],
    }


def by_split(items):
    d = defaultdict(list)
    for it in items:
        d[it["category"]].append(it)
    return d


# ----------------------------------------------------------------------------------
# Applying an aggregator (no_majority evaluation of aggregated_optimal)
# ----------------------------------------------------------------------------------
def apply_scorer(items, scorer):
    """Replay of the pipeline's aggregated_optimal path: sum_share group
    confidence, majority-vote answer with confidence tie-break, y_prob =
    winning group's sum-share."""
    y_true, y_prob = [], []
    for it in items:
        scores = np.asarray(scorer(it["feat"]), dtype=float)
        groups_arr = it["groups_arr"]
        n_groups = int(groups_arr.max()) + 1
        group_sum = np.bincount(groups_arr, weights=scores, minlength=n_groups)
        norm_sum = group_sum / (group_sum.sum() + 1e-9)
        majority_ids = it["majority_ids"]
        if len(majority_ids) == 1:
            winning_group = majority_ids[0]
        else:
            winning_group = max(majority_ids, key=lambda g: norm_sum[g])
        winning_index = it["groups"].index(winning_group)
        y_true.append(int(it["correctness"][winning_index]))
        y_prob.append(float(norm_sum[winning_group]))
    return np.array(y_true, dtype=int), np.array(y_prob, dtype=float)


def apply_scorer_per_category(items, scorers, fallback):
    """Evaluate items in their natural record order, picking each item's
    scorer by its category (mirrors the published Overall construction;
    element order matters for AURAC tie-breaking)."""
    y_true, y_prob, cats = [], [], []
    for it in items:
        scorer = scorers.get(it["category"], fallback)
        yt, yp = apply_scorer([it], scorer)
        y_true.append(int(yt[0]))
        y_prob.append(float(yp[0]))
        cats.append(it["category"])
    return (np.array(y_true, dtype=int), np.array(y_prob, dtype=float),
            np.array(cats))


def val_auroc_of_scorer(items, scorer):
    y_true, y_prob = apply_scorer(items, scorer)
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return None
    return float(fast_auroc(y_true, minmax_transform(y_prob)))


# ----------------------------------------------------------------------------------
# Refit of the published aggregator protocol on a set of validation items
# ----------------------------------------------------------------------------------
def fit_cchain(val_items, features, label=""):
    """final_evaluate.py's _find_weights_logistic_regression +
    find_best_aggregation_weights safety net, ported 1:1."""
    X = np.vstack([it["feat"] for it in val_items])
    y = np.concatenate([it["correctness"] for it in val_items])
    if len(X) < 2 or len(np.unique(y)) < 2:
        raise RuntimeError(f"[{label}] not enough data / single class")

    solver = "saga" if LR_PENALTY == "l1" else "lbfgs"
    best_model, best_val, best_c = None, -np.inf, None
    for c in C_GRID:
        # random_state=0 is the ONLY deviation from final_evaluate.py (which
        # leaves the saga solver unseeded): it fixes the solver's shuffling so
        # this script is run-to-run deterministic. The converged solutions
        # agree with unseeded runs to ~1e-6 in the coefficients.
        candidate = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=5000, C=c, penalty=LR_PENALTY,
                                      solver=solver, random_state=0)),
        ])
        try:
            candidate.fit(X, y)
        except Exception as e:
            print(f"[{label}] C={c} failed to fit: {e}")
            continue

        def lr_score(F, m=candidate):
            return m.predict_proba(F)[:, 1]

        value = val_auroc_of_scorer(val_items, lr_score)
        if value is None:
            continue
        if best_model is None or value > best_val:
            best_val, best_model, best_c = value, candidate, c

    if best_model is None:
        raise RuntimeError(f"[{label}] no valid LR across the C grid")

    # safety net: best single one-hot feature (strictly better wins)
    single_feat, single_val = None, -np.inf
    for idx, feat in enumerate(features):
        w = np.zeros(len(features))
        w[idx] = 1.0

        def dot_score(F, w=w):
            return F @ w

        value = val_auroc_of_scorer(val_items, dot_score)
        if value is None:
            continue
        if single_feat is None or value > single_val:
            single_feat, single_val = feat, value

    if single_feat is not None and single_val > best_val:
        idx = features.index(single_feat)
        w = np.zeros(len(features))
        w[idx] = 1.0

        def scorer(F, w=w):
            return F @ w

        return {"kind": "single", "feature": single_feat, "feature_idx": idx,
                "role": SEMANTIC_ROLES[idx], "scorer": scorer,
                "val_auroc": single_val,
                "desc": f"single:{SEMANTIC_ROLES[idx]}"}

    lr = best_model.named_steps["lr"]
    scaler = best_model.named_steps["scaler"]

    def scorer(F, m=best_model):
        return m.predict_proba(F)[:, 1]

    degenerate = bool(np.all(lr.coef_[0] == 0.0))
    return {"kind": "lr", "model": best_model, "C": best_c,
            "coef": lr.coef_[0].copy(), "intercept": float(lr.intercept_[0]),
            "scaler_mean": scaler.mean_.copy(),
            "scaler_scale": scaler.scale_.copy(),
            "scorer": scorer, "val_auroc": best_val,
            "degenerate": degenerate,
            "desc": f"LR(C={best_c})" + (" [all-zero coef]" if degenerate else "")}


# ----------------------------------------------------------------------------------
# Stored point estimates
# ----------------------------------------------------------------------------------
def load_point_estimates(path):
    """{(subset, method): {AUROC, AURAC}}"""
    out = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            out[(row["Subset"], row["Method"])] = {
                "AUROC": float(row["AUROC"]), "AURAC": float(row["AURAC"])}
    return out


# ----------------------------------------------------------------------------------
# Main analysis
# ----------------------------------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    data = {name: load_dataset(name) for name in CONFIG}
    stored_pe = {name: load_point_estimates(d["point_csv"])
                 for name, d in data.items()}

    # =========================== 1. VERIFICATION GATE ===========================
    print("=" * 70)
    print("STEP 1: verification gate (replay stored aggregators, no_majority)")
    gate_rows, gate_ok = [], True
    TOL = 1e-6  # point_estimates.csv is rounded to 6 decimals
    for name, d in data.items():
        bw = d["meta"]["best_weights"]
        fallback = make_scorer_from_entry(bw["fallback"], d["features"])
        scorers = {cat: make_scorer_from_entry(e, d["features"]) or fallback
                   for cat, e in bw["per_category"].items()}
        # evaluate every test item in its natural record order with its own
        # category's scorer (element order matters for AURAC tie-breaking)
        yt_all, yp_all, cats = apply_scorer_per_category(
            d["test_items"], scorers, fallback)
        for subset in SPLITS + ["Overall"]:
            mask = np.ones(len(cats), bool) if subset == "Overall" else cats == subset
            m = eval_metrics(yt_all[mask], yp_all[mask])
            for metric in ["AUROC", "AURAC"]:
                st = stored_pe[name][(subset, "aggregated_optimal")][metric]
                diff = abs(m[metric] - st)
                ok = diff <= TOL
                gate_ok &= ok
                gate_rows.append([name, subset, metric, m[metric], st, diff,
                                  "OK" if ok else "MISMATCH"])

    worst = max(r[5] for r in gate_rows)
    with open(os.path.join(OUT_DIR, "verification_gate.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Dataset", "Subset", "Metric", "Replayed", "Stored",
                    "AbsDiff", "Status"])
        for r in gate_rows:
            w.writerow([r[0], r[1], r[2], f"{r[3]:.10f}", f"{r[4]:.6f}",
                        f"{r[5]:.2e}", r[6]])
    print(f"Verification gate: worst |diff| = {worst:.2e} "
          f"({'PASS' if gate_ok else 'FAIL'})")
    if not gate_ok:
        raise SystemExit("VERIFICATION GATE FAILED — stopping before any "
                         "transfer computation.")

    # ====================== 2. refit + refit-vs-stored check ====================
    print("=" * 70)
    print("STEP 2: refit the published aggregator per split (source fits)")
    fits = {}       # (dataset, split_or_Pooled) -> fit dict
    refit_rows = []
    for name, d in data.items():
        val_by = by_split(d["val_items"])
        for split in SPLITS:
            fits[(name, split)] = fit_cchain(val_by[split], d["features"],
                                             label=f"{name}/{split}")
        fits[(name, "Pooled")] = fit_cchain(d["val_items"], d["features"],
                                            label=f"{name}/Pooled")

        # compare with the stored run_0 aggregator
        bw = d["meta"]["best_weights"]
        for split in SPLITS + ["Pooled"]:
            entry = (bw["fallback"] if split == "Pooled"
                     else bw["per_category"].get(split))
            fit = fits[(name, split)]
            stored_kind = ("lr" if entry and entry.get("lr_model") else "single")
            if stored_kind == "single" and entry:
                sw = entry["weights"]
                stored_desc = "single:" + SEMANTIC_ROLES[
                    int(np.argmax([sw[f] for f in d["features"]]))]
            elif entry:
                stored_desc = "lr"
            else:
                stored_desc = "none"
            match = (fit["kind"] == stored_kind)
            detail = ""
            if match and fit["kind"] == "single":
                match = stored_desc == fit["desc"]
            elif match and fit["kind"] == "lr":
                sc = np.array([entry["lr_model"]["coefficients"][f]
                               for f in d["features"]])
                dc = float(np.max(np.abs(sc - fit["coef"])))
                di = abs(entry["lr_model"]["intercept"] - fit["intercept"])
                detail = f"max|coef diff|={dc:.2e}, |intercept diff|={di:.2e}"
                if (dc == 0.0 and np.all(sc == 0.0) and np.all(fit["coef"] == 0.0)):
                    # all-zero coefficients -> constant score -> the intercept
                    # is irrelevant (confidence collapses to the vote share);
                    # predictions are bit-identical regardless of intercept.
                    detail += "; all-zero coefs, constant scorer => functionally identical"
                    match = True
                else:
                    match = dc < 1e-2 and di < 1e-2
            refit_rows.append([name, split, fit["desc"], stored_desc,
                               f"{fit['val_auroc']:.6f}",
                               "MATCH" if match else "DIFFERS", detail])
            print(f"  {name:8s} {SHORT.get(split, split):7s} refit={fit['desc']:20s}"
                  f" stored={stored_desc:20s} "
                  f"{'MATCH' if match else 'DIFFERS ' + detail}")
    with open(os.path.join(OUT_DIR, "refit_vs_stored.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Dataset", "Split", "RefitAggregator", "StoredAggregator",
                    "RefitValAUROC", "Agreement", "Detail"])
        w.writerows(refit_rows)

    # Tie-noise band for degenerate (all-zero-coefficient) LR fits: a constant
    # scorer collapses the confidence to the vote share, and the exact metric
    # value then depends on ulp-level floating-point ties (different constants
    # break the massive vote-share ties differently). Quantify the band.
    tie_bands = []
    for (name, split), fit in fits.items():
        if fit.get("degenerate") and split in SPLITS:
            items = by_split(data[name]["test_items"])[split]
            aurocs, auracs = [], []
            for c in np.linspace(0.25, 0.75, 11):
                yt, yp = apply_scorer(items, lambda F, c=c: np.full(len(F), c))
                m = eval_metrics(yt, yp)
                aurocs.append(m["AUROC"])
                auracs.append(m["AURAC"])
            tie_bands.append([name, SHORT[split],
                              min(aurocs), max(aurocs), min(auracs), max(auracs)])
            print(f"  NOTE: {name}/{SHORT[split]} fit is a constant scorer "
                  f"(all-zero coef) — tie-noise band AUROC "
                  f"[{min(aurocs):.3f},{max(aurocs):.3f}], AURAC "
                  f"[{min(auracs):.3f},{max(auracs):.3f}]")

    # ========================= 3. cross-split transfer ==========================
    print("=" * 70)
    print("STEP 3: cross-split transfer matrices")
    cross_split_rows = []
    for name, d in data.items():
        test_by = by_split(d["test_items"])
        for src in SPLITS + ["Pooled"]:
            fit = fits[(name, src)]
            for tgt in SPLITS:
                yt, yp = apply_scorer(test_by[tgt], fit["scorer"])
                m = eval_metrics(yt, yp)
                ins = stored_pe[name][(tgt, "aggregated_optimal")]
                cross_split_rows.append([
                    name, SHORT.get(src, src), SHORT[tgt], fit["desc"],
                    m["AUROC"], m["AURAC"],
                    ins["AUROC"], ins["AURAC"],
                    m["AUROC"] - ins["AUROC"], m["AURAC"] - ins["AURAC"]])
    with open(os.path.join(OUT_DIR, "cross_split_transfer.csv"), "w",
              newline="") as f:
        w = csv.writer(f)
        w.writerow(["Dataset", "SourceSplit", "TargetSplit", "SourceAggregator",
                    "AUROC", "AURAC", "InSplit_AUROC", "InSplit_AURAC",
                    "Delta_AUROC", "Delta_AURAC"])
        for r in cross_split_rows:
            w.writerow(r[:4] + [f"{x:.6f}" for x in r[4:]])

    # ========================= 4. cross-model transfer ==========================
    print("=" * 70)
    print("STEP 4: cross-model transfer (semantic-role feature alignment)")
    cross_model_rows = []
    for src_name, tgt_name in [("gemini", "minicpm"), ("minicpm", "gemini")]:
        tgt = data[tgt_name]
        test_by = by_split(tgt["test_items"])
        # per split (feature matrices are positionally aligned by semantic role,
        # so a source-fitted scorer — including its StandardScaler — applies
        # directly to the target's feature matrix)
        for split in SPLITS:
            fit = fits[(src_name, split)]
            yt, yp = apply_scorer(test_by[split], fit["scorer"])
            m = eval_metrics(yt, yp)
            nat = stored_pe[tgt_name][(split, "aggregated_optimal")]
            cross_model_rows.append([
                f"{src_name}->{tgt_name}", SHORT[split], fit["desc"],
                m["AUROC"], m["AURAC"], nat["AUROC"], nat["AURAC"],
                m["AUROC"] - nat["AUROC"], m["AURAC"] - nat["AURAC"]])
        # Overall = per-category transferred scorers, evaluated over all test
        # items in natural record order (mirrors the published Overall row)
        transferred = {split: fits[(src_name, split)]["scorer"]
                       for split in SPLITS}
        yt_all, yp_all, _cats = apply_scorer_per_category(
            tgt["test_items"], transferred, fits[(src_name, "Pooled")]["scorer"])
        m = eval_metrics(yt_all, yp_all)
        nat = stored_pe[tgt_name][("Overall", "aggregated_optimal")]
        cross_model_rows.append([
            f"{src_name}->{tgt_name}", "Overall", "per-split",
            m["AUROC"], m["AURAC"], nat["AUROC"], nat["AURAC"],
            m["AUROC"] - nat["AUROC"], m["AURAC"] - nat["AURAC"]])
        # Pooled source fit evaluated on all target test items
        fit = fits[(src_name, "Pooled")]
        yt, yp = apply_scorer(tgt["test_items"], fit["scorer"])
        m = eval_metrics(yt, yp)
        cross_model_rows.append([
            f"{src_name}->{tgt_name}", "Pooled", fit["desc"],
            m["AUROC"], m["AURAC"], nat["AUROC"], nat["AURAC"],
            m["AUROC"] - nat["AUROC"], m["AURAC"] - nat["AURAC"]])
    with open(os.path.join(OUT_DIR, "cross_model_transfer.csv"), "w",
              newline="") as f:
        w = csv.writer(f)
        w.writerow(["Direction", "Split", "SourceAggregator", "AUROC", "AURAC",
                    "Native_AUROC", "Native_AURAC", "Delta_AUROC",
                    "Delta_AURAC"])
        for r in cross_model_rows:
            w.writerow(r[:3] + [f"{x:.6f}" for x in r[3:]])

    # ================== 5. best single label-free score context =================
    print("=" * 70)
    print("STEP 5: best single label-free score per split (context)")
    LABEL_FREE_EXCLUDE = {"aggregated_optimal", "umpire_normal"}
    context_rows = []
    for name, d in data.items():
        chain_feats = set(d["features"])
        pe = stored_pe[name]
        for split in SPLITS + ["Overall"]:
            methods = [m for (s, m) in pe if s == split]
            chain = [m for m in methods if m in chain_feats]
            free = [m for m in methods if m not in LABEL_FREE_EXCLUDE]
            best_chain = max(chain, key=lambda m: pe[(split, m)]["AUROC"])
            best_free = max(free, key=lambda m: pe[(split, m)]["AUROC"])
            context_rows.append([
                name, SHORT.get(split, split),
                best_chain, pe[(split, best_chain)]["AUROC"],
                pe[(split, best_chain)]["AURAC"],
                best_free, pe[(split, best_free)]["AUROC"],
                pe[(split, best_free)]["AURAC"],
                pe[(split, "aggregated_optimal")]["AUROC"],
                pe[(split, "aggregated_optimal")]["AURAC"]])
    with open(os.path.join(OUT_DIR, "best_single_context.csv"), "w",
              newline="") as f:
        w = csv.writer(f)
        w.writerow(["Dataset", "Split", "BestChainFeature", "BestChain_AUROC",
                    "BestChain_AURAC", "BestLabelFree", "BestLabelFree_AUROC",
                    "BestLabelFree_AURAC", "CChain_AUROC", "CChain_AURAC"])
        for r in context_rows:
            w.writerow([x if isinstance(x, str) else f"{x:.6f}" for x in r])

    # ============================== 6. RESULTS.md ===============================
    write_results_md(gate_rows, worst, refit_rows, cross_split_rows,
                     cross_model_rows, context_rows, data, tie_bands)
    print(f"\nAll outputs written to {OUT_DIR}")


# ----------------------------------------------------------------------------------
# Markdown report
# ----------------------------------------------------------------------------------
def write_results_md(gate_rows, gate_worst, refit_rows, cross_split_rows,
                     cross_model_rows, context_rows, data, tie_bands):
    L = []
    L.append("# C_chain transferability (sampling-free / no_majority setting)\n")
    L.append("Canonical partition (split_seed=42, val_fraction=0.2); the "
             "aggregator is refit with the published protocol (L1 logistic "
             "regression over the 5 chain features, C grid "
             "{0.01,...,10}, selection by validation AUROC, single-feature "
             "safety net) on the SOURCE split's validation items and "
             "evaluated on the TARGET split's test items. Cross-model "
             "transfer aligns features by semantic role "
             "(S_smooth, S_goal, S_dens, G_avg, G_max) and carries the "
             "source-fitted StandardScaler with the model. Only deviation "
             "from the pipeline: the saga solver is seeded (random_state=0) "
             "for run-to-run determinism.\n")

    L.append("## Verification gate\n")
    L.append(f"Replayed in-split `aggregated_optimal` AUROC/AURAC match the "
             f"stored point estimates for all subsets of both datasets; "
             f"worst |replayed - stored| = {gate_worst:.2e} (within the "
             f"6-decimal rounding of point_estimates.csv). PASS.\n")

    L.append("## Refit vs stored run_0 aggregator\n")
    L.append("| Dataset | Split | Refit | Stored | Agreement |")
    L.append("| :-- | :-- | :-- | :-- | :-- |")
    for r in refit_rows:
        L.append(f"| {r[0]} | {SHORT.get(r[1], r[1])} | {r[2]} | {r[3]} | "
                 f"{r[5]}{(' (' + r[6] + ')') if r[6] else ''} |")
    L.append("")

    # cross-split matrices
    L.append("## Cross-split transfer (fit on source val, eval on target test)\n")
    for name in data:
        rows = [r for r in cross_split_rows if r[0] == name]
        for metric, i_val, i_ins in [("AUROC", 4, 6), ("AURAC", 5, 7)]:
            L.append(f"### {name} — {metric} (target in-split value in the "
                     f"diagonal position; delta = transfer − in-split)\n")
            L.append("| source \\ target | " + " | ".join(SHORT[s] for s in SPLITS)
                     + " |")
            L.append("| :-- " + "| :--: " * len(SPLITS) + "|")
            for src in [SHORT[s] for s in SPLITS] + ["Pooled"]:
                cells = []
                for tgt_s in SPLITS:
                    r = next(r for r in rows if r[1] == src and r[2] == SHORT[tgt_s])
                    v, ins = r[i_val], r[i_ins]
                    if src == SHORT[tgt_s]:
                        cells.append(f"**{v:.3f}**")
                    else:
                        cells.append(f"{v:.3f} ({v - ins:+.3f})")
                L.append(f"| {src} | " + " | ".join(cells) + " |")
            L.append("")

    L.append("## Cross-model transfer\n")
    L.append("| Direction | Split | Source aggregator | AUROC | native | dAUROC "
             "| AURAC | native | dAURAC |")
    L.append("| :-- | :-- | :-- | :--: | :--: | :--: | :--: | :--: | :--: |")
    for r in cross_model_rows:
        L.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]:.3f} | {r[5]:.3f} | "
                 f"{r[7]:+.3f} | {r[4]:.3f} | {r[6]:.3f} | {r[8]:+.3f} |")
    L.append("")

    L.append("## Context: best single label-free score per split "
             "(stored test-split point estimates)\n")
    L.append("| Dataset | Split | Best chain feature (AUROC/AURAC) | "
             "Best label-free method (AUROC/AURAC) | C_chain in-split |")
    L.append("| :-- | :-- | :-- | :-- | :-- |")
    for r in context_rows:
        L.append(f"| {r[0]} | {r[1]} | {r[2]} ({r[3]:.3f}/{r[4]:.3f}) | "
                 f"{r[5]} ({r[6]:.3f}/{r[7]:.3f}) | {r[8]:.3f}/{r[9]:.3f} |")
    L.append("")

    if tie_bands:
        L.append("## Caveat: degenerate (constant) fits\n")
        L.append("The following refit aggregators have ALL-ZERO L1 "
                 "coefficients: the scorer is constant, so the reported "
                 "confidence collapses to the majority-vote share and the "
                 "exact AUROC/AURAC depend on ulp-level floating-point "
                 "tie-breaking (different constants break the massive "
                 "vote-share ties differently). Their point values — "
                 "including the published in-split value they are compared "
                 "against — should be read within this band:\n")
        L.append("| Dataset | Split | AUROC band | AURAC band |")
        L.append("| :-- | :-- | :--: | :--: |")
        for b in tie_bands:
            L.append(f"| {b[0]} | {b[1]} | [{b[2]:.3f}, {b[3]:.3f}] | "
                     f"[{b[4]:.3f}, {b[5]:.3f}] |")
        L.append("")

    # --------- takeaways (computed) ---------
    L.append("## Takeaways\n")
    for name in data:
        off = [r for r in cross_split_rows if r[0] == name
               and r[1] != "Pooled" and r[1] != r[2]]
        d_auroc = [r[8] for r in off]
        d_aurac = [r[9] for r in off]
        L.append(f"- **{name} cross-split:** mean dAUROC = "
                 f"{np.mean(d_auroc):+.3f} (range {min(d_auroc):+.3f} to "
                 f"{max(d_auroc):+.3f}); mean dAURAC = {np.mean(d_aurac):+.3f} "
                 f"(range {min(d_aurac):+.3f} to {max(d_aurac):+.3f}).")
    for direction in ["gemini->minicpm", "minicpm->gemini"]:
        rows = [r for r in cross_model_rows
                if r[0] == direction and r[1] in SHORT.values()]
        d_auroc = [r[7] for r in rows]
        ov = next(r for r in cross_model_rows
                  if r[0] == direction and r[1] == "Overall")
        L.append(f"- **{direction}:** per-split mean dAUROC = "
                 f"{np.mean(d_auroc):+.3f}; Overall transfer AUROC "
                 f"{ov[3]:.3f} vs native {ov[5]:.3f} ({ov[7]:+.3f}), "
                 f"AURAC {ov[4]:.3f} vs {ov[6]:.3f} ({ov[8]:+.3f}).")
    L.append("")
    with open(os.path.join(OUT_DIR, "RESULTS.md"), "w") as f:
        f.write("\n".join(L))


if __name__ == "__main__":
    main()
