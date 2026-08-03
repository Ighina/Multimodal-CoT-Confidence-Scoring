#!/usr/bin/env python3
"""
Hybrid within-chain + between-sample experiment.

Fits the weighted_majority_aggregated_optimal pipeline of final_evaluate.py
(StandardScaler + L1 LogisticRegression per category with pooled fallback,
C grid selected by validation AUROC through the group-aggregation evaluation,
safety net against the best single feature) over the same 100 split seeds
(42..141) used by gemini-additional-baselines-multirun, for two feature sets:

  * chain5  : the five chain features (port validation: should reproduce the
              stored weighted_majority_aggregated_optimal row up to solver
              noise);
  * hybrid7 : chain5 + two question-level dispersion features computed from
              the per-candidate answer embeddings (negated RDS and Semantic
              Volume, constant within a question).

Reports mean +/- std per split and paired t-tests vs the stored rows.

Paper: the hybrid "C_chain + dispersion" aggregate of Appendix M (Table 21).
"""

import csv
import json
import os
from collections import Counter, defaultdict

import numpy as np
from scipy.stats import ttest_rel
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import bootstrap_significance as bs

# Working root holding the released data artifacts and stored run outputs.
# Override with the OMNIEVAL_ROOT environment variable; defaults to the
# current working directory (see evaluation/README.md).
ROOT = os.environ.get("OMNIEVAL_ROOT", os.getcwd())
MULTIRUN = os.path.join(ROOT, "gemini-additional-baselines-multirun")
SUBSETS = ["Overall", "UNOBench-Audio", "UNOBench-MC", "UNOBench-MO",
           "UNOBench-Visual"]
CHAIN5 = ["internal_smoothness", "internal_goal_directedness",
          "internal_semantic_density", "cross_modal_coherence",
          "cross_modal_grounding_max"]
C_GRID = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
METRICS = ["AUROC", "ECE", "AURAC"]


def load_questions():
    records = json.load(open(os.path.join(
        ROOT, "gemini_cots_with_additional_baselines.json")))
    cluster_info = json.load(open(os.path.join(
        ROOT, "gemini-majority-vote-clusters_top6.json")))
    embs = {str(e["question_id"]): e["embeddings"]
            for e in json.load(open(os.path.join(
                ROOT, "gemini-answer-embeddings.json")))}
    id_to_category = {}
    for line in open(os.path.join(ROOT, "unobench_processed.jsonl")):
        if line.strip():
            it = json.loads(line)
            id_to_category[it["question_id"]] = it.get("category", "Unknown")

    qs = {}
    for rec in records:
        qid = str(rec.get("question_id"))
        ci = cluster_info.get(qid)
        if not ci or not ci.get("groups"):
            continue
        groups = ci["groups"]
        n = len(groups)
        confs = rec.get("generations_confidence", []) or []
        corr = bs.get_correctness(rec, 0.5)
        E_all = embs.get(qid)
        if corr is None or len(confs) < n or len(corr) < n or E_all is None \
                or len(E_all) < n:
            continue
        E = np.asarray(E_all[:n], dtype=float)
        c = E.mean(axis=0)
        rds = -float(np.mean(np.linalg.norm(E - c, axis=1)))
        X = E - c
        _s, logdet = np.linalg.slogdet(X @ X.T + 1e-6 * np.eye(n))
        sv = -float(logdet)
        feat5 = np.array([[float(confs[i].get(f, 0.0)) for f in CHAIN5]
                          for i in range(n)])
        feat7 = np.hstack([feat5, np.full((n, 1), rds), np.full((n, 1), sv)])
        qs[qid] = {
            "cat": id_to_category.get(rec.get("question_id"), "Unknown"),
            "groups": np.array(groups, dtype=int),
            "majority_ids": ci["majority_ids"],
            "corr": np.array(corr[:n], dtype=int),
            "feat5": feat5,
            "feat7": feat7,
        }
    return qs


def eval_scores_wm(q, scores):
    """weighted_majority evaluation of per-sample scores for one question."""
    S_g, norm_sum, _ = bs.aggregate_group_scores(q["groups"], scores)
    wg = int(np.argmax(S_g))
    wi = int(np.argmax(q["groups"] == wg))
    return int(q["corr"][wi]), float(norm_sum[wg])


def summarize(y, p):
    y = np.array(y)
    p = bs.minmax_transform(np.array(p, dtype=float))
    if len(np.unique(y)) < 2:
        return None
    return {"AUROC": bs.fast_auroc(y, p), "ECE": bs.fast_ece(y, p),
            "AURAC": bs.fast_aurac(y, p)}


def val_auroc_for_scorer(qlist, scorer, key):
    ys, ps = [], []
    for q in qlist:
        yt, pp = eval_scores_wm(q, scorer(q[key]))
        ys.append(yt)
        ps.append(pp)
    s = summarize(ys, ps)
    return None if s is None else s["AUROC"]


def fit_group(val_qs, key, n_feat, label):
    """Port of _find_weights_logistic_regression + safety net for
    selection_mode='weighted_majority'. Returns a per-sample scorer fn."""
    X = np.vstack([q[key] for q in val_qs])
    y = np.concatenate([q["corr"] for q in val_qs])
    if len(np.unique(y)) < 2 or len(X) < 2:
        raise RuntimeError("degenerate validation")
    best_model, best_val = None, -np.inf
    for c in C_GRID:
        pipe = Pipeline([("scaler", StandardScaler()),
                         ("lr", LogisticRegression(max_iter=5000, C=c,
                                                   penalty="l1",
                                                   solver="saga"))])
        try:
            pipe.fit(X, y)
        except Exception:
            continue
        v = val_auroc_for_scorer(
            val_qs, lambda F, m=pipe: m.predict_proba(F)[:, 1], key)
        if v is not None and v > best_val:
            best_val, best_model = v, pipe
    if best_model is None:
        raise RuntimeError("no LR fit")
    # safety net: best single feature (one-hot weight vector)
    best_feat, best_feat_val = None, -np.inf
    for j in range(n_feat):
        v = val_auroc_for_scorer(val_qs, lambda F, jj=j: F[:, jj], key)
        if v is not None and v > best_feat_val:
            best_feat_val, best_feat = v, j
    if best_feat is not None and best_feat_val > best_val:
        return lambda F, jj=best_feat: F[:, jj]
    return lambda F, m=best_model: m.predict_proba(F)[:, 1]


def run_seed(qs, rec_stubs, id_to_category, seed, key, n_feat):
    test, val = bs.stratified_test_val_split(rec_stubs, id_to_category,
                                             0.2, seed)
    val_qs = [qs[str(r["question_id"])] for r in val]
    test_qs = [qs[str(r["question_id"])] for r in test]

    fallback = fit_group(val_qs, key, n_feat, "fallback")
    by_cat = defaultdict(list)
    for q in val_qs:
        by_cat[q["cat"]].append(q)
    scorers = {}
    for cat, cat_qs in by_cat.items():
        try:
            scorers[cat] = fit_group(cat_qs, key, n_feat, cat)
        except RuntimeError:
            scorers[cat] = fallback

    collected = defaultdict(lambda: {"y": [], "p": []})
    for q in test_qs:
        scorer = scorers.get(q["cat"], fallback)
        yt, pp = eval_scores_wm(q, np.asarray(scorer(q[key]), dtype=float))
        for subset in ["Overall", q["cat"]]:
            collected[subset]["y"].append(yt)
            collected[subset]["p"].append(pp)
    return {s: summarize(d["y"], d["p"]) for s, d in collected.items()}


def main():
    print("Loading ...")
    qs = load_questions()
    print(f"{len(qs)} questions")
    id_to_category = {}
    for line in open(os.path.join(ROOT, "unobench_processed.jsonl")):
        if line.strip():
            it = json.loads(line)
            id_to_category[it["question_id"]] = it.get("category", "Unknown")
    rec_stubs = [{"question_id": int(qid) if qid.isdigit() else qid}
                 for qid in qs]

    results = {"chain5": defaultdict(list), "hybrid7": defaultdict(list)}
    for i, seed in enumerate(range(42, 142)):
        for key, nf, tag in [("feat5", 5, "chain5"), ("feat7", 7, "hybrid7")]:
            r = run_seed(qs, rec_stubs, id_to_category, seed, key, nf)
            for subset in SUBSETS:
                results[tag][subset].append(
                    r.get(subset) or {m: np.nan for m in METRICS})
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/100 seeds done")

    # stored rows for comparison
    stored = {lbl: {s: [] for s in SUBSETS} for lbl in
              ["weighted_majority_aggregated_optimal",
               "majority_vote_selfconsistency"]}
    for i in range(100):
        for subset in SUBSETS:
            rows = {r["Method"]: r for r in csv.DictReader(open(os.path.join(
                MULTIRUN, f"run_{i}", "majority_weighted", f"{subset}.csv")))}
            for lbl in stored:
                stored[lbl][subset].append(
                    {m: float(rows[lbl][m]) if rows[lbl][m] not in ("N/A", "")
                     else np.nan for m in METRICS})

    def arr(d, subset, metric):
        return np.array([x[metric] for x in d[subset]])

    print("\n=== port validation: chain5 vs stored C_chain (means) ===")
    for subset in SUBSETS:
        for m in METRICS:
            a = np.nanmean(arr(results["chain5"], subset, m))
            b = np.nanmean(arr(stored["weighted_majority_aggregated_optimal"],
                               subset, m))
            flag = "  <-- CHECK" if abs(a - b) > 0.01 else ""
            print(f"{subset:16s} {m:6s} port={a:.4f} stored={b:.4f}{flag}")

    print("\n=== hybrid7 results (mean +/- std) ===")
    for subset in SUBSETS:
        cells = []
        for m in METRICS:
            v = arr(results["hybrid7"], subset, m)
            cells.append(f"{m}={np.nanmean(v):.4f}±{np.nanstd(v, ddof=1):.4f}")
        print(f"{subset:16s} " + "  ".join(cells))

    print("\n=== paired t-tests: hybrid7 vs stored rows ===")
    for ref in ["weighted_majority_aggregated_optimal",
                "majority_vote_selfconsistency"]:
        for subset in SUBSETS:
            for m in METRICS:
                a = arr(results["hybrid7"], subset, m)
                b = arr(stored[ref], subset, m)
                ok = ~(np.isnan(a) | np.isnan(b))
                _, p = ttest_rel(a[ok], b[ok])
                print(f"vs {ref[:28]:28s} {subset:16s} {m:6s} "
                      f"Δ={np.nanmean(a-b):+.4f} p={p:.2e}")

    json.dump({tag: {s: results[tag][s] for s in SUBSETS}
               for tag in results},
              open(os.path.join(ROOT, "bootstrap-results",
                                "hybrid_dispersion_runs.json"), "w"))
    print("\nSaved bootstrap-results/hybrid_dispersion_runs.json")


if __name__ == "__main__":
    main()
