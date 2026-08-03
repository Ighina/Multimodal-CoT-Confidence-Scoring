#!/usr/bin/env python3
"""
Evaluate two embedding-dispersion baselines (RDS and Semantic Volume) from
per-candidate answer embeddings, under the exact protocol of the
gemini-additional-baselines-multirun (same 100 split seeds 42..141, same
question-level treatment as Semantic Entropy / NumSets in final_evaluate.py:
answer = raw-frequency majority candidate, confidence = the question-level
score, min-max rescaled per split before metrics).

Scores (computed on the first n candidate answer embeddings, n = number of
clustered candidates, embeddings L2-normalised by the encoder):
  * RDS  (radial dispersion): mean Euclidean distance of the candidate
    embeddings from their centroid; uncertainty, so confidence = -RDS.
  * Semantic Volume: log-determinant of the Gram matrix of the mean-centred
    embeddings, logdet(X X^T + eps I); uncertainty, confidence = -SV.

Outputs mean +/- std across the 100 runs for AUROC/ECE/AURAC per split, plus
paired t-tests (best vs runner-up) against the existing table rows read from
the multirun's per-run CSVs, and LaTeX rows for the appendix table.

Paper: RDS and Semantic Volume rows of Tables 1 and 21 (Gemini-generated
answers) and the dispersion-baseline analysis of Appendix M.
"""

import csv
import json
import os
from collections import Counter

import numpy as np
from scipy.stats import ttest_rel

import bootstrap_significance as bs

# Working root holding the released data artifacts and stored run outputs.
# Override with the OMNIEVAL_ROOT environment variable; defaults to the
# current working directory (see evaluation/README.md).
ROOT = os.environ.get("OMNIEVAL_ROOT", os.getcwd())
MULTIRUN = os.path.join(ROOT, "gemini-additional-baselines-multirun")
SUBSETS = ["Overall", "UNOBench-Audio", "UNOBench-MC", "UNOBench-MO",
           "UNOBench-Visual"]
EXISTING = {
    "Self-Consistency": "majority_vote_selfconsistency",
    "Semantic Entropy": "weighted_majority_confidence_score_semantic_entropy",
    "Num. Semantic Sets": "weighted_majority_confidence_score_num_semsets",
    "C_chain": "weighted_majority_aggregated_optimal",
}
METRICS = ["AUROC", "ECE", "AURAC"]


def main():
    print("Loading data ...")
    records = json.load(open(os.path.join(ROOT, "gemini_cots_with_additional_baselines.json")))
    cluster_info = json.load(open(os.path.join(ROOT, "gemini-majority-vote-clusters_top6.json")))
    embs_list = json.load(open(os.path.join(ROOT, "gemini-answer-embeddings.json")))
    emb_by_qid = {str(e["question_id"]): e["embeddings"] for e in embs_list}
    id_to_category = {}
    for line in open(os.path.join(ROOT, "unobench_processed.jsonl")):
        if line.strip():
            it = json.loads(line)
            id_to_category[it["question_id"]] = it.get("category", "Unknown")

    # ---- per-question scores + majority-answer correctness ----
    qs = []  # (qid, category, y_true, conf_rds, conf_sv, sc_conf, sc_true)
    norms = []
    for rec in records:
        qid = str(rec.get("question_id"))
        ci = cluster_info.get(qid)
        if not ci or not ci.get("groups"):
            continue
        groups = ci["groups"]
        majority_ids = ci["majority_ids"]
        n = len(groups)
        corr = bs.get_correctness(rec, 0.5)
        E_all = emb_by_qid.get(qid)
        if corr is None or E_all is None or len(corr) < n or len(E_all) < n:
            continue
        E = np.asarray(E_all[:n], dtype=float)
        norms.append(float(np.linalg.norm(E[0])))
        c = E.mean(axis=0)
        rds = float(np.mean(np.linalg.norm(E - c, axis=1)))
        X = E - c
        G = X @ X.T
        sign, logdet = np.linalg.slogdet(G + 1e-6 * np.eye(n))
        sv = float(logdet)
        counts = Counter(groups)
        sc_group = majority_ids[0]
        idx = groups.index(sc_group)
        qs.append({
            "qid": qid,
            "cat": id_to_category.get(rec.get("question_id"), "Unknown"),
            "y": int(corr[idx]),
            "rds": -rds,          # confidence orientation
            "sv": -sv,
            "sc": counts[sc_group] / n,
        })
    print(f"{len(qs)} questions scored; first-embedding norm ~ {np.mean(norms):.3f}")

    # index by qid for the split replay
    by_qid = {q["qid"]: q for q in qs}
    rec_stubs = [{"question_id": rec.get("question_id")} for rec in records
                 if str(rec.get("question_id")) in by_qid]

    # ---- replay the 100 partitions ----
    new_methods = {"rds": {}, "sv": {}, "sc_check": {}}
    for m in new_methods:
        new_methods[m] = {s: [] for s in SUBSETS}
    for seed in range(42, 142):
        test, _val = bs.stratified_test_val_split(rec_stubs, id_to_category,
                                                  0.2, seed)
        items = [by_qid[str(r["question_id"])] for r in test]
        cats = np.array([q["cat"] for q in items])
        y = np.array([q["y"] for q in items])
        for key, field in [("rds", "rds"), ("sv", "sv"), ("sc_check", "sc")]:
            probs = np.array([q[field] for q in items], dtype=float)
            for subset in SUBSETS:
                mask = np.ones(len(y), bool) if subset == "Overall" else cats == subset
                yt = y[mask]
                if len(np.unique(yt)) < 2:
                    new_methods[key][subset].append({m: np.nan for m in METRICS})
                    continue
                pp = bs.minmax_transform(probs[mask])
                new_methods[key][subset].append({
                    "AUROC": bs.fast_auroc(yt, pp),
                    "ECE": bs.fast_ece(yt, pp),
                    "AURAC": bs.fast_aurac(yt, pp),
                })

    # ---- read existing per-run values from the multirun CSVs ----
    existing = {lbl: {s: [] for s in SUBSETS} for lbl in EXISTING}
    for i in range(100):
        for subset in SUBSETS:
            path = os.path.join(MULTIRUN, f"run_{i}", "majority_weighted",
                                f"{subset}.csv")
            rows = {r["Method"]: r for r in csv.DictReader(open(path))}
            for lbl, key in EXISTING.items():
                r = rows[key]
                existing[lbl][subset].append(
                    {m: float(r[m]) if r[m] not in ("N/A", "") else np.nan
                     for m in METRICS})

    # ---- verification: our SC replay vs stored SC ----
    diffs = []
    for subset in SUBSETS:
        for i in range(100):
            for m in METRICS:
                a = new_methods["sc_check"][subset][i][m]
                b = existing["Self-Consistency"][subset][i][m]
                if not (np.isnan(a) or np.isnan(b)):
                    diffs.append(abs(a - b))
    print(f"SC replay verification: worst |diff| = {max(diffs):.2e} over "
          f"{len(diffs)} run/subset/metric cells")
    assert max(diffs) < 5e-4, "SC replication failed"

    # ---- assemble table: mean +/- std, best/runner-up t-test stars ----
    table = {"RDS": new_methods["rds"], "Semantic Volume": new_methods["sv"]}
    table.update(existing)
    display = ["Self-Consistency", "Semantic Entropy", "Num. Semantic Sets",
               "RDS", "Semantic Volume", "C_chain"]

    def vals(lbl, subset, metric):
        return np.array([d[metric] for d in table[lbl][subset]])

    out = {}
    stars_map = {}
    for subset in SUBSETS:
        for metric in METRICS:
            means = {lbl: np.nanmean(vals(lbl, subset, metric)) for lbl in display}
            lower = metric == "ECE"
            ranked = sorted(means.items(), key=lambda t: t[1], reverse=not lower)
            best, run_up = ranked[0][0], ranked[1][0]
            a, b = vals(best, subset, metric), vals(run_up, subset, metric)
            ok = ~(np.isnan(a) | np.isnan(b))
            _, p = ttest_rel(a[ok], b[ok])
            stars = ("***" if p < 1e-4 else "**" if p < 1e-3 else
                     "*" if p < 1e-2 else "")
            stars_map[(subset, metric)] = (best, stars, float(p))
            for lbl in display:
                out[(lbl, subset, metric)] = (
                    float(np.nanmean(vals(lbl, subset, metric))),
                    float(np.nanstd(vals(lbl, subset, metric), ddof=1)))

    # ---- print summary + latex rows ----
    sub_order = ["Overall", "UNOBench-Audio", "UNOBench-MC", "UNOBench-MO",
                 "UNOBench-Visual"]
    print("\n--- means (AUROC/ECE/AURAC) ---")
    for lbl in display:
        for subset in sub_order:
            m = [f"{out[(lbl, subset, k)][0]:.4f}±{out[(lbl, subset, k)][1]:.4f}"
                 for k in METRICS]
            print(f"{lbl:20s} {subset:16s} " + "  ".join(m))
    print("\n--- best/runner-up stars ---")
    for k, v in stars_map.items():
        if v[1]:
            print(k, v)

    print("\n--- LaTeX rows (AUROC & ECE per split) ---")
    for lbl in display:
        cells = [lbl]
        for subset in sub_order:
            for metric in ["AUROC", "ECE"]:
                mu = out[(lbl, subset, metric)][0]
                s = f"{mu:.4f}"
                best, stars, _ = stars_map[(subset, metric)]
                if lbl == best:
                    s = r"\textbf{" + s + "}"
                    if stars:
                        s += r"\textsuperscript{" + stars + "}"
                cells.append(s)
        print(" & ".join(cells) + r" \\")
    print("\n--- AURAC overall values for prose ---")
    for lbl in display:
        print(lbl, f"{out[(lbl, 'Overall', 'AURAC')][0]:.4f}")

    json.dump({f"{lbl}|{s}|{m}": out[(lbl, s, m)] for lbl in display
               for s in SUBSETS for m in METRICS},
              open(os.path.join(ROOT, "bootstrap-results",
                                "dispersion_baselines_summary.json"), "w"),
              indent=1)
    print("\nSummary saved.")


if __name__ == "__main__":
    main()
