#!/usr/bin/env python3
r"""MiniCPM-o version of add_dispersion_baselines_gemini.py.

Computes the four question-level black-box baselines on MiniCPM-generated
answers and assembles the additional-baselines table:
  * Semantic Entropy (SE)  : exp(-H) over the cached answer clusters.
  * Num. Semantic Sets     : 1 - (k-1)/(n-1), k = number of clusters.
  * RDS                    : mean Euclidean distance of the n candidate answer
                             embeddings from their centroid; confidence = -RDS.
  * Semantic Volume (SV)   : logdet(X X^T + eps I) of the mean-centred
                             embeddings; confidence = -SV.
SE/NumSets/RDS/SV/SC are all replayed here over the 100 canonical partitions
(seeds 42..141); C_chain is read from the stored MiniCPM multirun. The SC replay
is gated against the stored Self-Consistency values before anything is emitted.

Paper: SE / Num. Semantic Sets / RDS / Semantic Volume rows of Table 2
(MiniCPM-generated answers) and the dispersion-baseline analysis of
Appendix M.
"""
import csv
import json
import math
import os
from collections import Counter

import numpy as np
from scipy.stats import ttest_rel

import bootstrap_significance as bs

# Working root holding the released data artifacts and stored run outputs.
# Override with the OMNIEVAL_ROOT environment variable; defaults to the
# current working directory (see evaluation/README.md).
ROOT = os.environ.get("OMNIEVAL_ROOT", os.getcwd())
MULTIRUN = os.path.join(ROOT, "minicpm-majority-multirun-logistic-reg")
RECORDS = os.path.join(ROOT, "minicpm_cots_with_correctness_and_my_scores.json")
# The stored MiniCPM runs cap at max_candidates_per_question = 6, so we use the
# top-6 cluster cache and the first 6 answer embeddings (matching the pipeline).
CLUSTERS = os.path.join(ROOT, "minicpm-majority-vote-clusters_top6.json")
EMBS = os.path.join(ROOT, "minicpmo-answer-embeddings.json")
SUBSETS = ["Overall", "UNOBench-Audio", "UNOBench-MC", "UNOBench-MO",
           "UNOBench-Visual"]
METRICS = ["AUROC", "ECE", "AURAC"]
# C_chain read from the stored multirun (majority_weighted, as in the Gemini
# additional-baselines table); SC is both replayed (for the gate) and read.
EXISTING = {"C_chain": "weighted_majority_aggregated_optimal"}
SC_KEY = "majority_vote_selfconsistency"


def main():
    print("Loading data ...")
    records = json.load(open(RECORDS))
    cluster_info = json.load(open(CLUSTERS))
    embs_list = json.load(open(EMBS))
    emb_by_qid = {str(e["question_id"]): e["embeddings"] for e in embs_list}
    id_to_category = {}
    for line in open(os.path.join(ROOT, "unobench_processed.jsonl")):
        if line.strip():
            it = json.loads(line)
            id_to_category[it["question_id"]] = it.get("category", "Unknown")

    qs = []
    skipped = 0
    for rec in records:
        qid = str(rec.get("question_id"))
        ci = cluster_info.get(qid)
        if not ci or not ci.get("groups"):
            skipped += 1
            continue
        groups = ci["groups"]
        majority_ids = ci["majority_ids"]
        n = len(groups)
        corr = bs.get_correctness(rec, 0.5)
        E_all = emb_by_qid.get(qid)
        if corr is None or E_all is None or len(corr) < n or len(E_all) < n:
            skipped += 1
            continue
        E = np.asarray(E_all[:n], dtype=float)
        c = E.mean(axis=0)
        rds = float(np.mean(np.linalg.norm(E - c, axis=1)))
        X = E - c
        _sign, logdet = np.linalg.slogdet(X @ X.T + 1e-6 * np.eye(n))
        sv = float(logdet)
        counts = Counter(groups)
        probs = [cnt / n for cnt in counts.values()]
        se = math.exp(-(-sum(p * math.log(p) for p in probs)))
        num = 1.0 if n <= 1 else 1.0 - (len(counts) - 1) / (n - 1)
        sc_group = majority_ids[0]
        idx = groups.index(sc_group)
        qs.append({
            "qid": qid,
            "cat": id_to_category.get(rec.get("question_id"), "Unknown"),
            "y": int(corr[idx]),
            "rds": -rds, "sv": -sv, "se": se, "num": num,
            "sc": counts[sc_group] / n,
        })
    print(f"{len(qs)} questions scored ({skipped} skipped).")

    by_qid = {q["qid"]: q for q in qs}
    rec_stubs = [{"question_id": rec.get("question_id")} for rec in records
                 if str(rec.get("question_id")) in by_qid]

    replay = {k: {s: [] for s in SUBSETS}
              for k in ["rds", "sv", "se", "num", "sc_check"]}
    for seed in range(42, 142):
        test, _ = bs.stratified_test_val_split(rec_stubs, id_to_category, 0.2, seed)
        items = [by_qid[str(r["question_id"])] for r in test]
        cats = np.array([q["cat"] for q in items])
        y = np.array([q["y"] for q in items])
        for key, field in [("rds", "rds"), ("sv", "sv"), ("se", "se"),
                           ("num", "num"), ("sc_check", "sc")]:
            probs = np.array([q[field] for q in items], dtype=float)
            for subset in SUBSETS:
                mask = np.ones(len(y), bool) if subset == "Overall" else cats == subset
                yt = y[mask]
                if len(np.unique(yt)) < 2:
                    replay[key][subset].append({m: np.nan for m in METRICS})
                    continue
                pp = bs.minmax_transform(probs[mask])
                replay[key][subset].append({"AUROC": bs.fast_auroc(yt, pp),
                                            "ECE": bs.fast_ece(yt, pp),
                                            "AURAC": bs.fast_aurac(yt, pp)})

    existing = {lbl: {s: [] for s in SUBSETS} for lbl in EXISTING}
    existing["Self-Consistency"] = {s: [] for s in SUBSETS}
    for i in range(100):
        for subset in SUBSETS:
            path = os.path.join(MULTIRUN, f"run_{i}", "majority_weighted",
                                f"{subset}.csv")
            rows = {r["Method"]: r for r in csv.DictReader(open(path))}
            for lbl, key in list(EXISTING.items()) + [("Self-Consistency", SC_KEY)]:
                r = rows[key]
                existing[lbl][subset].append(
                    {m: float(r[m]) if r[m] not in ("N/A", "") else np.nan
                     for m in METRICS})

    diffs = [abs(replay["sc_check"][s][i][m] - existing["Self-Consistency"][s][i][m])
             for s in SUBSETS for i in range(100) for m in METRICS
             if not (np.isnan(replay["sc_check"][s][i][m])
                     or np.isnan(existing["Self-Consistency"][s][i][m]))]
    print(f"SC replay verification: worst |diff| = {max(diffs):.2e} over "
          f"{len(diffs)} cells")
    assert max(diffs) < 5e-4, "SC replication failed"

    table = {"RDS": replay["rds"], "Semantic Volume": replay["sv"],
             "Semantic Entropy": replay["se"], "Num. Semantic Sets": replay["num"]}
    table.update(existing)
    display = ["Self-Consistency", "Semantic Entropy", "Num. Semantic Sets",
               "RDS", "Semantic Volume", "C_chain"]

    def vals(lbl, subset, metric):
        return np.array([d[metric] for d in table[lbl][subset]])

    out, stars_map = {}, {}
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

    sub_order = SUBSETS
    print("\n--- means (AUROC/ECE/AURAC) ---")
    for lbl in display:
        for subset in sub_order:
            m = [f"{out[(lbl, subset, k)][0]:.4f}" for k in METRICS]
            print(f"{lbl:20s} {subset:16s} " + "  ".join(m))

    print("\n--- LaTeX rows (AUROC & ECE per split) ---")
    for lbl in display:
        cells = [latex_label(lbl)]
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

    print("\n--- AURAC overall (prose) ---")
    for lbl in display:
        print(lbl, f"{out[(lbl, 'Overall', 'AURAC')][0]:.4f}")

    os.makedirs(os.path.join(ROOT, "bootstrap-results"), exist_ok=True)
    json.dump({f"{lbl}|{s}|{m}": out[(lbl, s, m)] for lbl in display
               for s in SUBSETS for m in METRICS},
              open(os.path.join(ROOT, "bootstrap-results",
                                "dispersion_baselines_minicpm_summary.json"), "w"),
              indent=1)
    print("\nSummary saved.")


def latex_label(lbl):
    return {"Num. Semantic Sets": r"Num.\ Semantic Sets",
            "C_chain": r"$\mathcal{C}_{\text{chain}}$"}.get(lbl, lbl)


if __name__ == "__main__":
    main()
