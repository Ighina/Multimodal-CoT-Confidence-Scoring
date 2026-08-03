#!/usr/bin/env python3
"""
Evaluate the attention-style grounding score (salience-weighted alignment for
unimodal inputs, entropy-gated routing for omnimodal inputs; stored as
cross_modal_grounding_attention in the Gemini+E5 run and
cross_modal_entropy_gated_routing in the MiniCPM/LCO runs) against the simple
grounding scores G_avg and G_max, under the exact no_majority protocol of the
published multiruns (same 100 seeds 42..141, majority-vote answer selection
with score tie-breaking, sum-share confidence, min-max preprocessing).

Verification gate: G_avg replayed with this script must match the stored
per-run no_majority CSVs. Paired t-tests are computed among the three
grounding scores per subset/metric.

Paper: Appendix I (attention-style grounding G_att, Table 14).
"""

import csv
import json
import os
from collections import defaultdict

import numpy as np
from scipy.stats import ttest_rel

import bootstrap_significance as bs

# Working root holding the released data artifacts and stored run outputs.
# Override with the OMNIEVAL_ROOT environment variable; defaults to the
# current working directory (see evaluation/README.md).
ROOT = os.environ.get("OMNIEVAL_ROOT", os.getcwd())
SUBSETS = ["Overall", "UNOBench-Audio", "UNOBench-MC", "UNOBench-MO",
           "UNOBench-Visual"]
METRICS = ["AUROC", "ECE", "AURAC"]
DS = {
    "gemini": dict(
        input_json="gemini_cots_reformatted.json",
        cache="gemini-majority-vote-clusters_top6.json",
        multirun="gemini-majority-multirun-logistic-reg",
        att="cross_modal_grounding_attention",
        gavg="cross_modal_coherence", gmax="cross_modal_grounding_max"),
    "lco": dict(
        input_json="lco_gemini_cots_reformatted.json",
        cache="gemini-majority-vote-clusters_top6.json",
        multirun="lco-gemini-majority-multirun-logistic-reg",
        att="cross_modal_entropy_gated_routing",
        gavg="cross_modal_overall", gmax="cross_modal_max_step_coherence"),
    "minicpm": dict(
        input_json="minicpm_cots_with_correctness_and_my_scores.json",
        cache="minicpm-majority-vote-clusters_top6.json",
        multirun="minicpm-majority-multirun-logistic-reg",
        att="cross_modal_entropy_gated_routing",
        gavg="cross_modal_overall", gmax="cross_modal_max_step_coherence"),
}


def eval_feature(q, scores):
    """no_majority evaluation for a MAJORITY_WEIGHTED baseline feature."""
    S_g, norm_sum, _ = bs.aggregate_group_scores(q["groups"], scores)
    ns = {g: float(norm_sum[g]) for g in range(len(norm_sum))}
    mids = q["majority_ids"]
    wg = mids[0] if len(mids) == 1 else max(mids, key=lambda g: ns[g])
    wi = int(np.argmax(q["groups"] == wg))
    return int(q["corr"][wi]), ns[wg]


def summarize(y, p):
    y = np.array(y)
    p = bs.minmax_transform(np.array(p, dtype=float))
    if len(np.unique(y)) < 2:
        return {m: np.nan for m in METRICS}
    return {"AUROC": bs.fast_auroc(y, p), "ECE": bs.fast_ece(y, p),
            "AURAC": bs.fast_aurac(y, p)}


def main():
    id_to_category = {}
    for line in open(os.path.join(ROOT, "unobench_processed.jsonl")):
        if line.strip():
            it = json.loads(line)
            id_to_category[it["question_id"]] = it.get("category", "Unknown")

    all_out = {}
    for ds, cfg in DS.items():
        print(f"=== {ds} ===")
        records = json.load(open(os.path.join(ROOT, cfg["input_json"])))
        cluster_info = json.load(open(os.path.join(ROOT, cfg["cache"])))
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
            if corr is None or len(confs) < n or len(corr) < n:
                continue
            qs[qid] = {
                "cat": id_to_category.get(rec.get("question_id"), "Unknown"),
                "groups": np.array(groups, dtype=int),
                "majority_ids": ci["majority_ids"],
                "corr": np.array(corr[:n], dtype=int),
                "att": np.array([float(confs[i].get(cfg["att"], 0.0))
                                 for i in range(n)]),
                "gavg_check": np.array([float(confs[i].get(cfg["gavg"], 0.0))
                                        for i in range(n)]),
            }
        rec_stubs = [{"question_id": int(q) if q.isdigit() else q} for q in qs]

        mine = {k: {s: [] for s in SUBSETS} for k in ["att", "gavg_check"]}
        for seed in range(42, 142):
            test, _ = bs.stratified_test_val_split(rec_stubs, id_to_category,
                                                   0.2, seed)
            tq = [qs[str(r["question_id"])] for r in test]
            for key in ["att", "gavg_check"]:
                coll = defaultdict(lambda: {"y": [], "p": []})
                for q in tq:
                    yt, pp = eval_feature(q, q[key])
                    for subset in ["Overall", q["cat"]]:
                        coll[subset]["y"].append(yt)
                        coll[subset]["p"].append(pp)
                for subset in SUBSETS:
                    d = coll.get(subset)
                    mine[key][subset].append(
                        summarize(d["y"], d["p"]) if d else
                        {m: np.nan for m in METRICS})

        # stored per-run rows for G_avg / G_max
        stored = {k: {s: [] for s in SUBSETS} for k in ["gavg", "gmax"]}
        for i in range(100):
            for subset in SUBSETS:
                path = os.path.join(ROOT, cfg["multirun"], f"run_{i}",
                                    "no_majority", f"{subset}.csv")
                rows = {r["Method"]: r for r in csv.DictReader(open(path))}
                for k in ["gavg", "gmax"]:
                    r = rows[cfg[k]]
                    stored[k][subset].append(
                        {m: float(r[m]) if r[m] not in ("N/A", "") else np.nan
                         for m in METRICS})

        # verification
        diffs = [abs(a[m] - b[m])
                 for subset in SUBSETS
                 for a, b in zip(mine["gavg_check"][subset],
                                 stored["gavg"][subset])
                 for m in METRICS if not (np.isnan(a[m]) or np.isnan(b[m]))]
        print(f"  G_avg replay verification: worst |diff| = {max(diffs):.2e}")
        assert max(diffs) < 5e-4

        table = {"G_avg": stored["gavg"], "G_max": stored["gmax"],
                 "G_att": mine["att"]}
        out = {}
        for subset in SUBSETS:
            for metric in METRICS:
                vals = {lbl: np.array([x[metric] for x in table[lbl][subset]])
                        for lbl in table}
                means = {lbl: np.nanmean(v) for lbl, v in vals.items()}
                lower = metric == "ECE"
                ranked = sorted(means.items(), key=lambda t: t[1],
                                reverse=not lower)
                best, run_up = ranked[0][0], ranked[1][0]
                a, b = vals[best], vals[run_up]
                ok = ~(np.isnan(a) | np.isnan(b))
                _, p = ttest_rel(a[ok], b[ok])
                stars = ("***" if p < 1e-4 else "**" if p < 1e-3 else
                         "*" if p < 1e-2 else "")
                out[(subset, metric)] = {
                    lbl: (means[lbl], float(np.nanstd(v, ddof=1)))
                    for lbl, v in vals.items()}
                out[(subset, metric)]["_best"] = (best, stars, float(p))
        all_out[ds] = out

        for lbl in ["G_avg", "G_max", "G_att"]:
            row = [lbl]
            for subset in SUBSETS:
                for metric in ["AUROC", "AURAC"]:
                    mu = out[(subset, metric)][lbl][0]
                    s = f"{mu:.4f}"
                    best, stars, _ = out[(subset, metric)]["_best"]
                    if lbl == best:
                        s = r"\textbf{" + s + "}" + (
                            r"\textsuperscript{" + stars + "}" if stars else "")
                    row.append(s)
            print("  " + " & ".join(row) + r" \\")

    json.dump({ds: {f"{s}|{m}": {lbl: v for lbl, v in d.items()
                                 if lbl != "_best"} | {"best": d["_best"]}
                    for (s, m), d in out.items()}
               for ds, out in all_out.items()},
              open(os.path.join(ROOT, "bootstrap-results",
                                "grounding_attention_summary.json"), "w"),
              indent=1, default=str)
    print("Saved summary.")


if __name__ == "__main__":
    main()
