#!/usr/bin/env python3
"""Strictly single-generation evaluation (paper: Appendix N, Table 22).

Scores EXACTLY ONE chain per question (candidate 0, the first sampled
generation) with its own per-chain score. No clustering, no majority vote,
no sum-share aggregation. y_true is candidate 0's own correctness
(uno_score >= 0.5). The canonical partition (split_seed=42, val_fraction=0.2)
is reused so C_chain's LR can be fitted on validation items under the SAME
protocol as the published pipeline (StandardScaler + L1 LR, saga, C grid,
val-AUROC selection, single-feature safety net) — with the sole difference
that validation selection is also single-chain, as it must be in this regime.

Outputs: bootstrap-results/single_chain/{single_chain_results.csv, RESULTS.md}
"""
import json
import numpy as np
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from bootstrap_significance import (
    fast_auroc, fast_ece, fast_aurac, minmax_transform,
    stratified_test_val_split,
)

INPUT_JSON = "gemini_cots_reformatted.json"
JSONL_PATH = "unobench_processed.jsonl"
SPLIT_SEED, VAL_FRACTION, THRESHOLD = 42, 0.2, 0.5
FEATURES = ["internal_smoothness", "internal_goal_directedness",
            "internal_semantic_density", "cross_modal_coherence",
            "cross_modal_grounding_max"]
ROLES = ["S_smooth", "S_goal", "S_dens", "G_avg", "G_max"]
BASELINES = ["confidence_score_level_based", "confidence_score_selfprobing",
             "confidence_score_verb_2s_cot"]
C_GRID = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
CATS = ["UNOBench-Audio", "UNOBench-MC", "UNOBench-MO", "UNOBench-Visual"]


def load():
    id2cat = {}
    with open(JSONL_PATH) as f:
        for line in f:
            d = json.loads(line)
            if d.get("question_id") is not None:
                id2cat[d["question_id"]] = d.get("category") or d.get("split")
    records = json.load(open(INPUT_JSON))
    return records, id2cat


def single_chain_items(records, id2cat):
    """One item per question: candidate 0's features/scores/correctness."""
    items = []
    for rec in records:
        confs = rec.get("generations_confidence") or []
        unos = rec.get("generations_uno_score") or []
        if not confs or not unos or unos[0] is None:
            continue
        c0 = confs[0]
        if any(c0.get(f) is None for f in FEATURES):
            continue
        items.append({
            "qid": rec.get("question_id"),
            "category": id2cat.get(rec.get("question_id"), "Unknown"),
            "y": int(float(unos[0]) >= THRESHOLD),
            "feat": np.array([float(c0[f]) for f in FEATURES]),
            "baselines": {b: (float(c0[b]) if c0.get(b) is not None else None)
                          for b in BASELINES},
        })
    return items


def fit_cchain_single(val_items, label=""):
    """Published fit protocol, transplanted to single-chain scoring."""
    X = np.vstack([it["feat"] for it in val_items])
    y = np.array([it["y"] for it in val_items])
    if len(np.unique(y)) < 2:
        raise RuntimeError(f"[{label}] single-class validation set")

    def val_auroc(scorer):
        p = minmax_transform(scorer(X))
        return fast_auroc(y, p)

    best_model, best_val, best_c = None, -np.inf, None
    for c in C_GRID:
        cand = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=5000, C=c, penalty="l1",
                                      solver="saga", random_state=0)),
        ])
        try:
            cand.fit(X, y)
        except Exception:
            continue
        v = val_auroc(lambda F, m=cand: m.predict_proba(F)[:, 1])
        if best_model is None or v > best_val:
            best_model, best_val, best_c = cand, v, c

    # single-feature safety net (strictly better wins), as in the pipeline
    single_idx, single_val = None, -np.inf
    for idx in range(len(FEATURES)):
        w = np.zeros(len(FEATURES)); w[idx] = 1.0
        v = val_auroc(lambda F, w=w: F @ w)
        if single_idx is None or v > single_val:
            single_idx, single_val = idx, v

    if best_model is None or single_val > best_val:
        w = np.zeros(len(FEATURES)); w[single_idx] = 1.0
        return (lambda F, w=w: F @ w), f"single:{ROLES[single_idx]}", single_val
    coef = best_model.named_steps["lr"].coef_[0]
    desc = "lr(C=%g): " % best_c + ", ".join(
        f"{r}={c:+.3f}" for r, c in zip(ROLES, coef))
    return (lambda F, m=best_model: m.predict_proba(F)[:, 1]), desc, best_val


def metrics(y, p_raw):
    y = np.asarray(y, dtype=int)
    p = minmax_transform(np.asarray(p_raw, dtype=float))
    return {"AUROC": fast_auroc(y, p), "ECE_minmax": fast_ece(y, p),
            "AURAC": fast_aurac(y, p)}


def main():
    records, id2cat = load()
    test_recs, val_recs = stratified_test_val_split(
        records, id2cat, VAL_FRACTION, SPLIT_SEED)
    test = single_chain_items(test_recs, id2cat)
    val = single_chain_items(val_recs, id2cat)
    print(f"test items: {len(test)}, val items: {len(val)}")

    # --- C_chain: per-category fit on validation, applied to test ---
    scorers, descs = {}, {}
    for cat in CATS:
        v = [it for it in val if it["category"] == cat]
        scorers[cat], descs[cat], vauc = fit_cchain_single(v, cat)
        print(f"  fit {cat}: {descs[cat]}  (val AUROC {vauc:.3f})")
    pooled_scorer, pooled_desc, _ = fit_cchain_single(val, "pooled-fallback")

    rows = []
    by_cat = defaultdict(list)
    for it in test:
        by_cat[it["category"]].append(it)

    def add_rows(method, score_fn):
        overall_y, overall_p = [], []
        for cat in CATS:
            its = by_cat[cat]
            y = [it["y"] for it in its]
            p = [score_fn(it) for it in its]
            keep = [(a, b) for a, b in zip(y, p) if b is not None]
            y, p = [a for a, _ in keep], [b for _, b in keep]
            m = metrics(y, p)
            rows.append({"Method": method, "Subset": cat, "N": len(y), **m})
            overall_y += y; overall_p += p
        m = metrics(overall_y, overall_p)
        rows.append({"Method": method, "Subset": "Overall",
                     "N": len(overall_y), **m})

    for i, role in enumerate(ROLES):
        add_rows(role, lambda it, i=i: float(it["feat"][i]))
    for b, name in zip(BASELINES, ["Level-Based", "Self-Probing",
                                   "Two-Step CoT"]):
        add_rows(name, lambda it, b=b: it["baselines"][b])
    add_rows("C_chain(single)", lambda it: float(
        scorers.get(it["category"], pooled_scorer)(
            it["feat"].reshape(1, -1))[0]))

    import csv, os
    os.makedirs("bootstrap-results/single_chain", exist_ok=True)
    out = "bootstrap-results/single_chain/single_chain_results.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nwritten: {out}\n")

    hdr = f"{'Method':18s}" + "".join(
        f"{c.replace('UNOBench-', ''):>9s}" for c in CATS + ["Overall"])
    for metric in ["AUROC", "AURAC"]:
        print(f"--- {metric} (single chain, candidate 0, test partition) ---")
        print(hdr)
        for meth in (ROLES + ["Level-Based", "Self-Probing", "Two-Step CoT",
                              "C_chain(single)"]):
            line = f"{meth:18s}"
            for cat in CATS + ["Overall"]:
                r = next(r for r in rows
                         if r["Method"] == meth and r["Subset"] == cat)
                line += f"{r[metric]:9.3f}"
            print(line)
        print()
    with open("bootstrap-results/single_chain/RESULTS.md", "w") as f:
        f.write("# Strictly single-generation evaluation\n\n"
                f"Candidate 0 per question, canonical partition "
                f"(seed {SPLIT_SEED}, val {VAL_FRACTION}). Fits:\n")
        for cat in CATS:
            f.write(f"- {cat}: {descs[cat]}\n")
        f.write(f"- pooled fallback: {pooled_desc}\n")


if __name__ == "__main__":
    main()
