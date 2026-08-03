#!/usr/bin/env python3
"""
Does adding self-verbalization scores as feature(s) to the C_chain
logistic-regression aggregator improve it, in the paper's MAIN sampling-free
setting (no_majority: every method scores the same count-majority answer)?

Paper: supports the discussion of the complementarity between the embedding
scores and the self-verbalisation baselines (Sections 5.2 and 6).

Variants (all no_majority, per-split L1-LR aggregator, refit per run):
  chain5              published C_chain (5 chain features)          [gate]
  chain5+selfprobing  + confidence_score_selfprobing                (= --add-self-probing)
  chain5+allverb      + selfprobing + verb_2s_cot + level_based

Fitting protocol (ported 1:1 from final_evaluate._find_weights_logistic_regression
+ find_best_aggregation_weights, mirrored in cchain_transfer_eval.fit_cchain):
  StandardScaler + LogisticRegression(penalty='l1', solver='saga', max_iter=5000,
  random_state=0) per split on that split's validation candidates -> correctness,
  C grid {0.01,0.03,0.1,0.3,1.0,3.0,10.0}, model selection by question-level
  validation AUROC through the same sum_share group aggregation + majority-vote
  answer selection, plus the single-feature (one-hot) safety net. Pooled fallback
  fit on all validation candidates. Each test item scored by its own split's
  aggregator; per-subset metrics use per-subset minmax (published construction).

Outputs (bootstrap-results/cchain_selfverb/{gemini,minicpm}/):
  verification.csv, means.csv, delta_vs_chain5.csv, delta_vs_sc.csv,
  bootstrap_deltas.csv, RESULTS.md
"""

import csv
import json
import os
import random
from collections import Counter, defaultdict

import numpy as np
from scipy.stats import rankdata, ttest_rel
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_BASE = os.path.join(ROOT, "bootstrap-results", "cchain_selfverb")
JSONL_PATH = os.path.join(ROOT, "unobench_processed.jsonl")

SPLITS = ["UNOBench-Audio", "UNOBench-MC", "UNOBench-MO", "UNOBench-Visual"]
SHORT = {"UNOBench-Audio": "Audio", "UNOBench-MC": "MC",
         "UNOBench-MO": "MO", "UNOBench-Visual": "Visual", "Overall": "Overall"}
SUBSETS = ["Overall"] + SPLITS
C_GRID = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
LR_PENALTY = "l1"
N_RUNS = 100
VERB = ["confidence_score_selfprobing", "confidence_score_verb_2s_cot",
        "confidence_score_level_based"]
SC_KEY = "majority_vote_selfconsistency"

CONFIG = {
    "gemini": {
        "input_json": os.path.join(ROOT, "gemini_cots_reformatted.json"),
        "cluster_cache": os.path.join(ROOT, "gemini-majority-vote-clusters_top6.json"),
        "run_base": os.path.join(ROOT, "gemini-majority-multirun-logistic-reg"),
        "chain5": ["internal_smoothness", "internal_goal_directedness",
                   "internal_semantic_density", "cross_modal_coherence",
                   "cross_modal_grounding_max"],
    },
    "minicpm": {
        "input_json": os.path.join(ROOT, "minicpm_cots_with_correctness_and_my_scores.json"),
        "cluster_cache": os.path.join(ROOT, "minicpm-majority-vote-clusters_top6.json"),
        "run_base": os.path.join(ROOT, "minicpm-majority-multirun-logistic-reg"),
        "chain5": ["internal_smoothness", "internal_goal_directedness",
                   "internal_semantic_density", "cross_modal_overall",
                   "cross_modal_max_step_coherence"],
    },
}


# ------------------------------------------------------------------ primitives
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


def minmax(y_prob):
    y_prob = np.asarray(y_prob, dtype=float)
    if len(y_prob) > 0:
        lo, hi = float(np.min(y_prob)), float(np.max(y_prob))
        y_prob = (y_prob - lo) / (hi - lo) if hi > lo else np.zeros_like(y_prob)
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


def fast_ece(y_true, y_prob, n_bins=10):
    edges = np.linspace(0, 1, n_bins + 1)
    bins = np.clip(np.searchsorted(edges, y_prob, side="right") - 1, 0, n_bins - 1)
    total = len(y_prob)
    if total == 0:
        return float("nan")
    counts = np.bincount(bins, minlength=n_bins)
    sconf = np.bincount(bins, weights=y_prob, minlength=n_bins)
    strue = np.bincount(bins, weights=y_true.astype(float), minlength=n_bins)
    nz = counts > 0
    return float(np.sum(np.abs(sconf[nz] / counts[nz] - strue[nz] / counts[nz])
                        * (counts[nz] / total)))


def metrics_from_raw(y_true, y_prob_raw):
    """Published metrics: minmax the subset's scores, then AUROC/ECE/AURAC."""
    y_true = np.asarray(y_true, dtype=int)
    yp = minmax(y_prob_raw)
    if len(np.unique(y_true)) < 2:
        return {"AUROC": np.nan, "ECE": fast_ece(y_true, yp), "AURAC": fast_aurac(y_true, yp)}
    return {"AUROC": float(fast_auroc(y_true, yp)),
            "ECE": fast_ece(y_true, yp),
            "AURAC": fast_aurac(y_true, yp)}


# ------------------------------------------------------------------ data prep
def load_static(name):
    cfg = CONFIG[name]
    records = json.load(open(cfg["input_json"]))
    cluster_info = json.load(open(cfg["cluster_cache"]))
    id_to_category = {}
    with open(JSONL_PATH) as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                id_to_category[item["question_id"]] = item.get("category", "Unknown")
    return records, cluster_info, id_to_category


def prepare_items(recs, cluster_info, id_to_category, union_feats, threshold=0.5):
    """Per-question prepared item; fvals is (n, len(union_feats))."""
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
        fvals = np.array(
            [[float(confs[i].get(f, 0.0)) for f in union_feats] for i in range(n)],
            dtype=float)
        items.append({
            "qid": qid,
            "category": id_to_category.get(rec.get("question_id"), "Unknown"),
            "groups": groups,
            "groups_arr": np.array(groups, dtype=int),
            "majority_ids": ci["majority_ids"],
            "correctness": np.array(correctness[:n], dtype=int),
            "fvals": fvals,
        })
    return items


def by_split(items):
    d = defaultdict(list)
    for it in items:
        d[it["category"]].append(it)
    return d


# ------------------------------------------------------------------ aggregator apply
def apply_scorer(items, scorer):
    """no_majority aggregated_optimal replay: sum_share group confidence,
    count-majority answer with sum-share tie-break, y_prob = winner's sum-share.
    Returns RAW y_prob (per-subset minmax applied later)."""
    y_true, y_prob = [], []
    for it in items:
        scores = np.asarray(scorer(it["fvals"]), dtype=float)
        ga = it["groups_arr"]
        ng = int(ga.max()) + 1
        gsum = np.bincount(ga, weights=scores, minlength=ng)
        norm = gsum / (gsum.sum() + 1e-9)
        mids = it["majority_ids"]
        win = mids[0] if len(mids) == 1 else max(mids, key=lambda g: norm[g])
        widx = it["groups"].index(win)
        y_true.append(int(it["correctness"][widx]))
        y_prob.append(float(norm[win]))
    return np.array(y_true, dtype=int), np.array(y_prob, dtype=float)


def apply_per_category(items, scorers, fallback):
    """Each item scored by its category's aggregator; returns raw per-item arrays
    in natural record order plus categories (matches published Overall)."""
    y_true, y_prob, cats = [], [], []
    for it in items:
        sc = scorers.get(it["category"], fallback)
        yt, yp = apply_scorer([it], sc)
        y_true.append(int(yt[0]))
        y_prob.append(float(yp[0]))
        cats.append(it["category"])
    return np.array(y_true, dtype=int), np.array(y_prob, dtype=float), np.array(cats)


def val_auroc(items, scorer):
    yt, yp = apply_scorer(items, scorer)
    if len(yt) == 0 or len(np.unique(yt)) < 2:
        return None
    return float(fast_auroc(yt, minmax(yp)))


# ------------------------------------------------------------------ fit (ported 1:1)
def fit_cchain(val_items, cols, feat_names, label=""):
    """L1-LR C-grid + single-feature safety net. `cols` selects columns of
    fvals for this variant. scorer accepts a full fvals matrix and selects cols."""
    X = np.vstack([it["fvals"][:, cols] for it in val_items])
    y = np.concatenate([it["correctness"] for it in val_items])
    if len(X) < 2 or len(np.unique(y)) < 2:
        raise RuntimeError(f"[{label}] not enough data / single class")

    best_model, best_val, best_c = None, -np.inf, None
    for c in C_GRID:
        cand = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=5000, C=c, penalty=LR_PENALTY,
                                      solver="saga", random_state=0)),
        ])
        try:
            cand.fit(X, y)
        except Exception as e:
            print(f"[{label}] C={c} failed: {e}")
            continue

        def lr_score(F, m=cand, cc=cols):
            return m.predict_proba(F[:, cc])[:, 1]

        v = val_auroc(val_items, lr_score)
        if v is None:
            continue
        if best_model is None or v > best_val:
            best_val, best_model, best_c = v, cand, c

    if best_model is None:
        raise RuntimeError(f"[{label}] no valid LR across C grid")

    # single-feature safety net (one-hot over the variant's own columns)
    single_col, single_val = None, -np.inf
    for j, col in enumerate(cols):
        def dot(F, col=col):
            return F[:, col]
        v = val_auroc(val_items, dot)
        if v is None:
            continue
        if single_col is None or v > single_val:
            single_col, single_val = col, v

    if single_col is not None and single_val > best_val:
        def scorer(F, col=single_col):
            return F[:, col]
        return {"kind": "single", "feature": feat_names[cols.index(single_col)],
                "scorer": scorer, "val_auroc": single_val, "degenerate": False}

    lr = best_model.named_steps["lr"]

    def scorer(F, m=best_model, cc=cols):
        return m.predict_proba(F[:, cc])[:, 1]

    return {"kind": "lr", "C": best_c, "coef": lr.coef_[0].copy(),
            "scorer": scorer, "val_auroc": best_val,
            "degenerate": bool(np.all(lr.coef_[0] == 0.0))}


def fit_variant(val_items, cols, feat_names, label=""):
    """Per-split fits + pooled fallback (mirrors find_best_weights_per_split)."""
    fallback = fit_cchain(val_items, cols, feat_names, label=f"{label}/Pooled")
    vbs = by_split(val_items)
    scorers, degen = {}, {}
    for sp in SPLITS:
        try:
            fit = fit_cchain(vbs[sp], cols, feat_names, label=f"{label}/{SHORT[sp]}")
        except RuntimeError:
            fit = fallback
        scorers[sp] = fit["scorer"]
        degen[sp] = fit.get("degenerate", False)
    return scorers, fallback["scorer"], degen


# ------------------------------------------------------------------ baselines (SC + standalone)
def collect_baseline_items(items, baseline):
    """Standalone no_majority baseline row (majority answer, sum-share tie-break,
    y_prob = winning candidate's raw score). `baseline` is a feature name."""
    y_true, y_prob, cats = [], [], []
    for it in items:
        groups = it["groups"]
        n = len(groups)
        ga = it["groups_arr"]
        # baseline column index within union: find via stored mapping on item
        scores = it["_baseline_scores"][baseline]
        ng = int(ga.max()) + 1
        gsum = np.bincount(ga, weights=scores, minlength=ng)
        norm = gsum / (gsum.sum() + 1e-9)
        mids = it["majority_ids"]
        win = mids[0] if len(mids) == 1 else max(mids, key=lambda g: norm[g])
        widx = groups.index(win)
        y_true.append(int(it["correctness"][widx]))
        y_prob.append(float(scores[widx]))
        cats.append(it["category"])
    return np.array(y_true, int), np.array(y_prob, float), np.array(cats)


def collect_sc_items(items):
    y_true, y_prob, cats = [], [], []
    for it in items:
        groups = it["groups"]
        n = len(groups)
        counts = Counter(groups)
        sc_group = it["majority_ids"][0]
        sc_index = groups.index(sc_group)
        y_true.append(int(it["correctness"][sc_index]))
        y_prob.append(counts[sc_group] / n)
        cats.append(it["category"])
    return np.array(y_true, int), np.array(y_prob, float), np.array(cats)


# ------------------------------------------------------------------ CSV readers
def read_agg_row(run_base, run_i, subset):
    """stored no_majority aggregated_optimal AUROC/ECE/AURAC for gate."""
    path = os.path.join(run_base, f"run_{run_i}", "no_majority", f"{subset}.csv")
    with open(path) as f:
        for row in csv.DictReader(f):
            if row["Method"] == "aggregated_optimal":
                return {"AUROC": float(row["AUROC"]), "ECE": float(row["ECE"]),
                        "AURAC": float(row["AURAC"])}
    return None


def read_sc_row(run_base, run_i, subset):
    """stored SC AUROC/ECE/AURAC from majority_weighted/{subset}.csv."""
    path = os.path.join(run_base, f"run_{run_i}", "majority_weighted", f"{subset}.csv")
    with open(path) as f:
        for row in csv.DictReader(f):
            if row["Method"] == SC_KEY:
                return {"AUROC": float(row["AUROC"]), "ECE": float(row["ECE"]),
                        "AURAC": float(row["AURAC"])}
    return None


def read_split_seed(run_base, run_i):
    m = json.load(open(os.path.join(run_base, f"run_{run_i}", "no_majority",
                                    "metadata.json")))
    return m["parameters"]["split_seed"]


# ------------------------------------------------------------------ per-run eval
def eval_variant_on_test(test_items, scorers, fallback):
    """Return {subset: {AUROC,ECE,AURAC}} using per-subset minmax."""
    yt, yp, cats = apply_per_category(test_items, scorers, fallback)
    out = {}
    for subset in SUBSETS:
        mask = np.ones(len(cats), bool) if subset == "Overall" else cats == subset
        out[subset] = metrics_from_raw(yt[mask], yp[mask])
    return out


# ------------------------------------------------------------------ main per dataset
def run_dataset(name):
    cfg = CONFIG[name]
    out_dir = os.path.join(OUT_BASE, name)
    os.makedirs(out_dir, exist_ok=True)
    chain5 = cfg["chain5"]
    union = chain5 + VERB
    variants = {
        "chain5": list(range(len(chain5))),
        "chain5+selfprobing": list(range(len(chain5))) + [len(chain5)],  # + selfprobing
        "chain5+allverb": list(range(len(union))),
    }
    print(f"\n########## {name} ##########")
    print(f"chain5 features: {chain5}")
    print(f"union feature order: {union}")

    records, cluster_info, id_to_category = load_static(name)

    # storage: per run, per variant, per subset metrics
    per_run = {v: {s: {m: [] for m in ["AUROC", "ECE", "AURAC"]}
                   for s in SUBSETS} for v in variants}
    sc_run = {s: {m: [] for m in ["AUROC", "ECE", "AURAC"]} for s in SUBSETS}
    gate_rows = []  # [run, subset, refit_auroc, stored_auroc, absdiff]
    degen_notes = []

    for run_i in range(N_RUNS):
        seed = read_split_seed(cfg["run_base"], run_i)
        test_recs, val_recs = stratified_test_val_split(
            records, id_to_category, 0.2, seed)
        test_items = prepare_items(test_recs, cluster_info, id_to_category, union)
        val_items = prepare_items(val_recs, cluster_info, id_to_category, union)

        for vname, cols in variants.items():
            scorers, fallback, degen = fit_variant(
                val_items, cols, [union[c] for c in cols],
                label=f"{name}/r{run_i}/{vname}")
            res = eval_variant_on_test(test_items, scorers, fallback)
            for s in SUBSETS:
                for m in ["AUROC", "ECE", "AURAC"]:
                    per_run[vname][s][m].append(res[s][m])
            if any(degen.values()):
                degen_notes.append((run_i, vname,
                                    [SHORT[s] for s in SPLITS if degen[s]]))

            if vname == "chain5":
                for s in SUBSETS:
                    stored = read_agg_row(cfg["run_base"], run_i, s)
                    d = abs(res[s]["AUROC"] - stored["AUROC"])
                    gate_rows.append([run_i, s, res[s]["AUROC"], stored["AUROC"], d])

        # SC from stored majority_weighted CSVs (same partition/seed)
        for s in SUBSETS:
            sc = read_sc_row(cfg["run_base"], run_i, s)
            for m in ["AUROC", "ECE", "AURAC"]:
                sc_run[s][m].append(sc[m])

        if (run_i + 1) % 10 == 0:
            print(f"  [{name}] completed run {run_i+1}/{N_RUNS} (seed={seed})")

    # ---- verification.csv ----
    worst = max(r[4] for r in gate_rows)
    # per-run pass: all subsets AUROC within 1e-3
    run_worst = defaultdict(float)
    for r in gate_rows:
        run_worst[r[0]] = max(run_worst[r[0]], r[4])
    runs_pass = sum(1 for ri in range(N_RUNS) if run_worst[ri] <= 1e-3)
    runs_investigate = [ri for ri in range(N_RUNS) if run_worst[ri] > 0.005]
    with open(os.path.join(out_dir, "verification.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "subset", "refit_chain5_AUROC", "stored_AUROC", "AbsDiff"])
        for r in gate_rows:
            w.writerow([r[0], r[1], f"{r[2]:.6f}", f"{r[3]:.6f}", f"{r[4]:.2e}"])
    print(f"[{name}] GATE: worst |diff|={worst:.2e}, runs passing (<=1e-3)="
          f"{runs_pass}/{N_RUNS}, runs to investigate (>0.005)={runs_investigate}")

    # ---- means.csv ----
    with open(os.path.join(out_dir, "means.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Variant", "Subset", "AUROC_mean", "AUROC_std",
                    "ECE_mean", "ECE_std", "AURAC_mean", "AURAC_std"])
        for vname in variants:
            for s in SUBSETS:
                row = [vname, SHORT[s]]
                for m in ["AUROC", "ECE", "AURAC"]:
                    arr = np.array(per_run[vname][s][m], float)
                    arr = arr[~np.isnan(arr)]
                    row += [f"{np.mean(arr):.4f}", f"{np.std(arr):.4f}"]
                w.writerow(row)
        # SC reference too
        for s in SUBSETS:
            row = ["self_consistency", SHORT[s]]
            for m in ["AUROC", "ECE", "AURAC"]:
                arr = np.array(sc_run[s][m], float)
                row += [f"{np.mean(arr):.4f}", f"{np.std(arr):.4f}"]
            w.writerow(row)

    # ---- delta tables (paired relative %, paired t-test) ----
    def star(p):
        return ("***" if p < 1e-4 else "**" if p < 1e-3 else "*" if p < 1e-2 else "")

    def write_delta(fname, ref_run, ref_label):
        rows = []
        for vname in ["chain5+selfprobing", "chain5+allverb"]:
            for s in SUBSETS:
                for m in ["AUROC", "ECE", "AURAC"]:
                    a = np.array(per_run[vname][s][m], float)
                    b = np.array(ref_run[s][m], float)
                    ok = ~(np.isnan(a) | np.isnan(b))
                    a, b = a[ok], b[ok]
                    rel = np.mean((a - b) / b) * 100.0
                    if len(a) > 1 and np.any(a != b):
                        t, p = ttest_rel(a, b)
                    else:
                        p = 1.0
                    rows.append([vname, SHORT[s], m, f"{np.mean(a):.4f}",
                                 f"{np.mean(b):.4f}", f"{rel:+.2f}%",
                                 f"{p:.2e}", star(p)])
        with open(os.path.join(out_dir, fname), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Variant", "Subset", "Metric", f"Variant_mean",
                        f"{ref_label}_mean", "RelDiff%", "p_ttest", "sig"])
            w.writerows(rows)
        return rows

    chain5_run = per_run["chain5"]
    d_chain = write_delta("delta_vs_chain5.csv", chain5_run, "chain5")
    d_sc = write_delta("delta_vs_sc.csv", sc_run, "SC")

    # ---- item-level bootstrap on canonical run_0 ----
    boot_rows = run0_bootstrap(name, cfg, records, cluster_info, id_to_category,
                               chain5, union, variants, out_dir)

    # ---- RESULTS.md ----
    write_results_md(name, out_dir, worst, runs_pass, runs_investigate,
                     per_run, sc_run, variants, d_chain, d_sc, boot_rows,
                     degen_notes, union)
    return {"worst": worst, "runs_pass": runs_pass,
            "runs_investigate": runs_investigate}


def run0_bootstrap(name, cfg, records, cluster_info, id_to_category,
                   chain5, union, variants, out_dir, B=10000, seed=0):
    print(f"[{name}] item-level bootstrap on run_0 (B={B})...")
    split_seed = read_split_seed(cfg["run_base"], 0)
    test_recs, val_recs = stratified_test_val_split(
        records, id_to_category, 0.2, split_seed)
    test_items = prepare_items(test_recs, cluster_info, id_to_category, union)
    val_items = prepare_items(val_recs, cluster_info, id_to_category, union)
    # attach baseline score columns for standalone selfprobing
    sp_col = union.index("confidence_score_selfprobing")
    for it in test_items:
        it["_baseline_scores"] = {"confidence_score_selfprobing":
                                  it["fvals"][:, sp_col]}

    # build per-item raw arrays for each method (aligned by test-item order)
    methods = {}
    for vname in ["chain5", "chain5+selfprobing"]:
        cols = variants[vname]
        scorers, fallback, _ = fit_variant(val_items, cols,
                                            [union[c] for c in cols],
                                            label=f"{name}/r0/{vname}")
        yt, yp, cats = apply_per_category(test_items, scorers, fallback)
        methods[vname] = (yt, yp, cats)
    yt, yp, cats = collect_baseline_items(test_items, "confidence_score_selfprobing")
    methods["selfprobing_standalone"] = (yt, yp, cats)
    yt, yp, cats = collect_sc_items(test_items)
    methods["self_consistency"] = (yt, yp, cats)

    cats_ref = methods["chain5"][2]
    rng = np.random.default_rng(seed)
    rows = []
    ref = "chain5+selfprobing"
    comparisons = ["chain5", "selfprobing_standalone", "self_consistency"]
    for subset in SUBSETS:
        mask = np.ones(len(cats_ref), bool) if subset == "Overall" else cats_ref == subset
        n = int(mask.sum())
        # pre-minmax each method's subset scores (rank-preserving for AUROC/AURAC)
        prep = {}
        for mname, (yt_m, yp_m, _c) in methods.items():
            prep[mname] = (yt_m[mask], minmax(yp_m[mask]))
        idxs = rng.integers(0, n, size=(B, n))
        # bootstrap metric arrays
        bmet = {mname: {"AUROC": np.empty(B), "AURAC": np.empty(B)}
                for mname in methods}
        for b in range(B):
            ix = idxs[b]
            for mname in methods:
                ytb, ypb = prep[mname]
                yts, yps = ytb[ix], ypb[ix]
                bmet[mname]["AUROC"][b] = fast_auroc(yts, yps)
                bmet[mname]["AURAC"][b] = fast_aurac(yts, yps)
        for other in comparisons:
            for metric in ["AUROC", "AURAC"]:
                da = bmet[ref][metric] - bmet[other][metric]
                da = da[~np.isnan(da)]
                if len(da) == 0:
                    continue
                p_two = min(1.0, 2 * min(np.mean(da <= 0), np.mean(da >= 0))
                            + 1.0 / len(da))
                lo, hi = np.percentile(da, 2.5), np.percentile(da, 97.5)
                rows.append([SHORT[subset], metric, ref, other, n,
                             f"{np.mean(da):+.4f}", f"{lo:+.4f}", f"{hi:+.4f}",
                             f"{p_two:.4f}"])
    with open(os.path.join(out_dir, "bootstrap_deltas.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Subset", "Metric", "Variant", "Comparison", "N",
                    "Delta_mean", "Delta_lo", "Delta_hi", "p_two"])
        w.writerows(rows)
    return rows


# ------------------------------------------------------------------ report
def write_results_md(name, out_dir, worst, runs_pass, runs_investigate,
                     per_run, sc_run, variants, d_chain, d_sc, boot_rows,
                     degen_notes, union):
    L = []
    L.append(f"# C_chain + self-verbalization features — {name} "
             f"(sampling-free / no_majority)\n")
    L.append("Main setting: every method scores the same count-majority answer. "
             "The aggregator is an L1 logistic regression refit per split per run "
             "(saga, random_state=0, C-grid selection by validation AUROC, "
             "single-feature safety net), evaluated with per-subset minmax. "
             "100 runs (canonical partition seed 42 = run_0, plus 99 reseeds).\n")

    L.append("## Verification gate (refit chain5 vs stored aggregated_optimal)\n")
    L.append(f"- Worst |refit − stored| AUROC across all runs/subsets: "
             f"**{worst:.2e}**")
    L.append(f"- Runs passing (all subsets AUROC within 1e-3): "
             f"**{runs_pass}/{N_RUNS}**")
    L.append(f"- Runs to investigate (>0.005): "
             f"{runs_investigate if runs_investigate else 'none'}\n")

    L.append("## 100-run means (mean ± std)\n")
    L.append("| Variant | Subset | AUROC | ECE | AURAC |")
    L.append("| :-- | :-- | :--: | :--: | :--: |")
    order = ["chain5", "chain5+selfprobing", "chain5+allverb"]
    for vname in order:
        for s in SUBSETS:
            a = np.array(per_run[vname][s]["AUROC"], float)
            a = a[~np.isnan(a)]
            e = np.array(per_run[vname][s]["ECE"], float)
            r = np.array(per_run[vname][s]["AURAC"], float)
            L.append(f"| {vname} | {SHORT[s]} | {np.mean(a):.4f}±{np.std(a):.4f} "
                     f"| {np.mean(e):.4f}±{np.std(e):.4f} "
                     f"| {np.mean(r):.4f}±{np.std(r):.4f} |")
    for s in SUBSETS:
        a = np.array(sc_run[s]["AUROC"], float)
        e = np.array(sc_run[s]["ECE"], float)
        r = np.array(sc_run[s]["AURAC"], float)
        L.append(f"| self_consistency | {SHORT[s]} | {np.mean(a):.4f}±{np.std(a):.4f} "
                 f"| {np.mean(e):.4f}±{np.std(e):.4f} "
                 f"| {np.mean(r):.4f}±{np.std(r):.4f} |")
    L.append("")

    def delta_table(title, rows, ref_label):
        L.append(f"## {title}\n")
        L.append("Mean paired relative % difference across 100 runs "
                 "(variant − reference)/reference; p from paired t-test. "
                 "For ECE lower is better, so a negative RelDiff% is an "
                 "improvement; for AUROC/AURAC positive is better. "
                 "Stars: * p<1e-2, ** p<1e-3, *** p<1e-4.\n")
        L.append(f"| Variant | Subset | Metric | Variant mean | {ref_label} mean "
                 "| RelDiff% | p | sig |")
        L.append("| :-- | :-- | :-- | :--: | :--: | :--: | :--: | :--: |")
        for r in rows:
            L.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]} "
                     f"| {r[6]} | {r[7]} |")
        L.append("")

    delta_table("Delta vs chain5", d_chain, "chain5")
    delta_table("Delta vs Self-Consistency", d_sc, "SC")

    L.append("## Item-level bootstrap (canonical run_0, B=10^4, shared indices)\n")
    L.append("Deltas are chain5+selfprobing − comparison. Two-sided bootstrap p.\n")
    L.append("| Subset | Metric | Comparison | N | Delta | 95% CI | p |")
    L.append("| :-- | :-- | :-- | :--: | :--: | :--: | :--: |")
    for r in boot_rows:
        L.append(f"| {r[0]} | {r[1]} | {r[3]} | {r[4]} | {r[5]} "
                 f"| [{r[6]}, {r[7]}] | {r[8]} |")
    L.append("")

    if degen_notes:
        L.append("## Degeneracies\n")
        agg = Counter()
        for run_i, vname, splits in degen_notes:
            for sp in splits:
                agg[(vname, sp)] += 1
        L.append("All-zero-coefficient (constant scorer) per-split LR fits — "
                 "the aggregator collapsed to the vote-share signal for that "
                 "split/run (count over 100 runs):\n")
        L.append("| Variant | Split | # runs degenerate |")
        L.append("| :-- | :-- | :--: |")
        for (vname, sp), cnt in sorted(agg.items()):
            L.append(f"| {vname} | {sp} | {cnt} |")
        L.append("")

    with open(os.path.join(out_dir, "RESULTS.md"), "w") as f:
        f.write("\n".join(L))


def main():
    os.makedirs(OUT_BASE, exist_ok=True)
    summary = {}
    for name in ["gemini", "minicpm"]:
        summary[name] = run_dataset(name)
    print("\n==================== DONE ====================")
    for name, s in summary.items():
        print(f"{name}: gate worst={s['worst']:.2e}, "
              f"pass={s['runs_pass']}/{N_RUNS}, "
              f"investigate={s['runs_investigate']}")


if __name__ == "__main__":
    main()
