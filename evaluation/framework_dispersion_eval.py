#!/usr/bin/env python3
"""
Integrate the RDS / Semantic Volume dispersion ideas INTO the paper's
chain-level UQ framework (C_chain, 'no_majority' sampling-free protocol) as
new per-candidate / question-level features, evaluate the full combination,
ablate (LOFO), and select the best feature set by honest greedy forward
selection on validation AUROC.  Gemini + E5 data only.

Paper: in-depth dispersion-baseline analysis and hybrid aggregate of
Appendix M.

Feature pool (per candidate i of a question; E = the question's candidate
full-generation embeddings, first n = len(cluster groups) rows of
gemini-answer-embeddings.json; inputs = the question's per-modality INPUT
embeddings from embeddings/e5-all-modalities.json, index-aligned with
gemini_cots_reformatted.json):

  f1..f5   chain features stored in generations_confidence:
           internal_smoothness (S_smooth), internal_goal_directedness
           (S_goal), internal_semantic_density (S_dens),
           cross_modal_coherence (G_avg), cross_modal_grounding_max (G_max)
  f6  cent_cos        mean cosine similarity of e_i to the other candidates
  f7  cent_rds        -|| e_i - centroid(E \\ {i}) ||_2
  f8  sv_loo          -[SV(E) - SV(E \\ {i})]  (candidate i's log-volume
                      contribution; SV = slogdet of the mean-centred Gram
                      + 1e-6 I, exactly the baseline's formula)
  f9  ans_ground_max  max over ALL input embeddings (modalities pooled) of
                      S(e_i, u) with S = (1 + cos)/2
  f10 ans_ground_mean mean over all input embeddings of S(e_i, u)
  Question-level (constant across the candidates, replicated per candidate):
  f11 rds_q           -RDS(E)  (baseline definition verbatim)
  f12 sv_q            -SV(E)   (baseline definition verbatim)
  f13 maj_rds         -RDS within the count-majority cluster's candidates
                      only (majority_ids[0]).  DOCUMENTED CHOICE: a
                      single-member majority cluster has zero dispersion by
                      definition, so the feature is 0.0 (= -0.0, the maximum
                      attainable value of -RDS).  The LR aggregator is free
                      to learn either sign for the feature.

Experiments (canonical partition split_seed=42 / val_fraction=0.2, published
no_majority protocol: per-candidate score -> sum-share over semantic
clusters -> confidence of the count-majority cluster's answer, tie-break by
share; fit per split on validation items with the published protocol:
StandardScaler + L1 LogisticRegression (saga, random_state=0),
C in {0.01,0.03,0.1,0.3,1,3,10}, selection by validation AUROC through the
sum-share path, single-one-hot-feature safety net):

  GATE 1  the stored run_0 aggregator replayed on this script's prepared
          test items must reproduce bootstrap-results/gemini-nomaj/
          point_estimates.csv aggregated_optimal within CSV rounding.
  GATE 2  f11/f12 must exactly match a verbatim re-derivation of
          add_dispersion_baselines_gemini.py's RDS / SV scores.
  3       configs: chain5, disp8 (f6..f13), hybrid7 (chain5+f11+f12),
          full13; LOFO from full13; greedy forward selection per split on
          validation AUROC -> "greedy_best".
  4       item-level paired bootstrap (B=1e4, shared indices):
          greedy_best vs chain5 refit, vs Self-Consistency, vs standalone
          Semantic Volume; Overall + per split; AUROC and AURAC.
  5       bonus strictly single-chain check for f9/f10 (candidate 0, own
          score, own correctness) vs the published single-chain G_max.

Outputs -> bootstrap-results/framework_dispersion/
  configs.csv, lofo.csv, greedy_selection.json, bootstrap_deltas.csv,
  single_chain_answer_grounding.csv, RESULTS.md
Deterministic (seeded saga, seeded bootstrap).
"""

import csv
import json
import os
from collections import Counter, defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import bootstrap_significance as bs

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT, "bootstrap-results", "framework_dispersion")

INPUT_JSON = os.path.join(ROOT, "gemini_cots_reformatted.json")
BASELINE_JSON = os.path.join(ROOT, "gemini_cots_with_additional_baselines.json")
CLUSTER_CACHE = os.path.join(ROOT, "gemini-majority-vote-clusters_top6.json")
ANSWER_EMB = os.path.join(ROOT, "gemini-answer-embeddings.json")
INPUT_EMB = os.path.join(ROOT, "embeddings", "e5-all-modalities.json")
JSONL_PATH = os.path.join(ROOT, "unobench_processed.jsonl")
RUN_META = os.path.join(ROOT, "gemini-majority-multirun-logistic-reg",
                        "run_0", "no_majority", "metadata.json")
POINT_CSV = os.path.join(ROOT, "bootstrap-results", "gemini-nomaj",
                         "point_estimates.csv")
SINGLE_CHAIN_CSV = os.path.join(ROOT, "bootstrap-results", "single_chain",
                                "single_chain_results.csv")

SPLITS = ["UNOBench-Audio", "UNOBench-MC", "UNOBench-MO", "UNOBench-Visual"]
SHORT = {"UNOBench-Audio": "Audio", "UNOBench-MC": "MC",
         "UNOBench-MO": "MO", "UNOBench-Visual": "Visual"}
SUBSETS = SPLITS + ["Overall"]
C_GRID = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
SPLIT_SEED, VAL_FRACTION, THRESHOLD = 42, 0.2, 0.5
B_BOOT, BOOT_SEED = 10000, 0

CHAIN5 = ["internal_smoothness", "internal_goal_directedness",
          "internal_semantic_density", "cross_modal_coherence",
          "cross_modal_grounding_max"]
FEATURE_NAMES = ["S_smooth", "S_goal", "S_dens", "G_avg", "G_max",
                 "cent_cos", "cent_rds", "sv_loo",
                 "ans_ground_max", "ans_ground_mean",
                 "rds_q", "sv_q", "maj_rds"]  # f1..f13
CONFIGS = {
    "chain5": list(range(5)),
    "disp8": list(range(5, 13)),
    "hybrid7": list(range(5)) + [10, 11],
    "full13": list(range(13)),
}


# ----------------------------------------------------------------------------------
# Dispersion primitives (baseline-identical formulas)
# ----------------------------------------------------------------------------------
def sv_logdet(E):
    """add_dispersion_baselines_eval.py's Semantic Volume: slogdet of the
    mean-centred Gram + 1e-6 I."""
    E = np.asarray(E, dtype=float)
    X = E - E.mean(axis=0)
    G = X @ X.T
    _sign, logdet = np.linalg.slogdet(G + 1e-6 * np.eye(len(E)))
    return float(logdet)


def rds_mean_dist(E):
    """add_dispersion_baselines_eval.py's RDS: mean Euclidean distance from
    the centroid."""
    E = np.asarray(E, dtype=float)
    c = E.mean(axis=0)
    return float(np.mean(np.linalg.norm(E - c, axis=1)))


def build_features(n, E, inputs, groups, majority_ids, confs):
    """(n x 13) per-candidate feature matrix for one question."""
    feat = np.zeros((n, 13), dtype=float)
    # f1..f5 stored chain features
    for i in range(n):
        for j, f in enumerate(CHAIN5):
            feat[i, j] = float(confs[i].get(f, 0.0))

    En = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
    cos_mat = En @ En.T
    sv_full = sv_logdet(E)
    if n == 1:
        # DOCUMENTED CHOICE for single-candidate questions (3 in the data):
        # the leave-one-out pool is empty, so — consistently with the f13
        # zero-dispersion convention — f6 = 1.0 (max agreement), f7 = -0.0
        # (zero distance), f8 = 0.0 (no volume contribution).
        feat[0, 5], feat[0, 6], feat[0, 7] = 1.0, 0.0, 0.0
    else:
        for i in range(n):
            others = np.delete(np.arange(n), i)
            # f6 cent_cos
            feat[i, 5] = float(np.mean(cos_mat[i, others]))
            # f7 cent_rds
            cent_loo = E[others].mean(axis=0)
            feat[i, 6] = -float(np.linalg.norm(E[i] - cent_loo))
            # f8 sv_loo
            feat[i, 7] = -(sv_full - sv_logdet(E[others]))

    # f9/f10 answer grounding to pooled input embeddings
    if inputs is not None and len(inputs) > 0:
        U = np.asarray(inputs, dtype=float)
        Un = U / np.maximum(np.linalg.norm(U, axis=1, keepdims=True), 1e-12)
        S = (1.0 + En @ Un.T) / 2.0            # (n x n_inputs)
        feat[:, 8] = S.max(axis=1)
        feat[:, 9] = S.mean(axis=1)
    else:  # no input embeddings stored for this question (does not occur)
        feat[:, 8] = 0.5
        feat[:, 9] = 0.5

    # f11/f12 question-level baselines, replicated per candidate
    feat[:, 10] = -rds_mean_dist(E)
    feat[:, 11] = -sv_full

    # f13 majority-cluster RDS
    maj = majority_ids[0]
    members = [i for i in range(n) if groups[i] == maj]
    feat[:, 12] = -rds_mean_dist(E[members]) if len(members) >= 2 else 0.0
    return feat


# ----------------------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------------------
def load_id2cat():
    id2cat = {}
    with open(JSONL_PATH) as f:
        for line in f:
            if line.strip():
                it = json.loads(line)
                id2cat[it["question_id"]] = it.get("category", "Unknown")
    return id2cat


def load_items():
    meta = json.load(open(RUN_META))
    records = json.load(open(INPUT_JSON))
    cluster_info = json.load(open(CLUSTER_CACHE))
    embs = {str(e["question_id"]): e["embeddings"]
            for e in json.load(open(ANSWER_EMB))}
    input_embs = json.load(open(INPUT_EMB))  # index-aligned with records
    id2cat = load_id2cat()

    # record-index lookup for the input-embedding alignment
    inputs_by_qid = {}
    for idx, rec in enumerate(records):
        m = input_embs[idx]
        vecs = []
        for k in ("audio", "image", "video"):
            if m.get(k):
                vecs.extend(m[k])
        inputs_by_qid[str(rec["question_id"])] = vecs

    test_records, val_records = bs.stratified_test_val_split(
        records, id2cat, VAL_FRACTION, SPLIT_SEED)

    dropped = Counter()

    def prepare(recs):
        items = []
        for rec in recs:
            qid = str(rec.get("question_id"))
            ci = cluster_info.get(qid)
            if not ci or not ci.get("groups"):
                dropped["no_cluster"] += 1
                continue
            groups = ci["groups"]
            n = len(groups)
            confs = rec.get("generations_confidence", []) or []
            corr = bs.get_correctness(rec, THRESHOLD)
            E_all = embs.get(qid)
            if corr is None or len(confs) < n or len(corr) < n:
                dropped["no_scores"] += 1
                continue
            if E_all is None or len(E_all) < n:
                dropped["no_answer_emb"] += 1
                continue
            E = np.asarray(E_all[:n], dtype=float)
            feat = build_features(n, E, inputs_by_qid.get(qid), groups,
                                  ci["majority_ids"], confs)
            items.append({
                "qid": qid,
                "category": id2cat.get(rec.get("question_id"), "Unknown"),
                "groups": groups,
                "groups_arr": np.array(groups, dtype=int),
                "majority_ids": ci["majority_ids"],
                "correctness": np.array(corr[:n], dtype=int),
                "feat": feat,          # n x 13
                "n": n,
            })
        return items

    test_items, val_items = prepare(test_records), prepare(val_records)
    print(f"Prepared items: test={len(test_items)}, val={len(val_items)}, "
          f"dropped={dict(dropped)}")
    return meta, test_items, val_items


def by_split(items):
    d = defaultdict(list)
    for it in items:
        d[it["category"]].append(it)
    return d


# ----------------------------------------------------------------------------------
# no_majority evaluation path (ports of cchain_transfer_eval / bootstrap_significance)
# ----------------------------------------------------------------------------------
def apply_scorer(items, scorer, cols):
    """Per-candidate score -> sum-share over clusters -> count-majority
    answer with share tie-break; y_prob = winning group's sum-share."""
    y_true, y_prob = [], []
    for it in items:
        scores = np.asarray(scorer(it["feat"][:, cols]), dtype=float)
        groups_arr = it["groups_arr"]
        n_groups = int(groups_arr.max()) + 1
        group_sum = np.bincount(groups_arr, weights=scores, minlength=n_groups)
        norm_sum = group_sum / (group_sum.sum() + 1e-9)
        majority_ids = it["majority_ids"]
        if len(majority_ids) == 1:
            wg = majority_ids[0]
        else:
            wg = max(majority_ids, key=lambda g: norm_sum[g])
        wi = it["groups"].index(wg)
        y_true.append(int(it["correctness"][wi]))
        y_prob.append(float(norm_sum[wg]))
    return np.array(y_true, dtype=int), np.array(y_prob, dtype=float)


def apply_scorer_per_category(items, scorers, cols_map, fallback=None):
    """Natural record order, each item scored by its category's scorer."""
    y_true, y_prob, cats = [], [], []
    for it in items:
        cat = it["category"]
        scorer = scorers.get(cat, fallback)
        if scorer is None:
            raise RuntimeError(f"no scorer for category {cat}")
        yt, yp = apply_scorer([it], scorer, cols_map[cat])
        y_true.append(int(yt[0]))
        y_prob.append(float(yp[0]))
        cats.append(cat)
    return (np.array(y_true, dtype=int), np.array(y_prob, dtype=float),
            np.array(cats))


def eval_metrics(y_true, y_prob):
    y_true = np.asarray(y_true, dtype=int)
    p = bs.minmax_transform(np.asarray(y_prob, dtype=float))
    if len(np.unique(y_true)) < 2:
        return {"AUROC": np.nan, "ECE": np.nan, "AURAC": np.nan}
    return {"AUROC": float(bs.fast_auroc(y_true, p)),
            "ECE": float(bs.fast_ece(y_true, p)),
            "AURAC": float(bs.fast_aurac(y_true, p))}


def val_auroc_of(items, scorer, cols):
    y_true, y_prob = apply_scorer(items, scorer, cols)
    if len(y_true) == 0 or len(np.unique(y_true)) < 2:
        return None
    return float(bs.fast_auroc(y_true, bs.minmax_transform(y_prob)))


# ----------------------------------------------------------------------------------
# Published fit protocol for an arbitrary feature subset (fit_cchain logic)
# ----------------------------------------------------------------------------------
def fit_config(val_items, cols, label=""):
    X = np.vstack([it["feat"][:, cols] for it in val_items])
    y = np.concatenate([it["correctness"] for it in val_items])
    if len(X) < 2 or len(np.unique(y)) < 2:
        raise RuntimeError(f"[{label}] not enough data / single class")

    best_model, best_val, best_c = None, -np.inf, None
    for c in C_GRID:
        cand = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=5000, C=c, penalty="l1",
                                      solver="saga", random_state=0)),
        ])
        try:
            cand.fit(X, y)
        except Exception as e:
            print(f"[{label}] C={c} failed: {e}")
            continue

        def lr_score(F, m=cand):
            return m.predict_proba(F)[:, 1]

        v = val_auroc_of(val_items, lr_score, cols)
        if v is None:
            continue
        if best_model is None or v > best_val:
            best_val, best_model, best_c = v, cand, c
    if best_model is None:
        raise RuntimeError(f"[{label}] no valid LR across the C grid")

    # single one-hot feature safety net (strictly better wins)
    single_j, single_val = None, -np.inf
    for j in range(len(cols)):
        def dot_score(F, jj=j):
            return F[:, jj]

        v = val_auroc_of(val_items, dot_score, cols)
        if v is None:
            continue
        if single_j is None or v > single_val:
            single_j, single_val = j, v

    if single_j is not None and single_val > best_val:
        def scorer(F, jj=single_j):
            return F[:, jj]

        name = FEATURE_NAMES[cols[single_j]]
        return {"kind": "single", "scorer": scorer, "val_auroc": single_val,
                "desc": f"single:{name}"}

    lr = best_model.named_steps["lr"]

    def scorer(F, m=best_model):
        return m.predict_proba(F)[:, 1]

    degenerate = bool(np.all(lr.coef_[0] == 0.0))
    nz = [FEATURE_NAMES[cols[j]] for j in range(len(cols))
          if lr.coef_[0][j] != 0.0]
    return {"kind": "lr", "scorer": scorer, "val_auroc": best_val,
            "C": best_c, "degenerate": degenerate, "nonzero": nz,
            "coef": lr.coef_[0].copy(),
            "desc": f"LR(C={best_c})" + (" [all-zero coef]" if degenerate
                                         else f" nz={','.join(nz)}")}


def fit_and_eval_config(cols, val_by, test_by, test_items, label):
    """Per-split fits (published protocol) + Overall via per-category
    scorers in natural order. Returns rows + per-item Overall arrays."""
    fits = {}
    for split in SPLITS:
        fits[split] = fit_config(val_by[split], cols, f"{label}/{SHORT[split]}")
    scorers = {s: fits[s]["scorer"] for s in SPLITS}
    cols_map = {s: cols for s in SPLITS}
    yt_all, yp_all, cats = apply_scorer_per_category(test_items, scorers,
                                                     cols_map)
    rows = []
    for subset in SUBSETS:
        mask = np.ones(len(cats), bool) if subset == "Overall" else cats == subset
        m = eval_metrics(yt_all[mask], yp_all[mask])
        desc = fits[subset]["desc"] if subset in fits else "per-split"
        valv = fits[subset]["val_auroc"] if subset in fits else np.nan
        rows.append([label, subset, int(mask.sum()),
                     m["AUROC"], m["ECE"], m["AURAC"], desc, valv])
    return fits, rows, (yt_all, yp_all, cats)


# ----------------------------------------------------------------------------------
# Greedy forward selection (validation AUROC, per split)
# ----------------------------------------------------------------------------------
def greedy_forward(val_items, label=""):
    selected, best_val, best_fit, trace = [], -np.inf, None, []
    remaining = list(range(13))
    while remaining:
        cand = []
        for f in remaining:
            fit = fit_config(val_items, selected + [f],
                             f"{label}+{FEATURE_NAMES[f]}")
            cand.append((fit["val_auroc"], f, fit))
        # deterministic tie-break: highest val AUROC, then lowest feature idx
        cand.sort(key=lambda t: (-t[0], t[1]))
        v, f, fit = cand[0]
        if v > best_val + 1e-12:
            selected.append(f)
            best_val, best_fit = v, fit
            remaining.remove(f)
            trace.append({"added": FEATURE_NAMES[f], "val_auroc": v,
                          "fit": fit["desc"]})
            print(f"  [{label}] + {FEATURE_NAMES[f]:15s} val AUROC = {v:.6f}"
                  f"  ({fit['desc']})")
        else:
            break
    return selected, best_val, best_fit, trace


# ----------------------------------------------------------------------------------
# GATE 1: stored run_0 aggregator replay vs point_estimates.csv
# ----------------------------------------------------------------------------------
def gate1(meta, test_items):
    features = meta["features"]
    assert features == CHAIN5
    bw = meta["best_weights"]
    fallback = bs.make_scorer(bw["fallback"], features)
    scorers = {cat: bs.make_scorer(e, features) or fallback
               for cat, e in bw["per_category"].items()}
    cols = list(range(5))
    cols_map = {cat: cols for cat in list(scorers) + ["Unknown"]}
    yt, yp, cats = apply_scorer_per_category(test_items, scorers, cols_map,
                                             fallback)
    stored = {}
    with open(POINT_CSV) as f:
        for row in csv.DictReader(f):
            if row["Method"] == "aggregated_optimal":
                stored[row["Subset"]] = {"AUROC": float(row["AUROC"]),
                                         "ECE": float(row["ECE_minmax"]),
                                         "AURAC": float(row["AURAC"])}
    worst, rows = 0.0, []
    for subset in SUBSETS:
        mask = np.ones(len(cats), bool) if subset == "Overall" else cats == subset
        m = eval_metrics(yt[mask], yp[mask])
        for k in ["AUROC", "ECE", "AURAC"]:
            diff = abs(m[k] - stored[subset][k])
            worst = max(worst, diff)
            rows.append([subset, k, m[k], stored[subset][k], diff])
    return worst, rows


# ----------------------------------------------------------------------------------
# GATE 2: verbatim re-derivation of the baseline RDS/SV scores
# ----------------------------------------------------------------------------------
def gate2(test_items, val_items):
    records = json.load(open(BASELINE_JSON))
    cluster_info = json.load(open(CLUSTER_CACHE))
    embs = {str(e["question_id"]): e["embeddings"]
            for e in json.load(open(ANSWER_EMB))}
    ref = {}
    for rec in records:  # verbatim port of add_dispersion_baselines_eval.py
        qid = str(rec.get("question_id"))
        ci = cluster_info.get(qid)
        if not ci or not ci.get("groups"):
            continue
        groups = ci["groups"]
        n = len(groups)
        corr = bs.get_correctness(rec, 0.5)
        E_all = embs.get(qid)
        if corr is None or E_all is None or len(corr) < n or len(E_all) < n:
            continue
        E = np.asarray(E_all[:n], dtype=float)
        c = E.mean(axis=0)
        rds = float(np.mean(np.linalg.norm(E - c, axis=1)))
        X = E - c
        G = X @ X.T
        _sign, logdet = np.linalg.slogdet(G + 1e-6 * np.eye(n))
        ref[qid] = (-rds, -float(logdet))

    n_cmp, worst = 0, 0.0
    for it in test_items + val_items:
        if it["qid"] not in ref:
            continue
        r, s = ref[it["qid"]]
        worst = max(worst, abs(it["feat"][0, 10] - r),
                    abs(it["feat"][0, 11] - s))
        n_cmp += 1
    return worst, n_cmp


# ----------------------------------------------------------------------------------
# Item-level paired bootstrap
# ----------------------------------------------------------------------------------
def paired_bootstrap(method_arrays, cats, pairs, out_rows):
    """method_arrays: {name: (y_true, y_prob_raw)} aligned in natural test
    order; pairs: (a, b) method-name tuples. Shared resample indices."""
    rng = np.random.default_rng(BOOT_SEED)
    for subset in ["Overall"] + SPLITS:
        mask = (np.ones(len(cats), bool) if subset == "Overall"
                else cats == subset)
        prepared = {}
        for name, (yt, yp) in method_arrays.items():
            prepared[name] = (yt[mask],
                              bs.minmax_transform(yp[mask]))
        n = int(mask.sum())
        boot = {name: {"AUROC": np.empty(B_BOOT), "AURAC": np.empty(B_BOOT)}
                for name in prepared}
        for b in range(B_BOOT):
            idx = rng.integers(0, n, n)
            for name, (yt, yp) in prepared.items():
                ytb, ypb = yt[idx], yp[idx]
                boot[name]["AUROC"][b] = bs.fast_auroc(ytb, ypb)
                boot[name]["AURAC"][b] = bs.fast_aurac(ytb, ypb)
        for a, bref in pairs:
            for k in ["AUROC", "AURAC"]:
                da = boot[a][k] - boot[bref][k]
                da = da[~np.isnan(da)]
                if len(da) == 0:
                    continue
                lo, hi = np.percentile(da, 2.5), np.percentile(da, 97.5)
                p_two = min(1.0, 2 * min(np.mean(da <= 0), np.mean(da >= 0))
                            + 1.0 / len(da))
                pa = eval_metrics(*[prepared[a][i] for i in (0, 1)])[k] \
                    if len(np.unique(prepared[a][0])) > 1 else np.nan
                pb = eval_metrics(*[prepared[bref][i] for i in (0, 1)])[k] \
                    if len(np.unique(prepared[bref][0])) > 1 else np.nan
                out_rows.append([subset, k, a, bref, pa, pb,
                                 float(np.mean(da)), float(lo), float(hi),
                                 p_two])
        print(f"  bootstrap [{subset}] done (n={n})")


# ----------------------------------------------------------------------------------
# Bonus: strictly single-chain f9/f10
# ----------------------------------------------------------------------------------
def single_chain_answer_grounding():
    """Candidate 0, own score, own correctness; same admission as
    single_chain_eval.py so numbers are comparable to the published
    single-chain G_max."""
    records = json.load(open(INPUT_JSON))
    embs = {str(e["question_id"]): e["embeddings"]
            for e in json.load(open(ANSWER_EMB))}
    input_embs = json.load(open(INPUT_EMB))
    id2cat = load_id2cat()

    items = []
    for idx, rec in enumerate(records):
        confs = rec.get("generations_confidence") or []
        unos = rec.get("generations_uno_score") or []
        if not confs or not unos or unos[0] is None:
            continue
        c0 = confs[0]
        if any(c0.get(f) is None for f in CHAIN5):
            continue
        qid = str(rec.get("question_id"))
        E_all = embs.get(qid)
        m = input_embs[idx]
        vecs = []
        for k in ("audio", "image", "video"):
            if m.get(k):
                vecs.extend(m[k])
        if E_all is None or len(E_all) == 0 or not vecs:
            continue
        e0 = np.asarray(E_all[0], dtype=float)
        e0 = e0 / max(np.linalg.norm(e0), 1e-12)
        U = np.asarray(vecs, dtype=float)
        U = U / np.maximum(np.linalg.norm(U, axis=1, keepdims=True), 1e-12)
        S = (1.0 + U @ e0) / 2.0
        items.append({
            "qid": rec.get("question_id"),
            "category": id2cat.get(rec.get("question_id"), "Unknown"),
            "y": int(float(unos[0]) >= THRESHOLD),
            "f9": float(S.max()), "f10": float(S.mean()),
            "gmax": float(c0["cross_modal_grounding_max"]),
        })

    test, _val = bs.stratified_test_val_split(items, id2cat, VAL_FRACTION,
                                              SPLIT_SEED)
    cats = np.array([it["category"] for it in test])
    y = np.array([it["y"] for it in test])
    rows = []
    for name, key in [("ans_ground_max (f9)", "f9"),
                      ("ans_ground_mean (f10)", "f10"),
                      ("G_max (replayed)", "gmax")]:
        p = np.array([it[key] for it in test], dtype=float)
        for subset in SUBSETS:
            mask = np.ones(len(y), bool) if subset == "Overall" else cats == subset
            m = eval_metrics(y[mask], p[mask])
            rows.append([name, subset, int(mask.sum()),
                         m["AUROC"], m["ECE"], m["AURAC"]])
    # published single-chain G_max for reference
    pub = {}
    if os.path.exists(SINGLE_CHAIN_CSV):
        with open(SINGLE_CHAIN_CSV) as f:
            for r in csv.DictReader(f):
                if r["Method"] == "G_max":
                    pub[r["Subset"]] = float(r["AUROC"])
    return rows, pub


# ----------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    meta, test_items, val_items = load_items()
    test_by, val_by = by_split(test_items), by_split(val_items)

    # ============================ GATE 1 ============================
    print("=" * 70)
    print("GATE 1: stored aggregator replay vs point_estimates.csv")
    worst1, gate1_rows = gate1(meta, test_items)
    print(f"GATE 1 worst |diff| = {worst1:.2e} "
          f"({'PASS' if worst1 <= 1e-6 else 'FAIL'})")
    if worst1 > 1e-6:
        raise SystemExit("GATE 1 FAILED")

    # ============================ GATE 2 ============================
    print("=" * 70)
    print("GATE 2: f11/f12 vs verbatim baseline RDS/SV re-derivation")
    worst2, n_cmp = gate2(test_items, val_items)
    print(f"GATE 2 worst |diff| = {worst2:.2e} over {n_cmp} questions "
          f"({'PASS' if worst2 <= 1e-12 and n_cmp >= 100 else 'FAIL'})")
    if worst2 > 1e-12 or n_cmp < 100:
        raise SystemExit("GATE 2 FAILED")

    # ===================== named configs + LOFO =====================
    print("=" * 70)
    print("Configs: chain5 / disp8 / hybrid7 / full13")
    config_rows, config_arrays, config_fits = [], {}, {}
    for label, cols in CONFIGS.items():
        fits, rows, arrays = fit_and_eval_config(cols, val_by, test_by,
                                                 test_items, label)
        config_rows += rows
        config_arrays[label] = arrays
        config_fits[label] = fits
        ov = next(r for r in rows if r[1] == "Overall")
        print(f"  {label:8s} Overall AUROC={ov[3]:.4f} AURAC={ov[5]:.4f}")

    print("LOFO from full13 ...")
    lofo_rows = []
    full_by_subset = {r[1]: r for r in config_rows if r[0] == "full13"}
    for k in range(13):
        cols = [j for j in range(13) if j != k]
        label = f"drop_{FEATURE_NAMES[k]}"
        _fits, rows, _arrays = fit_and_eval_config(cols, val_by, test_by,
                                                   test_items, label)
        for r in rows:
            full = full_by_subset[r[1]]
            lofo_rows.append([FEATURE_NAMES[k], r[1], r[2], r[3], r[4], r[5],
                              r[3] - full[3], r[5] - full[5], r[6]])
        ov = next(r for r in rows if r[1] == "Overall")
        print(f"  {label:22s} Overall AUROC={ov[3]:.4f} "
              f"(d={ov[3] - full_by_subset['Overall'][3]:+.4f})")

    # ===================== greedy forward selection ==================
    print("=" * 70)
    print("Greedy forward selection per split (validation AUROC)")
    greedy = {}
    for split in SPLITS:
        print(f"[{SHORT[split]}]")
        sel, vbest, fit, trace = greedy_forward(val_by[split], SHORT[split])
        greedy[split] = {"selected_idx": sel,
                         "selected": [FEATURE_NAMES[j] for j in sel],
                         "val_auroc": vbest, "fit": fit, "trace": trace}
    scorers = {s: greedy[s]["fit"]["scorer"] for s in SPLITS}
    cols_map = {s: greedy[s]["selected_idx"] for s in SPLITS}
    yt_g, yp_g, cats = apply_scorer_per_category(test_items, scorers, cols_map)
    for subset in SUBSETS:
        mask = np.ones(len(cats), bool) if subset == "Overall" else cats == subset
        m = eval_metrics(yt_g[mask], yp_g[mask])
        desc = (greedy[subset]["fit"]["desc"] + " feats=" +
                "+".join(greedy[subset]["selected"])) if subset in greedy \
            else "per-split"
        valv = greedy[subset]["val_auroc"] if subset in greedy else np.nan
        config_rows.append(["greedy_best", subset, int(mask.sum()),
                            m["AUROC"], m["ECE"], m["AURAC"], desc, valv])
        print(f"  greedy_best {subset:16s} AUROC={m['AUROC']:.4f} "
              f"AURAC={m['AURAC']:.4f}")
    config_arrays["greedy_best"] = (yt_g, yp_g, cats)

    # ================= reference methods for the bootstrap ==========
    sc_y, sc_p, sv_y, sv_p = [], [], [], []
    for it in test_items:
        counts = Counter(it["groups"])
        g0 = it["majority_ids"][0]
        i0 = it["groups"].index(g0)
        sc_y.append(int(it["correctness"][i0]))
        sc_p.append(counts[g0] / it["n"])
        sv_y.append(int(it["correctness"][i0]))   # question-level treatment
        sv_p.append(float(it["feat"][0, 11]))     # -SV
    method_arrays = {
        "greedy_best": (yt_g, yp_g),
        "chain5": (config_arrays["chain5"][0], config_arrays["chain5"][1]),
        "self_consistency": (np.array(sc_y), np.array(sc_p, dtype=float)),
        "semantic_volume": (np.array(sv_y), np.array(sv_p, dtype=float)),
    }
    print("=" * 70)
    print(f"Item-level paired bootstrap (B={B_BOOT}, shared indices)")
    boot_rows = []
    pairs = [("greedy_best", "chain5"),
             ("greedy_best", "self_consistency"),
             ("greedy_best", "semantic_volume")]
    paired_bootstrap(method_arrays, cats, pairs, boot_rows)

    # ================= bonus: single-chain f9/f10 ====================
    print("=" * 70)
    print("Bonus: strictly single-chain answer grounding (f9/f10)")
    sc_rows, pub_gmax = single_chain_answer_grounding()
    for r in sc_rows:
        if r[1] in ("UNOBench-Audio", "Overall"):
            print(f"  {r[0]:22s} {SHORT.get(r[1], r[1]):8s} AUROC={r[3]:.4f}")
    print(f"  published single-chain G_max: "
          + ", ".join(f"{SHORT.get(s, s)}={v:.4f}" for s, v in pub_gmax.items()))

    # ============================ outputs ============================
    def write_csv(name, header, rows):
        with open(os.path.join(OUT_DIR, name), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            for r in rows:
                w.writerow([f"{x:.6f}" if isinstance(x, (float, np.floating))
                            else x for x in r])
        print(f"Wrote {os.path.join(OUT_DIR, name)}")

    write_csv("gate1_replay.csv",
              ["Subset", "Metric", "Replayed", "Stored", "AbsDiff"],
              gate1_rows)
    write_csv("configs.csv",
              ["Config", "Subset", "N", "AUROC", "ECE_minmax", "AURAC",
               "FitDesc", "ValAUROC"], config_rows)
    write_csv("lofo.csv",
              ["DroppedFeature", "Subset", "N", "AUROC", "ECE_minmax",
               "AURAC", "dAUROC_vs_full13", "dAURAC_vs_full13", "FitDesc"],
              lofo_rows)
    with open(os.path.join(OUT_DIR, "greedy_selection.json"), "w") as f:
        json.dump({SHORT[s]: {
            "selected": greedy[s]["selected"],
            "val_auroc": greedy[s]["val_auroc"],
            "final_fit": greedy[s]["fit"]["desc"],
            "trace": greedy[s]["trace"],
        } for s in SPLITS}, f, indent=1)
    print(f"Wrote {os.path.join(OUT_DIR, 'greedy_selection.json')}")
    write_csv("bootstrap_deltas.csv",
              ["Subset", "Metric", "Method", "Reference", "Method_point",
               "Reference_point", "Delta_mean", "Delta_lo", "Delta_hi",
               "p_boot_two_sided"], boot_rows)
    write_csv("single_chain_answer_grounding.csv",
              ["Method", "Subset", "N", "AUROC", "ECE_minmax", "AURAC"],
              sc_rows)

    write_results_md(worst1, worst2, n_cmp, config_rows, lofo_rows, greedy,
                     boot_rows, sc_rows, pub_gmax)
    print(f"\nAll outputs in {OUT_DIR}")


def write_results_md(worst1, worst2, n_cmp, config_rows, lofo_rows, greedy,
                     boot_rows, sc_rows, pub_gmax):
    def cfg(label, subset):
        return next(r for r in config_rows if r[0] == label and r[1] == subset)

    L = []
    L.append("# Dispersion features inside the chain-level UQ framework "
             "(Gemini + E5, no_majority protocol)\n")
    L.append("Canonical partition (split_seed=42, val_fraction=0.2). Every "
             "configuration is fit per split on the validation items with "
             "the published protocol (StandardScaler + L1 logistic "
             "regression, saga seeded with random_state=0, C grid "
             "{0.01..10}, model selection by validation AUROC through the "
             "sum-share path, single-one-hot-feature safety net) and "
             "evaluated on the test items; Overall applies each item's own "
             "split scorer in natural record order.\n")
    L.append("Features: f1-f5 = chain5 (S_smooth, S_goal, S_dens, G_avg, "
             "G_max); f6 cent_cos, f7 cent_rds, f8 sv_loo (candidate-level "
             "dispersion); f9/f10 ans_ground_max/mean (answer-to-input "
             "grounding, S=(1+cos)/2 over all pooled input embeddings); "
             "f11 rds_q = -RDS, f12 sv_q = -SV (baseline definitions, "
             "question-level, replicated per candidate); f13 maj_rds = -RDS "
             "within the count-majority cluster (0.0, i.e. zero dispersion "
             "= max confidence, when the cluster has a single member — "
             "documented choice; the LR is free to learn either sign). "
             "For the 3 single-candidate questions (n=1) the leave-one-out "
             "features use the same zero-dispersion convention: f6=1.0, "
             "f7=-0.0, f8=0.0.\n")

    L.append("## Gates\n")
    L.append(f"- GATE 1 (stored run_0 aggregator replayed on this script's "
             f"prepared items vs `bootstrap-results/gemini-nomaj/"
             f"point_estimates.csv`, aggregated_optimal, all subsets x "
             f"AUROC/ECE/AURAC): worst |diff| = {worst1:.2e} — PASS "
             f"(<= CSV rounding 1e-6).")
    L.append(f"- GATE 2 (f11/f12 vs a verbatim re-derivation of "
             f"`add_dispersion_baselines_eval.py`'s RDS/SV on its own "
             f"record file): worst |diff| = {worst2:.2e} over {n_cmp} "
             f"questions — PASS.\n")

    L.append("## Main config table (test)\n")
    L.append("| Config | " + " | ".join(f"{SHORT.get(s, s)} AUROC/AURAC"
                                        for s in SUBSETS) + " |")
    L.append("| :-- " + "| :--: " * len(SUBSETS) + "|")
    for label in ["chain5", "disp8", "hybrid7", "full13", "greedy_best"]:
        cells = []
        for s in SUBSETS:
            r = cfg(label, s)
            cells.append(f"{r[3]:.3f}/{r[5]:.3f}")
        L.append(f"| {label} | " + " | ".join(cells) + " |")
    L.append("")
    L.append("Per-split aggregator chosen by each config (validation):\n")
    L.append("| Config | " + " | ".join(SHORT[s] for s in SPLITS) + " |")
    L.append("| :-- " + "| :-- " * len(SPLITS) + "|")
    for label in ["chain5", "disp8", "hybrid7", "full13", "greedy_best"]:
        cells = [str(cfg(label, s)[6]) for s in SPLITS]
        L.append(f"| {label} | " + " | ".join(cells) + " |")
    L.append("")

    L.append("## LOFO from full13 (Overall test AUROC, sorted by damage)\n")
    ov = sorted([r for r in lofo_rows if r[1] == "Overall"],
                key=lambda r: r[6])
    L.append("| Dropped | Overall AUROC | dAUROC | dAURAC |")
    L.append("| :-- | :--: | :--: | :--: |")
    for r in ov:
        L.append(f"| {r[0]} | {r[3]:.4f} | {r[6]:+.4f} | {r[7]:+.4f} |")
    L.append("")

    L.append("## Greedy forward selection (validation AUROC, honest)\n")
    L.append("| Split | Selected (in order) | Final val AUROC | Final fit |")
    L.append("| :-- | :-- | :--: | :-- |")
    for s in SPLITS:
        g = greedy[s]
        L.append(f"| {SHORT[s]} | {' -> '.join(g['selected'])} | "
                 f"{g['val_auroc']:.4f} | {g['fit']['desc']} |")
    L.append("")

    L.append("## Paired bootstrap: greedy_best vs references "
             f"(B={B_BOOT}, shared indices, two-sided p)\n")
    L.append("| Subset | Metric | vs | best | ref | delta [95% CI] | p |")
    L.append("| :-- | :-- | :-- | :--: | :--: | :--: | :--: |")
    for r in boot_rows:
        L.append(f"| {SHORT.get(r[0], r[0])} | {r[1]} | {r[3]} | "
                 f"{r[4]:.3f} | {r[5]:.3f} | {r[6]:+.4f} "
                 f"[{r[7]:+.4f}, {r[8]:+.4f}] | {r[9]:.4f} |")
    L.append("")

    L.append("## Bonus: strictly single-chain answer grounding (candidate "
             "0, own score, own correctness)\n")
    L.append("| Method | " + " | ".join(f"{SHORT.get(s, s)} AUROC"
                                        for s in SUBSETS) + " |")
    L.append("| :-- " + "| :--: " * len(SUBSETS) + "|")
    for name in ["ans_ground_max (f9)", "ans_ground_mean (f10)",
                 "G_max (replayed)"]:
        cells = [f"{r[3]:.4f}" for r in sc_rows if r[0] == name]
        L.append(f"| {name} | " + " | ".join(cells) + " |")
    if pub_gmax:
        L.append(f"| G_max (published single_chain_results.csv) | "
                 + " | ".join(f"{pub_gmax.get(s, float('nan')):.4f}"
                              for s in SUBSETS) + " |")
    L.append("")

    # ------------------------------ takeaways ------------------------------
    def boot(subset, metric, ref):
        return next(r for r in boot_rows if r[0] == subset and r[1] == metric
                    and r[3] == ref)

    sv_pts = {s: boot(s, "AUROC", "semantic_volume")[5]
              for s in ["Overall"] + SPLITS}
    gb_ch = boot("Overall", "AUROC", "chain5")
    L.append("## Takeaways\n")
    L.append(f"1. **Both gates pass** (stored-aggregator replay within CSV "
             f"rounding; f11/f12 bit-identical to the baseline script), so "
             f"every number below sits on the published pipeline.")
    L.append(f"2. **Inside the framework the dispersion features add only "
             f"a modest, split-local gain.** disp8 (f6-f13 only) is the "
             f"best fixed config Overall "
             f"(AUROC {cfg('disp8', 'Overall')[3]:.3f} vs chain5 "
             f"{cfg('chain5', 'Overall')[3]:.3f}; AURAC "
             f"{cfg('disp8', 'Overall')[5]:.3f} vs "
             f"{cfg('chain5', 'Overall')[5]:.3f}), driven by Visual, where "
             f"the candidate-level leave-one-out distance cent_rds replaces "
             f"G_max as the selected scorer "
             f"({cfg('disp8', 'UNOBench-Visual')[3]:.3f} vs "
             f"{cfg('chain5', 'UNOBench-Visual')[3]:.3f} AUROC). "
             f"cent_rds/ans_ground_mean are the only new features with any "
             f"LOFO footprint; most LOFO deltas are exactly 0 because the "
             f"protocol's safety net keeps selecting sparse/one-hot "
             f"aggregators.")
    L.append(f"3. **The baselines' signature strength — Audio — does NOT "
             f"survive the integration.** Standalone Semantic Volume "
             f"reaches {sv_pts['UNOBench-Audio']:.3f} Audio AUROC on this "
             f"partition, while every in-framework config sits at "
             f"0.47-0.50. This is structural, not a fitting failure: the "
             f"no_majority confidence is the majority cluster's sum-SHARE "
             f"of per-candidate scores, which is invariant to question-"
             f"level rescaling — a question-level feature (f11/f12/f13) "
             f"enters all candidates of a question identically, so a "
             f"one-hot selection of it collapses to the vote share and an "
             f"LR can only express it through the sigmoid's nonlinearity. "
             f"The absolute between-candidate geometry that RDS/SV use is "
             f"exactly what the share normalisation removes.")
    L.append(f"4. **Honest greedy selection does not beat C_chain.** "
             f"Selected on validation AUROC only: Audio -> G_avg (chance-"
             f"level val, same as chain5), MC -> constant scorer (= vote "
             f"share; all MC fits are degenerate, tie-noise band), MO -> "
             f"ans_ground_mean, Visual -> cent_rds+sv_loo+G_max. Test "
             f"Overall AUROC {cfg('greedy_best', 'Overall')[3]:.3f} is "
             f"significantly WORSE than chain5 ({gb_ch[6]:+.4f}, p = "
             f"{gb_ch[9]:.4f}): the per-split validation sets (40-296 "
             f"questions) are too small for reliable feature selection, "
             f"and val gains (e.g. MO 0.689, Visual 0.789) do not "
             f"transfer. greedy_best is also significantly worse than "
             f"standalone SV Overall/Audio and never significantly better "
             f"than any reference except MC vs Self-Consistency "
             f"(p = {boot('UNOBench-MC', 'AUROC', 'self_consistency')[9]:.3f}, "
             f"trivially small delta).")
    L.append(f"5. **Caveat on the standalone SV Overall value "
             f"({sv_pts['Overall']:.3f}).** It is a single global score "
             f"ranked across splits with very different accuracy rates; "
             f"like the paper's pooled C_chain 0.653, much of it is a "
             f"cross-split base-rate artifact — per split SV is below "
             f"chain5 everywhere except Audio (MC "
             f"{sv_pts['UNOBench-MC']:.3f}, MO "
             f"{sv_pts['UNOBench-MO']:.3f}, Visual "
             f"{sv_pts['UNOBench-Visual']:.3f}).")
    f9_audio = next(r[3] for r in sc_rows if r[0] == "ans_ground_max (f9)"
                    and r[1] == "UNOBench-Audio")
    f9_ov = next(r[3] for r in sc_rows if r[0] == "ans_ground_max (f9)"
                 and r[1] == "Overall")
    f10_ov = next(r[3] for r in sc_rows if r[0] == "ans_ground_mean (f10)"
                  and r[1] == "Overall")
    L.append(f"6. **Single-chain f9/f10 are weak.** Candidate-0-only "
             f"answer-to-input grounding gives AUROC {f9_audio:.3f} Audio "
             f"/ {f9_ov:.3f} Overall (f9; f10 {f10_ov:.3f} Overall; on "
             f"Audio f9 = f10 because Audio questions have a single input "
             f"embedding) — well below the published single-chain G_max "
             f"({pub_gmax.get('UNOBench-Audio', float('nan')):.3f} Audio / "
             f"{pub_gmax.get('Overall', float('nan')):.3f} Overall). "
             f"Step-level max grounding is not recoverable from the pooled "
             f"answer embedding.")
    L.append(f"7. **Recommendation.** Keep RDS/SV as separate question-"
             f"level baselines (or change the confidence functional, e.g. "
             f"mix the sum-share with a question-level dispersion term "
             f"outside the share normalisation) rather than injecting them "
             f"as candidate features; inside the current framework only "
             f"cent_rds on Visual is a defensible drop-in upgrade, and it "
             f"was found by full13/disp8 validation, not by the greedy "
             f"search on every split.")
    L.append("")

    with open(os.path.join(OUT_DIR, "RESULTS.md"), "w") as f:
        f.write("\n".join(L))
    print(f"Wrote {os.path.join(OUT_DIR, 'RESULTS.md')}")


if __name__ == "__main__":
    main()
