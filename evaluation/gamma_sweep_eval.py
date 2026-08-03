#!/usr/bin/env python3
r"""Sweep the gamma weight of the S_goal (answer-convergence) score.

    S_goal = gamma * mu_goal + (1 - gamma) * Delta_goal
    mu_goal    = mean_i  S(phi_{r_i}, phi_a)                       (avg similarity)
    Delta_goal = (mean_{i > m/2} sim - mean_{i <= m/2} sim + 1)/2  (progression)

Both mu_goal and Delta_goal are GAMMA-INDEPENDENT functions of the per-step
similarities to the answer, so once those similarities are known the whole gamma
sweep (and the m/2 split-point sweep) is a free re-combination -- no re-embedding
and no refitting per gamma.

THE CATCH: only the final S_goal at gamma=0.6 was ever persisted
(`internal_goal_directedness`); mu_goal / Delta_goal and the per-step embeddings
were computed in memory during the original run and discarded. So the sweep needs
the per-step->answer similarities to be recomputed ONCE by re-embedding the CoT
steps and answers with the same text encoder used for the internal scores. That
is Phase 1 below and REQUIRES A GPU + the same encoder used by the scoring
pipeline (src/embeddings). Phase 2 (the actual sweep + evaluation) is pure
offline re-combination and is already validated against the stored numbers.

Paper: the S_goal hyperparameter sensitivity check referenced in Appendix F
("Fixed hyperparameters of S_goal" — deferred to the code release).

Usage:
  # Phase 1 (GPU, one-time): re-embed and cache per-candidate sims-to-answer.
  python gamma_sweep_eval.py --embed \
      --steps-json gemini-cots-with-majority-vote.json \
      --encoder <e5-omni-model-or-path> \
      --out bootstrap-results/gamma_sweep/sims_to_goal.json

  # Phase 2 (offline): sweep gamma (and optionally the split point) and evaluate.
  python gamma_sweep_eval.py --sweep \
      --sims bootstrap-results/gamma_sweep/sims_to_goal.json

  # Self-test the Phase-2 plumbing with NO cache: feeds the stored gamma=0.6
  # S_goal through the evaluator and checks it reproduces the published AUROC.
  python gamma_sweep_eval.py --selftest
"""
import argparse
import json
import os

import numpy as np

import bootstrap_significance as bs

GAMMA_GRID = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
SPLITS = ["Overall", "UNOBench-Audio", "UNOBench-MC", "UNOBench-MO",
          "UNOBench-Visual"]


# --------------------------------------------------------------------------- #
# S_goal reconstruction from the per-step similarities to the answer
# --------------------------------------------------------------------------- #
def sgoal_from_sims(sims, gamma, split="half"):
    """Reconstruct S_goal for one chain from its list of step->answer sims.

    Mirrors InternalCoherenceMetric.compute_goal_directedness 1:1:
      - fewer than 2 steps  -> 1.0  (the metric's short-chain guard)
      - mu_goal    = mean(sims)
      - split at floor(m/2) (or m/3, 2m/3 when sweeping the split point)
      - Delta_goal = (second_half_mean - first_half_mean + 1)/2
      - S_goal     = gamma*mu_goal + (1-gamma)*Delta_goal
    """
    s = np.asarray(sims, dtype=float)
    m = len(s)
    if m < 2:
        return 1.0
    if split == "half":
        c = m // 2
    elif split == "third":
        c = m // 3
    elif split == "twothird":
        c = (2 * m) // 3
    else:
        raise ValueError(split)
    c = min(max(c, 1), m - 1)  # keep both halves non-empty
    mu = float(s.mean())
    delta = (float(s[c:].mean()) - float(s[:c].mean()) + 1.0) / 2.0
    return gamma * mu + (1.0 - gamma) * delta


# --------------------------------------------------------------------------- #
# Phase 2 evaluation: standalone-score no_majority AUROC/AURAC per split
# (reuses bootstrap_significance's collector -- validated identical to the
#  published pipeline in --selftest)
# --------------------------------------------------------------------------- #
def load_pipeline(run_dir, input_json, cluster_cache, jsonl_path):
    meta = json.load(open(os.path.join(run_dir, "no_majority", "metadata.json")))
    records = json.load(open(input_json))
    cluster_info = json.load(open(cluster_cache))
    id2cat = {}
    with open(jsonl_path) as f:
        for line in f:
            if line.strip():
                it = json.loads(line)
                id2cat[it["question_id"]] = it.get("category", "Unknown")
    return meta, records, cluster_info, id2cat


def eval_score_key(records, cluster_info, meta, id2cat, key):
    """Per-split AUROC/AURAC for a per-candidate confidence field `key`,
    in the no_majority setting (majority answer + sum-share confidence)."""
    features = meta["features"]
    bw = meta["best_weights"]
    fb = bs.make_scorer(bw["fallback"], features)
    sc = {c: bs.make_scorer(e, features) or fb
          for c, e in bw["per_category"].items()}
    p = meta["parameters"]
    test, _ = bs.stratified_test_val_split(records, id2cat, p["val_fraction"],
                                           p["split_seed"])
    _, items = bs.collect_per_item_nomaj(test, cluster_info, meta, sc, fb, id2cat)
    d = items[key]
    cats = np.array(d["category"])
    y = np.array(d["y_true"])
    pr = bs.minmax_transform(np.array(d["y_prob"]))
    out = {}
    for split in SPLITS:
        mask = np.ones(len(y), bool) if split == "Overall" else cats == split
        out[split] = (bs.fast_auroc(y[mask], pr[mask]),
                      bs.fast_aurac(y[mask], pr[mask]))
    return out


def inject_sgoal(records, sims_by_qc, gamma, split, key):
    """Write reconstructed S_goal(gamma) into each candidate's
    generations_confidence[i][key]. Candidate order matches the stored
    per-question generations_confidence list (verified: counts align)."""
    for rec in records:
        qid = rec.get("question_id")
        gc = rec.get("generations_confidence") or []
        for i in range(len(gc)):
            sims = sims_by_qc.get(f"{qid}:{i}")
            if sims is None:
                gc[i][key] = 0.0
            else:
                gc[i][key] = sgoal_from_sims(sims, gamma, split)


# --------------------------------------------------------------------------- #
# Phase 1: re-embed steps + answer, cache per-candidate sims-to-answer
# --------------------------------------------------------------------------- #
def phase1_embed(args):
    """GPU-ONLY. Re-embed CoT steps and answers with the internal-coherence text
    encoder and cache the per-step cosine similarities to the answer embedding.

    The encoder must be the SAME one that produced the stored
    `internal_goal_directedness` (E5-Omni text branch in the paper); the gate
    below catches a wrong choice by failing to reproduce the stored value.
    """
    import torch  # noqa: F401  (only needed here; keeps Phase 2 torch-free)

    steps_data = json.load(open(args.steps_json))
    encoder = _load_encoder(args.encoder)  # see helper; raises with guidance
    S = _make_similarity()  # (1 + cos)/2, matching methodology Eq.

    sims_by_qc, gate_pred, gate_true = {}, [], []
    for q in steps_data:
        cands = q if isinstance(q, list) else [q]
        for i, g in enumerate(cands):
            qid = g.get("id", None)
            steps = g.get("steps") or []
            ans = g.get("final_answer") or ""
            if len(steps) < 2 or not ans:
                continue
            step_emb = encoder.encode_text(steps)        # (m, d)
            ans_emb = encoder.encode_text([ans])[0]      # (d,)
            sims = [float(S(e, ans_emb)) for e in step_emb]
            key = f"{q_qid(q, i)}:{i}"
            sims_by_qc[key] = sims
            # gate: reconstruct gamma=0.6 / half-split, compare to stored value
            if g.get("goal_directedness") is not None:
                gate_pred.append(sgoal_from_sims(sims, 0.6, "half"))
                gate_true.append(float(g["goal_directedness"]))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(sims_by_qc, open(args.out, "w"))
    if gate_true:
        diff = np.max(np.abs(np.array(gate_pred) - np.array(gate_true)))
        print(f"[gate] max |reconstructed(gamma=0.6) - stored goal_directedness| "
              f"= {diff:.2e} over {len(gate_true)} chains")
        if diff > 1e-3:
            print("[gate] WARNING: >1e-3 -- wrong encoder or similarity/"
                  "split convention. Fix before trusting the sweep.")
    print(f"[phase1] cached {len(sims_by_qc)} chains -> {args.out}")


def q_qid(q, i):
    g = q[i] if isinstance(q, list) else q
    return g.get("id", g.get("question_id"))


def _load_encoder(name):
    raise SystemExit(
        "Phase 1 needs the text encoder from "
        "../Multimodal-CoT-Confidence-Scoring (src/embeddings). Wire "
        "OmnimodalEncoder / TextEncoder here with the E5-Omni checkpoint used "
        "for the internal scores, exposing encode_text(list[str]) -> (n, d) "
        "tensor. Run on a GPU box; this stub is intentional so Phase 2 stays "
        "dependency-free.")


def _make_similarity():
    def S(a, b):
        import numpy as _np
        a = _np.asarray(a, float); b = _np.asarray(b, float)
        cos = float(a @ b) / (float(_np.linalg.norm(a) * _np.linalg.norm(b)) + 1e-12)
        return (1.0 + cos) / 2.0
    return S


# --------------------------------------------------------------------------- #
def phase2_sweep(args):
    meta, records, cluster_info, id2cat = load_pipeline(
        args.run_dir, args.input_json, args.cluster_cache, args.jsonl_path)
    sims_by_qc = json.load(open(args.sims))
    key = "__sgoal_swept__"
    if key not in meta["parameters"]["baselines"]:
        meta["parameters"]["baselines"].append(key)  # let the collector emit it

    rows = []
    splits = [("half", GAMMA_GRID)]
    if args.sweep_split:
        splits = [("third", [0.6]), ("half", GAMMA_GRID), ("twothird", [0.6])]
    print(f"{'split':9s} {'gamma':>5s} " +
          " ".join(f"{s.replace('UNOBench-',''):>16s}" for s in SPLITS))
    for split, grid in splits:
        for gamma in grid:
            inject_sgoal(records, sims_by_qc, gamma, split, key)
            res = eval_score_key(records, cluster_info, meta, id2cat, key)
            rows.append({"split_point": split, "gamma": gamma,
                         **{f"{s}_AUROC": res[s][0] for s in SPLITS},
                         **{f"{s}_AURAC": res[s][1] for s in SPLITS}})
            print(f"{split:9s} {gamma:5.1f} " +
                  " ".join(f"{res[s][0]:.3f}/{res[s][1]:.3f}" for s in SPLITS))
    _write_csv(args.out_csv, rows)
    print(f"\n[phase2] written -> {args.out_csv}")


def _selftest(args):
    """Prove the Phase-2 evaluator on stored data: feeding the gamma=0.6 S_goal
    (`internal_goal_directedness`) must reproduce the published per-split AUROC."""
    meta, records, cluster_info, id2cat = load_pipeline(
        args.run_dir, args.input_json, args.cluster_cache, args.jsonl_path)
    res = eval_score_key(records, cluster_info, meta, id2cat,
                         "internal_goal_directedness")
    exp = {"Overall": 0.5960, "UNOBench-Audio": 0.4474, "UNOBench-MC": 0.5491,
           "UNOBench-MO": 0.6444, "UNOBench-Visual": 0.7300}
    ok = True
    for s in SPLITS:
        got = res[s][0]
        mark = "" if abs(got - exp[s]) < 5e-4 else "  <-- MISMATCH"
        if mark:
            ok = False
        print(f"  {s:18s} AUROC {got:.4f} (expected {exp[s]:.4f}){mark}")
    print("[selftest] PASS" if ok else "[selftest] FAIL")


def _write_csv(path, rows):
    import csv
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--embed", action="store_true", help="Phase 1 (GPU)")
    ap.add_argument("--sweep", action="store_true", help="Phase 2 (offline)")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--sweep-split", action="store_true",
                    help="also sweep the m/2 split point (third, half, 2/3)")
    ap.add_argument("--steps-json", default="gemini-cots-with-majority-vote.json")
    ap.add_argument("--encoder", default=None)
    ap.add_argument("--sims", default="bootstrap-results/gamma_sweep/sims_to_goal.json")
    ap.add_argument("--out", default="bootstrap-results/gamma_sweep/sims_to_goal.json")
    ap.add_argument("--out-csv", default="bootstrap-results/gamma_sweep/gamma_sweep.csv")
    ap.add_argument("--run-dir", default="gemini-majority-multirun-logistic-reg/run_0")
    ap.add_argument("--input-json", default="gemini_cots_reformatted.json")
    ap.add_argument("--cluster-cache", default="gemini-majority-vote-clusters_top6.json")
    ap.add_argument("--jsonl-path", default="unobench_processed.jsonl")
    args = ap.parse_args()

    if args.embed:
        phase1_embed(args)
    elif args.sweep:
        phase2_sweep(args)
    elif args.selftest:
        _selftest(args)
    else:
        ap.error("choose --embed (GPU), --sweep (offline), or --selftest")


if __name__ == "__main__":
    main()
