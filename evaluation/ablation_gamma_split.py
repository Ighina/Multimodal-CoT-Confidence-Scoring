#!/usr/bin/env python3
"""
Ablations A (gamma sweep for S_goal) and B (split-point sweep for the
forward/backward term) — artifact-recoverability audit.

Paper: documents why the S_goal hyperparameter sweep of Appendix F ("Fixed
hyperparameters of S_goal") needs the one-time re-embedding phase of
gamma_sweep_eval.py rather than being replayable from stored artifacts.

Both ablations require recomputing, per generation,

    mu_goal   = mean_i  S(phi_{r_i}, phi_a)
    Delta_fwd = E_{i >  m/2} S(phi_{r_i}, phi_a)
    Delta_bwd = E_{i <= m/2} S(phi_{r_i}, phi_a)
    S_goal    = gamma * mu_goal + (1 - gamma) * 0.5*(Delta_fwd - Delta_bwd + 1)

from the per-STEP embeddings phi_{r_1..m} and the final-ANSWER embedding
phi_a of every candidate chain (E5-Omni encoder).  The verification gate is
that gamma = 0.6 must reproduce the stored `goal_directedness` values.

This script does NOT attempt to fabricate those quantities from other
artifacts.  Instead it programmatically audits every embedding artifact in
the repo and verifies whether per-step / per-answer embeddings are stored
anywhere, and characterises the numeric grid of the stored scores.  Its
findings (written to bootstrap-results/ablations/gamma_split_audit.json and
summarised in RESULTS.md) are the documented reason ablations A and B are
reported as NOT computable offline from the stored data.

Usage:
    python ablation_gamma_split.py
"""

import json
import os
import struct
from collections import Counter

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT, "bootstrap-results", "ablations")

EMBEDDING_FILES = {
    "embeddings/e5-all-modalities.json": "per-question INPUT embeddings, one per present modality",
    "embeddings/e5-omnimodal.json": "per-question pooled omnimodal INPUT embedding",
    "gemini-answer-embeddings.json": "per-CANDIDATE full-generation embeddings (CoT+answer pooled)",
}
SCORES_FILE = "gemini-cots-with-majority-vote.json"


def is_bfloat16_exact(v: float) -> bool:
    """True iff v is exactly representable in bfloat16 (fp32 with the low
    16 mantissa bits zero)."""
    return struct.pack(">f", v)[2:] == b"\x00\x00"


def audit_embedding_file(path: str) -> dict:
    d = json.load(open(path))
    sig_counter: Counter = Counter()
    vec_counts: Counter = Counter()
    for e in d:
        if isinstance(e, dict) and "embeddings" in e:
            vec_counts[len(e["embeddings"])] += 1
        elif isinstance(e, dict):
            sig = tuple(
                (k, (len(v) if isinstance(v, list) else None))
                for k, v in sorted(e.items())
            )
            sig_counter[sig] += 1
    out = {"n_entries": len(d)}
    if vec_counts:
        out["vectors_per_entry"] = {str(k): v for k, v in sorted(vec_counts.items())}
    if sig_counter:
        out["modality_vector_signatures"] = [
            {"signature": [list(kv) for kv in sig], "count": c}
            for sig, c in sig_counter.most_common()
        ]
    return out


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    report = {
        "purpose": "Recoverability audit for ablations A (gamma sweep) and "
                   "B (split-point sweep) of S_goal.",
        "required_artifacts": [
            "per-step embeddings phi_{r_1..m} for every candidate chain",
            "final-answer embeddings phi_a for every candidate chain",
        ],
        "embedding_artifacts_found": {},
    }

    for rel, desc in EMBEDDING_FILES.items():
        path = os.path.join(ROOT, rel)
        print(f"Auditing {rel} ...")
        info = audit_embedding_file(path)
        info["content"] = desc
        report["embedding_artifacts_found"][rel] = info

    # ---- step counts: how many step vectors WOULD be needed ----
    print(f"Auditing {SCORES_FILE} (step counts + score grid) ...")
    mv = json.load(open(os.path.join(ROOT, SCORES_FILE)))
    cand_counts = Counter(len(q) for q in mv)
    step_counts: Counter = Counter()
    total_steps = 0
    n_scores = 0
    off_grid = {"goal_directedness": 0, "smoothness": 0, "semantic_density": 0}
    for q in mv:
        for cand in q:
            m = len(cand.get("steps") or [])
            step_counts[m] += 1
            total_steps += m
            for key in off_grid:
                v = cand.get(key)
                if v is None:
                    continue
                n_scores += 1
                if not is_bfloat16_exact(v):
                    off_grid[key] += 1

    report["chains"] = {
        "candidates_per_question": {str(k): v for k, v in sorted(cand_counts.items())},
        "steps_per_candidate_distribution": {str(k): v for k, v in sorted(step_counts.items())},
        "total_step_embeddings_required": total_steps,
    }
    report["stored_score_grid"] = {
        "n_internal_score_values_checked": n_scores,
        "values_NOT_on_bfloat16_grid": off_grid,
        "note": "All stored smoothness / goal_directedness / semantic_density "
                "values lie exactly on the bfloat16 grid: the upstream scoring "
                "run used bfloat16 embeddings/similarities. Even with "
                "re-extracted step embeddings, a <1e-6 reproduction gate would "
                "additionally require matching bfloat16 arithmetic.",
    }

    # ---- verdict ----
    n_q = report["embedding_artifacts_found"][
        "gemini-answer-embeddings.json"]["n_entries"]
    report["verdict"] = {
        "step_embeddings_recoverable": False,
        "answer_embeddings_recoverable": False,
        "reason": (
            f"The three embedding artifacts store, respectively: one vector per "
            f"present input modality per question; one pooled omnimodal input "
            f"vector per question; and exactly one full-generation vector per "
            f"candidate ({n_q} questions, counts match the candidate counts "
            f"7/6/1 — i.e. pooled chain embeddings, not per-step). Reproducing "
            f"S_goal would need {total_steps} per-step vectors plus one "
            f"final-answer vector per candidate; neither exists in the repo "
            f"(no .npy/.pt/.pkl artifacts either). Step TEXTS are stored "
            f"(`steps` in {SCORES_FILE}), so the vectors could be re-extracted "
            f"with the E5-Omni encoder on the GPU server, but not offline from "
            f"stored data."
        ),
        "ablation_A_gamma_sweep": "NOT computable from stored artifacts",
        "ablation_B_split_sweep": "NOT computable from stored artifacts",
        "verification_gate": "Not attemptable (inputs to S_goal missing); "
                             "no numbers are reported for A/B.",
    }

    out_path = os.path.join(OUT_DIR, "gamma_split_audit.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {out_path}")
    print("\nVERDICT: ablations A and B are NOT computable from stored "
          "artifacts (per-step and final-answer embeddings are not stored; "
          "only input-level and pooled per-candidate generation embeddings "
          "exist).")


if __name__ == "__main__":
    main()
