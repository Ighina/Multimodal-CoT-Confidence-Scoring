#!/usr/bin/env python3
"""
Add cluster-derived black-box baselines to a generation-confidence JSON:

  * confidence_score_semantic_entropy : exp(-H) where H is the discrete semantic
    entropy over the cached answer clusters (Farquhar et al., 2024, discrete variant).
  * confidence_score_num_semsets      : normalized inverse of the number of semantic
    clusters (Lin et al.-style "number of semantic sets" baseline).

Both scores are constant within a question (question-level baselines) and are
written to every candidate record so final_evaluate.py picks them up like any
other confidence_score_* baseline.

Paper: Semantic Entropy and Num. Semantic Sets baselines of Appendix M
(Table 21).

Usage:
    python add_semantic_baselines.py \
        --input-json gemini-cots-with-umpire-and-our-scores.json \
        --clusters gemini-majority-vote-clusters.json \
        --output-json gemini-cots-with-additional-baselines.json
"""

import argparse
import json
import math
from collections import Counter


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-json", required=True)
    p.add_argument("--clusters", required=True)
    p.add_argument("--output-json", required=True)
    args = p.parse_args()

    with open(args.input_json, encoding="utf-8") as f:
        data = json.load(f)
    with open(args.clusters, encoding="utf-8") as f:
        clusters = json.load(f)

    missing = 0
    for rec in data:
        entry = clusters.get(str(rec.get("question_id")))
        if entry is None:
            missing += 1
            se_conf = num_conf = None
        else:
            groups = entry["groups"]
            n = len(groups)
            counts = Counter(groups)
            probs = [c / n for c in counts.values()]
            entropy = -sum(pr * math.log(pr) for pr in probs)
            se_conf = math.exp(-entropy)
            num_sets = len(counts)
            num_conf = 1.0 if n <= 1 else 1.0 - (num_sets - 1) / (n - 1)
        for cand in rec.get("generations_confidence", []) or []:
            cand["confidence_score_semantic_entropy"] = se_conf
            cand["confidence_score_num_semsets"] = num_conf

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(data, f)
    print(f"Wrote {args.output_json} ({len(data)} questions, {missing} without clusters)")


if __name__ == "__main__":
    main()
