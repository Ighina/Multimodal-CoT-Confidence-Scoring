#!/usr/bin/env python3
"""
Merge the chain-level embedding scores produced by the scoring pipeline
(``experiments/run_experiments_temp.py``) into a CoT records file, so that
``final_evaluate.py`` can read them as per-candidate confidence features.

For every question and candidate generation, the ``internal_*`` (S_smooth,
S_dens, S_goal, ...) and ``cross_modal_*`` (G_avg, G_max, ...) scores are
written into the record's ``generations_confidence`` entry. The UMPIRE
length-normalised score stored on the record is converted to a confidence
(1 - UMPIRE_len-norm, NaN mapped to 0) and stored as ``umpire_normal``,
matching the baseline definition in Appendix B of the paper.

Usage:
    python add_my_scores.py \
        --scores-json minicpmo_scores.json \
        --data-json minicpm_cots_with_correctness.json \
        --output-json minicpm_cots_with_correctness_and_my_scores.json
"""

import argparse
import json

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Merge embedding-based chain scores into a CoT records file."
    )
    p.add_argument(
        "--scores-json",
        default="minicpmo_scores.json",
        help="Per-question list of per-candidate score dicts with 'internal' and "
        "'cross_modal' sub-dicts (default: %(default)s)",
    )
    p.add_argument(
        "--data-json",
        default="minicpm_cots_with_correctness.json",
        help="CoT records file with correctness labels and a "
        "'generations_confidence' list per question (default: %(default)s)",
    )
    p.add_argument(
        "--output-json",
        default="minicpm_cots_with_correctness_and_my_scores.json",
        help="Where to write the merged records (default: %(default)s)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.scores_json, "r") as f:
        my_scores = json.load(f)

    with open(args.data_json, "r") as f:
        records = json.load(f)

    for idx, question_scores in enumerate(my_scores):
        for idx2, candidate_scores in enumerate(question_scores):
            for metric, value in candidate_scores["internal"].items():
                records[idx]["generations_confidence"][idx2][
                    "internal_" + metric
                ] = value
            for metric, value in candidate_scores["cross_modal"].items():
                if metric != "per_step_coherence":
                    records[idx]["generations_confidence"][idx2][
                        "cross_modal_" + metric
                    ] = value
            if np.isnan(records[idx]["umpire_length_normalized"]):
                records[idx]["generations_confidence"][idx2]["umpire_normal"] = 0
            else:
                records[idx]["generations_confidence"][idx2]["umpire_normal"] = (
                    1 - records[idx]["umpire_length_normalized"]
                )

    with open(args.output_json, "w") as f:
        json.dump(records, f, indent=4)


if __name__ == "__main__":
    main()
