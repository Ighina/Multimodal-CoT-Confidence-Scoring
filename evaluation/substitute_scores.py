"""
Substitute (or add) my scores in a file produced by add_my_scores.py.

Takes a data file in the format written by add_my_scores.py (e.g.
minicpm_cots_with_correctness_and_my_scores.json) and a scores file in the
same format as minicpmo_scores.json, and overwrites the "internal_<metric>"
and "cross_modal_<metric>" keys of every generation with the new values,
adding the keys if they are not present yet.

Usage:
    python substitute_scores.py \
        --scores-json minicpmo_scores.json \
        --data-json minicpm_cots_with_correctness_and_my_scores.json \
        --output-json minicpm_cots_with_correctness_and_my_scores_v2.json
"""

import argparse
import json


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scores-json", required=True,
                   help="New scores file (same format as minicpmo_scores.json).")
    p.add_argument("--data-json", required=True,
                   help="Data file in the format output by add_my_scores.py.")
    p.add_argument("--output-json", default=None,
                   help="Where to write the result (default: overwrite --data-json).")
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.scores_json, "r") as f:
        my_scores = json.load(f)

    with open(args.data_json, "r") as f:
        data = json.load(f)

    assert len(my_scores) == len(data), (
        f"Question count mismatch: {len(my_scores)} in scores file "
        f"vs {len(data)} in data file."
    )

    added, substituted = 0, 0
    for idx, question in enumerate(data):
        # Align by question_id (falls back to position): the scores file is
        # ordered by qid, while the data file can have locally reordered or
        # truncated records (e.g. qids 2517-2521 in gemini_cots_reformatted).
        qid = question.get("question_id", idx)
        my_score_ = my_scores[qid]
        gen_confs = question["generations_confidence"]
        if len(my_score_) != len(gen_confs):
            print(f"WARNING: qid {qid}: {len(my_score_)} score entries but "
                  f"{len(gen_confs)} generations; merging the first "
                  f"{min(len(my_score_), len(gen_confs))} only.")
        for idx2, my_score in enumerate(my_score_[: len(gen_confs)]):
            gen_conf = gen_confs[idx2]
            for metric, value in my_score["internal"].items():
                key = "internal" + "_" + metric
                if key in gen_conf:
                    substituted += 1
                else:
                    added += 1
                gen_conf[key] = value
            for metric, value in my_score["cross_modal"].items():
                if metric != "per_step_coherence":
                    key = "cross_modal" + "_" + metric
                    if key in gen_conf:
                        substituted += 1
                    else:
                        added += 1
                    gen_conf[key] = value

    output_json = args.output_json or args.data_json
    with open(output_json, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Substituted {substituted} values, added {added} new keys -> {output_json}")


if __name__ == "__main__":
    main()
