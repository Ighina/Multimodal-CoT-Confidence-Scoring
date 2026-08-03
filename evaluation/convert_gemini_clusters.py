#!/usr/bin/env python3
"""
Extract answer clustering info from gemini-cots-with-majority-vote.json and write
it in the same format as minicpm-majority-vote-clusters.json.

The gemini file already contains pre-computed semantic clusters:
  - "answer_group" per candidate          → goes into "groups"
  - "_internal_majority_candidate_ids"    → goes into "majority_ids"

The output is a dict keyed by question_id (from metadata.original_idx), where each
value is {"groups": [...], "majority_ids": [...]}.

Paper: builds the semantic-cluster cache (Appendix A, "Self-Consistency
Clustering") consumed by final_evaluate.py and all replay scripts.

Usage:
    python convert_gemini_clusters.py [input_json] [output_json]

Defaults:
    input  = gemini-cots-with-majority-vote.json
    output = gemini-majority-vote-clusters.json
"""

import json
import sys


def convert(input_path: str, output_path: str) -> None:
    with open(input_path, "r", encoding="utf-8") as f:
        gemini_data = json.load(f)

    clusters: dict[str, dict] = {}

    for question_idx, candidates in enumerate(gemini_data):
        if not candidates:
            continue

        # Use the question_id from the first candidate's metadata
        qid = str(candidates[0]["metadata"]["original_idx"])

        # "answer_group" per candidate → groups list
        groups = [c["answer_group"] for c in candidates]

        # "_internal_majority_candidate_ids" → majority_ids
        majority_ids = candidates[0].get("_internal_majority_candidate_ids", [])

        clusters[qid] = {
            "groups": groups,
            "majority_ids": majority_ids,
        }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(clusters, f, indent=2, ensure_ascii=False)

    n_empty = sum(1 for v in clusters.values() if not v["majority_ids"])
    print(
        f"Converted {len(clusters)} questions → {output_path}\n"
        f"  Questions with empty majority_ids: {n_empty}"
    )


if __name__ == "__main__":
    input_path = sys.argv[1] if len(sys.argv) > 1 else "gemini-cots-with-majority-vote.json"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "gemini-majority-vote-clusters.json"
    convert(input_path, output_path)
