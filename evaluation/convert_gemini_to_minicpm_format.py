#!/usr/bin/env python3
"""
Convert gemini-cots-with-umpire-and-our-scores.json to the same format as
minicpm_cots_with_correctness_and_my_scores.json.

Usage:
    python convert_gemini_to_minicpm_format.py <output_filename>

The gemini file is a list-of-lists: data[question_idx][candidate_idx] where each
candidate dict has scores at the top level and metadata.correct for correctness.

The target format is a list of dicts (one per question) with a
"generations_confidence" key holding a list of per-candidate score dicts.
"""

import json
import sys

# ── Paths ──────────────────────────────────────────────────────────────────
INPUT_FILE = "gemini-cots-with-umpire-and-our-scores.json"

# Score keys to extract from each candidate (top-level keys in gemini format).
# These are the gemini-side key names; they get renamed to the minicpm convention
# via KEY_RENAME_MAP below before being written into generations_confidence.
SCORE_KEYS = [
    "confidence_score_level_based",
    "confidence_score_selfprobing",
    "confidence_score_verb_2s_cot",
    "grounding",
    "semantic_density",
    "goal_directedness",
    "smoothness",
    "mean_grounding_coherence",
    "internal_overall",
    "coherence",
    "mean_grounding_coherence_0307",
    "mean_grounding_coherence_0703",
    "grounding_attention",
    "grounding_max",
    "mean_grounding_attention_coherence_0703",
    "mean_grounding_attention_coherence_0307",
    "umpire_score",
]

# Map gemini key names → minicpm naming convention so the evaluation script's
# default BASELINES and AGG_FEATURES (which look for internal_* and cross_modal_*
# prefixes) can find them.
KEY_RENAME_MAP = {
    "smoothness": "internal_smoothness",
    "goal_directedness": "internal_goal_directedness",
    "semantic_density": "internal_semantic_density",
    "umpire_score": "umpire_normal",
    "coherence": "cross_modal_coherence",
    "grounding": "cross_modal_grounding",
    "grounding_attention": "cross_modal_grounding_attention",
    "grounding_max": "cross_modal_grounding_max",
    "mean_grounding_coherence": "cross_modal_mean_grounding_coherence",
    "mean_grounding_coherence_0307": "cross_modal_mean_grounding_coherence_0307",
    "mean_grounding_coherence_0703": "cross_modal_mean_grounding_coherence_0703",
    "mean_grounding_attention_coherence_0307": "cross_modal_mean_grounding_attention_coherence_0307",
    "mean_grounding_attention_coherence_0703": "cross_modal_mean_grounding_attention_coherence_0703",
}


def build_media_info(metadata: dict) -> dict:
    """Build a media_info dict from the candidate metadata."""
    info: dict[str, dict] = {"images": {}, "audios": {}, "videos": {}}
    # We only have counts, not paths, so store counts as placeholders.
    for i in range(metadata.get("num_images", 0)):
        info["images"][f"<image_{i+1}>"] = None
    for i in range(metadata.get("num_audios", 0)):
        info["audios"][f"<audio_{i+1}>"] = None
    for i in range(metadata.get("num_videos", 0)):
        info["videos"][f"<video_{i+1}>"] = None
    return info


def convert(input_path: str, output_path: str) -> None:
    with open(input_path, "r") as f:
        gemini_data = json.load(f)

    output: list[dict] = []

    for question_idx, candidates in enumerate(gemini_data):
        # Use the first candidate's metadata for question-level info
        first_meta = candidates[0]["metadata"]

        question_dict: dict = {}

        # ── Question-level fields ──────────────────────────────────────────
        question_dict["question_id"] = first_meta.get("original_idx", question_idx)
        question_dict["question_text"] = first_meta.get("question", "")
        question_dict["image"] = ""  # not available in gemini metadata
        question_dict["media_info"] = build_media_info(first_meta)

        # ── Generations text ────────────────────────────────────────────────
        question_dict["generations_text"] = [
            cand.get("text", "") for cand in candidates
        ]
        question_dict["most_likely_generation_text"] = candidates[0].get("text", "")

        # ── ROUGE / evaluation fields (not in gemini — set to None) ─────────
        question_dict["rouge1_reference_answers"] = None
        question_dict["rouge1_to_target"] = None
        question_dict["rouge2_reference_answers"] = None
        question_dict["rouge2_to_target"] = None
        question_dict["rougeL_reference_answers"] = None
        question_dict["rougeL_to_target"] = None
        question_dict["exact_match"] = None
        # uno_score = 1 if the most-likely (first) candidate is correct
        question_dict["uno_score"] = 1 if candidates[0]["metadata"].get("correct") else 0
        question_dict["answers"] = None
        question_dict["uno_score_threshold_0.4"] = None
        question_dict["uno_score_threshold_0.5"] = None
        question_dict["logdet"] = None
        question_dict["umpire_length_normalized"] = None
        question_dict["umpire_unnormalized"] = None

        # ── Most-likely confidence (from first candidate) ───────────────────
        question_dict["most_likely_confidence"] = {
            "confidence_score_level_based": candidates[0].get(
                "confidence_score_level_based"
            ),
            "confidence_score_selfprobing": candidates[0].get(
                "confidence_score_selfprobing"
            ),
            "confidence_score_verb_2s_cot": candidates[0].get(
                "confidence_score_verb_2s_cot"
            ),
        }

        # ── Generations confidence (per-candidate score dicts) ──────────────
        question_dict["generations_confidence"] = []
        for cand in candidates:
            conf: dict = {}
            # Copy all score keys, renaming to minicpm convention
            for key in SCORE_KEYS:
                renamed = KEY_RENAME_MAP.get(key, key)
                conf[renamed] = cand.get(key)
            question_dict["generations_confidence"].append(conf)

        # ── Generations UNO scores ──────────────────────────────────────────
        question_dict["generations_uno_score"] = [
            1 if cand["metadata"].get("correct") else 0 for cand in candidates
        ]
        question_dict["generations_uno_score_threshold_0.4"] = [
            1 if cand["metadata"].get("correct") else 0 for cand in candidates
        ]
        question_dict["generations_uno_score_threshold_0.5"] = [
            1 if cand["metadata"].get("correct") else 0 for cand in candidates
        ]

        output.append(question_dict)

    with open(output_path, "w") as f:
        json.dump(output, f, indent=4)

    print(f"Converted {len(output)} questions → {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <output_filename>")
        sys.exit(1)

    convert(INPUT_FILE, sys.argv[1])
