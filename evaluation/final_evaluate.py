#!/usr/bin/env python3
"""
Main evaluation protocol of the paper: scores every confidence method on a
generation-confidence JSON dataset (Gemini- or MiniCPM-generated UNO-Bench
answers) and reports AUROC / ECE / AURAC per split.

Phase 1 groups the sampled answers of each question into semantic clusters
(via a local LLM through vLLM, or a pre-computed cluster cache; Appendix A,
Figure 3). Phase 2 selects the count-majority answer, attaches each method's
confidence, fits the C_chain logistic-regression aggregator on the validation
items of the split (Section 3.4), and evaluates on the held-out test items:

  * no_majority setting       — every method GRADES the same majority-selected
                                answer (main protocol; Tables 1, 2 and 9).
  * majority_weighted setting — every method's score also re-WEIGHTS the votes
                                (geometry-weighted self-consistency,
                                Section 3.5; Appendices J-K, Tables 15-20).

One invocation is one evaluation run for a single train/validation/test
partition seed; the published numbers are means over 100 seeds, driven by
multi_seed_evaluate.py. Run `python final_evaluate.py --help` for all options.
"""

import csv
import os
import re
import json
import random
import argparse
from collections import defaultdict, Counter

import numpy as np
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from tqdm import tqdm


# ----------------------------------------------------------------------------------
# CLI argument parsing
# ----------------------------------------------------------------------------------
def parse_list_arg(value: str | None) -> list[str] | None:
    """Parse a comma-separated string into a list of trimmed strings, or return None."""
    if value is None:
        return None
    return [s.strip() for s in value.split(",") if s.strip()]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate confidence baselines, aggregated_optimal, and "
        "majority-voting methods on a generation-confidence JSON dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with all defaults (same as the previous hardcoded version):
  python final_evaluate.py

  # Override paths:
  python final_evaluate.py --input-json my_data.json --output-eval results.json

  # Use majority-vote answer selection with a custom seed:
  python final_evaluate.py --answer-selection-mode majority_vote --split-seed 123

  # Limit to 8 candidates, exclude unanimous questions:
  python final_evaluate.py --max-candidates 8 --exclude-unanimous
""",
    )

    # ── I/O paths ───────────────────────────────────────────────────────────
    p.add_argument(
        "--input-json",
        default="minicpm_cots_with_correctness_and_my_scores.json",
        help="Path to the JSON file with generation confidence scores "
        "(default: %(default)s)",
    )
    p.add_argument(
        "--cluster-cache-base",
        default="minicpm-majority-vote-clusters.json",
        help="Base filename for the clustering cache (default: %(default)s)",
    )
    p.add_argument(
        "--jsonl-path",
        default="unobench_processed.jsonl",
        help="Path to the JSONL file with question metadata / categories "
        "(default: %(default)s)",
    )
    p.add_argument(
        "--output-eval",
        default="evaluation_results.json",
        help="Path for the baseline + aggregated_optimal evaluation output "
        "(default: %(default)s)",
    )
    p.add_argument(
        "--output-eval-majority",
        default="evaluation_majority_voting_results.json",
        help="Path for the majority-voting evaluation output " "(default: %(default)s)",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        help="Directory for the structured CSV + metadata export. Two subfolders "
        "will be created inside it: 'no_majority/' and 'majority_weighted/', "
        "each containing per-split CSV files and a metadata.json.",
    )

    # ── Model ────────────────────────────────────────────────────────────────
    p.add_argument(
        "--model-id",
        default="Qwen/Qwen3.5-9B",
        help="HuggingFace model ID for the vLLM clustering engine "
        "(default: %(default)s)",
    )

    # ── Candidate / question filtering ───────────────────────────────────────
    p.add_argument(
        "--max-candidates",
        type=int,
        default=None,
        help="Cap the number of candidate generations used per question "
        "(None = use all; default: %(default)s)",
    )
    p.add_argument(
        "--exclude-unanimous",
        action="store_true",
        default=False,
        help="Drop questions where every sampled generation belongs to the same "
        "semantic group (majority vote share = 1.0)",
    )

    # ── Baseline lists ──────────────────────────────────────────────────────
    default_baselines = [
        "confidence_score_level_based",
        "confidence_score_selfprobing",
        "confidence_score_verb_2s_cot",
        # "internal_overall",
        "internal_smoothness",
        "internal_goal_directedness",
        "internal_semantic_density",
        # "cross_modal_entropy_weighted_alignment",
        "cross_modal_overall",
        # "cross_modal_coherence",
        "cross_modal_max_step_coherence",
        # "cross_modal_grounding_max",
        # "cross_modal_grounding_attention",
        # "cross_modal_geometric_mean",
        # "cross_modal_entropy_gated_routing",
        "umpire_normal",
    ]
    p.add_argument(
        "--baselines",
        type=str,
        default=None,
        help="Comma-separated list of baseline confidence methods. "
        "Default: internal_*, cross_modal_*, umpire_normal, and three "
        "confidence_score_* methods.",
    )
    default_majority_weighted = [
        # "internal_overall",
        "internal_smoothness",
        "internal_goal_directedness",
        "internal_semantic_density",
        # "cross_modal_entropy_weighted_alignment",
        "cross_modal_overall",
        # "cross_modal_coherence",
        # "cross_modal_max_step_coherence",
        "cross_modal_geometric_mean",
        # "cross_modal_grounding_max",
        # "cross_modal_grounding_attention",
        # "cross_modal_entropy_gated_routing",
    ]
    p.add_argument(
        "--majority-weighted-baselines",
        type=str,
        default=None,
        help="Comma-separated list of baselines that receive a genuine weighted "
        "majority-vote computation (others get a replica of the plain baseline "
        "score). Default: all internal_* and cross_modal_* baselines.",
    )

    # ── Split settings ───────────────────────────────────────────────────────
    p.add_argument(
        "--val-fraction",
        type=float,
        default=0.10,
        help="Fraction of data held out for validation / weight tuning "
        "(default: %(default)s)",
    )
    p.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Random seed for the stratified train/val split (default: %(default)s)",
    )

    # ── Answer selection ─────────────────────────────────────────────────────
    p.add_argument(
        "--answer-selection-mode",
        choices=["majority_vote", "random"],
        default="majority_vote",
        help="How the winning answer is picked for baseline / aggregated_optimal "
        "evaluation. 'majority_vote': raw-frequency majority group with "
        "tie-breaking by the method's own score. 'random': pick one candidate "
        "uniformly at random per question (default: %(default)s)",
    )
    p.add_argument(
        "--random-answer-seed",
        type=int,
        default=123,
        help="Random seed used when --answer-selection-mode=random "
        "(default: %(default)s)",
    )

    # ── Weight search ────────────────────────────────────────────────────────
    p.add_argument(
        "--weight-search-method",
        choices=["grid", "linear_regression", "logistic_regression"],
        default="grid",
        help="How to find the optimal feature weights for aggregated_optimal. "
        "'grid': exhaustive search over the probability simplex (default). "
        "'linear_regression': OLS with no intercept, fit per-sample then "
        "L1-normalised to the simplex. "
        "'logistic_regression': StandardScaler + LogisticRegression fitted on "
        "per-sample features predicting correctness; predict_proba is used "
        "as the per-sample score.",
    )
    p.add_argument(
        "--grid-step",
        type=float,
        default=0.1,
        help="Grid step size for the weight search over the probability simplex. "
        "0.1 -> ~19k candidates for 8 features; 0.2 -> ~792. "
        "Ignored when --weight-search-method=linear_regression. "
        "(default: %(default)s)",
    )
    p.add_argument(
        "--add-self-probing",
        action="store_true",
        default=False,
        help="Include confidence_score_selfprobing as an additional feature in "
        "the AGG_FEATURES set used by aggregated_optimal (on top of the "
        "internal_* and cross_modal_* baselines).",
    )
    p.add_argument(
        "--val-metric",
        default="AUROC",
        choices=["AUROC", "ECE", "Pearson", "AURAC", "TPR@10%FPR"],
        help="Metric used to select the best weight vector on the validation split "
        "(default: %(default)s)",
    )

    # ── Logistic-regression regularization search ──────────────────────────
    p.add_argument(
        "--lr-penalty",
        choices=["l1", "l2"],
        default="l1",
        help="Penalty used for --weight-search-method=logistic_regression. "
        "'l1' encourages sparse coefficients so correlated internal_*/"
        "cross_modal_* features don't dilute the strongest one "
        "(default: %(default)s)",
    )
    p.add_argument(
        "--lr-c-grid",
        type=str,
        default="0.01,0.03,0.1,0.3,1.0,3.0,10.0",
        help="Comma-separated list of inverse-regularization strengths (C) to "
        "try for logistic regression. The best C (by --val-metric, evaluated "
        "through the same group-aggregation pipeline used everywhere else) is "
        "kept. (default: %(default)s)",
    )

    # ── Uncertainty baselines ──────────────────────────────────────────────────
    p.add_argument(
        "--uncertainty-baselines",
        type=str,
        default=None,
        help="Comma-separated list of baselines where HIGHER score = MORE uncertain. "
        "For these, majority-vote tie-breaking picks the MINIMUM score instead of "
        "maximum, and their probabilities are flipped (1 - prob) before evaluation. "
        "Examples: umpire_normal, umpire_score. (default: none)",
    )

    # ── Metric-computation compatibility with eval_script_alt.py ────────────
    p.add_argument(
        "--confidence-mode",
        choices=["sum_share", "mean_share"],
        default="sum_share",
        help="How the reported per-question confidence is derived from the "
        "per-sample scores of the winning semantic group. "
        "'sum_share' = S_g / S_total where S_g is the SUM of the scores in the "
        "group (this is what eval_script_alt.py does with NORM_MODE='share'; "
        "because per-sample scores sit in a narrow band, the group SIZE "
        "dominates, so the confidence largely encodes the majority-vote "
        "frequency). 'mean_share' = group MEAN normalised across groups "
        "(the previous behaviour of this script, group-size-neutral). "
        "(default: %(default)s)",
    )
    p.add_argument(
        "--prob-transform",
        choices=["minmax", "clip"],
        default="minmax",
        help="Transform applied to each method's collected y_prob vector right "
        "before computing the metrics. 'minmax' = per-method min-max rescaling "
        "to [0,1] (rank-preserving; what eval_script_alt.py does), with "
        "uncertainty baselines flipped AFTER rescaling as 1-p. "
        "'clip' = np.clip(p, 0, 1) (the previous behaviour; destroys ranking "
        "for scores outside [0,1] and zeroes out negated uncertainty scores). "
        "(default: %(default)s)",
    )
    p.add_argument(
        "--no-question-level-autodetect",
        action="store_true",
        default=False,
        help="Disable the eval_script_alt.py-style auto-detection of baselines "
        "whose score is constant within every question (e.g. umpire-style "
        "question-level scores). When detection is ON (default), such "
        "baselines bypass the group aggregation entirely and their raw value "
        "is used as the confidence, exactly like eval_script_alt.py.",
    )

    # ── Correctness / caching ────────────────────────────────────────────────
    p.add_argument(
        "--correctness-threshold",
        type=float,
        default=0.5,
        help="Per-sample correctness = (generations_uno_score[i] >= threshold) "
        "(default: %(default)s)",
    )
    p.add_argument(
        "--cache-save-every",
        type=int,
        default=200,
        help="Checkpoint the clustering cache every N questions "
        "(default: %(default)s)",
    )

    args = p.parse_args()

    # Resolve baseline lists: use parsed comma-string or keep defaults.
    if args.baselines is not None:
        args.baselines = parse_list_arg(args.baselines)
    else:
        args.baselines = default_baselines

    if args.majority_weighted_baselines is not None:
        args.majority_weighted_baselines = parse_list_arg(
            args.majority_weighted_baselines
        )
    else:
        args.majority_weighted_baselines = default_majority_weighted

    if args.uncertainty_baselines is not None:
        args.uncertainty_baselines = parse_list_arg(args.uncertainty_baselines)
    else:
        args.uncertainty_baselines = []

    # Parse the LR regularization grid into floats.
    args.lr_c_grid = [float(x) for x in parse_list_arg(args.lr_c_grid)]

    # Derive the dynamic constants from the now-resolved baselines.
    args.agg_features = [
        b
        for b in args.baselines
        if b.startswith("internal_") or b.startswith("cross_modal_")
    ]

    # Add the injected feature to the optimization space
    # args.agg_features.append("consensus_fraction")

    # Optionally include confidence_score_selfprobing as an additional feature.
    if args.add_self_probing:
        if "confidence_score_selfprobing" not in args.agg_features:
            args.agg_features.append("confidence_score_selfprobing")
    args.test_methods = args.baselines + ["aggregated_optimal"]
    args.weighted_majority_keys = [f"weighted_majority_{b}" for b in args.baselines]
    args.majority_aggregated_key = "weighted_majority_aggregated_optimal"
    args.majority_methods = (
        ["majority_vote_selfconsistency"]
        + args.weighted_majority_keys
        + [args.majority_aggregated_key]
    )
    args.majority_weighted_methods = set(args.majority_weighted_baselines) | {
        "aggregated_optimal"
    }

    return args


# ----------------------------------------------------------------------------------
# Constants derived at runtime (kept as module-level names for backward compat within
# the rest of the code; populated in main() from the parsed args namespace).
# ----------------------------------------------------------------------------------
MODEL_ID: str
input_json_path: str
cluster_cache_path_base: str
jsonl_path: str
output_eval_path: str
output_eval_path_majority: str
MAX_CANDIDATES_PER_QUESTION: int | None
EXCLUDE_UNANIMOUS_QUESTIONS: bool
BASELINES: list[str]
MAJORITY_WEIGHTED_BASELINES: list[str]
MAJORITY_WEIGHTED_METHODS: set[str]
SELF_CONSISTENCY_KEY = "majority_vote_selfconsistency"
AGGREGATED_KEY = "aggregated_optimal"
AGG_FEATURES: list[str]
TEST_METHODS: list[str]
WEIGHTED_MAJORITY_KEYS: list[str]
WEIGHTED_MAJORITY_AGG_KEY: str
MAJORITY_METHODS: list[str]
VAL_FRACTION: float
SPLIT_SEED: int
ANSWER_SELECTION_MODE: str
RANDOM_ANSWER_SEED: int
GRID_STEP: float
VAL_METRIC: str
WEIGHT_SEARCH_METHOD: str
LR_PENALTY: str
LR_C_GRID: list[float]
UNCERTAINTY_BASELINES: list[str]
METRICS = ["AUROC", "ECE", "Pearson", "AURAC", "TPR@10%FPR"]
CORRECTNESS_THRESHOLD: float
CACHE_SAVE_EVERY: int
CONFIDENCE_MODE: str  # "sum_share" (like eval_script_alt.py) or "mean_share"
PROB_TRANSFORM: str  # "minmax" (like eval_script_alt.py) or "clip"
AUTO_DETECT_QUESTION_LEVEL: bool
# Baselines whose score is constant within every question (detected at runtime);
# they bypass group aggregation, exactly like eval_script_alt.py.
QUESTION_LEVEL_BASELINES: list[str] = []

# Logistic-regression models fitted per category (populated by weight-search phase).
# A value of None for a given label means the safety-net fallback (see
# `find_best_aggregation_weights`) rejected the fitted model in favor of the best
# single raw feature, and the corresponding weight vector in `weights_by_split` /
# `fallback_weights` should be used instead.
_LR_MODELS: dict[str, Pipeline | None] = {}
_LR_MODELS_BY_CATEGORY: dict[str, Pipeline | None] = {}
_LR_FALLBACK_MODEL: Pipeline | None = None


def _populate_globals(args: argparse.Namespace) -> None:
    """Copy every parsed/derived value into the module-level globals so the rest
    of the script can reference them without threading `args` through every
    function signature."""
    globals().update(
        {
            "MODEL_ID": args.model_id,
            "input_json_path": args.input_json,
            "cluster_cache_path_base": args.cluster_cache_base,
            "jsonl_path": args.jsonl_path,
            "output_eval_path": args.output_eval,
            "output_eval_path_majority": args.output_eval_majority,
            "output_dir": args.output_dir,
            "MAX_CANDIDATES_PER_QUESTION": args.max_candidates,
            "EXCLUDE_UNANIMOUS_QUESTIONS": args.exclude_unanimous,
            "BASELINES": args.baselines,
            "MAJORITY_WEIGHTED_BASELINES": args.majority_weighted_baselines,
            "AGG_FEATURES": args.agg_features,
            "TEST_METHODS": args.test_methods,
            "WEIGHTED_MAJORITY_KEYS": args.weighted_majority_keys,
            "WEIGHTED_MAJORITY_AGG_KEY": args.majority_aggregated_key,
            "MAJORITY_METHODS": args.majority_methods,
            "MAJORITY_WEIGHTED_METHODS": args.majority_weighted_methods,
            "VAL_FRACTION": args.val_fraction,
            "SPLIT_SEED": args.split_seed,
            "ANSWER_SELECTION_MODE": args.answer_selection_mode,
            "RANDOM_ANSWER_SEED": args.random_answer_seed,
            "GRID_STEP": args.grid_step,
            "VAL_METRIC": args.val_metric,
            "WEIGHT_SEARCH_METHOD": args.weight_search_method,
            "LR_PENALTY": args.lr_penalty,
            "LR_C_GRID": args.lr_c_grid,
            "UNCERTAINTY_BASELINES": args.uncertainty_baselines,
            "CORRECTNESS_THRESHOLD": args.correctness_threshold,
            "CACHE_SAVE_EVERY": args.cache_save_every,
            "CONFIDENCE_MODE": args.confidence_mode,
            "PROB_TRANSFORM": args.prob_transform,
            "AUTO_DETECT_QUESTION_LEVEL": not args.no_question_level_autodetect,
        }
    )


# ----------------------------------------------------------------------------------
# Answer extraction / correctness
# ----------------------------------------------------------------------------------
def extract_final_answer(text):
    """Pull the final answer out of a generation for clustering."""
    if not text:
        return ""
    t = re.sub(r"<\|[^|>]*\|>", "", text)  # strip special tokens (<|tts_eos|> etc.)
    t = re.sub(
        r"<think>.*?</think>", "", t, flags=re.DOTALL
    )  # drop the reasoning block
    hits = re.findall(r"ANSWER:\s*(.*)", t, flags=re.IGNORECASE)
    if hits:
        return hits[-1].strip()
    lines = [ln for ln in t.strip().splitlines() if ln.strip()]
    return lines[-1].strip() if lines else ""


def get_correctness(rec):
    """Per-sample binary correctness list (aligned with generations_text)."""
    raw = rec.get("generations_uno_score")
    if not isinstance(raw, list):
        raw = rec.get(f"generations_uno_score_threshold_{CORRECTNESS_THRESHOLD}")
    if not isinstance(raw, list):
        return None
    out = []
    for v in raw:
        try:
            out.append(int(float(v) >= CORRECTNESS_THRESHOLD))
        except (TypeError, ValueError):
            out.append(0)
    return out


# ----------------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------------
def calculate_ece(y_true, y_prob, n_bins=10):
    bin_limits = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_limits[i], bin_limits[i + 1]
        in_bin = (
            (y_prob >= lo) & (y_prob <= hi)
            if i == n_bins - 1
            else (y_prob >= lo) & (y_prob < hi)
        )
        prop = np.mean(in_bin)
        if prop > 0:
            acc = np.mean(y_true[in_bin])
            conf = np.mean(y_prob[in_bin])
            ece += np.abs(conf - acc) * prop
    return float(ece)


def calculate_aurac(y_true, y_prob):
    order = np.argsort(y_prob)[::-1]
    y_sorted = y_true[order]
    accs = np.cumsum(y_sorted) / np.arange(1, len(y_sorted) + 1)
    return float(np.mean(accs))


def calculate_tpr_at_fpr(y_true, y_prob, target_fpr=0.10):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.interp(target_fpr, fpr, tpr))


def summarize_method(y_true_list, y_prob_list, uncertainty=False):
    """Shared metric computation, used for baseline / majority-vote / aggregated methods.

    Mirrors eval_script_alt.py's evaluate_questions():
      * PROB_TRANSFORM='minmax' (default): the method's whole y_prob vector is
        min-max rescaled to [0,1]. This is rank-preserving, so AUROC / AURAC /
        TPR@FPR see the ORIGINAL ranking even when raw scores live outside
        [0,1] (clipping instead collapses those into ties at 0/1 and destroys
        the ranking).
      * uncertainty=True flips the rescaled probabilities (1 - p) AFTER the
        rescaling — the alt script's approach — instead of negating raw
        per-sample scores and then clipping (which zeroes everything out).
    """
    y_true = np.array(y_true_list)
    y_prob = np.array(y_prob_list, dtype=float)

    if len(y_prob) > 0:
        if PROB_TRANSFORM == "minmax":
            p_min, p_max = float(np.min(y_prob)), float(np.max(y_prob))
            if p_max > p_min:
                y_prob = (y_prob - p_min) / (p_max - p_min)
            else:
                y_prob = np.zeros_like(y_prob)
        else:  # "clip" — previous behaviour of this script
            y_prob = np.clip(y_prob, 0.0, 1.0)
        if uncertainty:
            y_prob = 1.0 - y_prob

    summary = {}
    if len(y_true) > 0 and len(np.unique(y_true)) > 1:
        summary["AUROC"] = float(roc_auc_score(y_true, y_prob))
        summary["TPR@10%FPR"] = calculate_tpr_at_fpr(y_true, y_prob)
        summary["ECE"] = calculate_ece(y_true, y_prob)
        summary["AURAC"] = calculate_aurac(y_true, y_prob)
        p = pearsonr(y_prob, y_true)[0] if len(y_true) > 1 else float("nan")
        summary["Pearson"] = float(p) if not np.isnan(p) else None
    else:
        summary["AUROC"] = None
        summary["TPR@10%FPR"] = None
        summary["ECE"] = calculate_ece(y_true, y_prob) if len(y_true) > 0 else None
        summary["AURAC"] = calculate_aurac(y_true, y_prob) if len(y_true) > 0 else None
        summary["Pearson"] = None
    return summary


def print_markdown_table(subset_name, subset_summary, method_names):
    print(f"### Subset: {subset_name}")
    print("| Method | AUROC | ECE | Pearson | AURAC | TPR@10%FPR |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: |")

    best = {}
    for metric in METRICS:
        best_m, best_v = None, None
        for m in method_names:
            v = subset_summary[m].get(metric)
            if v is None:
                continue
            if best_v is None or ((v < best_v) if metric == "ECE" else (v > best_v)):
                best_v, best_m = v, m
        best[metric] = best_m

    for m in method_names:
        row = f"| **{m}** "
        for metric in METRICS:
            v = subset_summary[m].get(metric)
            if v is None:
                row += "| N/A "
            else:
                star = "*" if best.get(metric) == m else ""
                row += f"| {v:.4f}{star} "
        print(row + "|")
    print("\n" + "-" * 40 + "\n")


# ----------------------------------------------------------------------------------
# Phase 1: semantic clustering of the sampled answers (vLLM)
# ----------------------------------------------------------------------------------
def salvage_groups(text, num_answers):
    """Recover index->group pairs from slightly-malformed JSON (a missing quote, a trailing
    comma, etc.) via regex, so one bad key doesn't force a string-matching fallback."""
    pairs = re.findall(r'["\']?(\d+)["\']?\s*:\s*(-?\d+)', text)
    d = {int(k): int(v) for k, v in pairs}
    if all(i in d for i in range(num_answers)):
        return [d[i] for i in range(num_answers)]
    return None


def get_answer_groups(answers, tokenizer, llm, max_retries=3):
    """Cluster answers by semantic equivalence / same MCQ option; returns a group id per answer."""
    if len(set(answers)) == 1:
        return [0] * len(answers)

    num_answers = len(answers)
    indexed_answers = {str(i): a for i, a in enumerate(answers)}

    prompt = f"""
    Given the following dictionary of exactly {num_answers} answers extracted from a language model
    (keyed by their original index, from "0" to "{num_answers - 1}"), group them based on their
    semantic equivalence or having selected the same multiple-choice option.

    Return ONLY a valid JSON object with a single key 'groups'. The value of 'groups' must be a
    JSON object (dictionary) mapping EVERY index string (from "0" to "{num_answers - 1}") to an
    integer group ID. You must include all {num_answers} indices, with no omissions or extras.

    Example Input Answers: {{"0": "Final Answer: A", "1": "The answer is A", "2": "Option B", "3": "B", "4": "A"}}
    Example Output: {{"groups": {{"0": 0, "1": 0, "2": 1, "3": 1, "4": 0}}}}

    Answers to cluster (Total items: {num_answers}):
    {json.dumps(indexed_answers, indent=2)}
    """

    messages = [
        {
            "role": "system",
            "content": "You are a precise data-processing AI. You strictly output valid JSON structures and nothing else.",
        },
        {"role": "user", "content": prompt},
    ]
    text_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    sampling_params = None  # created per-attempt so retries can bump temperature

    for attempt in range(max_retries):
        temp = 0.0 if attempt == 0 else 0.5
        sampling_params = SamplingParams(
            temperature=temp, max_tokens=16384, stop_token_ids=[tokenizer.eos_token_id]
        )
        raw_text = "Generation failed before producing text."
        try:
            outputs = llm.generate([text_prompt], sampling_params, use_tqdm=False)
            raw_text = outputs[0].outputs[0].text.strip()

            clean_text = (
                raw_text.split("</think>")[-1].strip()
                if "</think>" in raw_text
                else raw_text
            )
            m = re.search(r"\{.*\}", clean_text, re.DOTALL)
            if m:
                clean_text = m.group(0)

            try:
                data = json.loads(clean_text)
                groups_dict = data.get("groups") if isinstance(data, dict) else None
                if (
                    isinstance(groups_dict, dict)
                    and len(groups_dict) == num_answers
                    and all(str(i) in groups_dict for i in range(num_answers))
                ):
                    return [int(groups_dict[str(i)]) for i in range(num_answers)]
            except Exception:
                pass

            salvaged = salvage_groups(clean_text, num_answers)
            if salvaged is not None:
                return salvaged

            print(
                f"\n[Warning] Attempt {attempt+1}: could not parse a valid "
                f"{num_answers}-index grouping."
            )
        except Exception as e:
            print(
                f"\n[Error] Attempt {attempt+1} generation/parse failed: {e}\n"
                f"Raw:\n{raw_text}\n" + "-" * 40
            )

    print("Max retries reached. Falling back to exact string matching.")
    unique = list(set(answers))
    return [unique.index(a) for a in answers]


# ----------------------------------------------------------------------------------
# Stratified test/validation split
# ----------------------------------------------------------------------------------
def get_record_split_key(rec, id_to_category):
    if "split" in rec:
        return rec["split"]
    return id_to_category.get(rec.get("question_id"), "Unknown")


def stratified_test_val_split(records, id_to_category, val_fraction=None, seed=None):
    if val_fraction is None:
        val_fraction = VAL_FRACTION
    if seed is None:
        seed = SPLIT_SEED
    rng = random.Random(seed)
    by_stratum = defaultdict(list)
    for rec in records:
        key = get_record_split_key(rec, id_to_category)
        by_stratum[key].append(rec)

    test_records, val_records = [], []
    for key, recs in by_stratum.items():
        recs = recs[:]
        rng.shuffle(recs)
        n_val = max(1, round(len(recs) * val_fraction)) if len(recs) >= 2 else 0
        val_records.extend(recs[:n_val])
        test_records.extend(recs[n_val:])

    print(
        f"Stratified split: {len(test_records)} test / {len(val_records)} validation "
        f"records across {len(by_stratum)} strata (target val fraction={val_fraction})."
    )
    return test_records, val_records


# ----------------------------------------------------------------------------------
# aggregated_optimal weight search (validation only)
# ----------------------------------------------------------------------------------
def prepare_records_for_weight_search(records, cluster_info):
    prepared = []
    for rec in records:
        qid = str(rec.get("question_id"))
        ci = cluster_info.get(qid)
        if not ci or not ci.get("groups"):
            continue
        groups = ci["groups"]
        majority_ids = ci["majority_ids"]
        n = len(groups)

        confs = rec.get("generations_confidence", []) or []
        correctness = get_correctness(rec)
        if correctness is None or len(confs) < n or len(correctness) < n:
            continue

        # --- NEW: Inject consensus as a feature ---
        counts_g = Counter(groups)
        for i, g_id in enumerate(groups):
            # The fraction of votes this specific answer group received
            confs[i]["consensus_fraction"] = counts_g[g_id] / n

        feat_matrix = np.array(
            [[float(confs[i].get(f, 0.0)) for f in AGG_FEATURES] for i in range(n)]
        )
        prepared.append(
            {
                "groups": np.array(groups, dtype=int),
                "majority_ids": majority_ids,
                "correctness": np.array(correctness[:n], dtype=int),
                "features": feat_matrix,
            }
        )
    return prepared


def generate_weight_grid(k, step=None):
    """Yield all weight vectors of length k on the probability simplex (entries are
    non-negative multiples of `step` summing to 1.0)."""
    if step is None:
        step = GRID_STEP
    m = round(1.0 / step)

    def rec(remaining_slots, remaining_units):
        if remaining_slots == 1:
            yield (remaining_units,)
            return
        for u in range(remaining_units + 1):
            for tail in rec(remaining_slots - 1, remaining_units - u):
                yield (u,) + tail

    for units in rec(k, m):
        yield tuple(u * step for u in units)


def aggregate_group_scores(groups_arr, scores):
    """Turn per-sample scores into per-group aggregates.

    Returns (group_sum, norm_sum, norm_mean):
      - group_sum:  raw summed score per group (drives vote+confidence winner
                    selection in 'weighted_majority' mode — a group with more
                    supporting samples *should* win more often there).
      - norm_sum:   group_sum normalised to sum to 1 across groups. Biased
                    toward large groups when `scores` are bounded (e.g.
                    probabilities in [0, 1]), since summing more non-negative
                    bounded values can only grow the total — NOT used as the
                    reported confidence value, kept only for backward
                    reference / debugging.
      - norm_mean:  the group's *mean* per-sample score, normalised to sum to
                    1 across groups. This is what should be reported as the
                    method's confidence, since it doesn't automatically
                    reward larger groups the way a raw sum of bounded scores
                    does.
    """
    n_groups = int(groups_arr.max()) + 1
    group_sum = np.bincount(groups_arr, weights=scores, minlength=n_groups)
    group_count = np.bincount(groups_arr, minlength=n_groups)
    total_sum = group_sum.sum() + 1e-9
    norm_sum = group_sum / total_sum

    group_mean = group_sum / np.maximum(group_count, 1)
    total_mean = group_mean.sum() + 1e-9
    norm_mean = group_mean / total_mean

    # exp_scores = np.exp(group_sum / tau)
    # norm_mean = exp_scores / exp_scores.sum()

    return group_sum, norm_sum, norm_mean

    # # Calculate Entropy of the voting distribution
    # probs = group_count / group_count.sum()
    # entropy = -np.sum(probs * np.log(probs + 1e-9))

    # # High entropy (disagreement) reduces confidence
    # # Low entropy (unanimity) keeps confidence high
    # final_confidence = norm_mean * (1.0 - (entropy / np.log(len(probs))))

    # # Calculate the fraction of total votes this group received
    # group_fraction = group_count / len(groups_arr)

    # # Scale the mean confidence by the consensus fraction
    # consensus_weighted_mean = norm_mean * group_fraction

    # # Re-normalize so the final probabilities still sum to 1.0
    # total_consensus = consensus_weighted_mean.sum() + 1e-9
    # final_confidence = consensus_weighted_mean / total_consensus

    # return group_sum, norm_sum, norm_mean
    # return group_sum, norm_sum, final_confidence


def pick_group_confidence(norm_sum, norm_mean):
    """Choose which per-group normalised score array is REPORTED as confidence.

    'sum_share' reproduces eval_script_alt.py (NORM_MODE='share'): the group SUM
    normalised by the total. With narrow-band per-sample scores this is
    approximately n_g / N, i.e. the majority-vote frequency — the strong signal
    behind the alt script's high AUROC / AURAC.
    'mean_share' keeps this script's previous group-size-neutral behaviour.
    """
    return norm_sum if CONFIDENCE_MODE == "sum_share" else norm_mean


def detect_question_level_baselines(records, cluster_info, candidates, tol=1e-12):
    """Port of eval_script_alt.py's detector: a baseline is question-level if its
    per-sample value is constant across the samples of (almost) every question.
    Such baselines must bypass the group-sum/mean normaliser — pushing a constant
    through it either erases the signal (mean_share gives 1/n_groups for every
    group) or silently turns it into the vote frequency (sum_share)."""
    multi = Counter()
    constant = Counter()
    for rec in records:
        qid = str(rec.get("question_id"))
        ci = cluster_info.get(qid)
        if not ci or not ci.get("groups"):
            continue
        n = len(ci["groups"])
        if n < 2:
            continue
        confs = rec.get("generations_confidence", []) or []
        if len(confs) < n:
            continue
        for b in candidates:
            vals = [float(confs[i].get(b, 0.0)) for i in range(n)]
            multi[b] += 1
            if max(vals) - min(vals) <= tol:
                constant[b] += 1
    return [b for b in candidates if multi[b] > 0 and constant[b] / multi[b] > 0.99]


def score_weight_vector(
    weight_vec, prepared_data, selection_mode="raw_majority_tiebreak"
):
    w = np.asarray(weight_vec, dtype=float)
    y_true_all, y_prob_all = [], []

    for rec in prepared_data:
        scores = rec["features"] @ w  # (n,)
        groups_arr = rec["groups"]
        group_sum, norm_sum, norm_mean = aggregate_group_scores(groups_arr, scores)
        conf = pick_group_confidence(norm_sum, norm_mean)

        if selection_mode == "weighted_majority":
            # Winner combines vote count and confidence on purpose — keep sum-based.
            winning_group = int(np.argmax(group_sum))
        else:
            majority_ids = rec["majority_ids"]
            if len(majority_ids) == 1:
                winning_group = majority_ids[0]
            else:
                winning_group = max(majority_ids, key=lambda g: conf[g])

        winning_index = int(np.argmax(groups_arr == winning_group))
        y_true_all.append(int(rec["correctness"][winning_index]))
        y_prob_all.append(float(conf[winning_group]))

    y_true_all = np.array(y_true_all)
    y_prob_all = np.array(y_prob_all, dtype=float)
    if len(y_true_all) == 0 or len(np.unique(y_true_all)) < 2:
        return None
    return summarize_method(y_true_all, y_prob_all)


def best_single_feature(prepared_data, selection_mode="raw_majority_tiebreak"):
    """Evaluate each individual AGG_FEATURE alone (weight vector = one-hot) and
    return (feature_name, weight_vector, summary) for whichever single feature
    scores best on VAL_METRIC. Used as a safety net so a fitted linear/logistic
    combination is never silently adopted if it can't beat the best raw signal
    on its own."""
    best_feat, best_val, best_weights, best_summary = None, -np.inf, None, None
    for idx, feat in enumerate(AGG_FEATURES):
        w = tuple(1.0 if i == idx else 0.0 for i in range(len(AGG_FEATURES)))
        summary = score_weight_vector(w, prepared_data, selection_mode=selection_mode)
        if summary is None or summary.get(VAL_METRIC) is None:
            continue
        value = summary[VAL_METRIC]
        is_better = (value < best_val) if VAL_METRIC == "ECE" else (value > best_val)
        if best_feat is None or is_better:
            best_val, best_feat, best_weights, best_summary = value, feat, w, summary
    if best_feat is None:
        return None
    return best_feat, best_weights, best_summary


def _find_weights_linear_regression(
    val_records,
    cluster_info,
    label="validation",
    selection_mode="raw_majority_tiebreak",
):
    """Fit OLS (no intercept) on per-sample features → correctness, then L1-normalise
    the absolute coefficients to the probability simplex."""
    X_list: list[list[float]] = []
    y_list: list[int] = []
    for rec in val_records:
        qid = str(rec.get("question_id"))
        ci = cluster_info.get(qid)
        if not ci or not ci.get("groups"):
            continue
        n = len(ci["groups"])
        confs = rec.get("generations_confidence", []) or []
        correctness = get_correctness(rec)
        if correctness is None or len(confs) < n or len(correctness) < n:
            continue
        for i in range(n):
            X_list.append([float(confs[i].get(f, 0.0)) for f in AGG_FEATURES])
            y_list.append(correctness[i])

    if len(X_list) < 2:
        raise RuntimeError(
            f"[{label}] Linear regression needs at least 2 per-sample data points "
            "on the validation split. Increase --val-fraction or check the data."
        )

    X = np.array(X_list, dtype=float)
    y = np.array(y_list, dtype=float)

    # OLS: min ||X w - y||², no intercept
    w_raw, _residuals, _rank, _singulars = np.linalg.lstsq(X, y, rcond=None)

    # Convert to probability simplex: |w| / sum(|w|)
    w_abs = np.abs(w_raw)
    w = w_abs / (w_abs.sum() + 1e-9)
    weights = tuple(float(w_i) for w_i in w)

    print(
        f"[{label}] Linear-regression weights (raw → L1-normalised): "
        + ", ".join(
            f"{feat}={raw:+.4f}→{norm:.4f}"
            for feat, raw, norm in zip(AGG_FEATURES, w_raw, weights)
        )
    )

    # Evaluate on validation using the same group-aggregation logic
    prepared = prepare_records_for_weight_search(val_records, cluster_info)
    summary = score_weight_vector(weights, prepared, selection_mode=selection_mode)
    if summary is None:
        raise RuntimeError(
            f"[{label}] Linear-regression weights produced no valid evaluation "
            "on the validation split (single-class or empty)."
        )
    return weights, summary


def _find_weights_grid_search(
    val_records,
    cluster_info,
    label="validation",
    selection_mode="raw_majority_tiebreak",
):
    """Original exhaustive grid search over the probability simplex."""
    prepared = prepare_records_for_weight_search(val_records, cluster_info)
    print(
        f"[{label}] Grid-searching weights for {len(AGG_FEATURES)} features "
        f"({AGG_FEATURES}) on {len(prepared)} validation questions "
        f"(step={GRID_STEP}, selecting by {VAL_METRIC}, mode={selection_mode})..."
    )

    best_weights, best_value, best_summary = None, -np.inf, None
    grid = list(generate_weight_grid(len(AGG_FEATURES), GRID_STEP))
    for w in tqdm(grid, desc=f"Weight grid search [{label}]"):
        summary = score_weight_vector(w, prepared, selection_mode=selection_mode)
        if summary is None or summary.get(VAL_METRIC) is None:
            continue
        value = summary[VAL_METRIC]
        if value > best_value:
            best_value, best_weights, best_summary = value, w, summary

    if best_weights is None:
        raise RuntimeError(
            f"[{label}] Weight search found no valid candidate (validation split may be too "
            "small or single-class). Increase --val-fraction or check the data."
        )
    return best_weights, best_summary


def _evaluate_lr_model(model, val_records, cluster_info, selection_mode):
    """Run a fitted LR pipeline through the same group-aggregation logic used
    everywhere else (mean-based, group-size-neutral) and return its VAL_METRIC
    summary on `val_records`."""

    def _lr_score(features_2d: np.ndarray) -> np.ndarray:
        return model.predict_proba(features_2d)[:, 1]

    prepared = _prepare_records_with_lr_scores(val_records, cluster_info, _lr_score)
    if not prepared:
        return None

    y_true_all, y_prob_all = [], []
    for rec in prepared:
        scores = rec["scores"]
        groups_arr = rec["groups"]
        group_sum, norm_sum, norm_mean = aggregate_group_scores(groups_arr, scores)
        conf = pick_group_confidence(norm_sum, norm_mean)

        if selection_mode == "weighted_majority":
            winning_group = int(np.argmax(group_sum))
        else:
            majority_ids = rec["majority_ids"]
            if len(majority_ids) == 1:
                winning_group = majority_ids[0]
            else:
                winning_group = max(majority_ids, key=lambda g: conf[g])

        winning_index = int(np.argmax(groups_arr == winning_group))
        y_true_all.append(int(rec["correctness"][winning_index]))
        y_prob_all.append(float(conf[winning_group]))

    y_true_all = np.array(y_true_all)
    y_prob_all = np.array(y_prob_all, dtype=float)
    if len(y_true_all) == 0 or len(np.unique(y_true_all)) < 2:
        return None
    return summarize_method(y_true_all, y_prob_all)


def _find_weights_logistic_regression(
    val_records,
    cluster_info,
    label="validation",
    selection_mode="raw_majority_tiebreak",
):
    """Fit a StandardScaler + LogisticRegression on per-sample features → correctness,
    searching over --lr-penalty / --lr-c-grid and keeping whichever regularization
    strength scores best on VAL_METRIC (evaluated through the same mean-based,
    group-size-neutral aggregation used by the grid/linear-regression paths).

    Returns (weights_tuple, validation_summary) where weights_tuple is a sentinel
    (the model is stored in the global _LR_MODELS dict by the caller)."""
    X_list: list[list[float]] = []
    y_list: list[int] = []
    for rec in val_records:
        qid = str(rec.get("question_id"))
        ci = cluster_info.get(qid)
        if not ci or not ci.get("groups"):
            continue
        n = len(ci["groups"])
        confs = rec.get("generations_confidence", []) or []
        correctness = get_correctness(rec)
        if correctness is None or len(confs) < n or len(correctness) < n:
            continue
        for i in range(n):
            X_list.append([float(confs[i].get(f, 0.0)) for f in AGG_FEATURES])
            y_list.append(correctness[i])

    if len(X_list) < 2 or len(set(y_list)) < 2:
        raise RuntimeError(
            f"[{label}] Logistic regression needs at least 2 samples with both "
            "classes on the validation split. Increase --val-fraction or check the data."
        )

    X = np.array(X_list, dtype=float)
    y = np.array(y_list, dtype=int)

    solver = "saga" if LR_PENALTY == "l1" else "lbfgs"

    best_model, best_summary, best_val, best_c = None, None, -np.inf, None
    for c in LR_C_GRID:
        candidate = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "lr",
                    LogisticRegression(
                        max_iter=5000, C=c, penalty=LR_PENALTY, solver=solver
                    ),
                ),
            ]
        )
        try:
            candidate.fit(X, y)
        except Exception as e:
            print(f"[{label}] C={c} ({LR_PENALTY}) failed to fit: {e}")
            continue
        summary = _evaluate_lr_model(
            candidate, val_records, cluster_info, selection_mode
        )
        if summary is None or summary.get(VAL_METRIC) is None:
            continue
        value = summary[VAL_METRIC]
        is_better = (value < best_val) if VAL_METRIC == "ECE" else (value > best_val)
        if best_model is None or is_better:
            best_val, best_model, best_summary, best_c = value, candidate, summary, c

    if best_model is None:
        raise RuntimeError(
            f"[{label}] Logistic regression produced no valid evaluation across "
            f"the C grid {LR_C_GRID}."
        )

    coefs = best_model.named_steps["lr"].coef_[0]
    intercept = best_model.named_steps["lr"].intercept_[0]
    # Store the fitted model globally so Phase 2 evaluation can use it.
    _LR_MODELS[label] = best_model

    print(
        f"[{label}] LogisticRegression fitted on {len(X)} samples "
        f"(positive class={int(y.mean()*100)}%, penalty={LR_PENALTY}, best C={best_c}, "
        f"val {VAL_METRIC}={best_val:.4f}). "
        + ", ".join(f"{feat}={c:+.3f}" for feat, c in zip(AGG_FEATURES, coefs))
        + f", intercept={intercept:+.3f}"
    )

    return (0.0,) * len(AGG_FEATURES), best_summary


def _prepare_records_with_lr_scores(records, cluster_info, score_fn) -> list[dict]:
    """Like prepare_records_for_weight_search but uses *score_fn* on the
    stacked feature matrix to produce per-sample scores instead of a dot product."""
    prepared = []
    for rec in records:
        qid = str(rec.get("question_id"))
        ci = cluster_info.get(qid)
        if not ci or not ci.get("groups"):
            continue
        groups = ci["groups"]
        majority_ids = ci["majority_ids"]
        n = len(groups)
        confs = rec.get("generations_confidence", []) or []
        correctness = get_correctness(rec)
        if correctness is None or len(confs) < n or len(correctness) < n:
            continue
        feat_matrix = np.array(
            [[float(confs[i].get(f, 0.0)) for f in AGG_FEATURES] for i in range(n)]
        )
        scores = score_fn(feat_matrix)
        prepared.append(
            {
                "groups": np.array(groups, dtype=int),
                "majority_ids": majority_ids,
                "correctness": np.array(correctness[:n], dtype=int),
                "scores": np.asarray(scores, dtype=float),
            }
        )
    return prepared


def find_best_aggregation_weights(
    val_records,
    cluster_info,
    label="validation",
    selection_mode="raw_majority_tiebreak",
):
    """Dispatch to the active weight-search method, then apply a safety net:
    never adopt a fitted (linear/logistic) combination that underperforms the
    single best raw AGG_FEATURE on VAL_METRIC. Grid search already can't lose
    this comparison (the one-hot vectors are part of its search space), but
    linear/logistic regression optimize a different objective (per-sample
    loss, not the question-level metric) and have no such guarantee."""
    if WEIGHT_SEARCH_METHOD == "logistic_regression":
        weights, summary = _find_weights_logistic_regression(
            val_records, cluster_info, label, selection_mode
        )
    elif WEIGHT_SEARCH_METHOD == "linear_regression":
        weights, summary = _find_weights_linear_regression(
            val_records, cluster_info, label, selection_mode
        )
    else:
        weights, summary = _find_weights_grid_search(
            val_records, cluster_info, label, selection_mode
        )

    if WEIGHT_SEARCH_METHOD == "grid":
        # Grid search already includes every one-hot vector, so it can never
        # lose to a single feature — no safety-net check needed.
        return weights, summary

    prepared = prepare_records_for_weight_search(val_records, cluster_info)
    single = best_single_feature(prepared, selection_mode=selection_mode)
    if single is None:
        return weights, summary

    best_feat, single_weights, single_summary = single
    current_val = summary.get(VAL_METRIC)
    single_val = single_summary.get(VAL_METRIC)
    if single_val is None:
        return weights, summary

    single_is_better = (
        (single_val < current_val)
        if VAL_METRIC == "ECE"
        else (current_val is None or single_val > current_val)
    )
    if single_is_better:
        cur_str = f"{current_val:.4f}" if current_val is not None else "N/A"
        print(
            f"[{label}] Safety net: fitted {WEIGHT_SEARCH_METHOD} weights "
            f"({VAL_METRIC}={cur_str}) underperform single feature "
            f"'{best_feat}' ({VAL_METRIC}={single_val:.4f}). Falling back to "
            f"'{best_feat}' alone for this split."
        )
        # If an LR model had been stored for this label, drop it so downstream
        # code (Phase 2 eval, metadata export, printing) treats this label as
        # weight-vector-based instead of model-based.
        _LR_MODELS[label] = None
        return single_weights, single_summary

    return weights, summary


def find_best_weights_per_split(
    val_records,
    id_to_category,
    cluster_info,
    selection_mode="raw_majority_tiebreak",
    method_label="aggregated_optimal",
):
    global _LR_MODELS_BY_CATEGORY, _LR_FALLBACK_MODEL

    fallback_label = f"Overall (pooled fallback) [{method_label}]"
    fallback_weights, fallback_summary = find_best_aggregation_weights(
        val_records,
        cluster_info,
        label=fallback_label,
        selection_mode=selection_mode,
    )

    # When using logistic regression, store the fallback model.
    if WEIGHT_SEARCH_METHOD == "logistic_regression":
        _LR_FALLBACK_MODEL = _LR_MODELS.get(fallback_label)

    val_by_category = defaultdict(list)
    for rec in val_records:
        key = get_record_split_key(rec, id_to_category)
        val_by_category[key].append(rec)

    weights_by_split = {}
    for category, cat_val_records in val_by_category.items():
        cat_label = f"{category} [{method_label}]"
        try:
            weights, summary = find_best_aggregation_weights(
                cat_val_records,
                cluster_info,
                label=cat_label,
                selection_mode=selection_mode,
            )
            weights_by_split[category] = (weights, summary)
            # Store LR model for this category.
            if WEIGHT_SEARCH_METHOD == "logistic_regression":
                _LR_MODELS_BY_CATEGORY[category] = _LR_MODELS.get(cat_label)
        except RuntimeError as e:
            print(
                f"[{category}] {e}\n  -> falling back to the pooled weight vector for this split."
            )
            weights_by_split[category] = (fallback_weights, fallback_summary)
            if WEIGHT_SEARCH_METHOD == "logistic_regression":
                _LR_MODELS_BY_CATEGORY[category] = _LR_FALLBACK_MODEL

    return weights_by_split, fallback_weights, fallback_summary


def print_best_weights(
    method_label, split_label, best_weights, best_summary, lr_model=None
):
    print("\n" + "=" * 20 + f" BEST {method_label} WEIGHTS — {split_label} " + "=" * 20)
    if WEIGHT_SEARCH_METHOD == "logistic_regression" and lr_model is not None:
        lr = lr_model.named_steps["lr"]
        intercept = float(lr.intercept_[0])
        print(
            f"LogisticRegression coefficients  "
            f"(val {VAL_METRIC} = {best_summary.get(VAL_METRIC, 'N/A'):.4f}):\n"
        )
        print("| Feature | Coefficient |")
        print("| :--- | :---: |")
        for feat, c in zip(AGG_FEATURES, lr.coef_[0]):
            print(f"| {feat} | {c:+.4f} |")
        print(f"| (intercept) | {intercept:+.4f} |")
        print()
        return
    print(
        f"Selection metric: {VAL_METRIC} = {best_summary[VAL_METRIC]:.4f} on validation split\n"
    )
    print("| Feature | Weight |")
    print("| :--- | :---: |")
    for feat, w in zip(AGG_FEATURES, best_weights):
        print(f"| {feat} | {w:.3f} |")
    print(f"\nFull validation metrics for this weight vector: {best_summary}\n")


def print_all_best_weights(
    method_label,
    weights_by_split,
    fallback_weights,
    fallback_summary,
    lr_models_by_category=None,
    lr_fallback_model=None,
):
    print_best_weights(
        method_label,
        "Overall (pooled fallback)",
        fallback_weights,
        fallback_summary,
        lr_fallback_model,
    )
    for category, (weights, summary) in weights_by_split.items():
        lr_m = (lr_models_by_category or {}).get(category)
        print_best_weights(method_label, category, weights, summary, lr_m)


def is_unanimous_question(cluster_entry):
    """True if this question's clustering put every sampled generation into a single
    semantic group — i.e. the raw-frequency majority vote share is exactly 1.0."""
    groups = (cluster_entry or {}).get("groups") or []
    if not groups:
        return False
    return len(set(groups)) == 1


# ----------------------------------------------------------------------------------
# IO helpers
# ----------------------------------------------------------------------------------
def save_json(obj, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


# ----------------------------------------------------------------------------------
# CSV + metadata export
# ----------------------------------------------------------------------------------
def _format_value(v) -> str:
    """Format a float or None for CSV output."""
    if v is None:
        return "N/A"
    return f"{v:.4f}"


def write_subset_csv(
    folder: str, subset_name: str, subset_summary: dict, method_names: list[str]
) -> None:
    """Write a single per-split CSV file inside *folder*."""
    path = os.path.join(folder, f"{subset_name}.csv")
    header = ["Method"] + METRICS
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for method in method_names:
            summary = subset_summary.get(method, {})
            row = [method] + [_format_value(summary.get(m)) for m in METRICS]
            writer.writerow(row)
    print(f"  Wrote {path}")


def _lr_model_to_dict(model: Pipeline | None) -> dict | None:
    """Extract coefficients, intercept, and scaler params from a fitted LR pipeline."""
    if model is None:
        return None
    lr = model.named_steps["lr"]
    scaler = model.named_steps["scaler"]
    return {
        "coefficients": {feat: float(c) for feat, c in zip(AGG_FEATURES, lr.coef_[0])},
        "intercept": float(lr.intercept_[0]),
        "scaler_mean": {feat: float(m) for feat, m in zip(AGG_FEATURES, scaler.mean_)},
        "scaler_scale": {
            feat: float(s) for feat, s in zip(AGG_FEATURES, scaler.scale_)
        },
    }


def _weights_to_serializable(
    weights_by_split,
    fallback_weights,
    fallback_summary,
    lr_models_by_category=None,
    lr_fallback_model=None,
):
    """Convert weights into plain JSON-serializable dicts.
    In LR mode, extracts coefficients from the fitted Pipeline objects instead."""
    out: dict = {}

    use_lr = WEIGHT_SEARCH_METHOD == "logistic_regression"

    # Fallback entry: use the LR model if one survived the safety net, else the
    # (possibly safety-net-overridden) single-feature or fitted weight vector.
    if use_lr and lr_fallback_model is not None:
        out["fallback"] = {
            "lr_model": _lr_model_to_dict(lr_fallback_model),
            "validation_metrics": {
                k: (float(v) if v is not None else None)
                for k, v in fallback_summary.items()
            },
        }
    else:
        out["fallback"] = {
            "weights": {
                feat: float(w) for feat, w in zip(AGG_FEATURES, fallback_weights)
            },
            "validation_metrics": {
                k: (float(v) if v is not None else None)
                for k, v in fallback_summary.items()
            },
        }

    # Per-category entries: decide independently for each category, since the
    # safety net in find_best_aggregation_weights may have overridden only
    # some categories (clearing their stored LR model) while others kept it.
    out["per_category"] = {}
    for category, (weights, summary) in weights_by_split.items():
        lr_m = (lr_models_by_category or {}).get(category)
        if use_lr and lr_m is not None:
            out["per_category"][category] = {
                "lr_model": _lr_model_to_dict(lr_m),
                "validation_metrics": {
                    k: (float(v) if v is not None else None) for k, v in summary.items()
                },
            }
        else:
            out["per_category"][category] = {
                "weights": {feat: float(w) for feat, w in zip(AGG_FEATURES, weights)},
                "validation_metrics": {
                    k: (float(v) if v is not None else None) for k, v in summary.items()
                },
            }
    return out


def build_metadata(
    weights_by_split,
    fallback_weights,
    fallback_summary,
    method_label: str,
    lr_models_by_category=None,
    lr_fallback_model=None,
) -> dict:
    """Build the metadata.json content for one of the two subfolders."""
    return {
        "method": method_label,
        "val_metric": VAL_METRIC,
        "weight_search_method": WEIGHT_SEARCH_METHOD,
        "features": AGG_FEATURES,
        "grid_step": GRID_STEP,
        "best_weights": _weights_to_serializable(
            weights_by_split,
            fallback_weights,
            fallback_summary,
            lr_models_by_category,
            lr_fallback_model,
        ),
        "parameters": {
            "model_id": MODEL_ID,
            "val_fraction": VAL_FRACTION,
            "split_seed": SPLIT_SEED,
            "answer_selection_mode": ANSWER_SELECTION_MODE,
            "random_answer_seed": RANDOM_ANSWER_SEED,
            "correctness_threshold": CORRECTNESS_THRESHOLD,
            "max_candidates_per_question": MAX_CANDIDATES_PER_QUESTION,
            "exclude_unanimous_questions": EXCLUDE_UNANIMOUS_QUESTIONS,
            "baselines": BASELINES,
            "majority_weighted_baselines": MAJORITY_WEIGHTED_BASELINES,
            "input_json": input_json_path,
            "uncertainty_baselines": UNCERTAINTY_BASELINES,
            "confidence_mode": CONFIDENCE_MODE,
            "prob_transform": PROB_TRANSFORM,
            "question_level_baselines": QUESTION_LEVEL_BASELINES,
        },
    }


def export_csv_results(
    output_dir: str,
    final_results: dict,
    final_results_majority: dict,
    weights_by_split,
    fallback_weights,
    fallback_summary,
    wm_weights_by_split,
    wm_fallback_weights,
    wm_fallback_summary,
    lr_models_by_category=None,
    lr_fallback_model=None,
    lr_wm_models_by_category=None,
    lr_wm_fallback_model=None,
) -> None:
    """Create the full output-dir structure with CSVs and metadata.json files."""
    no_maj_dir = os.path.join(output_dir, "no_majority")
    maj_dir = os.path.join(output_dir, "majority_weighted")
    os.makedirs(no_maj_dir, exist_ok=True)
    os.makedirs(maj_dir, exist_ok=True)

    # ── Derive subset list from the data, with "Overall" always last ────────
    no_maj_subsets = sorted([s for s in final_results.keys() if s != "Overall"])
    if "Overall" in final_results:
        no_maj_subsets.append("Overall")
    maj_subsets = sorted([s for s in final_results_majority.keys() if s != "Overall"])
    if "Overall" in final_results_majority:
        maj_subsets.append("Overall")

    # ── no_majority CSVs ─────────────────────────────────────────────────────
    print(f"\nWriting no_majority CSVs to {no_maj_dir}/ ...")
    for subset in no_maj_subsets:
        write_subset_csv(no_maj_dir, subset, final_results[subset], TEST_METHODS)

    # ── majority_weighted CSVs ───────────────────────────────────────────────
    print(f"\nWriting majority_weighted CSVs to {maj_dir}/ ...")
    for subset in maj_subsets:
        write_subset_csv(
            maj_dir, subset, final_results_majority[subset], MAJORITY_METHODS
        )

    # ── metadata.json files ──────────────────────────────────────────────────
    meta_no_maj = build_metadata(
        weights_by_split,
        fallback_weights,
        fallback_summary,
        "aggregated_optimal",
        lr_models_by_category,
        lr_fallback_model,
    )
    meta_path = os.path.join(no_maj_dir, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_no_maj, f, indent=2, ensure_ascii=False)
    print(f"\n  Wrote {meta_path}")

    meta_maj = build_metadata(
        wm_weights_by_split,
        wm_fallback_weights,
        wm_fallback_summary,
        "weighted_majority_aggregated_optimal",
        lr_wm_models_by_category,
        lr_wm_fallback_model,
    )
    meta_path = os.path.join(maj_dir, "metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_maj, f, indent=2, ensure_ascii=False)
    print(f"  Wrote {meta_path}")


# ----------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------
def main():
    args = parse_args()
    _populate_globals(args)

    print(f"Loading data from {input_json_path} ...")
    with open(input_json_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    print(f"Loaded {len(records)} question records.")

    print(f"Loading {MODEL_ID} with vLLM engine...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    # llm = LLM(
    #     model=MODEL_ID, tensor_parallel_size=1, dtype="bfloat16", enforce_eager=False
    # )
    llm = None

    # Cache path is versioned by MAX_CANDIDATES_PER_QUESTION so a cache built for a
    # different candidate limit is never silently reused with the wrong truncation.
    cache_suffix = (
        ""
        if MAX_CANDIDATES_PER_QUESTION is None
        else f"_top{MAX_CANDIDATES_PER_QUESTION}"
    )
    cluster_cache_path = cluster_cache_path_base.replace(
        ".json", f"{cache_suffix}.json"
    )

    # ---- Phase 1: cluster the sampled answers for ALL records (resume from cache) ----
    print(
        "\n--- Phase 1: clustering sampled answers (most-likely generation excluded) ---"
    )
    if MAX_CANDIDATES_PER_QUESTION is not None:
        print(
            f"[note] Limiting analysis to the first {MAX_CANDIDATES_PER_QUESTION} "
            "candidate generations per question."
        )
    cluster_info = {}
    if os.path.exists(cluster_cache_path):
        with open(cluster_cache_path, "r", encoding="utf-8") as f:
            cluster_info = json.load(f)
        print(f"Resumed clustering cache with {len(cluster_info)} questions.")

    newly_clustered = 0
    for rec in tqdm(records, desc="Clustering"):
        qid = str(rec.get("question_id"))
        if qid in cluster_info:
            continue

        gens = rec.get("generations_text", []) or []
        if MAX_CANDIDATES_PER_QUESTION is not None:
            gens = gens[:MAX_CANDIDATES_PER_QUESTION]
        answers = [extract_final_answer(g) for g in gens]
        if not answers:
            cluster_info[qid] = {"groups": [], "majority_ids": []}
            continue

        groups = get_answer_groups(answers, tokenizer, llm)
        counts = Counter(groups)
        max_count = max(counts.values())
        majority_ids = [g for g, c in counts.items() if c == max_count]
        cluster_info[qid] = {"groups": groups, "majority_ids": majority_ids}

        newly_clustered += 1
        if newly_clustered % CACHE_SAVE_EVERY == 0:
            save_json(cluster_info, cluster_cache_path)

    save_json(cluster_info, cluster_cache_path)
    print(
        f"Clustering done ({newly_clustered} newly clustered). Cache: {cluster_cache_path}\n"
    )

    # ---- optionally drop unanimous-agreement questions (majority vote share = 1.0) ----
    if EXCLUDE_UNANIMOUS_QUESTIONS:
        before = len(records)
        records = [
            rec
            for rec in records
            if not is_unanimous_question(cluster_info.get(str(rec.get("question_id"))))
        ]
        print(
            f"[note] --exclude-unanimous: dropped {before - len(records)} "
            f"unanimous-agreement questions ({len(records)} remain).\n"
        )

    # ---- category / split lookup (needed for stratification) ----
    id_to_category = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                id_to_category[item["question_id"]] = item.get("category", "Unknown")

    # ---- auto-detect question-level baselines (constant within each question) ----
    global QUESTION_LEVEL_BASELINES
    if AUTO_DETECT_QUESTION_LEVEL:
        detected = detect_question_level_baselines(records, cluster_info, BASELINES)
        for b in detected:
            if b not in QUESTION_LEVEL_BASELINES:
                print(
                    f"[Notice] '{b}' is constant within every question -> treating it "
                    "as a question-level score (bypassing the group normaliser)."
                )
                QUESTION_LEVEL_BASELINES.append(b)
    print(f"Question-level baselines: {QUESTION_LEVEL_BASELINES}\n")

    # ---- stratified test/validation split ----
    test_records, val_records = stratified_test_val_split(records, id_to_category)

    # ---- search for aggregated_optimal's weights separately per split/category ----
    weights_by_split, fallback_weights, fallback_summary = find_best_weights_per_split(
        val_records,
        id_to_category,
        cluster_info,
        selection_mode="raw_majority_tiebreak",
        method_label="aggregated_optimal",
    )
    # Capture LR models for aggregated_optimal (this is a shallow copy — fine since
    # the next call to find_best_weights_per_split will overwrite the globals).
    lr_models_by_category = dict(_LR_MODELS_BY_CATEGORY)
    lr_fallback_model = _LR_FALLBACK_MODEL

    print_all_best_weights(
        "aggregated_optimal",
        weights_by_split,
        fallback_weights,
        fallback_summary,
        lr_models_by_category,
        lr_fallback_model,
    )

    # ---- separate per-split search for weighted_majority_aggregated_optimal. ----
    wm_weights_by_split, wm_fallback_weights, wm_fallback_summary = (
        find_best_weights_per_split(
            val_records,
            id_to_category,
            cluster_info,
            selection_mode="weighted_majority",
            method_label="weighted_majority_aggregated_optimal",
        )
    )
    # Capture LR models for weighted_majority_aggregated_optimal.
    lr_wm_models_by_category = dict(_LR_MODELS_BY_CATEGORY)
    lr_wm_fallback_model = _LR_FALLBACK_MODEL

    print_all_best_weights(
        "weighted_majority_aggregated_optimal",
        wm_weights_by_split,
        wm_fallback_weights,
        wm_fallback_summary,
        lr_wm_models_by_category,
        lr_wm_fallback_model,
    )

    # ---- Phase 2: evaluation on the TEST split only ----
    print(
        "--- Phase 2: evaluating baselines + majority-voting methods + aggregated_optimal (test set) ---"
    )

    subsets = defaultdict(list)
    for rec in test_records:
        category = id_to_category.get(rec.get("question_id"), "Unknown")
        subsets[category].append(rec)
        subsets["Overall"].append(rec)

    n_skipped = 0
    final_results = {}
    final_results_majority = {}

    for subset_name, subset_recs in tqdm(subsets.items(), desc="Subsets"):
        print(f"\nEvaluating subset: {subset_name} ({len(subset_recs)} questions)...")

        collected = {m: {"y_true": [], "y_prob": []} for m in TEST_METHODS}
        collected_majority = {m: {"y_true": [], "y_prob": []} for m in MAJORITY_METHODS}

        for rec in subset_recs:
            qid = str(rec.get("question_id"))
            ci = cluster_info.get(qid)
            if not ci or not ci.get("groups"):
                continue

            groups = ci["groups"]
            majority_ids = ci["majority_ids"]
            n = len(groups)

            confs = rec.get("generations_confidence", []) or []
            correctness = get_correctness(rec)
            if correctness is None or len(confs) < n or len(correctness) < n:
                if subset_name == "Overall":
                    n_skipped += 1
                continue

            # ---- self-consistency majority vote (baseline-agnostic) ----
            counts_g = Counter(groups)
            sc_winning_group = majority_ids[0]
            sc_index = groups.index(sc_winning_group)
            sc_score = counts_g[sc_winning_group] / n
            collected_majority[SELF_CONSISTENCY_KEY]["y_true"].append(
                int(correctness[sc_index])
            )
            collected_majority[SELF_CONSISTENCY_KEY]["y_prob"].append(sc_score)

            # ---- one random candidate index per question, reused across every method ----
            if ANSWER_SELECTION_MODE == "random":
                rand_index = random.Random(f"{RANDOM_ANSWER_SEED}:{qid}").randrange(n)

            for method in TEST_METHODS:
                if method == AGGREGATED_KEY:
                    rec_category = id_to_category.get(rec.get("question_id"), "Unknown")
                    # Use the LR model fitted for this category ONLY if the
                    # safety net in find_best_aggregation_weights didn't reject
                    # it in favor of a single raw feature. Otherwise fall back
                    # to the weight vector (which may itself be that single
                    # feature's one-hot vector) — never silently emit zeros.
                    lr_model = None
                    if WEIGHT_SEARCH_METHOD == "logistic_regression":
                        lr_model = lr_models_by_category.get(
                            rec_category, lr_fallback_model
                        )
                    if lr_model is not None:
                        feat_matrix = np.array(
                            [
                                [float(confs[i].get(f, 0.0)) for f in AGG_FEATURES]
                                for i in range(n)
                            ]
                        )
                        scores = list(lr_model.predict_proba(feat_matrix)[:, 1])
                    else:
                        rec_weights, _ = weights_by_split.get(
                            rec_category, (fallback_weights, fallback_summary)
                        )
                        scores = [
                            sum(
                                w * confs[i].get(feat, 0.0)
                                for w, feat in zip(rec_weights, AGG_FEATURES)
                            )
                            for i in range(n)
                        ]
                else:
                    scores = [float(confs[i].get(method, 0.0)) for i in range(n)]

                # NOTE: uncertainty baselines are NO LONGER negated here. Like
                # eval_script_alt.py, raw values flow through the pipeline;
                # tie-breaking picks the MIN group for uncertainty methods and
                # the probabilities are flipped (1 - p) AFTER per-method
                # min-max rescaling inside summarize_method. (The previous
                # negate-then-clip approach clipped every negated score to 0,
                # producing a constant confidence and chance-level AUROC.)
                is_uncertainty = method in UNCERTAINTY_BASELINES

                groups_arr = np.array(groups, dtype=int)
                scores_arr = np.array(scores, dtype=float)
                S_g, norm_sum, norm_mean = aggregate_group_scores(
                    groups_arr, scores_arr
                )
                # Reported confidence: 'sum_share' (default) reproduces
                # eval_script_alt.py — the group SUM normalised by the total,
                # which for narrow-band scores is dominated by the group SIZE
                # (i.e. it largely encodes the majority-vote frequency).
                # 'mean_share' restores this script's previous group-size-
                # neutral behaviour.
                conf_arr = pick_group_confidence(norm_sum, norm_mean)
                norm_scores = {g: float(conf_arr[g]) for g in range(len(conf_arr))}

                if method in QUESTION_LEVEL_BASELINES:
                    # Question-level score (constant across samples): bypass the
                    # group normaliser entirely, exactly like eval_script_alt.py.
                    # The answer is picked by plain majority vote; only the
                    # confidence comes from the raw score (min-max rescaled —
                    # and flipped if it's an uncertainty method — later on).
                    winning_group = majority_ids[0]
                    winning_index = groups.index(winning_group)
                    sel_y_true = int(correctness[winning_index])
                    sel_y_prob = float(scores[0])
                elif ANSWER_SELECTION_MODE == "random":
                    sel_y_true = int(correctness[rand_index])
                    sel_y_prob = float(scores[rand_index])
                else:
                    # Winner = raw-frequency majority group; among TIED majority
                    # groups the method's own score breaks the tie (min for
                    # uncertainty methods), mirroring eval_script_alt.py's
                    # score_question().
                    if len(majority_ids) == 1:
                        winning_group = majority_ids[0]
                    elif is_uncertainty:
                        winning_group = min(majority_ids, key=lambda g: norm_scores[g])
                    else:
                        winning_group = max(majority_ids, key=lambda g: norm_scores[g])
                    winning_index = groups.index(winning_group)
                    sel_y_true = int(correctness[winning_index])
                    if method in MAJORITY_WEIGHTED_METHODS:
                        sel_y_prob = norm_scores[winning_group]
                    else:
                        sel_y_prob = float(scores[winning_index])

                collected[method]["y_true"].append(sel_y_true)
                collected[method]["y_prob"].append(sel_y_prob)

                # weighted-majority row for every baseline
                if method in BASELINES:
                    wm_key = f"weighted_majority_{method}"
                    if (
                        method in MAJORITY_WEIGHTED_BASELINES
                        and method not in QUESTION_LEVEL_BASELINES
                    ):
                        # Winner selection combines vote count + confidence on
                        # purpose (sum-based). For uncertainty methods the
                        # LOWEST summed score wins (raw values are no longer
                        # negated upstream).
                        wm_winning_group = int(
                            np.argmin(S_g) if is_uncertainty else np.argmax(S_g)
                        )
                        wm_index = groups.index(wm_winning_group)
                        collected_majority[wm_key]["y_true"].append(
                            int(correctness[wm_index])
                        )
                        collected_majority[wm_key]["y_prob"].append(
                            norm_scores[wm_winning_group]
                        )
                    else:
                        collected_majority[wm_key]["y_true"].append(sel_y_true)
                        collected_majority[wm_key]["y_prob"].append(sel_y_prob)

            # ---- weighted_majority_aggregated_optimal ----
            rec_category = id_to_category.get(rec.get("question_id"), "Unknown")
            lr_wm_model = None
            if WEIGHT_SEARCH_METHOD == "logistic_regression":
                lr_wm_model = lr_wm_models_by_category.get(
                    rec_category, lr_wm_fallback_model
                )
            if lr_wm_model is not None:
                feat_matrix = np.array(
                    [
                        [float(confs[i].get(f, 0.0)) for f in AGG_FEATURES]
                        for i in range(n)
                    ]
                )
                wm_scores = list(lr_wm_model.predict_proba(feat_matrix)[:, 1])
            else:
                wm_rec_weights, _ = wm_weights_by_split.get(
                    rec_category, (wm_fallback_weights, wm_fallback_summary)
                )
                wm_scores = [
                    sum(
                        w * confs[i].get(feat, 0.0)
                        for w, feat in zip(wm_rec_weights, AGG_FEATURES)
                    )
                    for i in range(n)
                ]
            groups_arr = np.array(groups, dtype=int)
            wm_scores_arr = np.array(wm_scores, dtype=float)
            S_g_wm, norm_sum_wm, norm_mean_wm = aggregate_group_scores(
                groups_arr, wm_scores_arr
            )
            conf_wm = pick_group_confidence(norm_sum_wm, norm_mean_wm)
            # Winner combines vote count + confidence (sum-based) on purpose;
            # the reported confidence follows --confidence-mode.
            wm_agg_winning_group = int(np.argmax(S_g_wm))
            wm_agg_index = groups.index(wm_agg_winning_group)
            collected_majority[WEIGHTED_MAJORITY_AGG_KEY]["y_true"].append(
                int(correctness[wm_agg_index])
            )
            collected_majority[WEIGHTED_MAJORITY_AGG_KEY]["y_prob"].append(
                float(conf_wm[wm_agg_winning_group])
            )

        subset_summary = {
            m: summarize_method(
                collected[m]["y_true"],
                collected[m]["y_prob"],
                uncertainty=(m in UNCERTAINTY_BASELINES),
            )
            for m in TEST_METHODS
        }
        final_results[subset_name] = subset_summary

        def _wm_is_uncertainty(m):
            # weighted_majority_<b> inherits <b>'s uncertainty orientation.
            base = (
                m[len("weighted_majority_") :]
                if m.startswith("weighted_majority_")
                else m
            )
            return base in UNCERTAINTY_BASELINES

        subset_summary_majority = {
            m: summarize_method(
                collected_majority[m]["y_true"],
                collected_majority[m]["y_prob"],
                uncertainty=_wm_is_uncertainty(m),
            )
            for m in MAJORITY_METHODS
        }
        final_results_majority[subset_name] = subset_summary_majority

    save_json(final_results, output_eval_path)
    save_json(final_results_majority, output_eval_path_majority)
    print(f"\nTest-set evaluation metrics saved to {output_eval_path}")
    print(f"Majority-voting test-set metrics saved to {output_eval_path_majority}")
    if n_skipped:
        print(
            f"[note] {n_skipped} questions skipped (missing/short confidence or correctness lists)."
        )
    print()

    # ---- CSV + metadata export ----
    export_csv_results(
        output_dir,
        final_results,
        final_results_majority,
        weights_by_split,
        fallback_weights,
        fallback_summary,
        wm_weights_by_split,
        wm_fallback_weights,
        wm_fallback_summary,
        lr_models_by_category,
        lr_fallback_model,
        lr_wm_models_by_category,
        lr_wm_fallback_model,
    )

    # --- Recap of the winning weights right before the final tables ---
    print_all_best_weights(
        "aggregated_optimal",
        weights_by_split,
        fallback_weights,
        fallback_summary,
        lr_models_by_category,
        lr_fallback_model,
    )
    print_all_best_weights(
        "weighted_majority_aggregated_optimal",
        wm_weights_by_split,
        wm_fallback_weights,
        wm_fallback_summary,
        lr_wm_models_by_category,
        lr_wm_fallback_model,
    )

    print(
        "=" * 20
        + " BASELINE METHODS + aggregated_optimal (TEST SET) "
        + "=" * 20
        + "\n"
    )
    for subset_name, subset_summary in final_results.items():
        print_markdown_table(subset_name, subset_summary, TEST_METHODS)

    print("=" * 20 + " MAJORITY-VOTING METHODS (TEST SET) " + "=" * 20 + "\n")
    for subset_name, subset_summary in final_results_majority.items():
        print_markdown_table(subset_name, subset_summary, MAJORITY_METHODS)


if __name__ == "__main__":
    main()
