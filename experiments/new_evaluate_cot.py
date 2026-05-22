"""
Evaluation script for comparing different confidence aggregation methods.

This script evaluates Chain-of-Thought (CoT) results by comparing various
confidence aggregation methods using different score combinations, including
a majority vote baseline that uses the percentage of correct chains per group.

Features:
- Single evaluation: Run evaluation once with optional shuffling
- Multiple experiments: Run multiple iterations with shuffling to create confidence
  intervals and perform statistical tests comparing the top 3 methods in each category.
  Use --multiple_experiments with --shuffle and optionally --multiple_iterations to
  control the number of iterations (default: 1000).
- Subset evaluation: Metrics are computed separately for each subset defined by the
  "subset_name" column in the meituan-longcat/UNO-bench validation split, in addition to
  the overall dataset metrics.
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
from scipy import stats
from datasets import load_dataset

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.metrics import evaluate_confidence_scores, compare_methods
from src.coherence.expectation_maximization import compute_em_fused_embeddings_only


def load_json(file_path: str) -> List:
    """Load JSON file."""
    with open(file_path, "r") as f:
        return json.load(f)


def extract_final_answer(text: str) -> Optional[str]:
    """
    Extract the answer letter (A, B, C, or D) from a final_answer string.

    Strategy:
      1. Look for an isolated letter A-D (word-boundary or surrounded by
         punctuation / whitespace).
      2. If exactly one such letter is found, return it.
      3. If multiple are found, return the *last* one (often the concluded
         answer in a chain-of-thought string).
      4. If none is found, return None so callers can treat this chain as
         having a unique, unmatched answer.

    Args:
        text: The raw final_answer string from a chain.

    Returns:
        One of "A", "B", "C", "D", or None.
    """
    if not isinstance(text, str) or not text.strip():
        return None

    # Match stand-alone A/B/C/D: preceded and followed by non-alpha characters
    # (start/end of string, whitespace, punctuation, parentheses, etc.)
    matches = re.findall(r"(?<![A-Za-z])([A-D])(?![A-Za-z])", text.upper())

    if not matches:
        return None

    # Return the last match — in a CoT string the final stated answer tends to
    # appear at the end, after the reasoning.
    return matches[-1]


def extract_answer_arrays(
    cots_data: List[List[Dict]], n_chains: int, indices: List[int] = None
) -> np.ndarray:
    """
    Extract final answers from CoTs data as an object array of strings / None.

    Args:
        cots_data: List of lists of chain dictionaries.
        n_chains: Number of chains to consider per example.
        indices: Optional list of chain indices to select. Defaults to first n_chains.

    Returns:
        numpy object array of shape (num_examples, n_chains) containing
        "A", "B", "C", "D", or None for each chain.
    """
    answer_rows = []

    for example_chains in cots_data:
        if indices is not None:
            selected = [idx for idx in indices if idx < len(example_chains)]
        else:
            selected = list(range(min(n_chains, len(example_chains))))

        row = []
        for idx in selected:
            raw = example_chains[idx].get("final_answer", "")
            row.append(extract_final_answer(raw))

        # Pad with None if fewer chains than requested
        while len(row) < n_chains:
            row.append(None)

        answer_rows.append(row)

    return np.array(answer_rows, dtype=object)


def extract_labels(
    cots_data: List[List[Dict]], n_chains: int, indices: List[int] = None
) -> np.ndarray:
    """
    Extract labels from CoTs data.

    Args:
        cots_data: List of lists of dictionaries with "correct" key
        n_chains: Number of chains to consider per example
        indices: Optional list of indices to select specific chains. If None, selects first n_chains.

    Returns:
        numpy array of shape (num_examples, n_chains) with binary labels
    """
    labels = []

    for example_chains in cots_data:
        example_labels = []

        # Use provided indices or default to first n_chains
        if indices is not None:
            selected_indices = [idx for idx in indices if idx < len(example_chains)]
        else:
            selected_indices = list(range(min(n_chains, len(example_chains))))

        for idx in selected_indices:
            # Map True -> 1, False -> 0
            label = 1 if example_chains[idx]["metadata"].get("correct", False) else 0
            example_labels.append(label)

        # Pad if needed
        while len(example_labels) < n_chains:
            example_labels.append(0)

        labels.append(example_labels)

    return np.array(labels)

def extract_score_arrays(
    scores_data: List[List[Dict]], n_chains: int, indices: List[int] = None
) -> Dict[str, np.ndarray]:
    """
    Extract various score arrays from the scores data.

    Args:
        scores_data: List of lists of score dictionaries
        n_chains: Number of chains to consider per example
        indices: Optional list of indices to select specific chains. If None, selects first n_chains.

    Returns:
        Dictionary mapping score names to numpy arrays of shape (num_examples, n_chains)
    """
    score_arrays = {
        "internal_overall": [],
        "internal_smoothness": [],
        "internal_goal_directedness": [],
        "internal_semantic_density": [],
        "cross_modal_alignment": [],
        "cross_modal_alignment_max": [],
        "cross_modal_geometric_mean": [],
        "cross_modal_weighted_entropy": [],
        "cross_modal_variance_penalised": [],
        "cross_modal_weighted_alignment": [],
        "cross_modal_eb_variance_penalised": [],
        "cross_modal_optimal_transport": [],
        "cross_modal_ot_eb_variance_penalised": [],
        "cross_modal_entropy_no_temperature": [],
        "cross_modal_entropy_gated_routing": [],
        "cross_modal_max_coherence": [],
        "cross_modal_mean_coherence": [],
        "cross_modal_std_coherence": [],
        "cross_modal_min_coherence": [],
        "nli_overall": [],
        "nli_cumulative_step": [],
        "nli_goal": [],
        "confidence": [],
        "umpire": [],
        # Additional per-chain baselines
        "baseline_self_assessed_consistency": [],
        "baseline_pure_consistency": [],
        "baseline_correctness_vote": [],
        "baseline_self_assessed": [],
        "baseline_inverse_length": [],
        # Cross-modal pool (pool_grounding)
        "pool_grounding_composite": [],
        "pool_grounding_contrastive_z": [],
        "pool_grounding_eb_shrunk": [],
        "pool_grounding_entropy_gated": [],
        "pool_grounding_eb_penalised": [],
        "pool_grounding_pool_entropy": [],
        # Cross-modal pool — answer agreement signals
        "pool_grounding_answer_agreement_raw": [],
        "pool_grounding_answer_agreement_eb": [],
        "pool_grounding_answer_agreement_z": [],
        # Internal pool (pool_coherence)
        "pool_coherence_composite": [],
        "pool_coherence_contrastive_z": [],
        "pool_coherence_eb_shrunk": [],
        "pool_coherence_smoothness": [],
        "pool_coherence_goal_directedness": [],
        "pool_coherence_semantic_density": [],
        "pool_coherence_absolute_composite": [],
        # Internal pool — answer agreement signals
        "pool_coherence_answer_agreement_raw": [],
        "pool_coherence_answer_agreement_eb": [],
        "pool_coherence_answer_agreement_z": [],
    }

    for example_scores in scores_data:
        example_dict = {key: [] for key in score_arrays.keys()}

        # Use provided indices or default to first n_chains
        if indices is not None:
            selected_indices = [idx for idx in indices if idx < len(example_scores)]
        else:
            selected_indices = list(range(min(n_chains, len(example_scores))))

        for idx in selected_indices:
            score_dict = example_scores[idx]
            example_dict["internal_overall"].append(
                score_dict.get("internal", {}).get("overall", 0.0)
            )
            example_dict["internal_smoothness"].append(
                score_dict.get("internal", {}).get("smoothness", 0.0)
            )
            example_dict["internal_goal_directedness"].append(
                score_dict.get("internal", {}).get("goal_directedness", 0.0)
            )
            example_dict["internal_semantic_density"].append(
                score_dict.get("internal", {}).get("semantic_density", 0.0)
            )
            example_dict["cross_modal_alignment"].append(
                score_dict.get("cross_modal", {}).get("alignment", 0.0)
            )
            example_dict["cross_modal_alignment_max"].append(
                score_dict.get("cross_modal", {}).get("overall", 0.0)
            )
            example_dict["cross_modal_geometric_mean"].append(
                score_dict.get("cross_modal", {}).get("geometric_mean", 0.0)
            )
            example_dict["cross_modal_variance_penalised"].append(
                score_dict.get("cross_modal", {}).get("variance_penalised", 0.0)
            )
            example_dict["cross_modal_weighted_entropy"].append(
                score_dict.get("cross_modal", {}).get("entropy_weighted_alignment", 0.0)
            )
            example_dict["cross_modal_weighted_alignment"].append(
                score_dict.get("cross_modal", {}).get("weighted_alignment", 0.0)
            )
            example_dict["cross_modal_eb_variance_penalised"].append(
                score_dict.get("cross_modal", {}).get("eb_variance_penalised", 0.0)
            )
            example_dict["cross_modal_optimal_transport"].append(
                score_dict.get("cross_modal", {}).get(
                    "optimal_transport_alignment", 0.0
                )
            )
            example_dict["cross_modal_ot_eb_variance_penalised"].append(
                score_dict.get("cross_modal", {}).get("ot_eb_variance_penalised", 0.0)
            )
            example_dict["cross_modal_entropy_no_temperature"].append(
                score_dict.get("cross_modal", {}).get(
                    "entropy_no_temperature_alignment", 0.0
                )
            )
            example_dict["cross_modal_entropy_gated_routing"].append(
                score_dict.get("cross_modal", {}).get("entropy_gated_routing", 0.0)
            )
            example_dict["cross_modal_max_coherence"].append(
                score_dict.get("cross_modal", {}).get("max_step_coherence", 0.0)
            )
            example_dict["cross_modal_mean_coherence"].append(
                score_dict.get("cross_modal", {}).get("mean_step_coherence", 0.0)
            )
            example_dict["cross_modal_std_coherence"].append(
                score_dict.get("cross_modal", {}).get("std_step_coherence", 0.0)
            )
            example_dict["cross_modal_min_coherence"].append(
                score_dict.get("cross_modal", {}).get("min_step_coherence", 0.0)
            )
            example_dict["nli_overall"].append(
                score_dict.get("nli", {}).get("overall", 0.0)
            )
            example_dict["nli_cumulative_step"].append(
                score_dict.get("nli", {}).get("cumulative_step_nli", 0.0)
            )
            example_dict["nli_goal"].append(
                score_dict.get("nli", {}).get("goal_nli", 0.0)
            )
            example_dict["confidence"].append(score_dict.get("confidence", 0.0))
            example_dict["umpire"].append(
                score_dict.get("baseline", {}).get("umpire", 0.0)
            )
            baseline = score_dict.get("baseline", {}) or {}
            example_dict["baseline_self_assessed_consistency"].append(
                baseline.get("self_assessed-consistency", 0.0)
            )
            example_dict["baseline_pure_consistency"].append(
                baseline.get("pure-consistency", 0.0)
            )
            example_dict["baseline_correctness_vote"].append(
                baseline.get("correctness_vote", 0.0)
            )
            example_dict["baseline_self_assessed"].append(
                baseline.get("self_assessed", 0.0)
            )
            example_dict["baseline_inverse_length"].append(
                baseline.get("inverse_length", 0.0)
            )

            # ----------------------------------------------------------------
            # Pool-grounding scores
            # ----------------------------------------------------------------
            pg = score_dict.get("pool_grounding", {}) or {}
            example_dict["pool_grounding_composite"].append(
                pg.get("composite_score", 0.0)
            )
            example_dict["pool_grounding_contrastive_z"].append(
                pg.get("contrastive_z_score", 0.0)
            )
            example_dict["pool_grounding_eb_shrunk"].append(
                pg.get("pool_eb_shrunk", 0.0)
            )
            example_dict["pool_grounding_entropy_gated"].append(
                pg.get("absolute_entropy_gated", 0.0)
            )
            example_dict["pool_grounding_eb_penalised"].append(
                pg.get("absolute_eb_penalised", 0.0)
            )
            example_dict["pool_grounding_pool_entropy"].append(
                pg.get("pool_entropy_gated", 0.0)
            )
            # Answer-agreement signals emitted by CandidatePoolCoherenceMetric
            example_dict["pool_grounding_answer_agreement_raw"].append(
                pg.get("answer_agreement_raw", 0.0)
            )
            example_dict["pool_grounding_answer_agreement_eb"].append(
                pg.get("answer_agreement_eb", 0.0)
            )
            example_dict["pool_grounding_answer_agreement_z"].append(
                pg.get("answer_agreement_z", 0.0)
            )

            # ----------------------------------------------------------------
            # Pool-coherence scores
            # ----------------------------------------------------------------
            pc = score_dict.get("pool_coherence", {}) or {}
            example_dict["pool_coherence_composite"].append(
                pc.get("composite_score", 0.0)
            )
            example_dict["pool_coherence_contrastive_z"].append(
                pc.get("contrastive_z_score", 0.0)
            )
            example_dict["pool_coherence_eb_shrunk"].append(
                pc.get("pool_eb_shrunk", 0.0)
            )
            example_dict["pool_coherence_smoothness"].append(
                pc.get("absolute_smoothness", 0.0)
            )
            example_dict["pool_coherence_goal_directedness"].append(
                pc.get("absolute_goal_directedness", 0.0)
            )
            example_dict["pool_coherence_semantic_density"].append(
                pc.get("absolute_semantic_density", 0.0)
            )
            example_dict["pool_coherence_absolute_composite"].append(
                pc.get("absolute_composite", 0.0)
            )
            # Answer-agreement signals emitted by CandidatePoolInternalCoherenceMetric
            example_dict["pool_coherence_answer_agreement_raw"].append(
                pc.get("answer_agreement_raw", 0.0)
            )
            example_dict["pool_coherence_answer_agreement_eb"].append(
                pc.get("answer_agreement_eb", 0.0)
            )
            example_dict["pool_coherence_answer_agreement_z"].append(
                pc.get("answer_agreement_z", 0.0)
            )

        # Pad if needed
        for key in example_dict:
            while len(example_dict[key]) < n_chains:
                example_dict[key].append(0.0)

        # Append to arrays
        for key in score_arrays:
            score_arrays[key].append(example_dict[key])

    # Convert to numpy arrays
    return {key: np.array(val) for key, val in score_arrays.items()}


# ---------------------------------------------------------------------------
# Consensus-boosted aggregation helpers
# ---------------------------------------------------------------------------


def build_consensus_multipliers(answers: np.ndarray) -> np.ndarray:
    """
    For each example, compute a per-chain multiplier that reflects the
    fraction of the other chains in the same example that share the same
    final answer (i.e. a soft majority-vote weight).

    Chains whose answer could not be parsed (None) are treated as unique —
    they match no other chain, so their multiplier equals 1/n_chains.

    Args:
        answers: object array of shape (n_examples, n_chains) containing
                 "A"/"B"/"C"/"D" or None.

    Returns:
        float array of shape (n_examples, n_chains) with values in (0, 1].
        A chain that is the sole supporter of its answer gets 1/n_chains;
        a chain that is agreed upon by all chains gets 1.0.
    """
    n_examples, n_chains = answers.shape
    multipliers = np.zeros((n_examples, n_chains), dtype=float)

    for i in range(n_examples):
        row = answers[i]
        for j in range(n_chains):
            ans = row[j]
            if ans is None:
                # Treat as unique — no agreement
                multipliers[i, j] = 1.0 / n_chains
            else:
                # Count how many chains in this example share the same answer
                count = np.sum(row == ans)
                multipliers[i, j] = count / n_chains

    return multipliers


def build_plurality_mask(answers: np.ndarray) -> np.ndarray:
    """
    For each example, identify which answer letter has the most votes
    (plurality / majority) and return a binary mask marking those chains.

    Ties are broken by keeping all tied answers.  Chains with None answers
    are never included in the plurality group.

    Args:
        answers: object array of shape (n_examples, n_chains).

    Returns:
        bool array of shape (n_examples, n_chains); True where the chain
        belongs to the plurality answer group.
    """
    n_examples, n_chains = answers.shape
    mask = np.zeros((n_examples, n_chains), dtype=bool)

    for i in range(n_examples):
        row = answers[i]
        counts: Dict[str, int] = {}
        for ans in row:
            if ans is not None:
                counts[ans] = counts.get(ans, 0) + 1

        if not counts:
            # All None — no plurality possible; include everything
            mask[i, :] = True
            continue

        max_count = max(counts.values())
        plurality_answers = {a for a, c in counts.items() if c == max_count}

        for j in range(n_chains):
            if row[j] in plurality_answers:
                mask[i, j] = True

    return mask


def add_consensus_methods(
    methods: Dict[str, np.ndarray],
    score_arrays: Dict[str, np.ndarray],
    answers: np.ndarray,
) -> None:
    """
    Extend *methods* in-place with consensus-boosted variants.

    Two families are added for a selection of base score arrays:

    1. ``<base>_x_consensus_soft``
       Each chain's score is multiplied by its consensus multiplier
       (fraction of chains that share the same answer).  Chains that
       are isolated (or unparseable) receive a low weight; chains that
       belong to the majority receive a high weight.

    2. ``<base>_x_consensus_hard``
       Only chains belonging to the plurality-answer group retain their
       original score; all other chains are set to 0.  This is a hard
       filter: only the most-agreed-upon answer competes.

    3. ``consensus_soft_mean`` / ``consensus_hard_mean``
       The consensus multipliers / mask applied to the mean of *all*
       base scores, providing a single aggregate consensus signal.

    Additionally, ``majority_vote_x_<base>`` methods are added: these
    take the standard majority-vote score (fraction of same-answer chains)
    and multiply it element-wise by the normalised base score, rewarding
    chains that both belong to the consensus *and* have high individual
    quality.

    Args:
        methods: dict to extend in-place (already contains non-consensus methods).
        score_arrays: raw score arrays from ``extract_score_arrays``.
        answers: object array (n_examples, n_chains) from ``extract_answer_arrays``.
    """
    consensus_mult = build_consensus_multipliers(answers)  # (n_examples, n_chains)
    plurality_mask = build_plurality_mask(answers)  # (n_examples, n_chains)
    plurality_float = plurality_mask.astype(float)

    # Base scores to pair with consensus signals
    base_score_keys = [
        "confidence",
        "internal_overall",
        "cross_modal_alignment",
        "cross_modal_weighted_entropy",
        "cross_modal_geometric_mean",
        "cross_modal_variance_penalised",
        "cross_modal_weighted_alignment",
        "cross_modal_eb_variance_penalised",
        "cross_modal_optimal_transport",
        "cross_modal_ot_eb_variance_penalised",
        "cross_modal_entropy_no_temperature",
        "cross_modal_entropy_gated_routing",
        "umpire",
        "umpire_normalized",
        "mean_all",
        "weighted_50_50",
        "mean_internal",
        "mean_cross_modal",
        # Pool-level scores — now included since answer-agreement makes
        # these competitive alongside the token-frequency baseline
        "pool_grounding_composite",
        "pool_grounding_contrastive_z",
        "pool_grounding_eb_shrunk",
        "pool_grounding_entropy_gated",
        "pool_grounding_eb_penalised",
        "pool_grounding_pool_entropy",
        "pool_grounding_answer_agreement_raw",
        "pool_grounding_answer_agreement_eb",
        "pool_coherence_composite",
        "pool_coherence_contrastive_z",
        "pool_coherence_eb_shrunk",
        "pool_coherence_answer_agreement_raw",
        "pool_coherence_answer_agreement_eb",
        "mean_pool",
        "pool_x_internal_x_crossmodal",
        # Fused agreement methods
        "pool_grounding_agreement_x_composite",
        "pool_coherence_agreement_x_composite",
        "mean_pool_agreement",
    ]

    def _get(key: str) -> np.ndarray:
        """Retrieve from score_arrays or pre-built methods dict."""
        if key in score_arrays:
            return score_arrays[key]
        return methods[key]

    for key in base_score_keys:
        try:
            base = _get(key)
        except KeyError:
            continue  # Score not available in this run; skip silently

        # 1. Soft consensus: scale each score by fraction of agreeing chains
        methods[f"{key}_x_consensus_soft"] = base * consensus_mult

        # 2. Hard consensus: zero out chains not in the plurality group
        methods[f"{key}_x_consensus_hard"] = base * plurality_float

        # 3. Majority-vote quality: consensus fraction × normalised base score
        methods[f"majority_vote_x_{key}"] = consensus_mult * base

    # 4. Aggregate consensus signals across all base scores
    all_base = np.stack(
        [_get(k) for k in base_score_keys if k in score_arrays or k in methods], axis=0
    )
    mean_all_base = np.mean(all_base, axis=0)  # (n_examples, n_chains)

    methods["consensus_soft_mean"] = mean_all_base * consensus_mult
    methods["consensus_hard_mean"] = mean_all_base * plurality_float

    # 5. Pure answer-agreement score (no base quality signal) — equivalent
    #    to a soft majority vote that doesn't use labels.
    methods["answer_agreement_only"] = consensus_mult

    # 6. Plurality vote (hard, uniform within the plurality group)
    methods["plurality_vote"] = plurality_float / np.maximum(
        plurality_float.sum(axis=1, keepdims=True), 1.0
    )


# ---------------------------------------------------------------------------


def create_aggregation_methods(
    score_arrays: Dict[str, np.ndarray],
    labels: np.ndarray,
    answers: np.ndarray = None,
    em_methods: List[str] = None
) -> Dict[str, np.ndarray]:
    """
    Create various confidence aggregation methods.

    Args:
        score_arrays: Dictionary of individual score arrays
        labels: Binary labels array of shape (num_examples, n_chains)
        answers: Optional object array (num_examples, n_chains) of extracted
                 answer letters ("A"/"B"/"C"/"D" or None).  When provided,
                 consensus-boosted methods are added.
        em_methods: If included, it pass the relative methods to be aggregated with 
                    Expectation Maximization

    Returns:
        Dictionary mapping method names to confidence arrays
    """
    methods = {}

    # Random baseline - randomly select one chain per example
    n_examples, n_chains_count = labels.shape
    random_scores = np.random.uniform(0, 1, size=(n_examples, n_chains_count))
    methods["random"] = random_scores

    # Majority vote baseline - percentage of correct chains per group
    majority_vote_scores = np.mean(labels, axis=1, keepdims=True)  # (num_examples, 1)
    methods["majority_vote"] = np.where(
        labels == 1,
        majority_vote_scores,
        1 - majority_vote_scores,
    )

    # Individual raw scores
    methods["raw_confidence"] = score_arrays["confidence"]
    methods["internal_overall"] = score_arrays["internal_overall"]
    methods["internal_smoothness"] = score_arrays["internal_smoothness"]
    methods["internal_goal_directedness"] = score_arrays["internal_goal_directedness"]
    methods["internal_semantic_density"] = score_arrays["internal_semantic_density"]
    methods["cross_modal_alignment"] = score_arrays["cross_modal_alignment"]
    methods["cross_modal_min_coherence"] = score_arrays["cross_modal_min_coherence"]
    try:
        methods["cross_modal_alignment_max"] = score_arrays["cross_modal_alignment_max"]
        methods["cross_modal_weighted_entropy"] = score_arrays[
            "cross_modal_weighted_entropy"
        ]
        methods["cross_modal_geometric_mean"] = score_arrays[
            "cross_modal_geometric_mean"
        ]
        methods["cross_modal_variance_penalised"] = score_arrays[
            "cross_modal_variance_penalised"
        ]
        methods["cross_modal_weighted_alignment"] = score_arrays[
            "cross_modal_weighted_alignment"
        ]
        methods["cross_modal_eb_variance_penalised"] = score_arrays[
            "cross_modal_eb_variance_penalised"
        ]
        methods["cross_modal_optimal_transport"] = score_arrays[
            "cross_modal_optimal_transport"
        ]
        methods["cross_modal_ot_eb_variance_penalised"] = score_arrays[
            "cross_modal_ot_eb_variance_penalised"
        ]
        methods["cross_modal_entropy_no_temperature"] = score_arrays[
            "cross_modal_entropy_no_temperature"
        ]
        methods["cross_modal_entropy_gated_routing"] = score_arrays[
            "cross_modal_entropy_gated_routing"
        ]
        methods["cross_modal_max_coherence"] = score_arrays["cross_modal_max_coherence"]
        methods["cross_modal_mean_coherence"] = score_arrays[
            "cross_modal_mean_coherence"
        ]
        methods["cross_modal_std_coherence"] = score_arrays["cross_modal_std_coherence"]
    except KeyError:
        pass

    # Mean of all internal scores
    methods["mean_internal"] = np.mean(
        [
            score_arrays["internal_overall"],
            score_arrays["internal_smoothness"],
            score_arrays["internal_goal_directedness"],
            score_arrays["internal_semantic_density"],
        ],
        axis=0,
    )

    # Mean of all cross-modal scores
    methods["mean_cross_modal"] = np.mean(
        [
            score_arrays["cross_modal_alignment"],
            score_arrays["cross_modal_min_coherence"],
        ],
        axis=0,
    )

    # Mean of all scores (internal + cross-modal)
    methods["mean_all"] = np.mean(
        [
            score_arrays["internal_overall"],
            score_arrays["internal_smoothness"],
            score_arrays["internal_goal_directedness"],
            score_arrays["internal_semantic_density"],
            score_arrays["cross_modal_alignment"],
            score_arrays["cross_modal_min_coherence"],
        ],
        axis=0,
    )

    # Product of internal and cross-modal (geometric mean-like)
    methods["product_internal_crossmodal"] = (
        score_arrays["internal_overall"] * score_arrays["cross_modal_alignment"]
    )

    # Weighted sum: 70% internal, 30% cross-modal
    methods["weighted_70_30"] = (
        0.7 * score_arrays["internal_overall"]
        + 0.3 * score_arrays["cross_modal_alignment"]
    )

    # Weighted sum: 50% internal, 50% cross-modal
    methods["weighted_50_50"] = (
        0.5 * score_arrays["internal_overall"]
        + 0.5 * score_arrays["cross_modal_alignment"]
    )

    # Weighted sum: 80% internal, 20% cross-modal
    methods["weighted_80_20"] = (
        0.8 * score_arrays["internal_overall"]
        + 0.2 * score_arrays["cross_modal_alignment"]
    )

    # Max of internal and cross-modal
    methods["max_internal_crossmodal"] = np.maximum(
        score_arrays["internal_overall"], score_arrays["cross_modal_alignment"]
    )

    # Min of internal and cross-modal (conservative)
    methods["min_internal_crossmodal"] = np.minimum(
        score_arrays["internal_overall"], score_arrays["cross_modal_alignment"]
    )

    # Weighted combination emphasizing goal directedness
    methods["goal_directedness_focus"] = (
        0.6 * score_arrays["internal_goal_directedness"]
        + 0.3 * score_arrays["internal_smoothness"]
        + 0.1 * score_arrays["cross_modal_alignment"]
    )

    # Weighted combination emphasizing smoothness
    methods["smoothness_focus"] = (
        0.6 * score_arrays["internal_smoothness"]
        + 0.3 * score_arrays["internal_goal_directedness"]
        + 0.1 * score_arrays["cross_modal_alignment"]
    )

    # Harmonic mean of internal and cross-modal
    epsilon = 1e-8
    methods["harmonic_mean"] = 2 / (
        1 / (score_arrays["internal_overall"] + epsilon)
        + 1 / (score_arrays["cross_modal_alignment"] + epsilon)
    )

    # Umpire baseline score
    methods["umpire"] = score_arrays["umpire"]
    methods["umpire_normalized"] = normalize_confidences(score_arrays["umpire"])

    # Additional per-chain baselines (passed through from the score file's
    # `baseline` block).  Exposed both as raw values and normalised to [0, 1]
    # so they line up with the other methods on the same scale.
    for raw_key, method_name in [
        ("baseline_self_assessed_consistency", "self_assessed_consistency"),
        ("baseline_pure_consistency", "pure_consistency"),
        ("baseline_correctness_vote", "correctness_vote_baseline"),
        ("baseline_self_assessed", "self_assessed"),
        ("baseline_inverse_length", "inverse_length"),
    ]:
        if raw_key in score_arrays:
            methods[method_name] = score_arrays[raw_key]
            methods[f"{method_name}_normalized"] = normalize_confidences(
                score_arrays[raw_key]
            )

    # NLI individual scores
    methods["nli_overall"] = score_arrays["nli_overall"]
    methods["nli_cumulative_step"] = score_arrays["nli_cumulative_step"]
    methods["nli_goal"] = score_arrays["nli_goal"]

    # NLI fused with cross-modal alignment (simple mean)
    methods["nli_x_cross_modal"] = np.mean(
        [score_arrays["nli_overall"], score_arrays["cross_modal_alignment"]],
        axis=0,
    )

    # NLI fused with internal overall (simple mean)
    methods["nli_x_internal"] = np.mean(
        [score_arrays["nli_overall"], score_arrays["internal_overall"]],
        axis=0,
    )

    # NLI fused with both cross-modal alignment and internal overall (simple mean)
    methods["nli_x_cross_modal_x_internal"] = np.mean(
        [
            score_arrays["nli_overall"],
            score_arrays["cross_modal_alignment"],
            score_arrays["internal_overall"],
        ],
        axis=0,
    )

    # ------------------------------------------------------------------
    # Pool-level methods (contrastive, multi-candidate)
    # ------------------------------------------------------------------
    try:
        methods["pool_grounding_composite"] = score_arrays["pool_grounding_composite"]
        methods["pool_grounding_contrastive_z"] = score_arrays[
            "pool_grounding_contrastive_z"
        ]
        methods["pool_grounding_eb_shrunk"] = score_arrays["pool_grounding_eb_shrunk"]
        methods["pool_grounding_entropy_gated"] = score_arrays[
            "pool_grounding_entropy_gated"
        ]
        methods["pool_grounding_eb_penalised"] = score_arrays[
            "pool_grounding_eb_penalised"
        ]
        methods["pool_grounding_pool_entropy"] = score_arrays[
            "pool_grounding_pool_entropy"
        ]
        methods["pool_coherence_composite"] = score_arrays["pool_coherence_composite"]
        methods["pool_coherence_contrastive_z"] = score_arrays[
            "pool_coherence_contrastive_z"
        ]
        methods["pool_coherence_eb_shrunk"] = score_arrays["pool_coherence_eb_shrunk"]
        methods["pool_coherence_smoothness"] = score_arrays["pool_coherence_smoothness"]
        methods["pool_coherence_goal_directedness"] = score_arrays[
            "pool_coherence_goal_directedness"
        ]
        methods["pool_coherence_semantic_density"] = score_arrays[
            "pool_coherence_semantic_density"
        ]
        methods["pool_coherence_absolute_composite"] = score_arrays[
            "pool_coherence_absolute_composite"
        ]
        methods["mean_pool"] = np.mean(
            [
                score_arrays["pool_grounding_composite"],
                score_arrays["pool_coherence_composite"],
            ],
            axis=0,
        )
        methods["pool_x_internal_x_crossmodal"] = np.mean(
            [
                score_arrays["pool_grounding_composite"],
                score_arrays["pool_coherence_composite"],
                score_arrays["internal_overall"],
                score_arrays["cross_modal_alignment"],
            ],
            axis=0,
        )
    except KeyError:
        pass

    # ------------------------------------------------------------------
    # Pool answer-agreement methods
    # These use the pairwise answer-embedding similarity scores emitted
    # by CandidatePoolCoherenceMetric / CandidatePoolInternalCoherenceMetric
    # when answer_agreement_weight > 0.
    # ------------------------------------------------------------------
    try:
        # Raw pairwise agreement scores (soft majority vote in embedding space)
        methods["pool_grounding_answer_agreement_raw"] = score_arrays[
            "pool_grounding_answer_agreement_raw"
        ]
        methods["pool_grounding_answer_agreement_eb"] = score_arrays[
            "pool_grounding_answer_agreement_eb"
        ]
        methods["pool_grounding_answer_agreement_z"] = score_arrays[
            "pool_grounding_answer_agreement_z"
        ]
        methods["pool_coherence_answer_agreement_raw"] = score_arrays[
            "pool_coherence_answer_agreement_raw"
        ]
        methods["pool_coherence_answer_agreement_eb"] = score_arrays[
            "pool_coherence_answer_agreement_eb"
        ]
        methods["pool_coherence_answer_agreement_z"] = score_arrays[
            "pool_coherence_answer_agreement_z"
        ]

        # Agreement × composite: rewards chains that are both highly grounded
        # *and* agree with the pool majority — the core hypothesis from the
        # observation that answer agreement outperforms coherence metrics alone.
        methods["pool_grounding_agreement_x_composite"] = (
            score_arrays["pool_grounding_answer_agreement_eb"]
            * score_arrays["pool_grounding_composite"]
        )
        methods["pool_coherence_agreement_x_composite"] = (
            score_arrays["pool_coherence_answer_agreement_eb"]
            * score_arrays["pool_coherence_composite"]
        )

        # Three-way mean: grounding agreement + coherence agreement + composite
        # Blends the majority-vote signal from both pool metrics with their
        # composite quality scores in a single balanced aggregate.
        methods["mean_pool_agreement"] = np.mean(
            [
                score_arrays["pool_grounding_answer_agreement_eb"],
                score_arrays["pool_coherence_answer_agreement_eb"],
                score_arrays["pool_grounding_composite"],
                score_arrays["pool_coherence_composite"],
            ],
            axis=0,
        )

        # Agreement-only mean across both pool metrics (label-free majority vote
        # expressed purely in answer-embedding space, without any coherence signal).
        # Useful as a direct ablation against the token-frequency majority_vote.
        methods["pool_answer_agreement_mean"] = np.mean(
            [
                score_arrays["pool_grounding_answer_agreement_eb"],
                score_arrays["pool_coherence_answer_agreement_eb"],
            ],
            axis=0,
        )

        # EB z-score fusion: uses the z-normalised agreement (which is on the
        # same scale as contrastive_z) for a cleaner linear combination.
        methods["pool_grounding_agreement_z_x_composite"] = (
            score_arrays["pool_grounding_answer_agreement_z"]
            * score_arrays["pool_grounding_composite"]
        )
        methods["pool_coherence_agreement_z_x_composite"] = (
            score_arrays["pool_coherence_answer_agreement_z"]
            * score_arrays["pool_coherence_composite"]
        )

        # Full fusion: agreement (eb) + composite + internal + cross-modal
        # Tests whether the agreement signal complements the existing metrics
        # beyond what mean_pool_x_internal_x_crossmodal already captures.
        methods["pool_agreement_x_internal_x_crossmodal"] = np.mean(
            [
                score_arrays["pool_grounding_answer_agreement_eb"],
                score_arrays["pool_coherence_answer_agreement_eb"],
                score_arrays["pool_grounding_composite"],
                score_arrays["pool_coherence_composite"],
                score_arrays["internal_overall"],
                score_arrays["cross_modal_alignment"],
            ],
            axis=0,
        )
    except KeyError:
        pass

    # ------------------------------------------------------------------
    # Consensus-boosted methods (answer-agreement based)
    # ------------------------------------------------------------------
    if answers is not None:
        add_consensus_methods(methods, score_arrays, answers)

    # Ensure you are passing orthogonal metrics!
    # E.g., one for cross-modal, one for internal coherence, one for decisiveness.
    if em_methods is not None:
        em_scores = []
        for method in em_methods:
            try:
                assert method in em_methods, "The method defined for Expectation Maximization Fusion is not among the available methods"
            except AssertionError:
                pass
            em_scores.append(methods[method])
        assert len(em_methods)>1, "You passed less than two valid methods for Expectation Maximization Fusion: please check the names you passed are correct!"

    methods["em_embeddings_only"] = compute_em_fused_embeddings_only(embedding_features)

    return methods


def shuffle_data(
    labels: np.ndarray, score_arrays: Dict[str, np.ndarray], seed: int = 42
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Shuffle labels and corresponding scores within each group (example).
    """
    np.random.seed(seed)

    shuffled_labels = labels.copy()
    shuffled_score_arrays = {key: arr.copy() for key, arr in score_arrays.items()}

    for i in range(labels.shape[0]):
        perm = np.random.permutation(labels.shape[1])
        shuffled_labels[i] = labels[i, perm]
        for key in shuffled_score_arrays:
            shuffled_score_arrays[key][i] = score_arrays[key][i, perm]

    return shuffled_labels, shuffled_score_arrays


def normalize_confidences(confidences: np.ndarray) -> np.ndarray:
    """Normalize confidence scores to [0, 1] range."""
    min_val = confidences.min()
    max_val = confidences.max()

    if max_val - min_val < 1e-8:
        return np.ones_like(confidences) * 0.5

    return (confidences - min_val) / (max_val - min_val)


def calibrate_scores_holdout(
    confidence_methods: Dict[str, np.ndarray],
    labels: np.ndarray,
    holdout_frac: float = 0.3,
    calibrator_type: str = "logistic",
    seed: int = 42,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """
    Calibrate each method's raw scores using a single example-level holdout split.

    A fixed fraction of examples is reserved as the calibration set.  One
    calibrator is fit per method on the (score, label) pairs of that set and
    applied to the remaining evaluation examples.  Because every evaluation
    point is mapped through the *same* calibration curve, the global ranking
    induced by the raw scores is preserved up to a monotonic transform — so
    AUC-ROC / AUC-PR / AUC-ARC are unaffected (modulo ties), and ECE measures
    the actual calibration quality without per-fold scale artefacts.

    Supported calibrator_type values:
      "logistic"  — Platt scaling (L2-regularised logistic regression, C=1)
      "isotonic"  — Isotonic regression (sklearn IsotonicRegression)

    Args:
        confidence_methods: Dict mapping method name -> array (n_examples, n_chains).
        labels:             Binary correctness array (n_examples, n_chains).
        holdout_frac:       Fraction of examples reserved for fitting the calibrator.
        calibrator_type:    Calibration model ("logistic" or "isotonic").
        seed:               Random seed for the calibration / evaluation split.

    Returns:
        Tuple of:
          - calibrated:  Dict[name -> (n_eval, n_chains)] calibrated probabilities,
                         restricted to the evaluation split.
          - eval_labels: (n_eval, n_chains) labels for the evaluation examples.
          - eval_idx:    Indices of the evaluation examples in the original arrays.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.isotonic import IsotonicRegression

    n_examples, n_chains = labels.shape
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_examples)

    cal_size = int(round(holdout_frac * n_examples))
    # Need at least 2 examples on each side for a meaningful fit / evaluation
    cal_size = max(2, min(n_examples - 2, cal_size))
    cal_idx = perm[:cal_size]
    eval_idx = perm[cal_size:]

    cal_labels_flat = labels[cal_idx].flatten()
    eval_labels = labels[eval_idx]
    unique_classes = np.unique(cal_labels_flat)

    calibrated: Dict[str, np.ndarray] = {}
    for name, scores in confidence_methods.items():
        cal_scores = scores[cal_idx].flatten()
        eval_scores = scores[eval_idx].flatten()

        # If calibration labels are all one class the calibrator can't learn;
        # fall back to min-max normalised raw scores.
        if len(unique_classes) < 2:
            lo, hi = cal_scores.min(), cal_scores.max()
            span = hi - lo if hi - lo > 1e-8 else 1.0
            cal_probs = np.clip((eval_scores - lo) / span, 0.0, 1.0)
        else:
            try:
                if calibrator_type == "isotonic":
                    clf = IsotonicRegression(out_of_bounds="clip")
                    clf.fit(cal_scores, cal_labels_flat)
                    cal_probs = clf.predict(eval_scores).astype(float)
                else:  # "logistic" (default / Platt scaling)
                    clf = LogisticRegression(
                        C=1.0, solver="lbfgs", max_iter=1000, random_state=seed
                    )
                    clf.fit(cal_scores.reshape(-1, 1), cal_labels_flat)
                    cal_probs = clf.predict_proba(eval_scores.reshape(-1, 1))[:, 1]
            except Exception:
                lo, hi = cal_scores.min(), cal_scores.max()
                span = hi - lo if hi - lo > 1e-8 else 1.0
                cal_probs = np.clip((eval_scores - lo) / span, 0.0, 1.0)

        calibrated[name] = cal_probs.reshape(len(eval_idx), n_chains)

    return calibrated, eval_labels, eval_idx


def build_subset_index(cots_data: List, original_data) -> Dict[str, List[int]]:
    """
    Build a mapping from subset name to list of indices (positions) in cots_data.

    Each entry in cots_data is a list of chains; the first chain's
    metadata["original_idx"] is used to look up the subset in original_data.

    Args:
        cots_data: List of per-example chain lists
        original_data: HuggingFace dataset with a "subset_name" column

    Returns:
        Dict mapping subset name -> sorted list of cots_data row indices
    """
    subset_to_indices: Dict[str, List[int]] = {}
    for cot_idx, example_chains in enumerate(cots_data):
        original_idx = example_chains[0]["metadata"]["original_idx"]
        subset_name = original_data[original_idx]["subset_name"]
        subset_to_indices.setdefault(subset_name, []).append(cot_idx)
    return subset_to_indices


def filter_data_by_indices(
    cots_data: List, scores_data: List, indices: List[int]
) -> Tuple[List, List]:
    """Return cots_data and scores_data rows restricted to the given indices."""
    return [cots_data[i] for i in indices], [scores_data[i] for i in indices]


def run_single_evaluation(
    cots_data: List,
    scores_data: List,
    n_chains: int,
    normalize: bool,
    shuffle: bool,
    seed: int = None,
    use_max_confidence: bool = False,
    allowed_methods: Optional[List[str]] = None,
    global_max_available=None,
    calibrate: bool = False,
    calibration_holdout: float = 0.3,
    calibration_type: str = "logistic",
    em_methods: List[str] = None
) -> Tuple[Dict, Dict]:
    """
    Run a single evaluation iteration.

    Args:
        cots_data: List of CoT data
        scores_data: List of score data
        n_chains: Number of chains per example
        normalize: Whether to normalize confidence scores
        shuffle: Whether to shuffle the data
        seed: Random seed for reproducibility
        use_max_confidence: If True, evaluate using only the highest-confidence
            chain per example instead of flattening all chains.
        calibrate: If True, fit a single calibrator per method on a holdout
            split and evaluate only on the remaining examples.
        calibration_holdout: Fraction of examples used as the calibration set
            when calibrate=True.
        calibration_type: Calibrator type ("logistic" or "isotonic").
        em_methods: If included, it pass the relative methods to be aggregated with 
                    Expectation Maximization

    Returns:
        Tuple of (method_results, comparison)
    """
    random_indices = None
    if shuffle:
        if seed is not None:
            np.random.seed(seed)

        # Always use the GLOBAL max_available, not the local one
        pool_size = (
            global_max_available
            if global_max_available is not None
            else max(len(example) for example in cots_data)
        )

        # Use sampling WITH replacement (Bootstrapping)
        random_indices = np.random.choice(
            pool_size, size=n_chains, replace=True
        ).tolist()

    labels = extract_labels(cots_data, n_chains, indices=random_indices)
    score_arrays = extract_score_arrays(scores_data, n_chains, indices=random_indices)
    answers = extract_answer_arrays(cots_data, n_chains, indices=random_indices)

    confidence_methods = create_aggregation_methods(
        score_arrays, labels, answers=answers, em_methods=em_methods
    )

    if allowed_methods is not None:
        confidence_methods = {
            name: conf
            for name, conf in confidence_methods.items()
            if name in allowed_methods
        }

        if not confidence_methods:
            raise ValueError(
                "None of the specified --methods matched the generated methods."
            )

    if calibrate:
        confidence_methods, labels, _ = calibrate_scores_holdout(
            confidence_methods,
            labels,
            holdout_frac=calibration_holdout,
            calibrator_type=calibration_type,
            seed=seed if seed is not None else 42,
        )

    if normalize:
        confidence_methods = {
            name: normalize_confidences(conf)
            for name, conf in confidence_methods.items()
        }

    method_results = {}
    for method_name, confidences in confidence_methods.items():
        try:
            results = evaluate_confidence_scores(
                confidences, labels, use_max_confidence=use_max_confidence
            )
            method_results[method_name] = results
        except Exception as e:
            print(f"    ERROR evaluating {method_name}: {e}")
            continue

    comparison = compare_methods(method_results)
    return method_results, comparison


def compute_statistical_tests(
    all_results: List[Dict],
    ranking_key: str,
    metric_key: str,
    top_n: int = 3,
) -> Dict:
    """Compute statistical tests comparing top methods."""
    first_comparison = compare_methods(all_results[0])
    top_methods = first_comparison[ranking_key][:top_n]

    method_values = {method: [] for method in top_methods}
    for results in all_results:
        for method in top_methods:
            if method in results:
                method_values[method].append(results[method][metric_key])

    method_arrays = {
        method: np.array(values) for method, values in method_values.items()
    }

    statistical_tests = {}
    for i in range(len(top_methods) - 1):
        method_a = top_methods[i]
        method_b = top_methods[i + 1]

        if method_a in method_arrays and method_b in method_arrays:
            values_a = method_arrays[method_a]
            values_b = method_arrays[method_b]

            t_stat, t_pval = stats.ttest_rel(values_a, values_b)
            w_stat, w_pval = stats.wilcoxon(values_a, values_b)

            test_key = f"{method_a}_vs_{method_b}"
            statistical_tests[test_key] = {
                "method_a": method_a,
                "method_b": method_b,
                "mean_a": float(np.mean(values_a)),
                "mean_b": float(np.mean(values_b)),
                "std_a": float(np.std(values_a)),
                "std_b": float(np.std(values_b)),
                "mean_diff": float(np.mean(values_a - values_b)),
                "paired_t_test": {
                    "statistic": float(t_stat),
                    "p_value": float(t_pval),
                    "significant_at_0.05": t_pval < 0.05,
                    "significant_at_0.01": t_pval < 0.01,
                },
                "wilcoxon_test": {
                    "statistic": float(w_stat),
                    "p_value": float(w_pval),
                    "significant_at_0.05": w_pval < 0.05,
                    "significant_at_0.01": w_pval < 0.01,
                },
            }

    return statistical_tests


def aggregate_multiple_results(
    all_results: List[Dict],
    all_comparisons: List[Dict],
) -> Dict:
    """Aggregate results from multiple experiments."""
    all_method_names = set()
    for results in all_results:
        all_method_names.update(results.keys())

    aggregated = {}
    for method_name in all_method_names:
        metrics = {
            "in_group_accuracy": [],
            "auc_roc": [],
            "auc_pr": [],
            "auc_arc": [],
            "ece": [],
        }

        for results in all_results:
            if method_name in results:
                metrics["in_group_accuracy"].append(
                    results[method_name]["in_group_accuracy"]
                )
                metrics["auc_roc"].append(results[method_name]["auc_roc"])
                metrics["auc_pr"].append(results[method_name]["auc_pr"])
                metrics["auc_arc"].append(results[method_name]["auc_arc"])
                metrics["ece"].append(results[method_name]["ece"])

        aggregated[method_name] = {}
        for metric_name, values in metrics.items():
            if len(values) > 0:
                values_arr = np.array(values)
                mean = np.mean(values_arr)
                std = np.std(values_arr)
                n = len(values_arr)
                se = std / np.sqrt(n)
                ci_95 = stats.t.interval(0.95, n - 1, loc=mean, scale=se)

                aggregated[method_name][metric_name] = {
                    "mean": float(mean),
                    "std": float(std),
                    "median": float(np.median(values_arr)),
                    "min": float(np.min(values_arr)),
                    "max": float(np.max(values_arr)),
                    "ci_95_lower": float(ci_95[0]),
                    "ci_95_upper": float(ci_95[1]),
                    "n_iterations": n,
                }

    ranking_keys = [
        "in_group_accuracy_ranking",
        "auc_roc_ranking",
        "auc_pr_ranking",
        "auc_arc_ranking",
        "ece_ranking",
    ]
    aggregated_rankings = {}

    for ranking_key in ranking_keys:
        method_ranks = {method: [] for method in all_method_names}
        for comparison in all_comparisons:
            ranking = comparison[ranking_key]
            for rank, method in enumerate(ranking):
                method_ranks[method].append(rank)

        mean_ranks = {
            method: np.mean(ranks) if len(ranks) > 0 else float("inf")
            for method, ranks in method_ranks.items()
        }
        aggregated_rankings[ranking_key] = sorted(
            mean_ranks.keys(), key=lambda m: mean_ranks[m]
        )

    statistical_tests = {}
    test_configs = [
        ("in_group_accuracy_ranking", "in_group_accuracy", "in_group_accuracy_tests"),
        ("auc_roc_ranking", "auc_roc", "auc_roc_tests"),
        ("auc_pr_ranking", "auc_pr", "auc_pr_tests"),
        ("auc_arc_ranking", "auc_arc", "auc_arc_tests"),
        ("ece_ranking", "ece", "ece_tests"),
    ]

    for ranking_key, metric_key, test_name in test_configs:
        statistical_tests[test_name] = compute_statistical_tests(
            all_results, ranking_key, metric_key, top_n=3
        )

    return {
        "aggregated_metrics": aggregated,
        "aggregated_rankings": aggregated_rankings,
        "statistical_tests": statistical_tests,
    }


def run_evaluation_for_split(
    cots_data: List,
    scores_data: List,
    n_chains: int,
    normalize: bool,
    shuffle: bool,
    multiple_experiments: bool,
    multiple_iterations: int,
    label: str,
    use_max_confidence: bool = False,
    allowed_methods: Optional[List[str]] = None,
    global_max_available=None,
    calibrate: bool = False,
    calibration_holdout: float = 0.3,
    calibration_type: str = "logistic",
    em_methods: List[str] = None
) -> Dict:
    """
    Run evaluation (single or multiple experiments) for a given data split and
    return a result dict in the same structure used by the overall evaluation.
    """
    if multiple_experiments:
        all_results = []
        all_comparisons = []

        for i in range(multiple_iterations):
            method_results, comparison = run_single_evaluation(
                cots_data=cots_data,
                scores_data=scores_data,
                n_chains=n_chains,
                normalize=normalize,
                shuffle=shuffle,
                seed=i,
                use_max_confidence=use_max_confidence,
                allowed_methods=allowed_methods,
                global_max_available=global_max_available,
                calibrate=calibrate,
                calibration_holdout=calibration_holdout,
                calibration_type=calibration_type,
                em_methods=em_methods
            )
            all_results.append(method_results)
            all_comparisons.append(comparison)

        aggregation = aggregate_multiple_results(all_results, all_comparisons)
        method_results = all_results[0]
        comparison = all_comparisons[0]

        return {
            "method_results": method_results,
            "comparison": comparison,
            "aggregation": aggregation,
            "n_examples": len(cots_data),
        }
    else:
        method_results, comparison = run_single_evaluation(
            cots_data=cots_data,
            scores_data=scores_data,
            n_chains=n_chains,
            normalize=normalize,
            shuffle=shuffle,
            seed=42,
            use_max_confidence=use_max_confidence,
            global_max_available=global_max_available,
            calibrate=calibrate,
            calibration_holdout=calibration_holdout,
            calibration_type=calibration_type,
            em_methods=em_methods
        )
        return {
            "method_results": method_results,
            "comparison": comparison,
            "n_examples": len(cots_data),
        }


def print_split_summary(split_result: Dict, label: str, multiple_experiments: bool):
    """Print a compact summary for one data split."""
    print(f"\n{'=' * 60}")
    print(f"  {label}  (n={split_result['n_examples']})")
    print(f"{'=' * 60}")

    if multiple_experiments and "aggregation" in split_result:
        agg = split_result["aggregation"]

        print("  Top 3 by in-group accuracy:")
        for i, m in enumerate(
            agg["aggregated_rankings"]["in_group_accuracy_ranking"][:3], 1
        ):
            metrics = agg["aggregated_metrics"][m]["in_group_accuracy"]
            print(
                f"    {i}. {m}: {metrics['mean']:.4f} "
                f"(95% CI [{metrics['ci_95_lower']:.4f}, {metrics['ci_95_upper']:.4f}])"
            )

        print("  Top 3 by AUC-ROC:")
        for i, m in enumerate(agg["aggregated_rankings"]["auc_roc_ranking"][:3], 1):
            metrics = agg["aggregated_metrics"][m]["auc_roc"]
            print(
                f"    {i}. {m}: {metrics['mean']:.4f} "
                f"(95% CI [{metrics['ci_95_lower']:.4f}, {metrics['ci_95_upper']:.4f}])"
            )

        print("  Top 3 by ECE (lower is better):")
        for i, m in enumerate(agg["aggregated_rankings"]["ece_ranking"][:3], 1):
            metrics = agg["aggregated_metrics"][m]["ece"]
            print(
                f"    {i}. {m}: {metrics['mean']:.4f} "
                f"(95% CI [{metrics['ci_95_lower']:.4f}, {metrics['ci_95_upper']:.4f}])"
            )

        if "statistical_tests" in agg:
            print("\n  Statistical Significance (Top Methods):")
            for metric, tests in agg["statistical_tests"].items():
                if not tests:
                    continue
                print(f"    {metric}:")
                for test_name, test_data in tests.items():
                    is_sig = test_data["wilcoxon_test"]["significant_at_0.05"]
                    p_val = test_data["wilcoxon_test"]["p_value"]
                    print(f"      - {test_name}: Significant? {is_sig} (p={p_val:.4f})")
    else:
        method_results = split_result["method_results"]
        comparison = split_result["comparison"]

        print("  Top 3 by in-group accuracy:")
        for i, m in enumerate(comparison["in_group_accuracy_ranking"][:3], 1):
            print(f"    {i}. {m}: {method_results[m]['in_group_accuracy']:.4f}")

        print("  Top 3 by AUC-ROC:")
        for i, m in enumerate(comparison["auc_roc_ranking"][:3], 1):
            print(f"    {i}. {m}: {method_results[m]['auc_roc']:.4f}")

        print("  Top 3 by ECE (lower is better):")
        for i, m in enumerate(comparison["ece_ranking"][:3], 1):
            print(f"    {i}. {m}: {method_results[m]['ece']:.4f}")


def _method_rows(
    method_results: Dict, multiple_experiments: bool, aggregation: Dict = None
) -> List[Dict]:
    """
    Convert a method_results dict (and optional aggregation) into a list of row dicts
    suitable for csv.DictWriter.
    """
    rows = []
    for method, res in method_results.items():
        row = {"method": method}

        if multiple_experiments and aggregation is not None:
            agg_m = aggregation["aggregated_metrics"].get(method, {})
            for metric in ("in_group_accuracy", "auc_roc", "auc_pr", "auc_arc", "ece"):
                m = agg_m.get(metric, {})
                row[f"{metric}_mean"] = round(m.get("mean", float("nan")), 6)
                row[f"{metric}_std"] = round(m.get("std", float("nan")), 6)
                row[f"{metric}_ci_95_lower"] = round(
                    m.get("ci_95_lower", float("nan")), 6
                )
                row[f"{metric}_ci_95_upper"] = round(
                    m.get("ci_95_upper", float("nan")), 6
                )
        else:
            row["in_group_accuracy"] = round(
                res.get("in_group_accuracy", float("nan")), 6
            )
            row["auc_roc"] = round(res.get("auc_roc", float("nan")), 6)
            row["auc_pr"] = round(res.get("auc_pr", float("nan")), 6)
            row["auc_arc"] = round(res.get("auc_arc", float("nan")), 6)
            row["ece"] = round(res.get("ece", float("nan")), 6)

        rows.append(row)

    sort_key = (
        "in_group_accuracy_mean"
        if multiple_experiments and aggregation
        else "in_group_accuracy"
    )
    rows.sort(key=lambda r: r.get(sort_key, float("-inf")), reverse=True)
    return rows


def save_overall_csv(
    output_path: Path,
    overall_result: Dict,
    multiple_experiments: bool,
) -> None:
    """Write one CSV where every row is a method — full dataset metrics."""
    csv_path = output_path.with_name(output_path.stem + "_overall.csv")

    aggregation = overall_result.get("aggregation")
    rows = _method_rows(
        overall_result["method_results"], multiple_experiments, aggregation
    )

    if not rows:
        return

    fieldnames = list(rows[0].keys())

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Overall CSV saved to: {csv_path}")


def save_subsets_csv(
    output_path: Path,
    subset_results: Dict[str, Dict],
    multiple_experiments: bool,
) -> None:
    """Write one CSV where every row is a (subset, method) pair."""
    csv_path = output_path.with_name(output_path.stem + "_subsets.csv")

    all_rows = []
    for subset_name, split_result in sorted(subset_results.items()):
        aggregation = split_result.get("aggregation")
        rows = _method_rows(
            split_result["method_results"], multiple_experiments, aggregation
        )
        for row in rows:
            all_rows.append(
                {
                    "subset_name": subset_name,
                    "n_examples": split_result["n_examples"],
                    **row,
                }
            )

    if not all_rows:
        return

    fieldnames = list(all_rows[0].keys())

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"  Subsets CSV saved to: {csv_path}")


def _build_plot_data(
    split_result: Dict,
    multiple_experiments: bool,
    metric: str,
    lower_is_better: bool,
    top_n: int,
) -> Tuple[List[str], List[float], Optional[np.ndarray]]:
    """Extract sorted method names, values, and optional asymmetric CI error bars."""
    if multiple_experiments and "aggregation" in split_result:
        agg = split_result["aggregation"]["aggregated_metrics"]
        entries = []
        for method, metrics in agg.items():
            if metric in metrics:
                m = metrics[metric]
                entries.append((method, m["mean"], m["ci_95_lower"], m["ci_95_upper"]))
        entries.sort(key=lambda x: x[1], reverse=not lower_is_better)
        entries = entries[:top_n]
        if not entries:
            return [], [], None
        methods = [e[0] for e in entries]
        values = [e[1] for e in entries]
        err_lower = np.maximum(0.0, np.array([e[1] - e[2] for e in entries]))
        err_upper = np.maximum(0.0, np.array([e[3] - e[1] for e in entries]))
        return methods, values, np.array([err_lower, err_upper])
    else:
        method_results = split_result["method_results"]
        entries = []
        for method, results in method_results.items():
            val = results.get(metric)
            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                entries.append((method, float(val)))
        entries.sort(key=lambda x: x[1], reverse=not lower_is_better)
        entries = entries[:top_n]
        if not entries:
            return [], [], None
        methods = [e[0] for e in entries]
        values = [e[1] for e in entries]
        return methods, values, None


def plot_results(
    overall_result: Dict,
    subset_results: Dict[str, Dict],
    multiple_experiments: bool,
    output_path: Path,
    top_n: int = 20,
) -> None:
    """Create horizontal bar charts comparing methods for each split."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  Warning: matplotlib not available — install it to use --plot.")
        return

    metrics_config = [
        ("in_group_accuracy", "In-Group Accuracy", False),
        ("auc_roc", "AUC-ROC", False),
        ("auc_pr", "AUC-PR", False),
        ("auc_arc", "AUC-ARC", False),
        ("ece", "ECE  (lower ↓)", True),
    ]
    n_metrics = len(metrics_config)

    splits = {"OVERALL": overall_result, **subset_results}

    for split_name, split_result in splits.items():
        safe_name = split_name.replace("/", "_").replace(" ", "_").replace(":", "")
        n_examples = split_result.get("n_examples", "?")

        fig_height = max(6, top_n * 0.45 + 2)
        fig_width = n_metrics * 6

        fig, axes = plt.subplots(1, n_metrics, figsize=(fig_width, fig_height))
        if n_metrics == 1:
            axes = [axes]

        ci_note = "  (bars = 95 % CI)" if multiple_experiments else ""
        fig.suptitle(
            f"Method Comparison — {split_name}  (n={n_examples}){ci_note}",
            fontsize=13,
            fontweight="bold",
            y=1.01,
        )

        for ax, (metric, title, lower_is_better) in zip(axes, metrics_config):
            methods, values, xerr = _build_plot_data(
                split_result, multiple_experiments, metric, lower_is_better, top_n
            )

            if not methods:
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                ax.set_title(title, fontsize=10, fontweight="bold")
                continue

            methods = methods[::-1]
            values = values[::-1]
            if xerr is not None:
                xerr = xerr[:, ::-1]

            color = "#d62728" if lower_is_better else "#1f77b4"
            y_pos = range(len(methods))

            if xerr is not None:
                ax.barh(
                    y_pos,
                    values,
                    xerr=xerr,
                    color=color,
                    alpha=0.75,
                    error_kw={"ecolor": "dimgray", "capsize": 3, "linewidth": 1},
                )
            else:
                ax.barh(y_pos, values, color=color, alpha=0.75)

            ax.set_yticks(list(y_pos))
            ax.set_yticklabels(methods, fontsize=7)
            ax.set_title(title, fontsize=10, fontweight="bold")
            ax.set_xlabel("Value", fontsize=9)
            ax.margins(x=0.18)

            x_max = ax.get_xlim()[1]
            nudge = x_max * 0.01
            for i, v in enumerate(values):
                ax.text(v + nudge, i, f"{v:.4f}", va="center", fontsize=6.5)

        plt.tight_layout()
        plot_path = output_path.with_name(output_path.stem + f"_plot_{safe_name}.png")
        plt.savefig(plot_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"  Plot saved to: {plot_path}")


def _get_metric_value(
    split_result: Dict,
    multiple_experiments: bool,
    method: str,
    metric: str,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Return (mean, ci_lower, ci_upper) for a method/metric."""
    if multiple_experiments and "aggregation" in split_result:
        agg = split_result["aggregation"]["aggregated_metrics"]
        if method in agg and metric in agg[method]:
            m = agg[method][metric]
            return m["mean"], m["ci_95_lower"], m["ci_95_upper"]
        return None, None, None
    else:
        results = split_result["method_results"].get(method)
        if results is None:
            return None, None, None
        val = results.get(metric)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return None, None, None
        return float(val), None, None


def _select_methods_for_line_plot(
    n_values: List[int],
    all_n_overall: Dict[int, Dict],
    multiple_experiments: bool,
    top_n: int,
    rank_metric: str = "in_group_accuracy",
) -> List[str]:
    """Pick top methods by average value of rank_metric across all n evaluations."""
    method_values: Dict[str, List[float]] = {}
    for n in n_values:
        overall = all_n_overall[n]
        if multiple_experiments and "aggregation" in overall:
            for method, metrics in overall["aggregation"]["aggregated_metrics"].items():
                if rank_metric in metrics:
                    method_values.setdefault(method, []).append(
                        metrics[rank_metric]["mean"]
                    )
        else:
            for method, results in overall["method_results"].items():
                val = results.get(rank_metric)
                if val is not None:
                    method_values.setdefault(method, []).append(float(val))
    method_mean = {m: float(np.mean(v)) for m, v in method_values.items() if v}
    return sorted(method_mean, key=lambda m: method_mean[m], reverse=True)[:top_n]


def plot_multi_n_results(
    n_values: List[int],
    all_n_overall: Dict[int, Dict],
    all_n_subsets: Dict[int, Dict[str, Dict]],
    multiple_experiments: bool,
    experiment_dir: Path,
    experiment_name: str,
    top_n: int = 10,
    selected_methods: Optional[List[str]] = None,
) -> None:
    """Create one line-plot figure per split across evaluated n values."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.cm import get_cmap
    except ImportError:
        print("  Warning: matplotlib not available — install it to use --plot.")
        return

    metrics_config = [
        ("in_group_accuracy", "In-Group Accuracy", False),
        ("auc_roc", "AUC-ROC", False),
        ("auc_pr", "AUC-PR", False),
        ("auc_arc", "AUC-ARC", False),
        ("ece", "ECE  (lower ↓)", True),
    ]
    n_metrics = len(metrics_config)

    methods_to_plot = (
        selected_methods
        if selected_methods
        else _select_methods_for_line_plot(
            n_values, all_n_overall, multiple_experiments, top_n
        )
    )

    first_n = n_values[0]
    subset_names = sorted(all_n_subsets[first_n].keys())
    splits = [("OVERALL", None)] + [(name, name) for name in subset_names]

    cmap = get_cmap("tab20")
    n_methods = max(len(methods_to_plot), 1)
    colors = [cmap(i / n_methods) for i in range(len(methods_to_plot))]

    ci_note = "  (shaded = 95 % CI)" if multiple_experiments else ""

    for split_label, subset_key in splits:
        safe_name = split_label.replace("/", "_").replace(" ", "_").replace(":", "")

        fig, axes = plt.subplots(1, n_metrics, figsize=(n_metrics * 5, 5))
        if n_metrics == 1:
            axes = [axes]

        fig.suptitle(
            f"{experiment_name} — {split_label}{ci_note}",
            fontsize=13,
            fontweight="bold",
        )

        for ax, (metric, title, _) in zip(axes, metrics_config):
            for method, color in zip(methods_to_plot, colors):
                xs, ys, ci_los, ci_his = [], [], [], []
                for n in n_values:
                    split_result = (
                        all_n_overall[n]
                        if subset_key is None
                        else all_n_subsets[n].get(subset_key)
                    )
                    if split_result is None:
                        continue
                    val, ci_lo, ci_hi = _get_metric_value(
                        split_result, multiple_experiments, method, metric
                    )
                    if val is None:
                        continue
                    xs.append(n)
                    ys.append(val)
                    if ci_lo is not None:
                        ci_los.append(ci_lo)
                        ci_his.append(ci_hi)

                if not xs:
                    continue
                ax.plot(xs, ys, marker="o", color=color, label=method, linewidth=1.5)
                if ci_los and len(ci_los) == len(xs):
                    ax.fill_between(xs, ci_los, ci_his, color=color, alpha=0.15)

            ax.set_title(title, fontsize=10, fontweight="bold")
            ax.set_xlabel("n chains", fontsize=9)
            ax.set_ylabel("Value", fontsize=9)
            ax.set_xticks(n_values)
            ax.grid(True, alpha=0.3)

        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(
                handles,
                labels,
                loc="lower center",
                ncol=min(len(methods_to_plot), 5),
                fontsize=7,
                bbox_to_anchor=(0.5, -0.12),
            )

        plt.tight_layout()
        plot_path = experiment_dir / f"line_plot_{safe_name}.png"
        plt.savefig(plot_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        print(f"  Line plot saved to: {plot_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate CoT results with different confidence aggregation methods"
    )
    parser.add_argument("--cots_path", type=str, required=True)
    parser.add_argument("--scores_path", type=str, required=True)
    parser.add_argument("--n_chains", type=int, default=None)
    parser.add_argument("--n_chains_list", nargs="+", type=int, default=None)
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--multiple_experiments", action="store_true")
    parser.add_argument("--multiple_iterations", type=int, default=1000)
    parser.add_argument("--use_max_confidence", action="store_true")
    parser.add_argument("--methods", nargs="+", type=str, default=None)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--plot_top_n", type=int, default=20)
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Calibrate each method's scores via k-fold cross-validation before evaluation.",
    )
    parser.add_argument(
        "--calibration_holdout",
        type=float,
        default=0.3,
        help=(
            "Fraction of examples held out to fit the calibrator when "
            "--calibrate is enabled (default: 0.3). Metrics are reported on "
            "the remaining 1 - holdout fraction so the calibration curve is "
            "consistent across all evaluation points."
        ),
    )
    parser.add_argument(
        "--calibration_type",
        type=str,
        default="logistic",
        choices=["logistic", "isotonic"],
        help="Calibrator type: 'logistic' (Platt scaling) or 'isotonic' (default: logistic).",
    )
    parser.add_argument(
        "--expectation_maximization_methods",
        nargs="+",
        type=str,
        required=False,
        help="If included, it adds the expectation maximization method fusing the scores from the selected methods"
    )

    args = parser.parse_args()

    if args.multiple_experiments and not args.shuffle:
        parser.error("--multiple_experiments requires --shuffle to be enabled")
    if args.n_chains is None and args.n_chains_list is None:
        parser.error("Either --n_chains or --n_chains_list must be provided.")
    if args.n_chains is not None and args.n_chains_list is not None:
        parser.error("--n_chains and --n_chains_list are mutually exclusive.")
    if args.n_chains_list is not None and args.experiment_name is None:
        parser.error("--n_chains_list requires --experiment_name.")

    print(f"Loading CoTs from: {args.cots_path}")
    cots_data = load_json(args.cots_path)

    print(f"Loading scores from: {args.scores_path}")
    scores_data = load_json(args.scores_path)

    print("Loading original dataset (meituan-longcat/UNO-bench) to extract subsets...")
    original_data = load_dataset("meituan-longcat/UNO-bench")["validation"]

    print("Building subset index from cots metadata...")
    subset_index = build_subset_index(cots_data, original_data)
    subset_names = sorted(subset_index.keys())
    print(f"Found {len(subset_names)} subset(s): {subset_names}")

    global_max_available = max(len(example) for example in cots_data)

    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj

    def _run_subsets(n: int) -> Dict[str, Dict]:
        results: Dict[str, Dict] = {}
        for subset_name in subset_names:
            indices = subset_index[subset_name]
            sub_cots, sub_scores = filter_data_by_indices(
                cots_data, scores_data, indices
            )
            print(
                f"\nRunning evaluation for subset '{subset_name}' "
                f"(n_examples={len(sub_cots)}, n_chains={n})..."
            )
            results[subset_name] = run_evaluation_for_split(
                cots_data=sub_cots,
                scores_data=sub_scores,
                n_chains=n,
                normalize=args.normalize,
                shuffle=args.shuffle,
                multiple_experiments=args.multiple_experiments,
                multiple_iterations=args.multiple_iterations,
                label=subset_name,
                use_max_confidence=args.use_max_confidence,
                allowed_methods=args.methods,
                global_max_available=global_max_available,
                calibrate=args.calibrate,
                calibration_holdout=args.calibration_holdout,
                calibration_type=args.calibration_type,
                em_methods=args.expectation_maximization_methods
            )
            print_split_summary(
                results[subset_name],
                label=subset_name,
                multiple_experiments=args.multiple_experiments,
            )
        return results

    def _assemble_output(n: int, overall_result: Dict, subset_results: Dict) -> Dict:
        output = {
            "method_results": overall_result["method_results"],
            "comparison": overall_result["comparison"],
            "subset_results": subset_results,
            "metadata": {
                "n_chains": n,
                "n_examples": len(cots_data),
                "subsets": subset_names,
                "cots_path": args.cots_path,
                "scores_path": args.scores_path,
                "normalized": args.normalize,
                "shuffled": args.shuffle,
                "multiple_experiments": args.multiple_experiments,
                "multiple_iterations": (
                    args.multiple_iterations if args.multiple_experiments else 1
                ),
                "use_max_confidence": args.use_max_confidence,
                "calibrate": args.calibrate,
                "calibration_holdout": (
                    args.calibration_holdout if args.calibrate else None
                ),
                "calibration_type": args.calibration_type if args.calibrate else None,
            },
        }
        if "aggregation" in overall_result:
            output["aggregation"] = overall_result["aggregation"]
        return convert_to_serializable(output)

    # ------------------------------------------------------------------ #
    # Multi-n evaluation mode                                              #
    # ------------------------------------------------------------------ #
    if args.n_chains_list is not None:
        output_file_path = Path(args.output_file)
        experiment_dir = output_file_path.parent / args.experiment_name
        experiment_dir.mkdir(parents=True, exist_ok=True)

        all_n_overall: Dict[int, Dict] = {}
        all_n_subsets: Dict[int, Dict[str, Dict]] = {}

        for n in args.n_chains_list:
            print(f"\n{'#' * 80}")
            print(f"# Evaluating with n_chains = {n}")
            print(f"{'#' * 80}")

            n_output_path = experiment_dir / f"n_{n}" / output_file_path.name
            n_output_path.parent.mkdir(parents=True, exist_ok=True)

            print("\n" + "=" * 80)
            print(f"Running OVERALL evaluation (n_chains={n})...")
            print("=" * 80)

            n_overall = run_evaluation_for_split(
                cots_data=cots_data,
                scores_data=scores_data,
                n_chains=n,
                normalize=args.normalize,
                shuffle=args.shuffle,
                multiple_experiments=args.multiple_experiments,
                multiple_iterations=args.multiple_iterations,
                label="OVERALL",
                use_max_confidence=args.use_max_confidence,
                allowed_methods=args.methods,
                global_max_available=global_max_available,
                calibrate=args.calibrate,
                calibration_holdout=args.calibration_holdout,
                calibration_type=args.calibration_type,
                em_methods=args.expectation_maximization_methods
            )
            print_split_summary(n_overall, "OVERALL", args.multiple_experiments)

            n_subsets = _run_subsets(n)

            all_n_overall[n] = n_overall
            all_n_subsets[n] = n_subsets

            with open(n_output_path, "w") as f:
                json.dump(_assemble_output(n, n_overall, n_subsets), f, indent=2)
            print(f"  Results saved to: {n_output_path}")

            save_overall_csv(n_output_path, n_overall, args.multiple_experiments)
            save_subsets_csv(n_output_path, n_subsets, args.multiple_experiments)

            if args.plot:
                print(f"  Generating bar charts for n_chains={n}...")
                plot_results(
                    overall_result=n_overall,
                    subset_results=n_subsets,
                    multiple_experiments=args.multiple_experiments,
                    output_path=n_output_path,
                    top_n=args.plot_top_n,
                )

        if args.plot:
            print("\nGenerating multi-n line plots...")
            plot_multi_n_results(
                n_values=args.n_chains_list,
                all_n_overall=all_n_overall,
                all_n_subsets=all_n_subsets,
                multiple_experiments=args.multiple_experiments,
                experiment_dir=experiment_dir,
                experiment_name=args.experiment_name,
                top_n=args.plot_top_n,
                selected_methods=args.methods,
            )

        print("\nDone!")
        return

    # ------------------------------------------------------------------ #
    # Single-n evaluation mode                                             #
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 80)
    print("Running OVERALL evaluation...")
    print("=" * 80)

    overall_result = run_evaluation_for_split(
        cots_data=cots_data,
        scores_data=scores_data,
        n_chains=args.n_chains,
        normalize=args.normalize,
        shuffle=args.shuffle,
        multiple_experiments=args.multiple_experiments,
        multiple_iterations=args.multiple_iterations,
        label="OVERALL",
        use_max_confidence=args.use_max_confidence,
        allowed_methods=args.methods,
        calibrate=args.calibrate,
        calibration_holdout=args.calibration_holdout,
        calibration_type=args.calibration_type,
        em_methods=args.expectation_maximization_methods
    )
    print_split_summary(
        overall_result, label="OVERALL", multiple_experiments=args.multiple_experiments
    )

    subset_results = _run_subsets(args.n_chains)

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving results to: {args.output_file}")
    with open(output_path, "w") as f:
        json.dump(
            _assemble_output(args.n_chains, overall_result, subset_results), f, indent=2
        )

    print("\nGenerating CSV tables...")
    save_overall_csv(output_path, overall_result, args.multiple_experiments)
    save_subsets_csv(output_path, subset_results, args.multiple_experiments)

    if args.plot:
        print("\nGenerating plots...")
        plot_results(
            overall_result=overall_result,
            subset_results=subset_results,
            multiple_experiments=args.multiple_experiments,
            output_path=output_path,
            top_n=args.plot_top_n,
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
