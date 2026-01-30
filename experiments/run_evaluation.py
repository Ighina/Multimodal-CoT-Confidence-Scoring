"""
Evaluation script for comparing different confidence aggregation methods.

This script evaluates Chain-of-Thought (CoT) results by comparing various
confidence aggregation methods using different score combinations, including
a majority vote baseline that uses the percentage of correct chains per group.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.metrics import evaluate_confidence_scores, compare_methods


def load_json(file_path: str) -> List:
    """Load JSON file."""
    with open(file_path, "r") as f:
        return json.load(f)


def extract_labels(cots_data: List[List[Dict]], n_chains: int, indices: List[int] = None) -> np.ndarray:
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
        "cross_modal_min_coherence": [],
        "confidence": [],
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
            example_dict["cross_modal_min_coherence"].append(
                score_dict.get("cross_modal", {}).get("min_step_coherence", 0.0)
            )
            example_dict["confidence"].append(score_dict.get("confidence", 0.0))

        # Pad if needed
        for key in example_dict:
            while len(example_dict[key]) < n_chains:
                example_dict[key].append(0.0)

        # Append to arrays
        for key in score_arrays:
            score_arrays[key].append(example_dict[key])

    # Convert to numpy arrays
    return {key: np.array(val) for key, val in score_arrays.items()}


def create_aggregation_methods(
    score_arrays: Dict[str, np.ndarray], labels: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Create various confidence aggregation methods.

    Args:
        score_arrays: Dictionary of individual score arrays
        labels: Binary labels array of shape (num_examples, n_chains)

    Returns:
        Dictionary mapping method names to confidence arrays
    """
    methods = {}

    # Majority vote baseline - percentage of correct chains per group
    # Correct chains get score = percentage of correct chains
    # Incorrect chains get score = 1 - percentage of correct chains
    # This way, argmax will select correct chains when they're in the majority
    majority_vote_scores = np.mean(labels, axis=1, keepdims=True)  # (num_examples, 1)
    methods["majority_vote"] = np.where(
        labels == 1,
        majority_vote_scores,  # Correct chains get percentage of correct
        1 - majority_vote_scores,  # Incorrect chains get 1 - percentage of correct
    )

    # Individual raw scores
    methods["raw_confidence"] = score_arrays["confidence"]
    methods["internal_overall"] = score_arrays["internal_overall"]
    methods["internal_smoothness"] = score_arrays["internal_smoothness"]
    methods["internal_goal_directedness"] = score_arrays["internal_goal_directedness"]
    methods["internal_semantic_density"] = score_arrays["internal_semantic_density"]
    methods["cross_modal_alignment"] = score_arrays["cross_modal_alignment"]
    methods["cross_modal_min_coherence"] = score_arrays["cross_modal_min_coherence"]

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

    return methods


def shuffle_data(labels: np.ndarray, score_arrays: Dict[str, np.ndarray], seed: int = 42) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Shuffle labels and corresponding scores within each group (example).

    This maintains the correspondence between labels and scores while
    randomizing their order within each example's chains.

    Args:
        labels: Binary labels array of shape (num_examples, n_chains)
        score_arrays: Dictionary of score arrays, each of shape (num_examples, n_chains)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (shuffled_labels, shuffled_score_arrays)
    """
    np.random.seed(seed)

    shuffled_labels = labels.copy()
    shuffled_score_arrays = {key: arr.copy() for key, arr in score_arrays.items()}

    # Shuffle each example (row) independently
    for i in range(labels.shape[0]):
        # Generate a random permutation for this example
        perm = np.random.permutation(labels.shape[1])

        # Apply the same permutation to labels and all score arrays
        shuffled_labels[i] = labels[i, perm]
        for key in shuffled_score_arrays:
            shuffled_score_arrays[key][i] = score_arrays[key][i, perm]

    return shuffled_labels, shuffled_score_arrays


def normalize_confidences(confidences: np.ndarray) -> np.ndarray:
    """
    Normalize confidence scores to [0, 1] range.

    Args:
        confidences: Confidence array

    Returns:
        Normalized confidence array
    """
    min_val = confidences.min()
    max_val = confidences.max()

    if max_val - min_val < 1e-8:
        return np.ones_like(confidences) * 0.5

    return (confidences - min_val) / (max_val - min_val)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate CoT results with different confidence aggregation methods"
    )
    parser.add_argument(
        "--cots_path",
        type=str,
        required=True,
        help="Path to the CoTs JSON file (List of List of dictionaries)",
    )
    parser.add_argument(
        "--scores_path", type=str, required=True, help="Path to the scores JSON file"
    )
    parser.add_argument(
        "--n_chains",
        type=int,
        required=True,
        help="Number of chains to analyze per example",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to save the final JSON results",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize confidence scores to [0, 1] range",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Randomly select N chains using random indices instead of taking the first N (maintains correspondence between labels and scores)",
    )

    args = parser.parse_args()

    print(f"Loading CoTs from: {args.cots_path}")
    cots_data = load_json(args.cots_path)

    print(f"Loading scores from: {args.scores_path}")
    scores_data = load_json(args.scores_path)

    # Generate random indices if shuffle is requested
    random_indices = None
    if args.shuffle:
        print(f"Generating random indices for selecting {args.n_chains} chains")
        np.random.seed(42)
        random_indices = np.random.permutation(
            max(len(example) for example in cots_data)
        )[:args.n_chains].tolist()
        print(f"Selected indices: {random_indices}")

    print(f"Extracting labels (N={args.n_chains} chains per example)")
    labels = extract_labels(cots_data, args.n_chains, indices=random_indices)
    print(f"Labels shape: {labels.shape}")

    print(f"Extracting score arrays")
    score_arrays = extract_score_arrays(scores_data, args.n_chains, indices=random_indices)

    print(f"Creating aggregation methods")
    confidence_methods = create_aggregation_methods(score_arrays, labels)
    print(f"Created {len(confidence_methods)} aggregation methods")

    # Normalize if requested
    if args.normalize:
        print("Normalizing confidence scores")
        confidence_methods = {
            name: normalize_confidences(conf)
            for name, conf in confidence_methods.items()
        }

    # Evaluate each method
    print("\nEvaluating methods...")
    method_results = {}

    for method_name, confidences in confidence_methods.items():
        print(f"  Evaluating: {method_name}")
        try:
            results = evaluate_confidence_scores(confidences, labels)
            method_results[method_name] = results
            print(f"    AUC-ROC: {results['auc_roc']:.4f}, ECE: {results['ece']:.4f}")
        except Exception as e:
            print(f"    ERROR: {e}")
            continue

    # Compare methods
    print("\nComparing methods...")
    comparison = compare_methods(method_results)

    # Convert numpy types to native Python types for JSON serialization
    def convert_to_serializable(obj):
        """Recursively convert numpy types to native Python types."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj

    # Prepare output
    output = {
        "method_results": method_results,
        "comparison": comparison,
        "metadata": {
            "n_chains": args.n_chains,
            "n_examples": len(cots_data),
            "cots_path": args.cots_path,
            "scores_path": args.scores_path,
            "normalized": args.normalize,
            "shuffled": args.shuffle,
        },
    }

    # Convert entire output to serializable format
    output = convert_to_serializable(output)

    # Save results
    print(f"\nSaving results to: {args.output_file}")
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print("\nTop 3 methods by in-group accuracies:")
    for i, method_name in enumerate(comparison["in_group_accuracy_ranking"][:3], 1):
        in_group_acc = method_results[method_name]["in_group_accuracy"]
        print(f"  {i}. {method_name}: {in_group_acc:.4f}")

    print("\nTop 3 methods by AUC-ROC:")
    for i, method_name in enumerate(comparison["auc_roc_ranking"][:3], 1):
        auc_roc = method_results[method_name]["auc_roc"]
        print(f"  {i}. {method_name}: {auc_roc:.4f}")

    print("\nTop 3 methods by ECE (lower is better):")
    for i, method_name in enumerate(comparison["ece_ranking"][:3], 1):
        ece = method_results[method_name]["ece"]
        print(f"  {i}. {method_name}: {ece:.4f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
