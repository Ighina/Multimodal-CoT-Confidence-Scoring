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
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from scipy import stats

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


def run_single_evaluation(
    cots_data: List,
    scores_data: List,
    n_chains: int,
    normalize: bool,
    shuffle: bool,
    seed: int = None,
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

    Returns:
        Tuple of (method_results, comparison)
    """
    # Generate random indices if shuffle is requested
    random_indices = None
    if shuffle:
        if seed is not None:
            np.random.seed(seed)
        random_indices = np.random.permutation(
            max(len(example) for example in cots_data)
        )[:n_chains].tolist()

    # Extract labels and scores
    labels = extract_labels(cots_data, n_chains, indices=random_indices)
    score_arrays = extract_score_arrays(scores_data, n_chains, indices=random_indices)

    # Create aggregation methods
    confidence_methods = create_aggregation_methods(score_arrays, labels)

    # Normalize if requested
    if normalize:
        confidence_methods = {
            name: normalize_confidences(conf)
            for name, conf in confidence_methods.items()
        }

    # Evaluate each method
    method_results = {}
    for method_name, confidences in confidence_methods.items():
        try:
            results = evaluate_confidence_scores(confidences, labels)
            method_results[method_name] = results
        except Exception as e:
            print(f"    ERROR evaluating {method_name}: {e}")
            continue

    # Compare methods
    comparison = compare_methods(method_results)

    return method_results, comparison


def compute_statistical_tests(
    all_results: List[Dict],
    ranking_key: str,
    metric_key: str,
    top_n: int = 3,
) -> Dict:
    """
    Compute statistical tests comparing top methods.

    Args:
        all_results: List of method_results from each iteration
        ranking_key: Key for the ranking (e.g., "in_group_accuracy_ranking")
        metric_key: Key for the metric (e.g., "in_group_accuracy")
        top_n: Number of top methods to test

    Returns:
        Dictionary with statistical test results
    """
    # Get the top methods from the first iteration (as reference)
    first_comparison = compare_methods(all_results[0])
    top_methods = first_comparison[ranking_key][:top_n]

    # Collect metric values for each method across all iterations
    method_values = {method: [] for method in top_methods}
    for results in all_results:
        for method in top_methods:
            if method in results:
                method_values[method].append(results[method][metric_key])

    # Convert to arrays
    method_arrays = {method: np.array(values) for method, values in method_values.items()}

    # Perform pairwise tests (comparing each method to the next in ranking)
    statistical_tests = {}
    for i in range(len(top_methods) - 1):
        method_a = top_methods[i]
        method_b = top_methods[i + 1]

        if method_a in method_arrays and method_b in method_arrays:
            values_a = method_arrays[method_a]
            values_b = method_arrays[method_b]

            # Paired t-test (parametric)
            t_stat, t_pval = stats.ttest_rel(values_a, values_b)

            # Wilcoxon signed-rank test (non-parametric)
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
    """
    Aggregate results from multiple experiments.

    Args:
        all_results: List of method_results from each iteration
        all_comparisons: List of comparison dicts from each iteration

    Returns:
        Dictionary with aggregated results including confidence intervals
    """
    # Get all method names
    all_method_names = set()
    for results in all_results:
        all_method_names.update(results.keys())

    # Collect metric values for each method
    aggregated = {}
    for method_name in all_method_names:
        metrics = {
            "in_group_accuracy": [],
            "auc_roc": [],
            "ece": [],
        }

        for results in all_results:
            if method_name in results:
                metrics["in_group_accuracy"].append(results[method_name]["in_group_accuracy"])
                metrics["auc_roc"].append(results[method_name]["auc_roc"])
                metrics["ece"].append(results[method_name]["ece"])

        # Compute statistics
        aggregated[method_name] = {}
        for metric_name, values in metrics.items():
            if len(values) > 0:
                values_arr = np.array(values)
                mean = np.mean(values_arr)
                std = np.std(values_arr)
                n = len(values_arr)

                # 95% confidence interval using t-distribution
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

    # Aggregate rankings (use mean ranks)
    ranking_keys = ["in_group_accuracy_ranking", "auc_roc_ranking", "ece_ranking"]
    aggregated_rankings = {}

    for ranking_key in ranking_keys:
        # Collect ranks for each method across iterations
        method_ranks = {method: [] for method in all_method_names}

        for comparison in all_comparisons:
            ranking = comparison[ranking_key]
            for rank, method in enumerate(ranking):
                method_ranks[method].append(rank)

        # Compute mean rank and sort
        mean_ranks = {
            method: np.mean(ranks) if len(ranks) > 0 else float('inf')
            for method, ranks in method_ranks.items()
        }
        aggregated_rankings[ranking_key] = sorted(mean_ranks.keys(), key=lambda m: mean_ranks[m])

    # Compute statistical tests for top 3 methods in each category
    statistical_tests = {}
    test_configs = [
        ("in_group_accuracy_ranking", "in_group_accuracy", "in_group_accuracy_tests"),
        ("auc_roc_ranking", "auc_roc", "auc_roc_tests"),
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
    parser.add_argument(
        "--multiple_experiments",
        action="store_true",
        help="Run multiple experiments (requires --shuffle) to create confidence intervals and perform statistical tests",
    )
    parser.add_argument(
        "--multiple_iterations",
        type=int,
        default=1000,
        help="Number of iterations when --multiple_experiments is used (default: 1000)",
    )

    args = parser.parse_args()

    # Validate arguments
    if args.multiple_experiments and not args.shuffle:
        parser.error("--multiple_experiments requires --shuffle to be enabled")

    print(f"Loading CoTs from: {args.cots_path}")
    cots_data = load_json(args.cots_path)

    print(f"Loading scores from: {args.scores_path}")
    scores_data = load_json(args.scores_path)

    # Run evaluation(s)
    if args.multiple_experiments:
        print(f"\nRunning {args.multiple_iterations} experiments with shuffling...")
        all_results = []
        all_comparisons = []

        for i in range(args.multiple_iterations):
            if (i + 1) % 100 == 0:
                print(f"  Completed {i + 1}/{args.multiple_iterations} iterations")

            # Run evaluation with different seed for each iteration
            method_results, comparison = run_single_evaluation(
                cots_data=cots_data,
                scores_data=scores_data,
                n_chains=args.n_chains,
                normalize=args.normalize,
                shuffle=args.shuffle,
                seed=i,  # Different seed for each iteration
            )

            all_results.append(method_results)
            all_comparisons.append(comparison)

        print(f"\nCompleted all {args.multiple_iterations} iterations")
        print("Aggregating results and computing statistics...")

        # Aggregate results
        aggregation = aggregate_multiple_results(all_results, all_comparisons)

        # Use first iteration for detailed results
        method_results = all_results[0]
        comparison = all_comparisons[0]

    else:
        # Run single evaluation
        if args.shuffle:
            print(f"Generating random indices for selecting {args.n_chains} chains")

        print(f"Extracting labels (N={args.n_chains} chains per example)")
        print(f"Creating aggregation methods")
        print("\nEvaluating methods...")

        method_results, comparison = run_single_evaluation(
            cots_data=cots_data,
            scores_data=scores_data,
            n_chains=args.n_chains,
            normalize=args.normalize,
            shuffle=args.shuffle,
            seed=42,
        )

        # Print individual method results
        for method_name in method_results:
            results = method_results[method_name]
            print(f"  {method_name}: AUC-ROC: {results['auc_roc']:.4f}, ECE: {results['ece']:.4f}")

        print("\nComparing methods...")
        aggregation = None

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
            "multiple_experiments": args.multiple_experiments,
            "multiple_iterations": args.multiple_iterations if args.multiple_experiments else 1,
        },
    }

    # Add aggregation results if multiple experiments were run
    if aggregation is not None:
        output["aggregation"] = aggregation

    # Convert entire output to serializable format
    output = convert_to_serializable(output)

    # Save results
    print(f"\nSaving results to: {args.output_file}")
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    # Print results
    if args.multiple_experiments and aggregation is not None:
        print("\n" + "=" * 80)
        print("AGGREGATED RESULTS FROM MULTIPLE EXPERIMENTS")
        print("=" * 80)

        # Print top 3 methods with confidence intervals
        print("\nTop 3 methods by in-group accuracy (with 95% CI):")
        for i, method_name in enumerate(aggregation["aggregated_rankings"]["in_group_accuracy_ranking"][:3], 1):
            metrics = aggregation["aggregated_metrics"][method_name]["in_group_accuracy"]
            print(f"  {i}. {method_name}:")
            print(f"     Mean: {metrics['mean']:.4f} (95% CI: [{metrics['ci_95_lower']:.4f}, {metrics['ci_95_upper']:.4f}])")
            print(f"     Std: {metrics['std']:.4f}")

        print("\nTop 3 methods by AUC-ROC (with 95% CI):")
        for i, method_name in enumerate(aggregation["aggregated_rankings"]["auc_roc_ranking"][:3], 1):
            metrics = aggregation["aggregated_metrics"][method_name]["auc_roc"]
            print(f"  {i}. {method_name}:")
            print(f"     Mean: {metrics['mean']:.4f} (95% CI: [{metrics['ci_95_lower']:.4f}, {metrics['ci_95_upper']:.4f}])")
            print(f"     Std: {metrics['std']:.4f}")

        print("\nTop 3 methods by ECE (lower is better, with 95% CI):")
        for i, method_name in enumerate(aggregation["aggregated_rankings"]["ece_ranking"][:3], 1):
            metrics = aggregation["aggregated_metrics"][method_name]["ece"]
            print(f"  {i}. {method_name}:")
            print(f"     Mean: {metrics['mean']:.4f} (95% CI: [{metrics['ci_95_lower']:.4f}, {metrics['ci_95_upper']:.4f}])")
            print(f"     Std: {metrics['std']:.4f}")

        # Print statistical test results
        print("\n" + "=" * 80)
        print("STATISTICAL TESTS (Comparing Top 3 Methods)")
        print("=" * 80)

        test_categories = [
            ("in_group_accuracy_tests", "In-Group Accuracy"),
            ("auc_roc_tests", "AUC-ROC"),
            ("ece_tests", "ECE"),
        ]

        for test_key, test_name in test_categories:
            tests = aggregation["statistical_tests"][test_key]
            if tests:
                print(f"\n{test_name}:")
                for comparison_key, test_results in tests.items():
                    method_a = test_results["method_a"]
                    method_b = test_results["method_b"]
                    mean_diff = test_results["mean_diff"]
                    t_pval = test_results["paired_t_test"]["p_value"]
                    w_pval = test_results["wilcoxon_test"]["p_value"]

                    print(f"\n  {method_a} vs {method_b}:")
                    print(f"    Mean difference: {mean_diff:.4f}")
                    print(f"    Paired t-test: p={t_pval:.4f} {'***' if t_pval < 0.001 else '**' if t_pval < 0.01 else '*' if t_pval < 0.05 else 'n.s.'}")
                    print(f"    Wilcoxon test: p={w_pval:.4f} {'***' if w_pval < 0.001 else '**' if w_pval < 0.01 else '*' if w_pval < 0.05 else 'n.s.'}")

                    if t_pval < 0.05:
                        direction = "better" if mean_diff > 0 else "worse"
                        print(f"    Result: {method_a} is significantly {direction} than {method_b}")
                    else:
                        print(f"    Result: No significant difference")

    else:
        print("\nTop 3 methods by in-group accuracy:")
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
