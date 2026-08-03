"""
Evaluation script for WeightedFrequency aggregation method.

This script evaluates the confidence-weighted answer aggregation
and compares it with baseline methods.
"""

import argparse
import json
from pathlib import Path
import torch
import sys
import logging

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.evaluator import ConfidenceEvaluator
from src.coherence import ChainConfidenceScorer
from src.dataset.cot_generator import CoTChain


def setup_logger(log_file=None):
    """Setup logger."""
    logger = logging.getLogger("weighted_frequency_eval")
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def load_experiment_data(
    scores_path: str,
    cots_path: str,
    samples_path: str = None,
    logger=None
):
    """
    Load experiment data from saved files.

    Args:
        scores_path: Path to scores JSON file
        cots_path: Path to CoT chains JSON file
        samples_path: Optional path to samples data (for ground truth)
        logger: Optional logger

    Returns:
        Tuple of (scores, cots, samples)
    """
    if logger:
        logger.info(f"Loading scores from {scores_path}...")
    with open(scores_path, 'r') as f:
        scores = json.load(f)

    if logger:
        logger.info(f"Loading CoT chains from {cots_path}...")
    with open(cots_path, 'r') as f:
        cots_data = json.load(f)

    # Reconstruct CoTChain objects
    cots = [[CoTChain(**chain) for chain in sample_chains] for sample_chains in cots_data]

    samples = None
    if samples_path:
        if logger:
            logger.info(f"Loading samples from {samples_path}...")
        with open(samples_path, 'r') as f:
            samples = json.load(f)

    if logger:
        logger.info(f"Loaded {len(scores)} samples with scores and CoT chains")

    return scores, cots, samples


def prepare_evaluation_data(
    scores: list,
    cots: list,
    samples: list = None,
    device: str = "cpu"
):
    """
    Prepare data for evaluation.

    Args:
        scores: List of score dictionaries per sample
        cots: List of CoT chains per sample
        samples: Optional sample data with ground truth
        device: Device to load tensors to

    Returns:
        List of evaluation-ready samples
    """
    eval_data = []

    for idx, (sample_scores, sample_chains) in enumerate(zip(scores, cots)):
        # Get aggregation result (if available)
        aggregation_result = None
        chain_scores = sample_scores

        # Check if aggregation was already computed and stored
        if isinstance(sample_scores[-1], dict) and 'aggregation' in sample_scores[-1]:
            aggregation_result = sample_scores[-1]['aggregation']
            chain_scores = sample_scores[:-1]  # Exclude aggregation entry

        # Build chain data
        chains = []
        for chain, chain_score in zip(sample_chains, chain_scores):
            chain_data = {
                'answer': chain.final_answer,
                'final_answer': chain.final_answer,
                'confidence': chain_score.get('confidence', 0.0),
                # Embeddings would need to be loaded separately if needed
            }
            chains.append(chain_data)

        sample_data = {
            'sample_index': idx,
            'chains': chains,
            'true_answer': None,  # Will be filled from samples if available
        }

        # Add ground truth if available
        if samples and idx < len(samples):
            sample_info = samples[idx]
            sample_data['true_answer'] = sample_info.get('answer', None)
            sample_data['sample_id'] = sample_info.get('id', idx)

        # Add pre-computed aggregation if available
        if aggregation_result:
            sample_data['precomputed_aggregation'] = aggregation_result

        eval_data.append(sample_data)

    return eval_data


def compute_accuracy_from_aggregation(scores: list, samples: list = None):
    """
    Compute accuracy from pre-computed aggregation results in scores.

    Args:
        scores: List of score dictionaries (with aggregation results)
        samples: Optional samples with ground truth

    Returns:
        Dictionary with accuracy metrics
    """
    correct_count = 0
    total_count = 0
    aggregated_confidences = []

    for idx, sample_scores in enumerate(scores):
        # Check if aggregation result exists
        if not (isinstance(sample_scores[-1], dict) and 'aggregation' in sample_scores[-1]):
            continue

        agg_result = sample_scores[-1]['aggregation']
        aggregated_answer = agg_result['aggregated_answer']
        aggregated_confidence = agg_result['aggregated_confidence']

        # Get ground truth
        true_answer = None
        if samples and idx < len(samples):
            true_answer = samples[idx].get('answer', agg_result.get('true_answer'))
        else:
            true_answer = agg_result.get('true_answer')

        if true_answer is not None:
            is_correct = (aggregated_answer == true_answer)
            correct_count += is_correct
            total_count += 1
            aggregated_confidences.append(aggregated_confidence)

    accuracy = correct_count / total_count if total_count > 0 else 0.0

    return {
        'accuracy': accuracy,
        'correct': correct_count,
        'total': total_count,
        'avg_confidence': sum(aggregated_confidences) / len(aggregated_confidences) if aggregated_confidences else 0.0
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate WeightedFrequency aggregation method"
    )

    # Input files
    parser.add_argument(
        "--scores",
        type=str,
        required=True,
        help="Path to scores JSON file (from run_experiments_temp.py)"
    )
    parser.add_argument(
        "--cots",
        type=str,
        required=True,
        help="Path to CoT chains JSON file"
    )
    parser.add_argument(
        "--samples",
        type=str,
        default=None,
        help="Optional path to samples JSON with ground truth answers"
    )

    # Evaluation parameters
    parser.add_argument(
        "--aggregation_mode",
        type=str,
        default="exact",
        choices=["exact", "semantic"],
        help="Aggregation mode for evaluation"
    )
    parser.add_argument(
        "--similarity_threshold",
        type=float,
        default=0.85,
        help="Similarity threshold for semantic clustering"
    )
    parser.add_argument(
        "--use_precomputed",
        action="store_true",
        help="Use pre-computed aggregation results from scores file (if available)"
    )

    # Output
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save evaluation results"
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default=None,
        help="Path to log file"
    )

    # General
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for computation"
    )

    args = parser.parse_args()

    # Setup
    logger = setup_logger(args.log_file)
    logger.info("=" * 80)
    logger.info("WeightedFrequency Aggregation Evaluation")
    logger.info("=" * 80)

    # Load data
    scores, cots, samples = load_experiment_data(
        scores_path=args.scores,
        cots_path=args.cots,
        samples_path=args.samples,
        logger=logger
    )

    # Check if using pre-computed aggregation
    if args.use_precomputed:
        logger.info("Using pre-computed aggregation results from scores file")

        results = compute_accuracy_from_aggregation(scores, samples)

        logger.info("\n" + "=" * 80)
        logger.info("Evaluation Results (Pre-computed Aggregation)")
        logger.info("=" * 80)
        logger.info(f"Accuracy: {results['accuracy']:.4f} ({results['correct']}/{results['total']})")
        logger.info(f"Average confidence: {results['avg_confidence']:.4f}")

        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"\nSaved results to {args.output}")

    else:
        logger.info("Computing aggregation on-the-fly (requires confidence scorer)")
        logger.info("Note: This mode requires embeddings and a configured scorer.")
        logger.info("For pre-computed results, use --use_precomputed flag")

        # This would require setting up the full scorer and embeddings
        # For now, recommend using pre-computed results
        logger.warning(
            "On-the-fly evaluation requires embeddings. "
            "Please use --use_precomputed to evaluate pre-computed aggregation results."
        )

    logger.info("=" * 80)
    logger.info("Evaluation complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
