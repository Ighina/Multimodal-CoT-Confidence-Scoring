"""
Ablation study script to analyze component contributions.
"""

import argparse
import yaml
from pathlib import Path
import torch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.coherence import (
    InternalCoherenceMetric,
    CrossModalCoherenceMetric,
    ChainConfidenceScorer
)
from src.evaluation import ConfidenceEvaluator
from src.utils import ExperimentLogger, plot_ablation_results


def run_ablation_study(
    config: dict,
    test_data,
    exp_logger: ExperimentLogger
):
    """
    Run comprehensive ablation study.

    Args:
        config: Configuration dictionary
        test_data: Test dataset
        exp_logger: Experiment logger
    """
    exp_logger.logger.info("Starting ablation study...")

    # Define ablations to test
    ablations = [
        'internal_only',
        'cross_modal_only',
        'no_density',
        'clip_embeddings',
        'different_aggregations'
    ]

    results = {}

    for ablation in ablations:
        exp_logger.logger.info(f"\nRunning ablation: {ablation}")

        # Create ablated scorer
        scorer = create_ablated_scorer(config, ablation)

        # Create evaluator
        evaluator = ConfidenceEvaluator(scorer=scorer)

        # Evaluate
        ablation_results = evaluator.evaluate_scorer(
            test_data=test_data,
            show_progress=True
        )

        results[ablation] = ablation_results

        # Log results
        exp_logger.log_metrics(ablation_results, step=0)
        exp_logger.logger.info(f"AUC-ROC: {ablation_results['auc_roc']:.4f}")
        exp_logger.logger.info(f"ECE: {ablation_results['ece']:.4f}")

    # Compare all ablations
    exp_logger.logger.info("\n" + "="*50)
    exp_logger.logger.info("Ablation Study Summary")
    exp_logger.logger.info("="*50)

    for ablation, result in results.items():
        exp_logger.logger.info(f"\n{ablation}:")
        exp_logger.logger.info(f"  AUC-ROC: {result['auc_roc']:.4f}")
        exp_logger.logger.info(f"  AUC-PR:  {result['auc_pr']:.4f}")
        exp_logger.logger.info(f"  ECE:     {result['ece']:.4f}")

    # Visualize
    plot_ablation_results(
        results,
        metric='auc_roc',
        save_path=str(Path(config['output']['results_dir']) / 'ablation_auc_roc.png')
    )

    plot_ablation_results(
        results,
        metric='ece',
        save_path=str(Path(config['output']['results_dir']) / 'ablation_ece.png')
    )

    return results


def create_ablated_scorer(config: dict, ablation: str):
    """
    Create scorer with specified ablation.

    Args:
        config: Configuration
        ablation: Ablation type

    Returns:
        Ablated scorer
    """
    internal_metric = InternalCoherenceMetric(
        similarity_metric=config['coherence']['internal']['similarity_metric'],
        aggregation=config['coherence']['internal']['aggregation']
    )

    cross_modal_metric = CrossModalCoherenceMetric(
        similarity_metric=config['coherence']['cross_modal']['similarity_metric']
    )

    if ablation == 'internal_only':
        # Only internal coherence
        scorer = ChainConfidenceScorer(
            internal_metric=internal_metric,
            cross_modal_metric=cross_modal_metric,
            internal_weight=1.0,
            cross_modal_weight=0.0,
            density_weight=0.0
        )

    elif ablation == 'cross_modal_only':
        # Only cross-modal coherence
        scorer = ChainConfidenceScorer(
            internal_metric=internal_metric,
            cross_modal_metric=cross_modal_metric,
            internal_weight=0.0,
            cross_modal_weight=1.0,
            density_weight=0.0
        )

    elif ablation == 'no_density':
        # No density component
        scorer = ChainConfidenceScorer(
            internal_metric=internal_metric,
            cross_modal_metric=cross_modal_metric,
            internal_weight=0.6,
            cross_modal_weight=0.4,
            density_weight=0.0
        )

    elif ablation == 'clip_embeddings':
        # Use CLIP for text too (not sentence-transformers)
        scorer = ChainConfidenceScorer(
            internal_metric=internal_metric,
            cross_modal_metric=cross_modal_metric
        )

    elif ablation == 'different_aggregations':
        # Use min instead of mean
        internal_metric = InternalCoherenceMetric(
            similarity_metric='cosine',
            aggregation='min'
        )
        scorer = ChainConfidenceScorer(
            internal_metric=internal_metric,
            cross_modal_metric=cross_modal_metric
        )

    else:
        # Full model
        scorer = ChainConfidenceScorer(
            internal_metric=internal_metric,
            cross_modal_metric=cross_modal_metric
        )

    return scorer


def main():
    parser = argparse.ArgumentParser(description="Run ablation study")
    parser.add_argument('--config', type=str, default='config/config.yaml')
    parser.add_argument('--test_data', type=str, required=True)
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Setup logger
    exp_logger = ExperimentLogger(
        experiment_name='ablation_study',
        log_dir=config['logging']['log_dir']
    )

    # Load test data (placeholder)
    exp_logger.logger.info("Loading test data...")
    # test_data = load_test_data(args.test_data)
    test_data = []  # Placeholder

    # Run ablation study
    results = run_ablation_study(config, test_data, exp_logger)

    exp_logger.logger.info("Ablation study complete!")
    exp_logger.close()


if __name__ == '__main__':
    main()
